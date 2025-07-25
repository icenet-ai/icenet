import logging

import numpy as np
import pandas as pd
import xarray as xr

from download_toolbox.dataset import DatasetConfig
from icenet.plotting.utils import (get_seas_forecast_init_dates,
                                   filter_forecast_da_by_obs,
                                   get_seas_forecast_da,
                                   process_regions)


def compute_binary_accuracy(masks: object,
                            fc_da: object,
                            obs_da: object,
                            threshold: float) -> object:
    """
    Compute the binary class accuracy of a forecast,
    where we consider a binary class prediction of ice with SIC > 15%.
    In particular, we compute the mean percentage of correct
    classifications over the active grid cell area.

    Params:
        masks: an icenet Masks object
        fc_da: the forecasts given as an xarray.DataArray object
                  with time, xc, yc coordinates
        obs_da: the "ground truth" given as an xarray.DataArray object
                   with time, xc, yc coordinates
        threshold: the SIC threshold of interest (in percentage as a fraction),
                      i.e. threshold is between 0 and 1

    Returns:
        binary accuracy for forecast as xarray.DataArray object
    """
    threshold = 0.15 if threshold is None else threshold
    if (threshold < 0) or (threshold > 1):
        raise ValueError("threshold must be a float between 0 and 1")

    agcm = masks.get_active_cell_da(obs_da).rename({"time": "leadtime"}).fillna(0)
    agcm.coords['leadtime'] = obs_da.leadtime
    binary_obs_da = obs_da > threshold
    binary_fc_da = fc_da > threshold

    # compute binary accuracy metric
    binary_fc_da = (binary_fc_da == binary_obs_da).astype(np.float16).weighted(agcm)
    binacc_fc = (binary_fc_da.mean(dim=['yc', 'xc'], skipna=True) * 100)
    return binacc_fc


def compute_sea_ice_extent_error(masks: object,
                                 fc_da: object,
                                 obs_da: object,
                                 threshold: float) -> object:
    """
    Compute sea ice extent (SIE) error of a forecast, where SIE error is
    defined as the total area covered by grid cells with SIC > (threshold*100)%.

    :param masks: an icenet Masks object
    :param fc_da: the forecasts given as an xarray.DataArray object
                  with time, xc, yc coordinates
    :param obs_da: the "ground truth" given as an xarray.DataArray object
                   with time, xc, yc coordinates
    :param threshold: the SIC threshold of interest (in percentage as a fraction),
                      i.e. threshold is between 0 and 1

    :return: SIE error for forecast as xarray.DataArray object
    """
    grid_area_size = float(abs(fc_da.xc[1] - fc_da.xc[0])) / 1000.
    threshold = 0.15 if threshold is None else threshold
    if (threshold < 0) or (threshold > 1):
        raise ValueError("threshold must be a float between 0 and 1")

    # obtain mask
    agcm = masks.get_active_cell_da(obs_da).rename({"time": "leadtime"}).fillna(0)
    agcm.coords['leadtime'] = obs_da.leadtime

    # binary for observed (i.e. truth)
    binary_obs_da = obs_da > threshold
    binary_obs_weighted_da = binary_obs_da.astype(int).weighted(agcm)

    # binary for forecast
    binary_fc_da = fc_da > threshold
    binary_fc_weighted_da = binary_fc_da.astype(int).weighted(agcm)

    # sie error
    forecast_sie_error = (binary_fc_weighted_da.sum(['xc', 'yc']) -
                          binary_obs_weighted_da.sum(['xc', 'yc'])) * (
                              grid_area_size**2)

    return forecast_sie_error


def compute_metrics(metrics: object,
                    masks: object,
                    fc_da: object,
                    obs_da: object) -> object:
    """
    Computes metrics based on SIC error which are passed in as a list of strings.
    Returns a dictionary where the keys are the metrics,
    and the values are the computed metrics.

    :param metrics: a list of strings
    :param masks: an icenet Masks object
    :param fc_da: an xarray.DataArray object with time, xc, yc coordinates
    :param obs_da: an xarray.DataArray object with time, xc, yc coordinates

    :return: dictionary with keys as metric names and values as
             xarray.DataArray's storing the computed metrics for each forecast
    """
    # check requested metrics have been implemented
    implemented_metrics = ["mae", "mse", "rmse"]
    for metric in metrics:
        if metric not in implemented_metrics:
            raise NotImplementedError(
                f"{metric} metric has not been implemented. "
                f"Please only choose out of {implemented_metrics}.")

    # obtain mask
    agcm = masks.get_active_cell_da(obs_da)

    metric_dict = {}
    # compute raw error
    err_da = (fc_da - obs_da) * 100
    if "mae" in metrics:
        # compute absolute SIC errors
        abs_err_da = da.fabs(err_da)
        abs_weighted_da = abs_err_da.weighted(~agcm)
    if "mse" in metrics or "rmse" in metrics:
        # compute squared SIC errors
        square_err_da = err_da**2
        square_weighted_da = square_err_da.weighted(~agcm)

    for metric in metrics:
        if metric == "mae":
            metric_dict[metric] = abs_weighted_da.mean(dim=['yc', 'xc'])
        elif metric == "mse":
            if "mse" not in metric_dict.keys():
                # might've already been computed if RMSE came first
                metric_dict["mse"] = square_weighted_da.mean(dim=['yc', 'xc'])
        elif metric == "rmse":
            if "mse" not in metric_dict.keys():
                # check if MSE already been computed
                metric_dict["mse"] = square_weighted_da.mean(dim=['yc', 'xc'])
            metric_dict[metric] = da.sqrt(metric_dict["mse"])

    # only return metrics requested (might've computed MSE when computing RMSE)
    return {k: metric_dict[k] for k in metrics}


def compute_metric_as_dataframe(metric: object,
                                masks: object,
                                init_date: object,
                                fc_da: object,
                                obs_da: object,
                                obs_ds_config: object,
                                **kwargs) -> pd.DataFrame:
    """
    Computes a metric for each leadtime in a forecast and stores the
    results in a pandas dataframe with columns 'date' (which is the
    initialisation date passed in), 'leadtime' and the metric name(s).

    :param metric: string, or list of strings, specifying which metric(s) to compute
    :param masks: an icenet Masks object
    :param init_date: forecast initialisation date which gets
                      added to pandas dataframe (as string, or datetime object)
    :param fc_da: an xarray.DataArray object with time, xc, yc coordinates
    :param obs_da: an xarray.DataArray object with time, xc, yc coordinates
    :param obs_ds_config:
    :param kwargs: any keyword arguments that are required for the computation
                   of the metric, e.g. 'threshold' for SIE error and binary accuracy
                   metrics

    :return: computed metric in a pandas dataframe with columns 'date',
             'leadtime' and 'met' for each metric, met, in metric
    """
    if isinstance(metric, str):
        metric = [metric]
    metric_dict = {}
    for met in metric:
        if met in ["mae", "mse", "rmse"]:
            metric_dict[met] = compute_metrics(metrics=[met],
                                               masks=masks,
                                               fc_da=fc_da,
                                               obs_da=obs_da)[met].values
        elif met == "binacc":
            if "threshold" not in kwargs.keys():
                raise KeyError(
                    "if met = 'binacc', must pass in argument for threshold")
            metric_dict[met] = compute_binary_accuracy(
                masks=masks,
                fc_da=fc_da,
                obs_da=obs_da,
                threshold=kwargs["threshold"]).values
        elif met == "sie":
            if "threshold" not in kwargs.keys():
                raise KeyError(
                    "if met = 'sie', must pass in argument for threshold")
            metric_dict[met] = compute_sea_ice_extent_error(
                masks=masks,
                fc_da=fc_da,
                obs_da=obs_da,
                threshold=kwargs["threshold"]).values
        else:
            raise NotImplementedError(f"{met} is not implemented")

    # create dataframe from metric_dict
    metric_df = pd.DataFrame(metric_dict)

    init_date = pd.to_datetime(init_date)
    # compute day of year after first converting year to a non-leap year
    # avoids issue where 2016-03-31 is different to 2015-03-31
    if init_date.strftime("%m-%d") == "02-29":
        # if date is 29th Feb on a leap year, use dayofyear 59
        # (corresponds to 28th Feb in non-leap years)
        dayofyear = 59
    else:
        dayofyear = init_date.replace(year=2001).dayofyear
    month = init_date.month
    # get target dates
    leadtime = list(range(1, len(metric_df.index) + 1, 1))
    leadtime_attr = "{}s".format(obs_ds_config.frequency.attribute)
    target_date = pd.Series([init_date + relativedelta(**{leadtime_attr: d}) for d in leadtime])

    # obtain day of year using same method above to avoid any leap-year issues
    target_dayofyear = pd.Series([
        59 if d.strftime("%m-%d") == "02-29" else d.replace(
            year=2001).dayofyear for d in target_date
    ])
    target_month = target_date.dt.month
    return pd.concat([
        pd.DataFrame({
            "date": init_date,
            "dayofyear": dayofyear,
            "month": month,
            "target_date": target_date,
            "target_dayofyear": target_dayofyear,
            "target_month": target_month,
            "leadtime": leadtime
        }), metric_df
    ], axis=1)


def compute_metrics_leadtime_avg(metric: str,
                                 forecast_file: str,
                                 ds_config: DatasetConfig,
                                 ecmwf: bool,
                                 data_path: str,
                                 bias_correct: bool = False,
                                 region: tuple = None,
                                 **kwargs) -> object:
    """
    Given forecast file, for each initialisation date in the xarrray.DataArray
    we compute the metric for each leadtime and store the results
    in a pandas dataframe with columns 'date' (specifying the initialisation date),
    'leadtime' and the metric name. This pandas dataframe can then be used
    to average over leadtime to obtain leadtime averaged metrics.

    # TODO: ensure able to calculate metrics across different temporal domains

    :param metric: string specifying which metric to compute
    :param forecast_file: string specifying a path to a .nc file
    :param ds_config: ground truth dataset config appropriate to the forecast file
    :param ecmwf: bool to indicate whether or not to compare
                  with ECMWF SEAS forecast. If True, will only average
                  over forecasts where the initialisation dates between IceNet
                  and SEAS are the same
    :param data_path: string specifying where to save the metrics dataframe.
                      If None, dataframe is not saved
    :param bias_correct: bool to indicate whether or not to
                         perform a bias correction on SEAS forecast,
                         by default False. Ignored if ecmwf=False
    :param region: region to zoom in to
    :param kwargs: any keyword arguments that are required for the computation
                   of the metric, e.g. 'threshold' for SIE error and binary accuracy
                   metrics

    :return: pandas dataframe with columns 'date', 'leadtime' and the metric name.
    """
    # open forecast file
    fc_ds = xr.open_dataset(forecast_file)
    masks = get_implementation(fc_ds.attrs["icenet_mask_implementation"])(ds_config)

    if ecmwf:
        # find out what dates cross over with the SEAS5 predictions
        (fc_start_date, fc_end_date) = (fc_ds.time.values.min(),
                                        fc_ds.time.values.max())
        dates = get_seas_forecast_init_dates(fc_ds.attrs["hemisphere_string"])
        dates = dates[(dates > fc_start_date) & (dates <= fc_end_date)]
        times = [x for x in fc_ds.time.values if x in dates]
        fc_ds = fc_ds.sel(time=times)

    logging.info(f"Computing {metric} for {len(fc_ds.time.values)} forecasts")
    # obtain metric for each leadtime at each initialised date in the forecast file

    fc_metrics_list = []
    if ecmwf:
        seas_metrics_list = []
    for time in fc_ds.time.values:
        # obtain forecast
        fc = fc_ds.sel(time=slice(time, time))["sic_mean"]
        obs = ds_config.get_dataset(var_names=["siconca"]).siconca
        obs = obs.sel(time=slice(
            pd.to_datetime(time),
            pd.to_datetime(time) + relativedelta(**{
                "{}s".format(ds_config.frequency.attribute): int(fc.leadtime.max())})
        ))
        fc = filter_forecast_da_by_obs(fc, obs, ds_config.frequency)

        if ecmwf:
            # obtain SEAS forecast
            seas = get_seas_forecast_da(obs_ds_config=ds_config,
                                        date=pd.to_datetime(time),
                                        bias_correct=bias_correct)
            # remove the initialisation date from dataarray
            seas = seas.assign_coords(dict(xc=seas.xc / 1e3, yc=seas.yc / 1e3))
            seas = seas.isel(time=slice(1, None))
        else:
            seas = None

        if region is not None:
            seas, fc, obs, masks = process_regions(region,
                                                   [seas, fc, obs, masks])

        # compute metrics
        fc_metrics_list.append(
            compute_metric_as_dataframe(metric=metric,
                                        masks=masks,
                                        init_date=time,
                                        fc_da=fc,
                                        obs_da=obs,
                                        obs_ds_config=ds_config,
                                        **kwargs))
        if seas is not None:
            seas_metrics_list.append(
                compute_metric_as_dataframe(metric=metric,
                                            masks=masks,
                                            init_date=time,
                                            fc_da=seas,
                                            obs_da=obs,
                                            obs_ds_config=ds_config,
                                            **kwargs))

    # groupby the leadtime and compute the mean average of the metric
    fc_metric_df = pd.concat(fc_metrics_list)
    fc_metric_df["forecast_name"] = "IceNet"
    if ecmwf:
        seas_metric_df = pd.concat(seas_metrics_list)
        seas_metric_df["forecast_name"] = "SEAS"
        fc_metric_df = pd.concat([fc_metric_df, seas_metric_df])

    if data_path is not None:
        logging.info(f"Saving the metric dataframe in {data_path}")
        try:
            fc_metric_df.to_csv(data_path)
        except OSError:
            # don't break if not successful, still return dataframe
            logging.info(
                "Save not successful! Make sure the data_path directory exists"
            )

    return fc_metric_df.reset_index(drop=True)
