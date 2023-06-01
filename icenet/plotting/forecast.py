import argparse
import datetime as dt
import logging
import os
import sys

from datetime import timedelta

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_pdf import PdfPages

import seaborn as sns

import numpy as np
import pandas as pd
import dask.array as da
import xarray as xr

from icenet import __version__ as icenet_version
from icenet.data.cli import date_arg
from icenet.data.sic.mask import Masks
from icenet.plotting.utils import (
    filter_ds_by_obs,
    get_forecast_ds,
    get_obs_da,
    get_seas_forecast_da,
    get_seas_forecast_init_dates,
    show_img,
    get_plot_axes,
    process_probes,
    process_regions
)
from icenet.plotting.video import xarray_to_video


def parse_location_or_region(argument: str):
    separator = ','
    # Allow ValueError to propagate if not given sequence of integers
    return tuple(int(s) for s in argument.split(separator))


def location_arg(argument: str):
    try:
        x, y = parse_location_or_region(argument)
        return (x, y)
    except ValueError:
        argparse.ArgumentTypeError(
            "Expected a location (pair of integers separated by a comma)"
        )


def region_arg(argument: str):
    """type handler for region arguments with argparse

    :param argument:

    :return:
    """
    try:
        x1, y1, x2, y2 = parse_location_or_region(argument)

        assert x2 > x1 and y2 > y1, "Region is not valid"
        return x1, y1, x2, y2
    except TypeError:
        raise argparse.ArgumentTypeError(
            "Region argument must be list of four integers")


def compute_binary_accuracy(masks: object,
                            fc_da: object,
                            obs_da: object,
                            threshold: float) -> object:
    """
    Compute the binary class accuracy of a forecast,
    where we consider a binary class prediction of ice with SIC > 15%.
    In particular, we compute the mean percentage of correct
    classifications over the active grid cell area.

    :param masks: an icenet Masks object
    :param fc_da: the forecasts given as an xarray.DataArray object
                  with time, xc, yc coordinates
    :param obs_da: the "ground truth" given as an xarray.DataArray object
                   with time, xc, yc coordinates
    :param threshold: the SIC threshold of interest (in percentage as a fraction),
                      i.e. threshold is between 0 and 1

    :return: binary accuracy for forecast as xarray.DataArray object
    """
    threshold = 0.15 if threshold is None else threshold
    if (threshold < 0) or (threshold > 1):
        raise ValueError("threshold must be a float between 0 and 1")

    # obtain mask
    agcm = masks.get_active_cell_da(obs_da)

    # binary for observed (i.e. truth)
    binary_obs_da = obs_da > threshold

    # binary for forecast
    binary_fc_da = fc_da > threshold

    # compute binary accuracy metric
    binary_fc_da = (binary_fc_da == binary_obs_da). \
        astype(np.float16).weighted(agcm)
    binacc_fc = (binary_fc_da.mean(dim=['yc', 'xc']) * 100)

    return binacc_fc


def plot_binary_accuracy(masks: object,
                         fc_da: object,
                         cmp_da: object,
                         obs_da: object,
                         output_path: object,
                         threshold: float = 0.15) -> object:
    """
    Compute and plot the binary class accuracy of a forecast,
    where we consider a binary class prediction of ice with SIC > 15%.
    In particular, we compute the mean percentage of correct
    classifications over the active grid cell area.
    
    :param masks: an icenet Masks object
    :param fc_da: the forecasts given as an xarray.DataArray object 
                  with time, xc, yc coordinates
    :param cmp_da: a comparison forecast / sea ice data given as an 
                   xarray.DataArray object with time, xc, yc coordinates.
                   If None, will ignore plotting a comparison forecast
    :param obs_da: the "ground truth" given as an xarray.DataArray object
                   with time, xc, yc coordinates
    :param output_path: string specifying the path to store the plot
    :param threshold: the SIC threshold of interest (in percentage as a fraction),
                      i.e. threshold is between 0 and 1
    
    :return: tuple of (binary accuracy for forecast (fc_da), 
                       binary accuracy for comparison (cmp_da))
    """
    binacc_fc = compute_binary_accuracy(masks=masks,
                                        fc_da=fc_da,
                                        obs_da=obs_da,
                                        threshold=threshold)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title(f"Binary accuracy comparison (threshold SIC = {threshold*100}%)")
    ax.plot(binacc_fc.time, binacc_fc.values, label="IceNet")

    if cmp_da is not None:
        binacc_cmp = compute_binary_accuracy(masks=masks,
                                             fc_da=cmp_da,
                                             obs_da=obs_da,
                                             threshold=threshold)
        ax.plot(binacc_cmp.time, binacc_cmp.values, label="SEAS")
    else:
        binacc_cmp = None

    ax.set_ylabel("Binary accuracy (%)")
    ax.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_minor_locator(mdates.DayLocator())
    ax.set_xlabel("Date")
    ax.legend(loc='lower right')

    output_path = os.path.join("plot", "binacc.png") \
        if not output_path else output_path
    logging.info(f"Saving to {output_path}")
    plt.savefig(output_path)

    return binacc_fc, binacc_cmp


def compute_sea_ice_extent_error(masks: object,
                                 fc_da: object,
                                 obs_da: object,
                                 grid_area_size: int,
                                 threshold: float) -> object:
    """
    Compute sea ice extent (SIE) error of a forecast, where SIE error is
    defined as the total area covered by grid cells with SIC > (threshold*100)%.

    :param masks: an icenet Masks object
    :param fc_da: the forecasts given as an xarray.DataArray object 
                  with time, xc, yc coordinates
    :param obs_da: the "ground truth" given as an xarray.DataArray object
                   with time, xc, yc coordinates
    :param grid_area_size: the length of the sides of the grid (in km),
                           by default set to 25 (so area of grid is 25*25)
    :param threshold: the SIC threshold of interest (in percentage as a fraction),
                      i.e. threshold is between 0 and 1

    :return: SIE error for forecast as xarray.DataArray object
    """
    grid_area_size = 25 if grid_area_size is None else grid_area_size
    threshold = 0.15 if threshold is None else threshold
    if (threshold < 0) or (threshold > 1):
        raise ValueError("threshold must be a float between 0 and 1")
    
    # obtain mask
    agcm = masks.get_active_cell_da(obs_da)
    
    # binary for observed (i.e. truth)
    binary_obs_da = obs_da > threshold
    binary_obs_weighted_da = binary_obs_da.astype(int).weighted(agcm)

    # binary for forecast
    binary_fc_da = fc_da > threshold
    binary_fc_weighted_da = binary_fc_da.astype(int).weighted(agcm)
    
    # sie error
    forecast_sie_error = (
        binary_fc_weighted_da.sum(['xc', 'yc']) -
        binary_obs_weighted_da.sum(['xc', 'yc'])
    ) * (grid_area_size**2)
    
    return forecast_sie_error


def plot_sea_ice_extent_error(masks: object,
                              fc_da: object,
                              cmp_da: object,
                              obs_da: object,
                              output_path: object,
                              grid_area_size: int = 25,
                              threshold: float = 0.15) -> object:
    """
    Compute and plot sea ice extent (SIE) error of a forecast, where SIE error is
    defined as the total area covered by grid cells with SIC > (threshold*100)%.
    
    :param masks: an icenet Masks object
    :param fc_da: the forecasts given as an xarray.DataArray object 
                  with time, xc, yc coordinates
    :param cmp_da: a comparison forecast / sea ice data given as an 
                   xarray.DataArray object with time, xc, yc coordinates.
                   If None, will ignore plotting a comparison forecast
    :param obs_da: the "ground truth" given as an xarray.DataArray object
                   with time, xc, yc coordinates
    :param output_path: string specifying the path to store the plot
    :param grid_area_size: the length of the sides of the grid (in km),
                           by default set to 25 (so area of grid is 25*25)
    :param threshold: the SIC threshold of interest (in percentage as a fraction),
                      i.e. threshold is between 0 and 1
    
    :return: tuple of (SIE error for forecast (fc_da), SIE error for comparison (cmp_da))
    """
    forecast_sie_error = compute_sea_ice_extent_error(masks=masks,
                                                      fc_da=fc_da,
                                                      obs_da=obs_da,
                                                      grid_area_size=grid_area_size,
                                                      threshold=threshold)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title(f"SIE error comparison ({grid_area_size} km grid resolution) "
                 f"(threshold SIC = {threshold*100}%)")
    ax.plot(forecast_sie_error.time, forecast_sie_error.values, label="IceNet")

    if cmp_da is not None:
        cmp_sie_error = compute_sea_ice_extent_error(masks=masks,
                                                     fc_da=cmp_da,
                                                     obs_da=obs_da,
                                                     grid_area_size=grid_area_size,
                                                     threshold=threshold)
        ax.plot(cmp_sie_error.time, cmp_sie_error.values, label="SEAS")
    else:
        cmp_sie_error = None

    ax.set_ylabel("Sea ice extent error (km)")
    ax.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_minor_locator(mdates.DayLocator())
    ax.set_xlabel("Date")
    ax.legend(loc='lower right')

    output_path = os.path.join("plot", "sie_error.png") \
        if not output_path else output_path
    logging.info(f"Saving to {output_path}")
    plt.savefig(output_path)

    return forecast_sie_error, cmp_sie_error


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
            raise NotImplementedError(f"{metric} metric has not been implemented. "
                                      f"Please only choose out of {implemented_metrics}.")
    
    # obtain mask
    mask_da = masks.get_active_cell_da(obs_da)
    
    metric_dict = {}
    # compute raw error
    err_da = (fc_da-obs_da)*100
    if "mae" in metrics:
        # compute absolute SIC errors
        abs_err_da = da.fabs(err_da)
        abs_weighted_da = abs_err_da.weighted(mask_da)
    if "mse" in metrics or "rmse" in metrics:
        # compute squared SIC errors
        square_err_da = err_da**2
        square_weighted_da = square_err_da.weighted(mask_da)
        
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
    
    
def plot_metrics(metrics: object,
                 masks: object,
                 fc_da: object,
                 cmp_da: object,
                 obs_da: object,
                 output_path: object,
                 separate: bool = False) -> object:
    """
    Computes metrics which are passed in as a list of strings,
    and plots them for each forecast.
    Returns a dictionary where the keys are the metrics,
    and the values are the computed metrics.

    :param metrics: a list of strings
    :param masks: an icenet Masks object
    :param fc_da: an xarray.DataArray object with time, xc, yc coordinates
    :param cmp_da: a comparison forecast / sea ice data given as an 
                   xarray.DataArray object with time, xc, yc coordinates.
                   If None, will ignore plotting a comparison forecast
    :param obs_da: an xarray.DataArray object with time, xc, yc coordinates
    :param output_path: string specifying the path to store the plot(s).
                        If separate=True, this should be a directory
    :param separate: bool specifying whether there is a plot created for
                     each metric (True) or not (False), default is False
    
    :return: dictionary with keys as metric names and values as 
             xarray.DataArray's storing the computed metrics for each forecast
    """
    # compute metrics
    fc_metric_dict = compute_metrics(metrics=metrics,
                                     masks=masks,
                                     fc_da=fc_da,
                                     obs_da=obs_da)
    if cmp_da is not None:
        cmp_metric_dict = compute_metrics(metrics=metrics,
                                          masks=masks,
                                          fc_da=cmp_da,
                                          obs_da=obs_da)
    else:
        cmp_metric_dict = None
    
    if separate:
        # produce separate plots for each metric
        for metric in metrics:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.set_title(f"{metric.upper()} comparison")
            ax.plot(fc_metric_dict[metric].time,
                    fc_metric_dict[metric].values,
                    label="IceNet")
            if cmp_metric_dict is not None:
                ax.plot(cmp_metric_dict[metric].time,
                        cmp_metric_dict[metric].values,
                        label="SEAS")
                
            ax.xaxis.set_major_formatter(
                mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.xaxis.set_minor_locator(mdates.DayLocator())
            ax.legend(loc='lower right')
            
            outpath = os.path.join("plot", f"{metric}.png") \
                if not output_path else os.path.join(output_path, f"{metric}.png")
            logging.info(f"Saving to {outpath}")
            plt.savefig(outpath)
    else:
        # produce one plot for all metrics
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_title("Metric comparison")
        for metric in metrics:
            ax.plot(fc_metric_dict[metric].time,
                    fc_metric_dict[metric].values,
                    label=f"IceNet {metric.upper()}")
            if cmp_metric_dict is not None:
                ax.plot(cmp_metric_dict[metric].time,
                        cmp_metric_dict[metric].values,
                        label=f"SEAS {metric.upper()}",
                        linestyle="dotted")
        
        ax.set_ylabel("SIC (%)")
        ax.xaxis.set_major_formatter(
            mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_minor_locator(mdates.DayLocator())
        ax.set_xlabel("Date")
        ax.legend(loc='lower right')
        
        output_path = os.path.join("plot", "metrics.png") \
            if not output_path else output_path
        logging.info(f"Saving to {output_path}")
        plt.savefig(output_path)
    
    return fc_metric_dict, cmp_metric_dict


def compute_metric_as_dataframe(metric: object,
                                masks: object,
                                init_date: object,
                                fc_da: object,
                                obs_da: object,
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
    :param kwargs: any keyword arguments that are required for the computation
                   of the metric, e.g. 'threshold' for SIE error and binary accuracy
                   metrics, or 'grid_area_size' for SIE error metric
    
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
                raise KeyError("if met = 'binacc', must pass in argument for threshold")
            metric_dict[met] = compute_binary_accuracy(masks=masks,
                                                       fc_da=fc_da,
                                                       obs_da=obs_da,
                                                       threshold=kwargs["threshold"]).values
        elif met == "sie":
            if "grid_area_size" not in kwargs.keys():
                raise KeyError("if met = 'sie', must pass in argument for grid_area_size")
            if "threshold" not in kwargs.keys():
                raise KeyError("if met = 'sie', must pass in argument for threshold")
            metric_dict[met] = compute_sea_ice_extent_error(masks=masks,
                                                            fc_da=fc_da,
                                                            obs_da=obs_da,
                                                            grid_area_size=kwargs["grid_area_size"],
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
    leadtime = list(range(1, len(metric_df.index)+1, 1))
    target_date = pd.Series([init_date + timedelta(days=d) for d in leadtime])
    target_dayofyear = target_date.dt.dayofyear
    # obtain day of year using same method above to avoid any leap-year issues
    target_dayofyear = pd.Series([59 if d.strftime("%m-%d")=="02-29"
                                  else d.replace(year=2001).dayofyear
                                  for d in target_date])
    target_month = target_date.dt.month
    return pd.concat([pd.DataFrame({"date": init_date,
                                    "dayofyear": dayofyear,
                                    "month": month,
                                    "target_date": target_date,
                                    "target_dayofyear": target_dayofyear,
                                    "target_month": target_month,
                                    "leadtime": leadtime}),
                      metric_df],
                      axis=1)


def compute_metrics_leadtime_avg(metric: str,
                                 masks: object,
                                 hemisphere: str,
                                 forecast_file: str,
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
    
    :param metric: string specifying which metric to compute
    :param masks: an icenet Masks object
    :param hemisphere: string, typically either 'north' or 'south'
    :param forecast_file: string specifying a path to a .nc file
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
                   metrics, or 'grid_area_size' for SIE error metric
    
    :return: pandas dataframe with columns 'date', 'leadtime' and the metric name.
    """
    # open forecast file
    fc_ds = xr.open_dataset(forecast_file)
    
    if ecmwf:
        # find out what dates cross over with the SEAS5 predictions
        (fc_start_date, fc_end_date) = (fc_ds.time.values.min(), fc_ds.time.values.max())
        dates = get_seas_forecast_init_dates(hemisphere)
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
        obs = get_obs_da(hemisphere=hemisphere,
                         start_date=pd.to_datetime(time) + timedelta(days=1),
                         end_date=pd.to_datetime(time) + timedelta(days=int(fc.leadtime.max())))
        fc = filter_ds_by_obs(fc, obs, time)
        
        if ecmwf:
            # obtain SEAS forecast
            seas = get_seas_forecast_da(hemisphere=hemisphere,
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
        fc_metrics_list.append(compute_metric_as_dataframe(metric=metric,
                                                           masks=masks,
                                                           init_date=time,
                                                           fc_da=fc,
                                                           obs_da=obs,
                                                           **kwargs))
        if seas is not None:
            seas_metrics_list.append(compute_metric_as_dataframe(metric=metric,
                                                                 masks=masks,
                                                                 init_date=time,
                                                                 fc_da=seas,
                                                                 obs_da=obs,
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
            logging.info("Save not successful! Make sure the data_path directory exists")
        
    return fc_metric_df.reset_index(drop=True)


def _parse_day_of_year(dayofyear: int, leapyear: bool = False) -> int:
    """
    Private function which takes in a day of year (integer or float) and returns
    the integer day of year. Useful for ensuring consistency over leap years,
    as dates after March could have different day of years due to leap years.
    For example, 01/03/00 is 60th day in 2000 but 01/03/01 is the 59th day in 2001.
    
    :param dayofyear: integer as int or float type
    :param leapyear: bool to indicate if we want to convert a leapyear dayofyear to
                     non-leapyear
    
    :return: int dayofyear
    """
    if leapyear:
        return (pd.Timestamp("2000-01-01") + timedelta(days=int(dayofyear) - 1)).strftime("%m-%d")
    else:
        return (pd.Timestamp("2001-01-01") + timedelta(days=int(dayofyear) - 1)).strftime("%m-%d")


def _heatmap_ylabels(metrics_df: pd.DataFrame, average_over: str, groupby_col: str) -> object:
    """
    Private function to return the labels for the y-axis in heatmap plots.

    :param metrics_df: pandas dataframe with columns 'date', 'leadtime' and the metric name
    :param average_over: string to specify how to average the metrics.
                         If average_over="all", averages over all possible
                         forecasts and produces line plot.
                         If average_over="month" or "day", averages
                         over the month or day respectively and produces
                         heat map plot.
    :param groupby_col: string to specify how we are grouping the data.
                        If average_over="all", this is typically "dayofyear"
                        or "target_dayofyear".
                        If average_over="month" or "day", this is typically
                        "month" or "target_month".
    
    :return: list of labels for the y-axis
    """
    if average_over == "day":
        # only add labels to the start, end dates
        # and any days that represent the start of months
        days_of_interest = np.array([metrics_df[groupby_col].min(),
                                     1, 32, 60, 91, 121, 152,
                                     182, 213, 244, 274, 305, 335,
                                     metrics_df[groupby_col].max()])
        labels = [_parse_day_of_year(day)
                    if day in days_of_interest else ""
                    for day in sorted(metrics_df[groupby_col].unique())]
    else:
        # find out what months have been plotted and add their names
        month_names = np.array(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                "Jul", "Aug", "Sept", "Oct", "Nov", "Dec"])
        labels = [month_names[month-1]
                    for month in sorted(metrics_df[groupby_col].unique())]

    return labels


def standard_deviation_heatmap(metric: str,
                               model_name: str,
                               metrics_df: pd.DataFrame,
                               average_over: str,
                               target_date_avg: bool = False,
                               output_path: str = None,
                               fc_std_metric: pd.DataFrame = None,
                               vmax: float = None,
                               **kwargs) -> object:
    """
    Produces a heatmap plot of the leadtime standard deviation of a metric.

    :param metric: string specifying which metric to compute
    :param model_name: string specifying the name of the model or forecast
    :param metrics_df: pandas dataframe with columns 'date', 'leadtime' and the metric name
    :param average_over: string to specify how to average the metrics.
                         If average_over="all", averages over all possible
                         forecasts and produces line plot.
                         If average_over="month" or "day", averages
                         over the month or day respectively and produces
                         heat map plot.
    :param target_date_avg: bool to indiciate whether or not to average over
                            the target date of forecast (as opposed to averaging
                            over the initialisation date), by default False
    :param output_path: string specifying the path to store the plot, default None
    :param fc_std_metric: pandas dataframe of the standard deviation of the metric over leadtime.
                          If None, this will be computed here
    :param vmax: float specifying maximum to anchor the colourmap. If None,
                 this is inferred from the data
    
    :return: dataframe of the standard deviation of the metric over leadtime
    """
    logging.info(f"Creating standard deviation over leadtime plot for "
                 f"{metric} metric for {model_name} forecasts")
    if average_over == "day":
        groupby_col = "dayofyear"
    else:
        groupby_col = "month"
    if target_date_avg:
        groupby_col = "target_" + groupby_col
    
    if fc_std_metric is None:
        # compute standard deviation of metric
        fc_std_metric = metrics_df.groupby([groupby_col, "leadtime"]).std(numeric_only=True).\
            reset_index().pivot(index=groupby_col, columns="leadtime", values=metric).\
                sort_values(groupby_col, ascending=True)
    n_forecast_days = fc_std_metric.shape[1]
    
    # set ylabel (if average_over == "all"), or legend label (otherwise)
    if metric in ["mae", "mse", "rmse"]:
        ylabel = f"SIC {metric.upper()} (%)"
    elif metric == "binacc":
        ylabel = f"Binary accuracy (%)"
    elif metric == "sie":
        ylabel = f"SIE error (km)"
        
    # plot heatmap of standard deviation for IceNet
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(data=fc_std_metric,
                ax=ax,
                vmin=0,
                vmax=vmax,
                cmap="mako_r",
                cbar_kws=dict(label=f"Standard deviation of {ylabel}"))
    
    # y-axis
    labels = _heatmap_ylabels(metrics_df=metrics_df,
                              average_over=average_over,
                              groupby_col=groupby_col)
    ax.set_yticks(np.arange(len(metrics_df[groupby_col].unique()))+0.5)
    ax.set_yticklabels(labels)
    plt.yticks(rotation=0)
    if target_date_avg:
        ax.set_ylabel("Target date of forecast")
    else:
        ax.set_ylabel("Initialisation date of forecast")
    
    # x-axis
    ax.set_xticks(np.arange(30, n_forecast_days, 30))
    ax.set_xticklabels(np.arange(30, n_forecast_days, 30))
    plt.xticks(rotation=0)
    ax.set_xlabel("Lead time (days)")
    
    # add plot title
    (start_date, end_date) = (metrics_df["date"].min().strftime('%d/%m/%Y'),
                              metrics_df["date"].max().strftime('%d/%m/%Y'))
    time_coverage = "\nStandard deviation over a minimum of " + \
        f"{round((metrics_df[groupby_col].value_counts()/n_forecast_days).min())} " + \
        f"forecasts between {start_date} - {end_date}"
    if metric in ["mae", "mse", "rmse"]:
        title = f"{metric.upper()} comparison"
    elif metric == "binacc":
        title = f"Binary accuracy comparison (threshold SIC = {kwargs['threshold'] * 100}%)"
    elif metric == "sie":
        title = f"SIE error comparison ({kwargs['grid_area_size']} km grid resolution, " +\
            f"threshold SIC = {kwargs['threshold'] * 100}%)"
    ax.set_title(title + f" ({model_name})" + time_coverage)

    # save plot
    targ = "target" if target_date_avg and average_over != "all" else "init"
    filename = f"leadtime_averaged_{targ}_{average_over}_{metric}_{model_name}_std.png"
    output_path = os.path.join("plot", filename) \
        if not output_path else output_path
    logging.info(f"Saving to {output_path}")
    plt.savefig(output_path)
    
    return fc_std_metric


def plot_metrics_leadtime_avg(metric: str,
                              masks: object,
                              hemisphere: str,
                              forecast_file: str,
                              ecmwf: bool,
                              output_path: str,
                              average_over: str,
                              plot_std: bool = False,
                              std_mult: float = 1.0,
                              data_path: str = None,
                              target_date_avg: bool = False,
                              bias_correct: bool = False,
                              region: tuple = None,
                              **kwargs) -> object:
    """
    Plots leadtime averaged metrics either using all the forecasts
    in the forecast file, or averaging them over by month or day.
    
    :param metric: string specifying which metric to compute
    :param masks: an icenet Masks object
    :param hemisphere: string, typically either 'north' or 'south'
    :param forecast_file: a path to a .nc file
    :param ecmwf: bool to indicate whether or not to compare
                  with ECMWF SEAS forecast. If True, will only average
                  over forecasts where the initialisation dates between IceNet
                  and SEAS are the same
    :param output_path: string specifying the path to store the plot
    :param average_over: string to specify how to average the metrics.
                         If average_over="all", averages over all possible
                         forecasts and produces line plot.
                         If average_over="month" or "day", averages
                         over the month or day respectively and produces
                         heat map plot.
    :param target_date_avg: bool to indiciate whether or not to average over
                            the target date of forecast (as opposed to averaging
                            over the initialisation date), by default False
    :param data_path: string specifying a CSV file where metrics dataframe
                      could be loaded from. If loading in the dataframe is
                      not possible, it will compute the metrics dataframe
                      and try to save the dataframe
    :param bias_correct: bool to indicate whether or not to
                         perform a bias correction on SEAS forecast,
                         by default False. Ignored if ecmwf=False
    :param region: region to zoom in to
    :param kwargs: any keyword arguments that are required for the computation
                   of the metric, e.g. 'threshold' for SIE error and binary accuracy
                   metrics, or 'grid_area_size' for SIE error metric
    
    :return: pandas dataframe with columns 'date', 'leadtime' and the metric name
    """
    implemented_metrics = ["binacc", "sie", "mae", "mse", "rmse"]
    if metric not in implemented_metrics:
        raise NotImplementedError(f"{metric} metric has not been implemented. "
                                  f"Please only choose out of {implemented_metrics}.")
    if metric == "binacc":
        # add default kwargs
        if "threshold" not in kwargs.keys():
            kwargs["threshold"] = 0.15
    elif metric == "sie":
        # add default kwargs
        if "grid_area_size" not in kwargs.keys():
            kwargs["grid_area_size"] = 25
        if "threshold" not in kwargs.keys():
            kwargs["threshold"] = 0.15
    elif metric in ["mae", "mse", "rmse"]:
        # remove grid_area_size and threshold kwargs if passed
        if "grid_area_size" in kwargs.keys():
            del kwargs["grid_area_size"]
        if "threshold" in kwargs.keys():
            del kwargs["threshold"]
    
    do_compute_metrics = True
    if data_path is not None:
        # loading in precomputed dataframes for the metrics
        logging.info(f"Attempting to read in metrics dataframe from {data_path}")
        try:
            metric_df = pd.read_csv(data_path)
            metric_df["date"] = pd.to_datetime(metric_df["date"])
            do_compute_metrics = False
        except OSError:
            logging.info(f"Couldn't load in dataframe from {data_path}, "
                         f"will compute metric dataframe and try save to {data_path}")

    if do_compute_metrics:
        # computing the dataframes for the metrics
        # will save dataframe in data_path if data_path is not None
        metric_df = compute_metrics_leadtime_avg(metric=metric,
                                                 hemisphere=hemisphere,
                                                 forecast_file=forecast_file,
                                                 ecmwf=ecmwf,
                                                 masks=masks,
                                                 data_path=data_path,
                                                 bias_correct=bias_correct,
                                                 region=region,
                                                 **kwargs)

    fc_metric_df = metric_df[metric_df["forecast_name"] == "IceNet"]
    seas_metric_df = metric_df[metric_df["forecast_name"] == "SEAS"]
    seas_metric_df = seas_metric_df if (len(seas_metric_df) != 0) and ecmwf else None

    logging.info(f"Creating leadtime averaged plot for {metric} metric")
    fig, ax = plt.subplots(figsize=(12, 6))
    (start_date, end_date) = (fc_metric_df["date"].min().strftime('%d/%m/%Y'),
                              fc_metric_df["date"].max().strftime('%d/%m/%Y'))
    
    # set ylabel (if average_over == "all"), or legend label (otherwise)
    if metric in ["mae", "mse", "rmse"]:
        ylabel = f"SIC {metric.upper()} (%)"
    elif metric == "binacc":
        ylabel = f"Binary accuracy (%)"
    elif metric == "sie":
        ylabel = f"SIE error (km)"
    
    if average_over == "all":
        # averaging metric over leadtime for all forecasts
        fc_avg_metric = fc_metric_df.groupby("leadtime").mean(metric).\
            sort_values("leadtime", ascending=True)[metric]
        n_forecast_days = fc_avg_metric.index.max()
        
        # plot leadtime averaged metrics
        ax.plot(fc_avg_metric.index, fc_avg_metric, label="IceNet", color="blue")
        if plot_std:
            # obtaining the standard deviation of the metric
            fc_std_metric = fc_metric_df.groupby("leadtime")[metric].std().\
                sort_index(ascending=True)
            ci = std_mult * fc_std_metric
            ax.fill_between(x=fc_avg_metric.index,
                            y1=fc_avg_metric - ci,
                            y2=fc_avg_metric + ci,
                            color="blue",
                            alpha=0.1)
        if seas_metric_df is not None:
            seas_avg_metric = seas_metric_df.groupby("leadtime").mean(metric).\
                sort_values("leadtime", ascending=True)[metric]
            ax.plot(seas_avg_metric.index, seas_avg_metric, label="SEAS", color="darkorange")
            if plot_std:
                # obtaining the standard deviation of the metric
                seas_std_metric = seas_metric_df.groupby("leadtime")[metric].std().\
                    sort_index(ascending=True)
                ci = std_mult * seas_std_metric
                ax.fill_between(x=seas_avg_metric.index,
                                y1=seas_avg_metric - ci,
                                y2=seas_avg_metric + ci,
                                color="darkorange",
                                alpha=0.1)

        # string to add in plot title
        time_coverage = f"\nAveraged over {len(fc_metric_df['date'].unique())} " + \
            f"forecasts between {start_date} - {end_date}"
        if plot_std:
            time_coverage += f"\n(with +/-{std_mult} standard deviations)"
        ax.set_ylabel(ylabel)
        ax.legend(loc='lower right')
    elif average_over in ["day", "month"]:
        if average_over == "day":
            groupby_col = "dayofyear"
        else:
            groupby_col = "month"
        if target_date_avg:
            groupby_col = "target_" + groupby_col
        
        # compute metric by first grouping the dataframe by groupby_col and leadtime
        fc_avg_metric = fc_metric_df.groupby([groupby_col, "leadtime"]).mean(metric).\
            reset_index().pivot(index=groupby_col, columns="leadtime", values=metric).\
                sort_values(groupby_col, ascending=True)
        n_forecast_days = fc_avg_metric.shape[1]
        
        if seas_metric_df is not None:
            # compute the difference in leadtime average to SEAS forecast
            seas_avg_metric = seas_metric_df.groupby([groupby_col, "leadtime"]).mean(metric).\
                reset_index().pivot(index=groupby_col, columns="leadtime", values=metric).\
                    sort_values(groupby_col, ascending=True)
            heatmap_df_diff = fc_avg_metric - seas_avg_metric
            max = np.nanmax(np.abs(heatmap_df_diff.values))
            
            # plot heatmap of the difference between IceNet and SEAS
            sns.heatmap(data=heatmap_df_diff,
                        ax=ax,
                        vmax=max,
                        vmin=-max,
                        cmap="seismic_r" if metric in ["binacc", "sie"] else "seismic",
                        cbar_kws=dict(label=f"{ylabel} difference between IceNet and SEAS"))

        else:
            # plot heatmap of the leadtime averaged metric when grouped by groupby_col
            sns.heatmap(data=fc_avg_metric, 
                        ax=ax,
                        cmap="inferno" if metric in ["binacc", "sie"] else "inferno_r",
                        cbar_kws=dict(label=ylabel))

        # string to add in plot title
        time_coverage = "\nAveraged over a minimum of " + \
            f"{round((fc_metric_df[groupby_col].value_counts()/n_forecast_days).min())} " + \
            f"forecasts between {start_date} - {end_date}"

        # y-axis
        labels = _heatmap_ylabels(metrics_df=fc_metric_df,
                                  average_over=average_over,
                                  groupby_col=groupby_col)
        ax.set_yticks(np.arange(len(fc_metric_df[groupby_col].unique()))+0.5)
        ax.set_yticklabels(labels)
                
        plt.yticks(rotation=0)
        if target_date_avg:
            ax.set_ylabel("Target date of forecast")
        else:
            ax.set_ylabel("Initialisation date of forecast")
    else:
        raise NotImplementedError(f"averaging over {average_over} not a valid option.")
    
    # add plot title
    if metric in ["mae", "mse", "rmse"]:
        title = f"{metric.upper()} comparison"
    elif metric == "binacc":
        title = f"Binary accuracy comparison (threshold SIC = {kwargs['threshold'] * 100}%)"
    elif metric == "sie":
        title = f"SIE error comparison ({kwargs['grid_area_size']} km grid resolution, " +\
            f"threshold SIC = {kwargs['threshold'] * 100}%)"
    ax.set_title(title + time_coverage)
        
    # x-axis
    ax.set_xticks(np.arange(30, n_forecast_days, 30))
    ax.set_xticklabels(np.arange(30, n_forecast_days, 30))
    plt.xticks(rotation=0)
    ax.set_xlabel("Lead time (days)")
    
    # save plot
    targ = "target" if target_date_avg and average_over != "all" else "init"
    filename = f"leadtime_averaged_{targ}_{average_over}_{metric}" + \
        ("_comp" if seas_metric_df is not None else "") + ".png"
    output_path = os.path.join("plot", filename) \
        if not output_path else output_path
    logging.info(f"Saving to {output_path}")
    plt.savefig(output_path)
    
    if plot_std and average_over in ["day", "month"]:
        # create heapmap for the standard deviation
        if seas_metric_df is not None:
            # compute the standard deviation of the metric for both the forecast and SEAS5
            fc_std_metric = fc_metric_df.groupby([groupby_col, "leadtime"]).std(numeric_only=True).\
                reset_index().pivot(index=groupby_col, columns="leadtime", values=metric).\
                    sort_values(groupby_col, ascending=True)
            seas_std_metric = seas_metric_df.groupby([groupby_col, "leadtime"]).std(numeric_only=True).\
                reset_index().pivot(index=groupby_col, columns="leadtime", values=metric).\
                    sort_values(groupby_col, ascending=True)
            # compute the maximum standard deviation to obtain a common scale
            vmax = np.nanmax([np.nanmax(fc_std_metric.values), np.nanmax(seas_std_metric.values)])
            # create heapmap for the standard deviation
            standard_deviation_heatmap(metric=metric,
                                       model_name="SEAS",
                                       metrics_df=seas_metric_df,
                                       average_over=average_over,
                                       output_path=output_path.replace(".png", "_SEAS_std.png"),
                                       target_date_avg=target_date_avg,
                                       fc_std_metric=seas_std_metric,
                                       vmax=vmax,
                                       **kwargs)
        else:
            # no need to pre-compute the scale for the heatmap, hence
            # pre-computing the standard deviation of the metric is not required
            fc_std_metric = None
            vmax = None
        standard_deviation_heatmap(metric=metric,
                                   model_name="IceNet",
                                   metrics_df=fc_metric_df,
                                   average_over=average_over,
                                   output_path=output_path.replace(".png", "_IceNet_std.png"),
                                   target_date_avg=target_date_avg,
                                   fc_std_metric=fc_std_metric,
                                   vmax=vmax,
                                   **kwargs)

    return fc_metric_df, seas_metric_df


def sic_error_video(fc_da: object,
                    obs_da: object,
                    land_mask: object,
                    output_path: object) -> object:
    """

    :param fc_da:
    :param obs_da:
    :param land_mask:
    :param output_path:
    
    :return: matplotlib animation
    """

    diff = fc_da - obs_da
    fig, maps = plt.subplots(nrows=1,
                             ncols=3,
                             figsize=(16, 6),
                             layout="tight")
    fig.set_dpi(150)

    leadtime = 0
    fc_plot = fc_da.isel(time=leadtime).to_numpy()
    obs_plot = obs_da.isel(time=leadtime).to_numpy()
    diff_plot = diff.isel(time=leadtime).to_numpy()

    upper_bound = np.max([np.abs(np.min(diff_plot)), np.abs(np.max(diff_plot))])
    diff_vmin = -upper_bound
    diff_vmax = upper_bound
    logging.debug("Bounds of differences: {} - {}".format(diff_vmin, diff_vmax))

    sic_cmap = mpl.cm.get_cmap("Blues_r", 20)
    contour_kwargs = dict(
        vmin=0,
        vmax=1,
        cmap=sic_cmap
    )

    diff_cmap = mpl.cm.get_cmap("RdBu_r", 20)
    im1 = maps[0].imshow(fc_plot, **contour_kwargs)
    im2 = maps[1].imshow(obs_plot, **contour_kwargs)
    im3 = maps[2].imshow(diff_plot,
                         vmin=diff_vmin,
                         vmax=diff_vmax,
                         cmap=diff_cmap)

    tic = maps[0].set_title("IceNet "
                            f"{pd.to_datetime(fc_da.isel(time=leadtime).time.values).strftime('%d/%m/%Y')}")
    tio = maps[1].set_title("OSISAF Obs "
                            f"{pd.to_datetime(obs_da.isel(time=leadtime).time.values).strftime('%d/%m/%Y')}")
    maps[2].set_title("Diff")

    p0 = maps[0].get_position().get_points().flatten()
    p1 = maps[1].get_position().get_points().flatten()
    p2 = maps[2].get_position().get_points().flatten()

    ax_cbar = fig.add_axes([p0[0]-0.05, 0.04, p1[2]-p0[0], 0.02])
    plt.colorbar(im1, orientation='horizontal', cax=ax_cbar)

    ax_cbar1 = fig.add_axes([p2[0]+0.05, 0.04, p2[2]-p2[0], 0.02])
    plt.colorbar(im3, orientation='horizontal', cax=ax_cbar1)

    for m_ax in maps[0:3]:
        m_ax.tick_params(
            labelbottom=False,
            labelleft=False,
        )
        m_ax.contourf(land_mask,
                      levels=[.5, 1],
                      colors=[mpl.cm.gray(180)],
                      zorder=3)

    def update(date):
        logging.debug(f"Plotting {date}")

        fc_plot = fc_da.isel(time=date).to_numpy()
        obs_plot = obs_da.isel(time=date).to_numpy()
        diff_plot = diff.isel(time=date).to_numpy()

        tic.set_text("IceNet {}".format(
            pd.to_datetime(fc_da.isel(time=date).time.values).strftime("%d/%m/%Y")))
        tio.set_text("OSISAF Obs {}".format(
            pd.to_datetime(obs_da.isel(time=date).time.values).strftime("%d/%m/%Y")))

        im1.set_data(fc_plot)
        im2.set_data(obs_plot)
        im3.set_data(diff_plot)

        return tic, tio, im1, im2, im3

    animation = FuncAnimation(fig,
                              update,
                              range(0, len(fc_da.time)),
                              interval=100)

    plt.close()

    output_path = os.path.join("plot", "sic_error.mp4") \
        if not output_path else output_path
    logging.info(f"Saving to {output_path}")
    animation.save(output_path,
                   fps=10,
                   extra_args=['-vcodec', 'libx264'])
    return animation


def sic_error_local_header_data(da: xr.DataArray):
    n_probe = len(da.probe)
    return {
        "probe array index": {
            i_probe: (
                f"{da.xi.values[i_probe]},"
                f"{da.yi.values[i_probe]}"
            )
            for i_probe in range(n_probe)
        },
        "probe location (EASE)": {
            i_probe: (
                f"{da.xc.values[i_probe]},"
                f"{da.yc.values[i_probe]}"
            )
            for i_probe in range(n_probe)
        },
        "probe location (lat, lon)": {
            i_probe: (
                f"{da.lat.values[i_probe]},"
                f"{da.lon.values[i_probe]}"
            )
            for i_probe in range(n_probe)
        },
        "obs_kind": {
            0: "forecast",
            1: "observation",
            2: "forecast error ('0' - '1')",
            # optionally, include SEAS comparison
        },
    }


def sic_error_local_write_fig(combined_da: xr.DataArray,
                              output_prefix: str):
    """A helper function for `sic_error_local_plots`: plot error and
    forecast/observation data.

    :param combined_da: A DataArray with dims ('time', 'probe', 'obs_kind')

    :param output_prefix: A string from which to produce the output
    path (the probe index and file extension will be appended).
    """

    OBS_KIND_FC = 0
    OBS_KIND_OBS = 1
    OBS_KIND_ERR = 2

    plot_series = combined_da.to_dataframe(name="SIC")["SIC"]

    outfile = output_prefix + ".fc.pdf"

    all_figs = []
    with PdfPages(outfile) as output_pdf:
        n_probe = len(combined_da.probe)
        for i_probe in range(n_probe):

            #### Forecast/observed plots ####

            lat = combined_da.probe.lat.values[i_probe]
            lon = combined_da.probe.lon.values[i_probe]

            lat_h = "N" if lat >= 0.0 else "S"
            lat = abs(lat)

            lon_h = "E" if lon >= 0.0 else "W"
            lon = abs(lon)

            fig, ax = plt.subplots()
            all_figs.append(fig)

            ax.set_title(
                f"Sea ice concentration at location {i_probe + 1}\n"
                f"{lat:.3f}° {lat_h}, {lon:.3f}° {lon_h}"
            )

            ax.set_xlabel("Date")
            ax.set_ylabel("SIC (%)")

            # dims: (obs_kind, time, probe)
            ax.plot(plot_series.loc[OBS_KIND_FC, :, i_probe], label="IceNet")
            ax.plot(plot_series.loc[OBS_KIND_OBS, :, i_probe], label="Observed")
            ax.legend()

            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            output_pdf.savefig(fig, bbox_inches='tight')

            #### Error plot ####
            fig2, ax2 = plt.subplots()
            all_figs.append(fig2)

            ax2.set_title(
                f"Sea ice concentration error at location {i_probe + 1}\n"
                f"{lat:.3f}° {lat_h}, {lon:.3f}° {lon_h}"
            )

            ax2.set_xlabel("Date")
            ax2.set_ylabel("SIC error (%)")

            ax2.axhline(color='k', lw=0.5, ls='--')
            ax2.plot(plot_series.loc[OBS_KIND_ERR, :, i_probe], color='C2')

            plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
            output_pdf.savefig(fig2, bbox_inches='tight')

    return all_figs


def sic_error_local_plots(fc_da: object,
                          obs_da: object,
                          output_path: object,
                          as_command: bool = False):

    """
    :param fc_da: a DataArray with dims ('time', 'probe')
    :param obs_da: a DataArray with dims ('time', 'probe')
    """

    # convert SIC to percentages (ranging from 0% to 100%) rather than a fraction
    fc_da = fc_da*100
    obs_da = obs_da*100
    err_da = (fc_da-obs_da)
    combined_da = xr.concat(
        [fc_da, obs_da, err_da],
        dim="obs_kind", coords="minimal"
    )

    # convert to a dataframe for csv output
    df = (
        combined_da
        .to_dataframe(name="SIC")
        # drop unneeded coords (lat, lon, xc, yc)
        .loc[:, "SIC"]
        # Convert mult-indices
        .unstack(2).unstack(0)
    )

    if output_path is None:
        output_path = "sic_error_local.csv"

    header_info = sic_error_local_header_data(combined_da)

    header = "# icenet_plot_sic_error_local\n"
    header += f"# Part of Icenet, version {icenet_version}\n"
    header += "#\n"

    if as_command:
        header += "# Command output from \n"
        cmd = ' '.join([a.__repr__() for a in sys.argv])
        header += f"#   {cmd}\n"
        header += "#\n"

    header += "# Key\n"
    for header_kind, header_data in header_info.items():
        header += f"# {header_kind}\n"
        for k, v in header_data.items():
            header += f"#   {k}: {v}\n"

    with open(output_path, "w") as outfile:
        outfile.write(header)
        df.to_csv(outfile)

    figs = sic_error_local_write_fig(combined_da, "sic_error_local")

    return figs


class ForecastPlotArgParser(argparse.ArgumentParser):

    """An ArgumentParser specialised to support forecast plot arguments

    Additional argument enabled by allow_ecmwf() etc.

    The 'allow_*' methods return self to permit method chaining.

    :param forecast_date: allows this positional argument to be disabled
    """

    def __init__(self, *args,
                 forecast_date: bool = True,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.add_argument("hemisphere", choices=("north", "south"))
        self.add_argument("forecast_file", type=str)
        if forecast_date:
            self.add_argument("forecast_date", type=date_arg)

        self.add_argument("-o", "--output-path", type=str, default=None)
        self.add_argument("-v", "--verbose", action="store_true", default=False)
        self.add_argument("-r", "--region", default=None, type=region_arg,
                          help="Region specified x1, y1, x2, y2")

    def allow_ecmwf(self):
        self.add_argument("-b", "--bias-correct",
                          help="Bias correct SEAS forecast array",
                          action="store_true",
                          default=False)
        self.add_argument("-e", "--ecmwf", action="store_true", default=False)
        return self

    def allow_threshold(self):
        self.add_argument("-t",
                          "--threshold",
                          help="The SIC threshold of interest",
                          type=float,
                          default=0.15)
        return self

    def allow_sie(self):
        self.add_argument("-ga",
                          "--grid-area",
                          help="The length of the sides of the grid used (in km)",
                          type=int,
                          default=25)
        return self

    def allow_metrics(self):
        self.add_argument("-m", 
                          "--metrics",
                          help="Which metrics to compute and plot",
                          type=str,
                          default="mae,mse,rmse")
        self.add_argument("-s",
                          "--separate",
                          help="Whether or not to produce separate plots for each metric",
                          action="store_true",
                          default=False)
        return self

    def allow_probes(self):
        self.add_argument(
            "-p", "--probe", action="append", dest="probes",
            type=location_arg, metavar="LOCATION",
            help="Sample at LOCATION",
        )
        return self

    def parse_args(self, *args, **kwargs):
        args = super().parse_args(*args, **kwargs)

        logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
        logging.getLogger("matplotlib").setLevel(logging.WARNING)

        return args

##
# CLI endpoints
#


def binary_accuracy():
    """
    Produces plot of the binary classification accuracy of forecasts.
    """
    ap = (
        ForecastPlotArgParser()
        .allow_ecmwf()
        .allow_threshold()
    )
    args = ap.parse_args()

    masks = Masks(north=args.hemisphere == "north",
                  south=args.hemisphere == "south")

    fc = get_forecast_ds(args.forecast_file,
                         args.forecast_date)
    obs = get_obs_da(args.hemisphere,
                     pd.to_datetime(args.forecast_date) +
                     timedelta(days=1),
                     pd.to_datetime(args.forecast_date) +
                     timedelta(days=int(fc.leadtime.max())))
    fc = filter_ds_by_obs(fc, obs, args.forecast_date)

    if args.ecmwf:
        seas = get_seas_forecast_da(
            args.hemisphere,
            args.forecast_date,
            bias_correct=args.bias_correct) \
            if args.ecmwf else None

        if seas is not None:
            seas = seas.assign_coords(dict(xc=seas.xc / 1e3, yc=seas.yc / 1e3))
            seas = seas.isel(time=slice(1, None))
    else:
        seas = None

    if args.region:
        seas, fc, obs, masks = process_regions(args.region,
                                               [seas, fc, obs, masks])
    
    plot_binary_accuracy(masks=masks,
                         fc_da=fc,
                         cmp_da=seas,
                         obs_da=obs,
                         output_path=args.output_path,
                         threshold=args.threshold)


def sie_error():
    """
    Produces plot of the sea ice extent (SIE) error of forecasts.
    """
    ap = (
        ForecastPlotArgParser()
        .allow_ecmwf()
        .allow_threshold()
        .allow_sie()
    )
    args = ap.parse_args()

    masks = Masks(north=args.hemisphere == "north",
                  south=args.hemisphere == "south")

    fc = get_forecast_ds(args.forecast_file,
                         args.forecast_date)
    obs = get_obs_da(args.hemisphere,
                     pd.to_datetime(args.forecast_date) +
                     timedelta(days=1),
                     pd.to_datetime(args.forecast_date) +
                     timedelta(days=int(fc.leadtime.max())))
    fc = filter_ds_by_obs(fc, obs, args.forecast_date)

    if args.ecmwf:
        seas = get_seas_forecast_da(
            args.hemisphere,
            args.forecast_date,
            bias_correct=args.bias_correct) \
            if args.ecmwf else None

        if seas is not None:
            seas = seas.assign_coords(dict(xc=seas.xc / 1e3, yc=seas.yc / 1e3))
            seas = seas.isel(time=slice(1, None))
    else:
        seas = None

    if args.region:
        seas, fc, obs, masks = process_regions(args.region,
                                               [seas, fc, obs, masks])

    plot_sea_ice_extent_error(masks=masks,
                              fc_da=fc,
                              cmp_da=seas,
                              obs_da=obs,
                              output_path=args.output_path,
                              grid_area_size=args.grid_area,
                              threshold=args.threshold)


def plot_forecast():
    """CLI entry point for icenet_plot_forecast

    :return:
    """
    ap = ForecastPlotArgParser()
    ap.add_argument("-l", "--leadtimes",
                    help="Leadtimes to output, multiple as CSV, range as n..n",
                    type=lambda s: [int(i) for i in
                                    list(s.split(",") if "," in s else
                                         range(int(s.split("..")[0]),
                                               int(s.split("..")[1]) + 1) if ".." in s else
                                         [s])])
    ap.add_argument("-c", "--no-coastlines",
                    help="Turn off cartopy integration",
                    action="store_true", default=False)
    ap.add_argument("-f", "--format",
                    help="Format to output in",
                    choices=("mp4", "png", "svg", "tiff"),
                    default="png")
    ap.add_argument("-n", "--cmap-name",
                    help="Color map name if not wanting to use default",
                    default=None)
    ap.add_argument("-s", "--stddev",
                    help="Plot the standard deviation from the ensemble",
                    action="store_true",
                    default=False)
    args = ap.parse_args()

    fc = get_forecast_ds(args.forecast_file,
                         args.forecast_date,
                         stddev=args.stddev)
    fc = fc.transpose(..., "yc", "xc")

    output_path = "." if args.output_path is None else args.output_path

    if not os.path.isdir(output_path):
        logging.warning("No directory at: {}".format(output_path))
        os.makedirs(output_path)
    elif os.path.isfile(output_path):
        raise RuntimeError("{} should be a directory and not existent...".
                           format(output_path))

    forecast_name = "{}.{}".format(
        os.path.splitext(os.path.basename(args.forecast_file))[0],
        args.forecast_date)

    cmap_name = "BuPu_r" if args.stddev else "Blues_r"
    if args.cmap_name is not None:
        cmap_name = args.cmap_name

    logging.info("Using cmap {}".format(cmap_name))
    cmap = cm.get_cmap(cmap_name)
    cmap.set_bad("dimgrey")

    if args.region is not None:
        fc = process_regions(args.region, [fc])[0]

    vmax = 1.

    if args.stddev:
        vmax = float(fc.max())
        logging.info("Calculated vmax to be: {}".format(vmax))

    leadtimes = args.leadtimes \
        if args.leadtimes is not None \
        else list(range(1, int(max(fc.leadtime.values)) + 1))

    if args.format == "mp4":
        pred_da = fc.isel(time=0).sel(leadtime=leadtimes)

        if "forecast_date" not in pred_da:
            forecast_dates = [
                pd.Timestamp(args.forecast_date) + dt.timedelta(lt)
                for lt in args.leadtimes]
            pred_da = pred_da.assign_coords(
                forecast_date=("leadtime", forecast_dates))

        pred_da = pred_da.drop("time").drop("leadtime").\
            rename(leadtime="time", forecast_date="time").set_index(time="time")

        anim_args = dict(
            figsize=5
        )
        if not args.no_coastlines:
            logging.warning("Coastlines will not work with the current "
                            "implementation of xarray_to_video")

        output_filename = os.path.join(output_path, "{}.{}.{}{}".format(
            forecast_name,
            args.forecast_date.strftime("%Y%m%d"),
            "" if not args.stddev else "stddev.",
            args.format
        ))
        xarray_to_video(pred_da, fps=1, cmap=cmap,
                        imshow_kwargs=dict(vmin=0., vmax=vmax)
                        if not args.stddev else None,
                        video_path=output_filename,
                        **anim_args)
    else:
        for leadtime in leadtimes:
            pred_da = fc.sel(leadtime=leadtime).isel(time=0)
            bound_args = dict(north=args.hemisphere == "north",
                              south=args.hemisphere == "south")

            if args.region is not None:
                bound_args.update(x1=args.region[0],
                                  x2=args.region[2],
                                  y1=args.region[1],
                                  y2=args.region[3])

            ax = get_plot_axes(**bound_args,
                               do_coastlines=not args.no_coastlines)

            bound_args.update(cmap=cmap)

            im = show_img(ax, pred_da, **bound_args, vmax=vmax,
                          do_coastlines=not args.no_coastlines)

            plt.colorbar(im, ax=ax)
            plot_date = args.forecast_date + dt.timedelta(leadtime)
            ax.set_title("{:04d}/{:02d}/{:02d}".format(plot_date.year,
                                                       plot_date.month,
                                                       plot_date.day))
            output_filename = os.path.join(output_path, "{}.{}.{}{}".format(
                forecast_name,
                (args.forecast_date + dt.timedelta(
                    days=leadtime)).strftime("%Y%m%d"),
                "" if not args.stddev else "stddev.",
                args.format
            ))

            logging.info("Saving to {}".format(output_filename))
            plt.savefig(output_filename)
            plt.clf()


def parse_metrics_arg(argument: str) -> object:
    """
    Splits a string into a list by separating on commas.
    Will remove any whitespace and removes duplicates.
    Used to parsing metrics argument in metric_plots.
    
    :param argument: string
    
    :return: list of metrics to compute
    """
    return list(set([s.replace(" ", "") for s in argument.split(",")]))


def metric_plots():
    """
    Produces plot of requested metrics for forecasts.
    """
    ap = (
        ForecastPlotArgParser()
        .allow_ecmwf()
        .allow_metrics()
    )
    args = ap.parse_args()

    masks = Masks(north=args.hemisphere == "north",
                  south=args.hemisphere == "south")

    fc = get_forecast_ds(args.forecast_file,
                         args.forecast_date)
    obs = get_obs_da(args.hemisphere,
                     pd.to_datetime(args.forecast_date) +
                     timedelta(days=1),
                     pd.to_datetime(args.forecast_date) +
                     timedelta(days=int(fc.leadtime.max())))
    fc = filter_ds_by_obs(fc, obs, args.forecast_date)

    metrics = parse_metrics_arg(args.metrics)

    if args.ecmwf:
        seas = get_seas_forecast_da(
            args.hemisphere,
            args.forecast_date,
            bias_correct=args.bias_correct) \
            if args.ecmwf else None

        if seas is not None:
            seas = seas.assign_coords(dict(xc=seas.xc / 1e3, yc=seas.yc / 1e3))
            seas = seas.isel(time=slice(1, None))
    else:
        seas = None

    if args.region:
        seas, fc, obs, masks = process_regions(args.region,
                                               [seas, fc, obs, masks])

    plot_metrics(metrics=metrics,
                 masks=masks,
                 fc_da=fc,
                 cmp_da=seas,
                 obs_da=obs,
                 output_path=args.output_path,
                 separate=args.separate)


def leadtime_avg_plots():
    """
    Produces plot of leadtime averaged metrics for forecasts.
    """
    ap = (
        ForecastPlotArgParser(forecast_date=False)
        .allow_ecmwf()
        .allow_threshold()
        .allow_sie()
    )
    ap.add_argument("-m",
                    "--metric",
                    help="Which metric to compute and plot",
                    type=str)
    ap.add_argument("-dp",
                    "--data_path",
                    help="Where to find (or store) metrics dataframe",
                    type=str,
                    default=None)
    ap.add_argument("-ao",
                    "--average_over",
                    help="How to average the forecast metrics",
                    type=str,
                    choices=["all", "month", "day"])
    ap.add_argument("-s",
                    "--std",
                    help="If flagged, standard deviation is plotted",
                    action="store_true",
                    default=False)
    ap.add_argument("-sm",
                    "--std_mult",
                    help="What multiple of the standard deviation to plot",
                    type=float,
                    default=1.0)
    ap.add_argument("-td",
                    "--target_date_average",
                    help="Averages metric over target date instead of init date",
                    action="store_true",
                    default=False)
    args = ap.parse_args()
    masks = Masks(north=args.hemisphere == "north",
                  south=args.hemisphere == "south")
    
    plot_metrics_leadtime_avg(metric=args.metric,
                              masks=masks,
                              hemisphere=args.hemisphere,
                              forecast_file=args.forecast_file,
                              ecmwf=args.ecmwf,
                              output_path=args.output_path,
                              average_over=args.average_over,
                              plot_std=args.std,
                              std_mult=args.std_mult,
                              data_path=args.data_path,
                              target_date_avg=args.target_date_average,
                              bias_correct=args.bias_correct,
                              region=args.region,
                              threshold=args.threshold,
                              grid_area_size=args.grid_area)


def sic_error():
    """
    Produces video visualisation of SIC of forecast and ground truth.
    """
    ap = ForecastPlotArgParser()
    args = ap.parse_args()

    masks = Masks(north=args.hemisphere == "north",
                  south=args.hemisphere == "south")

    fc = get_forecast_ds(args.forecast_file,
                         args.forecast_date)
    obs = get_obs_da(args.hemisphere,
                     pd.to_datetime(args.forecast_date) +
                     timedelta(days=1),
                     pd.to_datetime(args.forecast_date) +
                     timedelta(days=int(fc.leadtime.max())))
    fc = filter_ds_by_obs(fc, obs, args.forecast_date)

    if args.region:
        fc, obs, masks = process_regions(args.region, [fc, obs, masks])

    sic_error_video(fc_da=fc,
                    obs_da=obs,
                    land_mask=masks.get_land_mask(),
                    output_path=args.output_path)


def sic_error_local():
    """
    Entry point for the icenet_plot_sic_error_local command
    """

    ap = (
        ForecastPlotArgParser()
        .allow_probes()
    )
    args = ap.parse_args()

    fc = get_forecast_ds(args.forecast_file,
                         args.forecast_date)
    obs = get_obs_da(args.hemisphere,
                     pd.to_datetime(args.forecast_date) +
                     timedelta(days=1),
                     pd.to_datetime(args.forecast_date) +
                     timedelta(days=int(fc.leadtime.max())))
    fc = filter_ds_by_obs(fc, obs, args.forecast_date)

    fc, obs = process_probes(args.probes, [fc, obs])

    sic_error_local_plots(fc,
                          obs,
                          args.output_path,
                          as_command=True)
