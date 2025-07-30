import datetime as dt
import glob
import logging
import operator
import os
import re

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from dateutil.relativedelta import relativedelta
from ibicus.debias import LinearScaling

from download_toolbox.dataset import DatasetConfig
from download_toolbox.interface import get_dataset_config_implementation, Frequency, get_implementation
from preprocess_toolbox.processor import Processor


def broadcast_forecast(start_date: object,
                       end_date: object,
                       datafiles: list | None = None,
                       dataset: xr.Dataset | None = None,
                       target: os.PathLike | str = None,
                       frequency: Frequency = Frequency.DAY) -> object:
    """

    :param start_date:
    :param end_date:
    :param datafiles:
    :param dataset:
    :param target:
    :param frequency:
    :return:
    """

    if not ((datafiles is None) ^ (dataset is None)):
        raise RuntimeError("Only one of datafiles and dataset can be set")

    if datafiles:
        logging.info("Using {} to generate forecast through {} to {}".format(
            ", ".join(datafiles), start_date, end_date))
        dataset = xr.open_mfdataset(datafiles, engine="netcdf4")

    dates = pd.date_range(start_date, end_date, freq=frequency.freq)
    i = 0

    logging.debug("Dataset summary: \n{}".format(dataset))

    if len(dataset.time.values) > 1:
        while dataset.time.values[i + 1] < dates[0]:
            i += 1

    logging.info("Starting index will be {} for {} - {}".format(i, dates[0], dates[-1]))
    dt_arr = []

    for d in dates:
        logging.debug("Looking for date {}".format(d))
        arr = None

        while arr is None:
            if d >= dataset.time.values[i]:
                delta_attribute = "{}s".format(frequency.attribute)
                delta_lead = relativedelta(pd.to_datetime(d), pd.to_datetime(dataset.time.values[i]))
                # TODO: d_lead used to use as is, but forecasts start at leadtime 1, so genuine fix
                #  or red herring? Validate with daily forecasts as well: we introduced +1
                d_lead = getattr(delta_lead, delta_attribute) + 1

                if i + 1 < len(dataset.time.values):
                    if pd.to_datetime(dataset.time.values[i]) + relativedelta(**{delta_attribute: d_lead}) >= \
                       pd.to_datetime(dataset.time.values[i + 1]) + relativedelta(**{delta_attribute: 1}):
                        i += 1
                        continue

                logging.debug("Selecting date {} and lead {}".format(
                    pd.to_datetime(dataset.time.values[i]).strftime("%D"),
                    d_lead))

                arr = dataset.sel(time=dataset.time.values[i],
                                  leadtime=d_lead).\
                    copy().\
                    drop("time").\
                    assign_coords(dict(time=d)).\
                    drop("leadtime")
            else:
                i += 1

        dt_arr.append(arr)

    target_ds = xr.concat(dt_arr, dim="time")

    if target:
        logging.info("Saving dataset to {}".format(target))
        target_ds.to_netcdf(target)
    return target_ds


def get_seas_forecast_init_dates(seas_config_path: os.PathLike) -> list:
    """
    Obtains list of dates for which we have SEAS forecasts

    :param seas_config_path: path to a configuration

    :return: list of dates
    """

    seas_ds_config = get_dataset_config_implementation(seas_config_path)
    return seas_ds_config.existing_dates


def get_seas_forecast_da(
        seas_config_path: os.PathLike,
        date: str,
        bias_correct: bool = True,
) -> tuple:
    """
    Atmospheric model Ensemble 15-day forecast (Set III - ENS)

    Coordinates:
      * time                          (time) datetime64[ns] 2022-04-01 ... 2022-0...
      * yc                            (yc) float64 5.388e+06 ... -5.388e+06
      * xc                            (xc) float64 -5.388e+06 ... 5.388e+06

    :param seas_config_path: dataset config path for the comparison dataset
    :param date:
    :param bias_correct:
    """
    seas_ds_config = get_dataset_config_implementation(seas_config_path)
    seas_file = seas_ds_config.var_filepath(seas_ds_config.var_config('siconca'), [date,])

    if os.path.exists(seas_file):
        seas_da = xr.open_dataset(seas_file).siconca
    else:
        logging.warning("No SEAS data available at {}".format(seas_file))
        return None

    if bias_correct:
        raise NotImplementedError("BIAS correction not currently refactored")
        # Let's have some maximum, though it's quite high
        (start_date, end_date) = (date - dt.timedelta(days=10 * 365),
                                  date + dt.timedelta(days=10 * 365))
        obs_ds = ds_config.get_dataset(var_names=["siconca"])
        obs_da = obs_ds.sel(time=slice(
            pd.to_datetime(start_date),
            pd.to_datetime(end_date))).siconca
        # TODO: this is no longer valid, use ds_config
        seas_hist_files = dict(
            sorted({
                os.path.abspath(el):
                    dt.datetime.strptime(os.path.basename(el)[0:8], obs_ds_config.frequency.date_format)
                for el in glob.glob(
                    os.path.join(ds_config.path.replace(ds_config.identifier, "seas"),
                                 "siconca", "*.nc"))
                if re.search(r'^\d{8}\.nc$', os.path.basename(el)) and
                el != seas_file
            }.items()))

        def strip_overlapping_time(ds):
            data_file = os.path.abspath(ds.encoding["source"])

            try:
                idx = list(seas_hist_files.keys()).index(data_file)
            except ValueError:
                logging.exception("\n{} not in \n\n{}".format(
                    data_file, seas_hist_files))
                return None

            if idx < len(seas_hist_files) - 1:
                max_date = seas_hist_files[
                               list(seas_hist_files.keys())[idx + 1]] \
                           - dt.timedelta(days=1)
                logging.debug("Stripping {} to {}".format(data_file, max_date))
                return ds.sel(time=slice(None, max_date))
            else:
                logging.debug("Not stripping {}".format(data_file))
                return ds

        hist_da = xr.open_mfdataset(seas_hist_files,
                                    preprocess=strip_overlapping_time).siconca
        debiaser = LinearScaling(delta_type="additive",
                                 variable="siconc",
                                 reasonable_physical_range=[0., 1.])

        logging.info("Debiaser input ranges: obs {:.2f} - {:.2f}, "
                     "hist {:.2f} - {:.2f}, fut {:.2f} - {:.2f}".format(
                         float(obs_da.min()), float(obs_da.max()),
                         float(hist_da.min()), float(hist_da.max()),
                         float(seas_da.min()), float(seas_da.max())))

        seas_array = debiaser.apply(obs_da.values, hist_da.values,
                                    seas_da.values)
        seas_da.values = seas_array
        logging.info("Debiaser output range: {:.2f} - {:.2f}".format(
            float(seas_da.min()), float(seas_da.max())))

    logging.info("Returning SEAS data from {} from {}".format(seas_file, date))

    # This isn't great looking, but we know we're not dealing with huge
    # indexes in here
    date_location = list(seas_da.time.values).index(pd.Timestamp(date))
    if date_location > 0:
        logging.warning("SEAS forecast started {} day before the requested "
                        "date {}, make sure you account for this!".format(
                            date_location, date))

    seas_da = seas_da.sel(time=pd.Timestamp(date))
    # Regridding references determine the coordinates, so this might not be xc-yc
    if 'x' in seas_da.coords:
        seas_da = seas_da.rename(dict(x="xc", y="yc"))
    seas_da.coords['leadtime'] = range(1, len(seas_da.leadtime) + 1)
    return seas_da


def get_forecast_data(forecast_file: os.PathLike,
                      forecast_date: str,
                      stddev: bool = False) -> object:
    """

    :param forecast_file: a path to a .nc file
    :param forecast_date: initialisation date of the forecast
    :param stddev: initialisation date of the forecast
    :returns fc_ds:
    """
    logging.info("Opening forecast {} for date {}".format(forecast_file, forecast_date))
    forecast_date = pd.to_datetime(forecast_date)
    forecast_ds = xr.open_dataset(forecast_file, decode_coords="all")
    forecast_ds = forecast_ds.sel(time=forecast_date)

    return forecast_ds.sic_mean if not stddev else forecast_ds.sic_stddev


def get_forecast_obs_data(forecast_file: os.PathLike,
                          obs_ds_config: os.PathLike,
                          forecast_date: str,
                          stddev: bool = False) -> tuple[xr.DataArray, xr.DataArray, Processor]:
    """Method to retrieve forecast and equivalent observational data

    Args:
        forecast_file: a path to a .nc file
        obs_ds_config:
        forecast_date: initialisation date of the forecast
        stddev:
    Returns:
        tuple(forecast data,
              observational data,
              masks processor instance)

    """
    forecast_da = get_forecast_data(forecast_file, forecast_date, stddev)
    ds_config = get_dataset_config_implementation(obs_ds_config)
    obs_ds = ds_config.get_dataset(var_names=["siconca"])

    # Forecast date is initialisation date, leadtime == 1, so we need to offset indexes
    # For monthly calculations taking deltas won't work correctly
    obs_ds = obs_ds.sel(time=slice(
        forecast_da.forecast_date.min(),
        forecast_da.forecast_date.max()
    ))

    masks = get_implementation(xr.open_dataset(forecast_file).attrs["icenet_mask_implementation"])(ds_config)
    forecast_da = filter_forecast_da_by_obs(forecast_da, obs_ds, ds_config.frequency)

    # TODO: Naive manner by which to detect AMSR data - can we generalise / make clearer
    #  and also, we should consider using the icenet.data.processor functionality
    #  for converging the data, as this is duplicated functionality
    if "x" in obs_ds.coords:
        # AMSR clause
        obs_da = obs_ds.rename(dict(x="xc", y="yc", time="leadtime")).siconca
    else:
        # OSISAF clause
        obs_da = obs_ds.rename(dict(time="leadtime")).siconca
        obs_da.coords['xc'] = obs_da.coords['xc'] * 1e3
        obs_da.coords['yc'] = obs_da.coords['yc'] * 1e3

    # All data returned is mapped as a single forecast
    obs_da.coords['leadtime'] = forecast_da.coords['leadtime']
    obs_da /= 100.

    return forecast_da.load(), obs_da.load(), masks


def filter_forecast_da_by_obs(da: xr.DataArray,
                              obs_da: object,
                              frequency: Frequency = Frequency.DAY) -> object:
    """

    :param da:
    :param obs_da:
    :param forecast_date: initialisation date of the forecast
    :param frequency: frequency of the observational dataset
    :return:
    """
    if len(obs_da.time) < len(da.leadtime):
        (start_date, end_date) = (obs_da.time.to_series()[0],
                                  obs_da.time.to_series()[-1])

        if len(obs_da.time) < 1:
            raise RuntimeError("No observational data available between {} "
                               "and {}".format(start_date.strftime(frequency.date_format),
                                               end_date.strftime(frequency.date_format)))

        logging.warning("Observational data not available for full range of "
                        "forecast lead times: obs {}-{} vs fc {}-{}".format(
                            obs_da.time.to_series()[0].strftime(frequency.date_format),
                            obs_da.time.to_series()[-1].strftime(frequency.date_format),
                            start_date.strftime(frequency.date_format),
                            end_date.strftime(frequency.date_format)))

        # We subset to get a nicely compatible dataset for plotting
        return da.sel(time=slice(start_date, end_date))

    # Otherwise we're assuming all obs data is covering the da provided
    return da


def calculate_extents(da: xr.DataArray,
                      x1: int | None = None,
                      x2: int | None = None,
                      y1: int | None = None,
                      y2: int | None = None):
    """

    :param da:
    :param x1:
    :param x2:
    :param y1:
    :param y2:
    :return:
    """

    x1 = x1 if x1 is not None else 0
    x2 = x2 if x2 is not None else len(da.xc)
    y1 = y1 if y1 is not None else 0
    y2 = y2 if y2 is not None else len(da.yc)
    xc_sz = da.xc[1] - da.xc[0]
    yc_sz = da.yc[1] - da.yc[0]

    extents = [
        da.xc[0] + (x1 * xc_sz),
        da.xc[-1] - ((len(da.xc) - x2) * xc_sz),
        da.yc[-1] - ((len(da.yc) - y2) * yc_sz),
        da.yc[0] + (y1 * yc_sz),
    ]
    logging.debug("Data extents: {}".format(extents))
    return extents


def get_plot_axes(da: xr.DataArray,
                  x1: int | None = None,
                  x2: int | None = None,
                  y1: int | None = None,
                  y2: int | None = None,
                  do_coastlines: bool = True,
                  north: bool = True,
                  south: bool = False):
    """

    :param da:
    :param x1:
    :param x2:
    :param y1:
    :param y2:
    :param do_coastlines:
    :param north:
    :param south:
    :return:
    """
    if not (north ^ south):
        raise RuntimeError("One hemisphere only must be selected")

    fig = plt.figure(figsize=(10, 8), dpi=150, layout='tight')

    if do_coastlines:
        pole = 1 if north else -1
        proj = ccrs.LambertAzimuthalEqualArea(0, pole * 90)
        ax = fig.add_subplot(1, 1, 1, projection=proj)
        extents = calculate_extents(da, x1, x2, y1, y2)
        ax.set_extent(extents, crs=proj)
    else:
        ax = fig.add_subplot(1, 1, 1)

    return ax


def show_img(ax,
             da,
             x1: int | None = None,
             x2: int | None = None,
             y1: int | None = None,
             y2: int | None = None,
             cmap: object = None,
             do_coastlines: bool = True,
             vmin: float = 0.,
             vmax: float = 1.,
             north: bool = True,
             south: bool = False):
    """

    :param ax:
    :param da:
    :param x1:
    :param x2:
    :param y1:
    :param y2:
    :param cmap:
    :param do_coastlines:
    :param vmin:
    :param vmax:
    :param north:
    :param south:
    :return:
    """

    assert north ^ south, "One hemisphere only must be selected"

    if do_coastlines:
        pole = 1 if north else -1
        data_crs = ccrs.LambertAzimuthalEqualArea(0, pole * 90)
        extents = calculate_extents(da, x1, x2, y1, y2)
        im = ax.imshow(da,
                       vmin=vmin,
                       vmax=vmax,
                       cmap=cmap,
                       transform=data_crs,
                       extent=extents)
        ax.coastlines()
    else:
        im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)

    return im


def process_probes(probes, data) -> tuple:
    """
    :param probes: A sequence of locations (pairs)
    :param data: A sequence of xr.DataArray
    """

    # index into each element of data with a xr.DataArray, for pointwise
    # selection.  Construct the indexing DataArray as follows:

    probes_da = xr.DataArray(probes, dims=('probe', 'coord'))
    xcs, ycs = probes_da.sel(coord=0), probes_da.sel(coord=1)

    for idx, arr in enumerate(data):
        arr = arr.assign_coords({
            "xi": ("xc", np.arange(len(arr.xc))),
            "yi": ("yc", np.arange(len(arr.yc))),
        })
        if arr is not None:
            data[idx] = arr.isel(xc=xcs, yc=ycs)

    return data


def process_regions(region: tuple, data: tuple) -> tuple:
    """

    :param region:
    :param data:

    :return:
    """

    if len(region) != 4:
        raise RuntimeError("Region needs to be a list of four integers")

    x1, y1, x2, y2 = region

    if x2 < x1 or y2 < y1:
        raise RuntimeError("Region is not valid")

    for idx, arr in enumerate(data):
        if arr is not None:
            data[idx] = arr[..., (432 - y2):(432 - y1), x1:x2]
    return data
