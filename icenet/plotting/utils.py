import datetime as dt
import glob
import logging
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
from download_toolbox.interface import get_dataset_config_implementation, Frequency
from preprocess_toolbox.utils import get_implementation


def broadcast_forecast(start_date: object,
                       end_date: object,
                       datafiles: object = None,
                       dataset: object = None,
                       target: object = None,
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


def get_seas_forecast_init_dates(
    hemisphere: str,
    source_path: object = os.path.join(".", "data", "mars.seas")
) -> object:
    """
    Obtains list of dates for which we have SEAS forecasts we have.

    :param hemisphere: string, typically either 'north' or 'south'
    :param source_path: path where north and south SEAS forecasts are stored

    :return: list of dates
    """
    # list the files in the path where SEAS forecasts are stored
    filenames = os.listdir(os.path.join(source_path, hemisphere, "siconca"))
    # obtain the dates from files with YYYYMMDD.nc format
    return pd.to_datetime(
        [x.split('.')[0] for x in filenames if re.search(r'^\d{8}\.nc$', x)])


def get_seas_forecast_da(
        obs_ds_config: DatasetConfig,
        date: str,
        bias_correct: bool = True,
) -> tuple:
    """
    Atmospheric model Ensemble 15-day forecast (Set III - ENS)

    Coordinates:
      * time                          (time) datetime64[ns] 2022-04-01 ... 2022-0...
      * yc                            (yc) float64 5.388e+06 ... -5.388e+06
      * xc                            (xc) float64 -5.388e+06 ... 5.388e+06

    TODO: we need to be supplying the download toolbox SEAS configuration for this dataset

    :param obs_ds_config: dataset config for the ground truth dataset
    :param date:
    :param bias_correct:
    """

    ds_config = get_dataset_config_implementation(obs_ds_config)
    seas_file = os.path.join(
        ds_config.path.replace(ds_config.identifier, "seas"),
        "siconca",
        "{}.nc".format(date.replace(day=1).strftime("%Y%m%d")))

    if os.path.exists(seas_file):
        seas_da = xr.open_dataset(seas_file).siconc
    else:
        logging.warning("No SEAS data available at {}".format(seas_file))
        return None

    if bias_correct:
        # Let's have some maximum, though it's quite high
        (start_date, end_date) = (date - dt.timedelta(days=10 * 365),
                                  date + dt.timedelta(days=10 * 365))
        obs_ds = ds_config.get_dataset(var_names=["siconca"])
        obs_da = obs_ds.sel(time=slice(
            pd.to_datetime(start_date),
            pd.to_datetime(end_date))).siconca
        seas_hist_files = dict(
            sorted({
                os.path.abspath(el):
                    dt.datetime.strptime(os.path.basename(el)[0:8], "%Y%m%d")
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
                                    preprocess=strip_overlapping_time).siconc
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

    seas_da = seas_da.sel(time=slice(date, None))
    logging.debug("SEAS data range: {} - {}, {} dates".format(
        pd.to_datetime(min(seas_da.time.values)).strftime("%Y-%m-%d"),
        pd.to_datetime(max(seas_da.time.values)).strftime("%Y-%m-%d"),
        len(seas_da.time)))

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
    forecast_ds = forecast_ds.sel(time=slice(forecast_date, forecast_date))

    return forecast_ds.sic_mean if not stddev else forecast_ds.sic_stddev


def get_forecast_obs_data(forecast_file: os.PathLike,
                          obs_ds_config: DatasetConfig,
                          forecast_date: str,
                          stddev: bool = False) -> object:
    """

    :param forecast_file: a path to a .nc file
    :param obs_ds_config:
    :param forecast_date: initialisation date of the forecast
    :param stddev: initialisation date of the forecast
    :returns fc_da, obs_da, masks:
    """
    forecast_da = get_forecast_data(forecast_file, forecast_date, stddev)
    ds_config = get_dataset_config_implementation(obs_ds_config)
    obs_ds = ds_config.get_dataset(var_names=["siconca"])
    obs_ds = obs_ds.sel(time=slice(
        pd.to_datetime(forecast_date),
        pd.to_datetime(forecast_date) + relativedelta(**{
            "{}s".format(ds_config.frequency.attribute): int(forecast_da.leadtime.max())})
    ))
    masks = get_implementation(xr.open_dataset(forecast_file).attrs["icenet_mask_implementation"])(ds_config)
    forecast_da = filter_ds_by_obs(forecast_da, obs_ds, forecast_date, ds_config.frequency)
    obs_ds['siconca'] /= 100
    return forecast_da, obs_ds.siconca, masks


def filter_ds_by_obs(ds: object,
                     obs_da: object,
                     forecast_date: str,
                     frequency: Frequency = Frequency.DAY) -> object:
    """

    :param ds:
    :param obs_da:
    :param forecast_date: initialisation date of the forecast
    :param frequency: frequency of the observational dataset
    :return:
    """
    forecast_date = pd.to_datetime(forecast_date)
    delta_attribute = "{}s".format(frequency.attribute)
    (start_date, end_date) = (forecast_date + relativedelta(**{delta_attribute: int(ds.leadtime.min())}),
                              forecast_date + relativedelta(**{delta_attribute: int(ds.leadtime.max())}))

    if len(obs_da.time) < len(ds.leadtime):
        if len(obs_da.time) < 1:
            raise RuntimeError("No observational data available between {} "
                               "and {}".format(start_date.strftime("%D"),
                                               end_date.strftime("%D")))

        logging.warning("Observational data not available for full range of "
                        "forecast lead times: obs {}-{} vs fc {}-{}".format(
                            obs_da.time.to_series()[0].strftime(frequency.date_format),
                            obs_da.time.to_series()[-1].strftime(frequency.date_format),
                            start_date.strftime(frequency.date_format),
                            end_date.strftime(frequency.date_format)))

        (start_date, end_date) = (obs_da.time.to_series()[0],
                                  obs_da.time.to_series()[-1])

    # We broadcast to get a nicely compatible dataset for plotting
    return broadcast_forecast(start_date=start_date,
                              end_date=end_date,
                              dataset=ds,
                              frequency=frequency)


def calculate_extents(x1: int, x2: int, y1: int, y2: int):
    """

    :param x1:
    :param x2:
    :param y1:
    :param y2:
    :return:
    """
    data_extent_base = 5387500

    extents = [
        -data_extent_base + (x1 * 25000),
        data_extent_base - ((432 - x2) * 25000),
        -data_extent_base + (y1 * 25000),
        data_extent_base - ((432 - y2) * 25000),
    ]

    logging.debug("Data extents: {}".format(extents))
    return extents


def get_plot_axes(x1: int = 0,
                  x2: int = 432,
                  y1: int = 0,
                  y2: int = 432,
                  do_coastlines: bool = True,
                  north: bool = True,
                  south: bool = False):
    """

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
        extents = calculate_extents(x1, x2, y1, y2)
        ax.set_extent(extents, crs=proj)
    else:
        ax = fig.add_subplot(1, 1, 1)

    return ax


def show_img(ax,
             arr,
             x1: int = 0,
             x2: int = 432,
             y1: int = 0,
             y2: int = 432,
             cmap: object = None,
             do_coastlines: bool = True,
             vmin: float = 0.,
             vmax: float = 1.,
             north: bool = True,
             south: bool = False):
    """

    :param ax:
    :param arr:
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
        extents = calculate_extents(x1, x2, y1, y2)
        im = ax.imshow(arr,
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
