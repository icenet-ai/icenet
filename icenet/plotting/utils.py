import datetime as dt
import glob
import logging
import os

import pandas as pd
import xarray as xr

from icenet.process.forecasts import broadcast_forecast
from icenet.data.sic.mask import Masks


def get_forecast_hres_obs_da(hemisphere: str,
                             start_date: str,
                             end_date: str,
                             bias_correct: bool = True,
                             obs_source: object =
                             os.path.join(".", "data", "osisaf"),
                             source_path: object =
                             os.path.join(".", "data", "mars.hres"),
                             ) -> tuple:
    """

    :param hemisphere:
    :param start_date:
    :param end_date:
    :param bias_correct:
    :param obs_source:
    :param source_path:
    """
    masks = Masks(
        north=hemisphere == "north",
        south=hemisphere == "south")
    land_mask = masks.get_land_mask()

    start_date, end_date = pd.to_datetime(start_date), pd.to_datetime(end_date)
    hres_years = list(set([start_date.year, end_date.year]))
    hres_files = [os.path.join(source_path, hemisphere, "siconca", "{}.nc".format(yr))
                  for yr in hres_years]
    hres_ds = xr.open_mfdataset(hres_files)

    hres_ds = hres_ds.assign_coords(
        dict(xc=hres_ds.xc / 1e3, yc=hres_ds.yc / 1e3))

    obs_da = get_obs_da(hemisphere, start_date, end_date, obs_source)

    return hres_ds.siconc, obs_da, land_mask


def get_forecast_obs_da(hemisphere: str,
                        forecast_file: object,
                        forecast_date: str,
                        obs_source: object =
                        os.path.join(".", "data", "osisaf"),
                        stddev: bool = False
                        ) -> tuple:
    """

    :param hemisphere:
    :param forecast_file:
    :param forecast_date:
    :param obs_source:
    :param stddev:
    :returns tuple(fc_ds, obs_ds, land_mask):
    """
    land_mask = Masks(
        north=hemisphere == "north",
        south=hemisphere == "south").get_land_mask()
    forecast_date = pd.to_datetime(forecast_date)

    forecast_ds = xr.open_dataset(forecast_file)
    forecast_ds = forecast_ds.sel(time=slice(forecast_date, forecast_date))

    if len(forecast_ds.time) != 1:
        raise ValueError("Dataset does not contain {}: \n{}".format(forecast_date, forecast_ds))

    (start_date, end_date) = (
            forecast_date + dt.timedelta(days=int(forecast_ds.leadtime.min())),
            forecast_date + dt.timedelta(days=int(forecast_ds.leadtime.max()))
    )

    obs_da = get_obs_da(hemisphere, start_date, end_date, obs_source)

    if len(obs_da.time) < len(forecast_ds.leadtime):
        logging.warning("Observational data not available for full range of "
                        "forecast leadtimes: {}-{} vs {}-{}".format(
                         obs_da.time.to_series()[0].strftime("%D"),
                         obs_da.time.to_series()[-1].strftime("%D"),
                         start_date.strftime("%D"),
                         end_date.strftime("%D")))
        (start_date, end_date) = (
            obs_da.time.to_series()[0],
            obs_da.time.to_series()[-1]
        )

    # We broadcast to get a nicely compatible dataset for plotting
    forecast_ds = broadcast_forecast(
        start_date, end_date, dataset=forecast_ds)

    get_key = "sic_mean" if not stddev else "sic_stddev"
    return getattr(forecast_ds, get_key), obs_da, land_mask


def get_obs_da(hemisphere: str,
               start_date: str,
               end_date: str,
               obs_source: object =
               os.path.join(".", "data", "osisaf"),
               ) -> object:
    """

    :param hemisphere:
    :param start_date:
    :param end_date:
    :param obs_source:
    :return:
    """
    obs_years = pd.Series(pd.date_range(start_date, end_date)).dt.year.unique()
    obs_dfs = [el for yr in obs_years for el in
               glob.glob(os.path.join(obs_source,
                                      hemisphere,
                                      "siconca", "{}.nc".format(yr)))]

    if len(obs_dfs) < len(obs_years):
        logging.warning("Cannot find necessary obs source files for {} in {}".
                        format(obs_years, obs_source))

    obs_ds = xr.open_mfdataset(obs_dfs)
    obs_ds = obs_ds.sel(time=slice(start_date, end_date))

    return obs_ds.ice_conc
