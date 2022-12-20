import datetime as dt
import glob
import logging
import os

import pandas as pd
import xarray as xr

from icenet.process.forecasts import broadcast_forecast
from icenet.data.sic.mask import Masks


def get_seas_forecast_da(hemisphere: str,
                         date: str,
                         bias_correct: bool = True,
                         source_path: object = os.path.join(".", "data", "mars.seas"),
                         ) -> tuple:
    """
    Atmospheric model Ensemble 15-day forecast (Set III - ENS)

    :param hemisphere:
    :param date:
    :param bias_correct:
    :param source_path:
    """

    # TODO: why aren't we using SEASDownloader?
    # TODO: we could download here potentially

    seas_file = os.path.join(
        source_path,
        hemisphere,
        "siconca",
        "{}.nc".format(date.strftime("%Y%m%d")))
    seas_ds = xr.open_dataset(seas_file)

    return seas_ds.siconc


def get_forecast_ds(forecast_file: object,
                    forecast_date: str,
                    stddev: bool = False
                    ) -> tuple:
    """

    :param forecast_file:
    :param forecast_date:
    :param stddev:
    :returns tuple(fc_ds, obs_ds, land_mask):
    """
    forecast_date = pd.to_datetime(forecast_date)

    forecast_ds = xr.open_dataset(forecast_file)
    get_key = "sic_mean" if not stddev else "sic_stddev"

    forecast_ds = getattr(
        forecast_ds.sel(time=slice(forecast_date, forecast_date)),
        get_key)

    return forecast_ds


def filter_ds_by_obs(ds: object,
                     obs_da: object,
                     forecast_date: str) -> object:
    """

    :param ds:
    :param obs_da:
    :param forecast_date:
    :return:
    """
    forecast_date = pd.to_datetime(forecast_date)
    (start_date, end_date) = (
            forecast_date + dt.timedelta(days=int(ds.leadtime.min())),
            forecast_date + dt.timedelta(days=int(ds.leadtime.max()))
    )

    if len(obs_da.time) < len(ds.leadtime):
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
    return broadcast_forecast(start_date=start_date,
                              end_date=end_date,
                              dataset=ds)


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
