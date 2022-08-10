import datetime as dt
import glob
import logging
import os

import pandas as pd
import xarray as xr

from icenet2.process.forecasts import broadcast_forecast
from icenet2.data.sic.mask import Masks


def get_forecast_obs_ds(hemisphere: str,
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
    forecast_ds = forecast_ds.sel(time=slice(forecast_date,forecast_date))

    if len(forecast_ds.time) != 1:
        raise ValueError("Dataset does not contain {}: \n{}".format(forecast_date, forecast_ds))

    obs_years = [forecast_date.year]
    max_lead_year = (forecast_date +
                     dt.timedelta(days=int(forecast_ds.leadtime.max()))).year

    if max_lead_year == forecast_date.year + 1:
        obs_years.append(max_lead_year)

    obs_dfs = [el for yr in obs_years for el in
               glob.glob(os.path.join(obs_source,
                                      hemisphere,
                                      "siconca", "{}.nc".format(yr)))]

    if len(obs_dfs) < len(obs_years):
        logging.warning("Cannot find necessary obs source files for {} in {}".
                        format(obs_years, obs_source))

    obs_ds = xr.open_mfdataset(obs_dfs)
    (start_date, end_date) = (
            forecast_date + dt.timedelta(days=int(forecast_ds.leadtime.min())),
            forecast_date + dt.timedelta(days=int(forecast_ds.leadtime.max()))
    )
    obs_ds = obs_ds.sel(time=slice(start_date, end_date))

    if len(obs_ds.time) < len(forecast_ds.leadtime):
        logging.warning("Observational data not available for full range of "
                        "forecast leadtimes: {}-{} vs {}-{}".format(
                         obs_ds.time.to_series()[0].strftime("%D"),
                         obs_ds.time.to_series()[-1].strftime("%D"),
                         start_date.strftime("%D"),
                         end_date.strftime("%D")))
        (start_date, end_date) = (
            obs_ds.time.to_series()[0],
            obs_ds.time.to_series()[-1]
        )

    # We broadcast to get a nicely compatible dataset for plotting
    forecast_ds = broadcast_forecast(
        start_date, end_date, dataset=forecast_ds)

    get_key = "sic_mean" if not stddev else "sic_stddev"
    return getattr(forecast_ds, get_key), obs_ds.ice_conc, land_mask
