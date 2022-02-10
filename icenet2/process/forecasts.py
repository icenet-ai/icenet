import argparse
import datetime as dt
import logging
import os

import iris
import pandas as pd
import xarray as xr

import iris.analysis

from icenet2.process.utils import date_arg


def broadcast_forecast(start_date, end_date, datafiles, target=None):
    logging.info("Using {} to generate forecast through {} to {}".
                 format(", ".join(datafiles), start_date, end_date))
    ds = xr.open_mfdataset(datafiles, engine="netcdf4")
    dates = pd.date_range(start_date, end_date)
    i = 0

    if len(ds.time.values) > 1:
        while ds.time.values[i + 1] < dates[0]:
            i += 1

    logging.info("Starting index will be {} for {} - {}".
                 format(i, dates[0], dates[-1]))
    dt_arr = []

    for d in dates:
        logging.debug("Looking for date {}".format(d))
        arr = None

        while not arr:
            if d >= ds.time.values[i]:
                d_lead = (d - ds.time.values[i]).days

                if i + 1 < len(ds.time.values):
                    if pd.to_datetime(ds.time.values[i]) + \
                            dt.timedelta(days=d_lead) >= \
                            pd.to_datetime(ds.time.values[i + 1]) + \
                            dt.timedelta(days=1):
                        i += 1
                        continue

                logging.debug("Selecting date {} and lead {}".
                              format(ds.time.values[i], d_lead))

                arr = ds.sel(time=ds.time.values[i], leadtime=d_lead).\
                    copy().\
                    drop("time").\
                    assign_coords(dict(time=d)).\
                    drop("leadtime")
            else:
                i += 1

        dt_arr.append(arr)

    target_ds = xr.concat(dt_arr, dim="time")

    if target:
        logging.info("Generating dataset to {}".format(target))
        target_ds.to_netcdf(target)
    return target_ds


def reproject_output(forecast_file, proj_file, save_file):
    logging.info("Loading forecast {}".format(forecast_file))
    forecast_cube = iris.load_cube(forecast_file)

    logging.info("Projecting as per {}".format(proj_file))
    gp = iris.load_cube(proj_file)

    forecast_cube.coord('projection_y_coordinate').convert_units('meters')
    forecast_cube.coord('projection_x_coordinate').convert_units('meters')

    logging.info("Attempting to reproject and save to {}".
                 format(save_file))
    latlon_cube = forecast_cube.regrid(gp, iris.analysis.Linear())
    iris.save(latlon_cube, save_file)


def broadcast_main():
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output-target", dest="target", default=None)
    ap.add_argument("start_date", type=date_arg)
    ap.add_argument("end_date", type=date_arg)
    ap.add_argument("datafiles", nargs="+")
    args = ap.parse_args()
    broadcast_forecast(args.start_date, args.end_date,
                       args.datafiles, args.target)


def reproject_main():
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("forecast_file")
    ap.add_argument("proj_file")
    ap.add_argument("save_file")
    args = ap.parse_args()
    reproject_output(args.forecast_file, args.proj_file, args.save_file)
