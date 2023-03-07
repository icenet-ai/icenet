import argparse
import datetime as dt
import logging
import os

import cf_units
import iris
import pandas as pd
import rasterio

import iris.analysis

from icenet.process.utils import date_arg
from icenet.utils import setup_logging

from icenet.plotting.utils import broadcast_forecast, get_forecast_ds


def reproject_output(forecast_file: object,
                     proj_file: object,
                     save_file: object) -> object:
    """

    :param forecast_file:
    :param proj_file:
    :param save_file:
    """
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


@setup_logging
def broadcast_args() -> argparse.Namespace:
    """CLI arguments for broadcasting several forecasts linearly through time

    """
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output-target", dest="target", default=None)
    ap.add_argument("start_date", type=date_arg)
    ap.add_argument("end_date", type=date_arg)
    ap.add_argument("datafiles", nargs="+")
    args = ap.parse_args()
    return args


def broadcast_main():
    """CLI entry point for icenet_output_broadcast

    """
    args = broadcast_args()
    broadcast_forecast(args.start_date,
                       args.end_date,
                       args.datafiles,
                       args.target)


@setup_logging
def reproject_args() -> argparse.Namespace:
    """CLI args for reprojecting against another file

    """
    ap = argparse.ArgumentParser()
    ap.add_argument("forecast_file")
    ap.add_argument("proj_file")
    ap.add_argument("save_file")
    args = ap.parse_args()
    return args


def reproject_main():
    """CLI entry point for icenet_output_reproject

    """
    args = reproject_args()
    reproject_output(args.forecast_file, args.proj_file, args.save_file)


@setup_logging
def geotiff_args() -> argparse.Namespace:
    """CLI args for creating geotiffs

    """
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output-path", default=".")
    ap.add_argument("-s", "--stddev",
                    help="Plot the standard deviation from the ensemble",
                    action="store_true",
                    default=False)
    ap.add_argument("-v", "--verbose", default=False, action="store_true")
    ap.add_argument("forecast_file")
    ap.add_argument("forecast_date")
    ap.add_argument("leadtimes",
                    help="Leadtimes to output, multiple as CSV, range as n..n",
                    type=lambda s: [int(i) for i in
                                    list(s.split(",") if "," in s else
                                         range(int(s.split("..")[0]),
                                               int(s.split("..")[1]) + 1) if ".." in s else
                                         [s])])

    args = ap.parse_args()
    return args


def create_geotiff_output():
    """CLI entry point for icenet_output_geotiff

    """
    args = geotiff_args()

    if not os.path.isdir(args.output_path):
        logging.warning("No directory at: {}, creating".
                        format(args.output_path))
        os.makedirs(args.output_path)
    elif os.path.isfile(args.output_path):
        raise RuntimeError("{} should be a directory and not existent...".
                           format(args.output_path))

    ds = get_forecast_ds(args.forecast_file,
                         args.forecast_date,
                         stddev=args.stddev)
    ds = ds.isel(time=0).transpose(..., "yc", "xc")

    # The projection information set when we create NetCDF output compliant
    # with CF standards still has units as meters, but the attributes on the
    # variables is 1000 meters. It's easier to reset this for the GeoTIFF output
    # else you'll get scale errors that are a pain to fix in downstream
    x_meters = ds.xc * 1000
    y_meters = ds.yc * 1000
    x_attrs = ds.xc.attrs
    y_attrs = ds.yc.attrs

    ds = ds.assign_coords(xc=x_meters, yc=y_meters)
    ds['xc'].attrs = x_attrs
    ds['yc'].attrs = y_attrs
    ds['xc'].attrs['units'] = cf_units.Unit('meters')
    ds['yc'].attrs['units'] = cf_units.Unit('meters')

    if type(ds.rio.crs) != rasterio.crs.CRS:
        raise RuntimeError("Did not extract CRS via the coordinates, ds.rio.crs"
                           " is not of type rasterio.crs.CRS")

    leadtimes = args.leadtimes \
        if args.leadtimes is not None \
        else list(range(1, int(max(ds.leadtime.values)) + 1))

    forecast_name = "{}.{}".format(
        os.path.splitext(os.path.basename(args.forecast_file))[0],
        args.forecast_date)

    logging.info("Selecting and outputting files from {} for {}".
                 format(args.forecast_file, args.forecast_date))

    for leadtime in leadtimes:
        pred_da = ds.sel(leadtime=leadtime)

        output_filename = os.path.join(args.output_path, "{}.{}.{}tiff".format(
            forecast_name,
            (pd.to_datetime(args.forecast_date) + dt.timedelta(
                days=leadtime)).strftime("%Y-%m-%d"),
            "" if not args.stddev else "stddev."
        ))

        logging.debug("Outputting leadtime {} to {}".
                      format(leadtime, output_filename))
        pred_da.rio.to_raster(output_filename)
