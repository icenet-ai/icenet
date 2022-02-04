import argparse
import logging
import os
import shutil

import xarray as xr


from icenet2.process.utils import date_arg, destination_filename


def upload_parse_args():
    a = argparse.ArgumentParser()

    a.add_argument("filename")
    a.add_argument("destination")
    a.add_argument("date", default=None, type=date_arg, nargs="?")

    a.add_argument("-v", "--verbose", default=False, action="store_true")

    return a.parse_args()


def upload():
    args = upload_parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    logging.info("Local upload facility")

    if not os.path.isdir(args.destination):
        raise RuntimeError("Destination {} does not exist".
                           format(args.destination))

    if args.date:
        ds = xr.open_dataset(args.filename)
        ds = ds.sel(time=slice(args.date, args.date))

        if len(ds.time) < 1:
            raise ValueError("No elements in {} for {}".format(
                args.filename, args.date
            ))

        filename = destination_filename(args.destination,
                                        args.filename,
                                        args.date)
        ds.to_netcdf(filename)
        ds.close()
    else:
        newname = shutil.copy(args.filename, args.destination)
        logging.info("Copied to {}".format(newname))
