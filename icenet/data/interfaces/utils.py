import argparse
import collections
import glob
import logging
import os

import pandas as pd
import xarray as xr

from icenet.utils import setup_logging


def batch_requested_dates(dates: object,
                          attribute: str = "month") -> object:
    """

    TODO: should be using Pandas DatetimeIndexes / Periods for this, but the
     need to refactor slightly, and this is working for the moment

    :param dates:
    :param attribute:
    :return:
    """
    dates = collections.deque(sorted(dates))

    batched_dates = []
    batch = []

    while len(dates):
        if not len(batch):
            batch.append(dates.popleft())
        else:
            if getattr(batch[-1], attribute) == getattr(dates[0], attribute):
                batch.append(dates.popleft())
            else:
                batched_dates.append(batch)
                batch = []

    if len(batch):
        batched_dates.append(batch)

    if len(dates) > 0:
        raise RuntimeError("Batching didn't work!")

    return batched_dates


def reprocess_monthlies(source: str,
                        hemisphere: str,
                        identifier: str,
                        output_base: str,
                        dry: bool = False,
                        var_names: object = None):
    """

    :param source:
    :param hemisphere:
    :param identifier:
    :param output_base:
    :param dry:
    :param var_names:
    """
    if not var_names:
        var_names = []

    for var_name in var_names:
        var_path = os.path.join(source, hemisphere, var_name)
        files = glob.glob("{}/{}_*.nc".format(var_path, var_name))

        for file in files:
            _, year = os.path.basename(os.path.splitext(file)[0]).\
                          split("_")[0:2]

            try:
                year = int(year)

                if not (1900 < year < 2200):
                    logging.warning("File is not between 1900-2200, probably "
                                    "not something to convert: {}".format(year))
            except ValueError:
                logging.warning("Cannot derive year from {}".format(year))
                continue

            destination = os.path.join(output_base,
                                       identifier,
                                       hemisphere,
                                       var_name,
                                       str(year))

            if not os.path.exists(destination):
                os.makedirs(destination, exist_ok=True)

            logging.info("Processing {} from {} to {}".
                         format(var_name, year, destination))

            ds = xr.open_dataset(file)

            var_names = [name for name in list(ds.data_vars.keys())
                         if not name.startswith("lambert_")]

            var_names = set(var_names)
            logging.debug("Files have var names {} which will be renamed to {}".
                          format(", ".join(var_names), var_name))

            ds = ds.rename({k: var_name for k in var_names})
            da = getattr(ds, var_name)

            for date in da.time.values:
                date = pd.Timestamp(date)
                fname = '{:04d}_{:02d}_{:02d}.nc'. \
                    format(date.year, date.month, date.day)
                daily = da.sel(time=slice(date, date))

                output_path = os.path.join(destination, fname)

                if dry or os.path.exists(output_path):
                    continue
                else:
                    daily.to_netcdf(output_path)


def add_time_dim(source: str,
                 hemisphere: str,
                 identifier: str,
                 dry: bool = False,
                 var_names: object = []):
    """

    :param source:
    :param hemisphere:
    :param identifier:
    :param dry:
    :param var_names:
    """
    files = {}

    for var_name in var_names:
        var_path = os.path.join(source, identifier, hemisphere, var_name)

        if var_name not in files:
            files[var_name] = {}

        file_list = glob.glob("{}/*/*.nc".format(var_path))

        for path, filename in [os.path.split(el) for el in file_list]:
            if filename.startswith("{}_".format(var_name)):
                raise RuntimeError("{} starts with var name, we only want "
                                   "correctly named files to convert".
                                   format(filename))
            year = str(path.split(os.sep)[-1])

            if year not in files[var_name]:
                files[var_name][year] = []

            src = os.path.join(path, filename)
            dest = os.path.join(path, "{}_{}".format(var_name, filename))

            if not dry:
                try:
                    os.rename(src, dest)
                except OSError as e:
                    logging.exception("Not able to move file to temporary"
                                      "destination {}".format(dest))
                    raise e
            else:
                logging.info("{} -> {}".format(src, dest))

            files[var_name][year].append(dest)

    for year_files in [files[var][el] for var in files for el in files[var]]:
        if not dry:
            ds = xr.open_mfdataset(year_files,
                                   combine="nested",
                                   concat_dim="time",
                                   parallel=True)

            if "siconca" in year_files[0]:
                ds = ds.rename_vars({"siconca": "ice_conc"})
                ds = ds.sortby("time")
                ds['time'] = [pd.Timestamp(el) for el in
                              ds.indexes['time'].normalize()]

            for d in ds.time.values:
                dt = pd.to_datetime(d)
                date_str = dt.strftime("%Y_%m_%d")
                fpath = os.path.join(os.path.split(year_files[0])[0],
                                     "{}.nc".format(date_str))

                if not os.path.exists(fpath):
                    dw = ds.sel(time=slice(dt, dt))

                    logging.info("Writing {}".format(fpath))
                    dw.to_netcdf(fpath)
                else:
                    raise RuntimeError("Already exists: {}".format(fpath))

            ds.close()

            for orig_file in year_files:
                logging.info("Removing {}".format(orig_file))
                os.unlink(orig_file)
        else:
            logging.info("Would process out: {}".format(year_files))


@setup_logging
def get_args():
    """

    :return:
    """
    a = argparse.ArgumentParser()
    a.add_argument("-d", "--dry", default=False, action="store_true")
    a.add_argument("-o", "--output", default="./data")
    a.add_argument("-v", "--verbose", default=False, action="store_true")
    a.add_argument("source")
    a.add_argument("hemisphere", choices=["nh", "sh"])
    a.add_argument("identifier")
    a.add_argument("vars", nargs='+')
    return a.parse_args()


def add_time_dim_main():
    """CLI stub to sort out missing time dimensions in daily data
    """

    args = get_args()
    logging.info("Temporary solution for sorting missing time dim in files")

    if args.output != "./data":
        raise RuntimeError("output is not used for this command: {}".format(
            args.output))

    add_time_dim(args.source, args.hemisphere, args.identifier,
                 dry=args.dry,
                 var_names=args.vars)


def reprocess_main():
    """CLI stub solution for reprocessing monthly files
    """
    args = get_args()
    logging.info("Temporary solution for reprocessing monthly files")
    reprocess_monthlies(args.source, args.hemisphere, args.identifier,
                        output_base=args.output,
                        dry=args.dry,
                        var_names=args.vars)

