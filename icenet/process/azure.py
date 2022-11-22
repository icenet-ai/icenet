import argparse
import configparser
import logging
import os
import shutil
import tempfile

import xarray as xr

from icenet.process.utils import date_arg, destination_filename
from icenet.utils import setup_logging

from azure.storage.blob import ContainerClient

# https://docs.microsoft.com/en-us/azure/developer/python/sdk/storage/storage-blob-readme?view=storage-py-v12#next-steps


@setup_logging
def upload_parse_args():
    """

    :return:
    """
    a = argparse.ArgumentParser()

    a.add_argument("filename")
    a.add_argument("date", default=None, type=date_arg, nargs="?")

    a.add_argument("-c", "--container", default="input", type=str)
    a.add_argument("-l", "--leave", default=False, action="store_true")
    a.add_argument("-o", "--overwrite", default=False, action="store_true")
    a.add_argument("-v", "--verbose", default=False, action="store_true")

    return a.parse_args()


def upload():
    """

    """
    args = upload_parse_args()
    logging.info("Azure upload facility")

    url = os.getenv("AZURE_STORAGE_CONNECTION_STRING")

    if not url:
        try:
            ini = configparser.RawConfigParser()
            ini.read(os.path.expandvars("$HOME/.icenet.conf"))
            url = ini.get("azure", "connection_string")
        except configparser.Error as e:
            logging.exception("Configuration is not correctly set up")
            raise e

    try:
        if args.date:
            tmpdir = tempfile.mkdtemp(dir=".")
            ds = xr.open_dataset(args.filename)
            ds = ds.sel(time=slice(args.date, args.date))

            if len(ds.time) < 1:
                raise ValueError("No elements in {} for {}".format(
                    args.filename, args.date
                ))

            filename = destination_filename(tmpdir, args.filename, args.date)
            ds.to_netcdf(filename)
            ds.close()
        else:
            filename = args.filename

        with open(filename, "rb") as data:
            logging.info("Uploading {}".format(filename))
            logging.info("Connecting client")

            container_client = \
                ContainerClient.\
                from_connection_string(url, container_name=args.container)
            container_client.upload_blob(
                os.path.basename(filename), data, overwrite=args.overwrite)
    finally:
        if args.date and not args.leave:
            logging.info("Removing {}".format(tmpdir))
            shutil.rmtree(tmpdir)

