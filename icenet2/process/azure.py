import argparse
import datetime as dt
import configparser
import logging
import os
import re
import shutil
import tempfile

import xarray as xr

from azure.storage.blob import ContainerClient

# https://docs.microsoft.com/en-us/azure/developer/python/sdk/storage/storage-blob-readme?view=storage-py-v12#next-steps


def date_arg(string):
    d_match = re.search(r'^(\d+)-(\d+)-(\d+)$', string).groups()

    if d_match:
        return dt.date(*[int(s) for s in d_match])


def upload_parse_args():
    a = argparse.ArgumentParser()

    a.add_argument("filename")
    a.add_argument("date", default=None, type=date_arg, nargs="?")

    a.add_argument("-c", "--container", default="input", type=str)
    a.add_argument("-l", "--leave", default=False, action="store_true")
    a.add_argument("-o", "--overwrite", default=False, action="store_true")
    a.add_argument("-v", "--verbose", default=False, action="store_true")

    return a.parse_args()


def upload():
    args = upload_parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
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

            filename = os.path.join(tmpdir, args.filename)
            ds.to_netcdf(filename)
        else:
            filename = args.filename

        with open(filename, "rb") as data:
            logging.info("Uploading {}".format(filename))
            logging.info("Connecting client")

            container_client = \
                ContainerClient.from_connection_string(url,
                                                       container_name=args.container)
            container_client.upload_blob(
                filename, data, overwrite=args.overwrite)
    finally:
        if args.date and not args.leave:
            logging.info("Removing {}".format(tmpdir))
            shutil.rmtree(tmpdir)


if __name__ == "__main__":
    upload()
