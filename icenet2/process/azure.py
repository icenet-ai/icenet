import argparse
import configparser
import logging
import os

from azure.storage.blob import ContainerClient

# https://docs.microsoft.com/en-us/azure/developer/python/sdk/storage/storage-blob-readme?view=storage-py-v12#next-steps

def upload_parse_args():
    a = argparse.ArgumentParser()

    a.add_argument("filename")

    a.add_argument("-c", "--container", default="input", type=str)
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

    logging.info("Connecting client")

    container_client = \
        ContainerClient.from_connection_string(url,
                                               container_name=args.container)

    blobs = container_client.list_blobs()
    filenames = [b['name'] for b in blobs]

    logging.info("{} files already in container {}".format(
        len(filenames), args.container
    ))

    if args.filename not in filenames:
        with open(args.filename, "rb") as data:
            logging.info("Uploading {}".format(args.filename))
            container_client.upload_blob(
                args.filename, data, overwrite=args.overwrite)


if __name__ == "__main__":
    upload()
