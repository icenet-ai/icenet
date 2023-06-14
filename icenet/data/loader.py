import argparse
import logging
import os

import numpy as np

from icenet.data.loaders import IceNetDataLoaderFactory
from icenet.data.cli import add_date_args, process_date_args
from icenet.utils import setup_logging

"""

"""


@setup_logging
def create_get_args():
    """

    :return:
    """
    implementations = list(IceNetDataLoaderFactory().loader_map)

    ap = argparse.ArgumentParser()
    ap.add_argument("name", type=str)
    ap.add_argument("hemisphere", choices=("north", "south"))

    ap.add_argument("-c", "--cfg-only", help="Do not generate data, "
                                             "only config", default=False,
                    action="store_true", dest="cfg")
    ap.add_argument("-d", "--dry",
                    help="Don't output files, just generate data",
                    default=False, action="store_true")
    ap.add_argument("-dt", "--dask-timeouts", type=int, default=120)
    ap.add_argument("-dp", "--dask-port", type=int, default=8888)
    ap.add_argument("-f", "--futures-per-worker", type=float, default=2.,
                    dest="futures")
    ap.add_argument("-fn", "--forecast-name", dest="forecast_name",
                    default=None, type=str)
    ap.add_argument("-fd", "--forecast-days", dest="forecast_days",
                    default=93, type=int)

    ap.add_argument("-i", "--implementation", type=str,
                    choices=implementations,
                    default=implementations[0])
    ap.add_argument("-l", "--lag", type=int, default=2)

    ap.add_argument("-ob", "--output-batch-size", dest="batch_size", type=int,
                    default=8)

    ap.add_argument("-p", "--pickup", help="Skip existing tfrecords",
                    default=False, action="store_true")
    ap.add_argument("-t", "--tmp-dir", help="Temporary directory",
                    default="/local/tmp", dest="tmp_dir", type=str)

    ap.add_argument("-v", "--verbose", action="store_true", default=False)
    ap.add_argument("-w", "--workers", help="Number of workers to use "
                                            "generating sets",
                    type=int, default=2)

    add_date_args(ap)
    args = ap.parse_args()
    return args


def create():
    """

    """
    args = create_get_args()
    dates = process_date_args(args)

    dl = IceNetDataLoaderFactory().create_data_loader(
        args.implementation,
        "loader.{}.json".format(args.name),
        args.forecast_name if args.forecast_name else args.name,
        args.lag,
        dates_override=dates
        if sum([len(v) for v in dates.values()]) > 0 else None,
        dry=args.dry,
        n_forecast_days=args.forecast_days,
        north=args.hemisphere == "north",
        south=args.hemisphere == "south",
        output_batch_size=args.batch_size,
        pickup=args.pickup,
        generate_workers=args.workers,
        dask_port=args.dask_port,
        futures_per_worker=args.futures)

    if args.cfg:
        dl.write_dataset_config_only()
    else:
        dl.generate()


def save_sample(output_folder: str,
                date: object,
                sample: tuple):
    """

    :param output_folder:
    :param date:
    :param sample:
    """
    net_input, net_output, sample_weights = sample

    if os.path.exists(output_folder):
        logging.warning("{} output already exists".format(output_folder))
    os.makedirs(output_folder, exist_ok=output_folder)

    for date, output, directory in ((date, net_input, "input"),
                                    (date, net_output, "outputs"),
                                    (date, sample_weights, "weights")):
        output_directory = os.path.join(output_folder, "loader", directory)
        os.makedirs(output_directory, exist_ok=True)
        loader_output_path = os.path.join(output_directory,
                                          date.strftime("%Y_%m_%d.npy"))

        logging.info("Saving {} - generated {} {}".
                     format(date, directory, output.shape))
        np.save(loader_output_path, output)
