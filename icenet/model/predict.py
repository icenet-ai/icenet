import argparse
import datetime as dt
import logging
import os
import re

import numpy as np
import pandas as pd
import tensorflow as tf

import icenet.model.models as models

from icenet.data.dataset import IceNetDataSet
from icenet.utils import setup_logging

"""

"""


def predict_forecast(
    dataset_config: object,
    network_name: object,
    dataset_name: object = None,
    model_func: callable = models.unet_batchnorm,
    n_filters_factor: float = 1 / 8,
    network_folder: object = None,
    output_folder: object = None,
    seed: int = 42,
    start_dates: object = tuple([dt.datetime.now().date()]),
    test_set: bool = False,
) -> object:
    """

    :param dataset_config:
    :param network_name:
    :param dataset_name:
    :param model_func:
    :param n_filters_factor:
    :param network_folder:
    :param output_folder:
    :param seed:
    :param start_dates:
    :param test_set:
    :return:
    """
    # TODO: going to need to be able to handle merged datasets
    ds = IceNetDataSet(dataset_config)
    dl = ds.get_data_loader()

    if not network_folder:
        network_folder = os.path.join(".", "results", "networks", network_name)

    dataset_name = dataset_name if dataset_name else ds.identifier
    network_path = os.path.join(network_folder,
                                "{}.network_{}.{}.h5".format(network_name,
                                                             dataset_name,
                                                             seed))

    logging.info("Loading model from {}...".format(network_path))

    network = model_func(
        (*ds.shape, dl.num_channels),
        [],
        [],
        n_filters_factor=n_filters_factor,
        n_forecast_days=ds.n_forecast_days
    )
    network.load_weights(network_path)

    if not test_set:
        logging.info("Generating forecast inputs from processed/ files")

        for date in start_dates:
            run_prediction(network,
                           date,
                           output_folder,
                           *dl.generate_sample(date, prediction=True))
    else:
        # TODO: This is horrible behaviour, rethink and refactor: we should
        #  be able to pull from the test set in a nicer and more efficient
        #  fashion
        _, _, test_inputs = ds.get_split_datasets()

        source_key = [k for k in dl.config['sources'].keys() if k != "meta"][0]
        # FIXME: should be using date format from class
        test_dates = [dt.date(*[int(v) for v in d.split("_")]) for d in
                      dl.config["sources"][source_key]["dates"]["test"]]

        if len(test_dates) == 0:
            raise RuntimeError("No processed files were produced for the test "
                               "set")

        missing = set(start_dates).difference(test_dates)
        if len(missing) > 0:
            raise RuntimeError("{} are not in the test set".
                               format(", ".join([str(pd.to_datetime(el).date()) 
                                                 for el in missing])))

        data_iter = test_inputs.as_numpy_iterator()
        # FIXME: this is broken, this entry never gets added to the set?
        data = next(data_iter)
        x, y, sw = data
        batch = 0

        for i, idx in enumerate([test_dates.index(sd) for sd in start_dates]):
            while batch < int(idx / ds.batch_size):
                data = next(data_iter)
                x, y, sw = data
                batch += 1
            arr_idx = idx % ds.batch_size
            logging.info("Processing test batch {}, item {} (date {})".format(
                batch + 1, arr_idx, test_dates[idx]))

            run_prediction(network,
                           test_dates[idx],
                           output_folder,
                           x[arr_idx, ...],
                           y[arr_idx, ...],
                           sw[arr_idx, ...])


def run_prediction(network, date, output_folder,
                   net_input, net_output, sample_weights):
    logging.info("Running prediction {}".format(date))
    pred = network(tf.convert_to_tensor([net_input]), training=False)

    if os.path.exists(output_folder):
        logging.warning("{} output already exists".format(output_folder))
    os.makedirs(output_folder, exist_ok=output_folder)
    output_path = os.path.join(output_folder, date.strftime("%Y_%m_%d.npy"))

    logging.info("Saving {} - forecast output {}".format(date, pred.shape))
    np.save(output_path, pred)

    logging.debug("Saving loader generated data for reference...")

    for date, output, directory in \
            ((date, net_input, "input"),
             (date, net_output, "outputs"),
             (date, sample_weights, "weights")):
        output_directory = os.path.join(output_folder, "loader", directory)
        os.makedirs(output_directory, exist_ok=True)
        loader_output_path = os.path.join(output_directory,
                                          date.strftime("%Y_%m_%d.npy"))

        logging.info("Saving {} - generated {} {}".
                     format(date, directory, output.shape))
        np.save(loader_output_path, output)

    return output_path


def date_arg(string: str) -> object:
    """

    :param string:
    :return:
    """
    date_match = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", string)
    return dt.date(*[int(s) for s in date_match.groups()])


@setup_logging
def get_args():
    """

    :return:
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("network_name")
    ap.add_argument("output_name")
    ap.add_argument("seed", type=int, default=42)
    ap.add_argument("datefile", type=argparse.FileType("r"))

    ap.add_argument("-i", "--train-identifier", dest="ident",
                    help="Train dataset identifier", type=str, default=None)
    ap.add_argument("-n", "--n-filters-factor", type=float, default=1.)
    ap.add_argument("-t", "--testset", default=False, action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true", default=False)

    return ap.parse_args()


def main():
    args = get_args()

    dataset_config = \
        os.path.join(".", "dataset_config.{}.json".format(args.dataset))

    date_content = args.datefile.read()
    dates = [dt.date(*[int(v) for v in s.split("-")])
             for s in date_content.split()]
    args.datefile.close()

    output_folder = os.path.join(".", "results", "predict",
                                 args.output_name,
                                 "{}.{}".format(args.network_name, args.seed))

    predict_forecast(dataset_config,
                     args.network_name,
                     # FIXME: this is turning into a mapping mess,
                     #  do we need to retain the train SD name in the
                     #  network?
                     dataset_name=
                     args.ident if args.ident else args.dataset,
                     n_filters_factor=
                     args.n_filters_factor,
                     output_folder=output_folder,
                     seed=args.seed,
                     start_dates=dates,
                     test_set=args.testset)

