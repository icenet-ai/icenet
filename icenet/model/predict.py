import argparse
import datetime as dt
import logging
import os
import re

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import load_model

from icenet.data.loader import save_sample
from icenet.data.network_dataset import IceNetDataSet
from icenet.model.cli import predict_args

"""

"""


def predict_forecast(
    dataset_config: object,
    network_name: object,
    dataset_name: object = None,
    network_folder: object = None,
    output_folder: object = None,
    save_args: bool = False,
    seed: int = 42,
    start_dates: object = tuple([dt.datetime.now().date()]),
    test_set: bool = False,
) -> object:
    """

    :param dataset_config:
    :param network_name:
    :param dataset_name:
    :param legacy_rounding:
    :param model_func:
    :param n_filters_factor:
    :param network_folder:
    :param output_folder:
    :param save_args:
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
    model_path = os.path.join(
        network_folder, "{}.model_{}.{}".format(network_name,
                                                dataset_name,
                                                seed))

    logging.info("Loading model from {}...".format(model_path))

    network = load_model(model_path, compile=False)

    if not test_set:
        logging.info("Generating forecast inputs from processed/ files")

        for date in start_dates:
            data_sample = dl.generate_sample(date, prediction=True)
            run_prediction(network=network,
                           date=date,
                           output_folder=output_folder,
                           data_sample=data_sample,
                           save_args=save_args)
    else:
        # TODO: This is horrible behaviour, rethink and refactor: we should
        #  be able to pull from the test set in a nicer and more efficient
        #  fashion
        _, _, test_inputs = ds.get_split_datasets()

        source_key = [k for k in dl.config['sources'].keys() if k != "meta"][0]
        # FIXME: should be using date format from class
        test_dates = [
            dt.date(*[int(v)
                      for v in d.split("_")])
            for d in dl.config["sources"][source_key]["dates"]["test"]
        ]

        if len(test_dates) == 0:
            raise RuntimeError("No processed files were produced for the test "
                               "set")

        missing = set(start_dates).difference(test_dates)
        if len(missing) > 0:
            raise RuntimeError("{} are not in the test set".format(", ".join(
                [str(pd.to_datetime(el).date()) for el in missing])))

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

            run_prediction(network=network,
                           date=test_dates[idx],
                           output_folder=output_folder,
                           data_sample=(x[arr_idx, ...],
                                        y[arr_idx, ...],
                                        sw[arr_idx, ...]),
                           save_args=save_args)


def run_prediction(network, date, output_folder, data_sample, save_args):
    net_input, net_output, sample_weights = data_sample

    logging.info("Running prediction {}".format(date))
    pred = network(tf.convert_to_tensor([net_input]), training=False)

    if os.path.exists(output_folder):
        logging.warning("{} output already exists".format(output_folder))
    os.makedirs(output_folder, exist_ok=output_folder)
    output_path = os.path.join(output_folder, date.strftime("%Y_%m_%d.npy"))

    logging.info("Saving {} - forecast output {}".format(date, pred.shape))
    np.save(output_path, pred)

    if save_args:
        logging.debug("Saving loader generated data for reference...")
        save_sample(os.path.join(output_path, "loader"), date, data_sample)

    return output_path


def main():
    args = predict_args()

    dataset_config = \
        os.path.join(".", "dataset_config.{}.json".format(args.dataset))

    date_content = args.datefile.read()
    dates = [
        dt.date(*[int(v) for v in s.split("-")]) for s in date_content.split()
    ]
    args.datefile.close()

    output_folder = os.path.join(".", "results", "predict", args.output_name,
                                 "{}.{}".format(args.network_name, args.seed))

    predict_forecast(
        dataset_config,
        args.network_name,
        # FIXME: this is turning into a mapping mess,
        #  do we need to retain the train SD name in the
        #  network?
        dataset_name=args.ident if args.ident else args.dataset,
        output_folder=output_folder,
        save_args=args.save_args,
        seed=args.seed,
        start_dates=dates,
        test_set=args.testset)
