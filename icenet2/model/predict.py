import collections
import datetime as dt
import json
import logging
import os

import icenet2.model.models as models

from icenet2.data.loader import IceNetDataSet

import numpy as np
import tensorflow as tf

from tensorflow.keras.models import load_model


def predict_forecast(
    dataset_config,
    network_name,
    dataset_name=None,
    model_func=models.unet_batchnorm,
    n_filters_factor=1/8,
    network_folder=None,
    seed=42,
    start_dates=tuple([dt.datetime.now().date()]),
    testset=False,
):
    ds = IceNetDataSet(dataset_config)
    dl = ds.get_data_loader()

    if not testset:
        logging.info("Generating forecast inputs from processed/ files")

        forecast_inputs, gen_outputs = \
            list(zip(*[dl.generate_sample(date) for date in start_dates]))
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
                               format(", ".join(missing)))

        forecast_inputs, gen_outputs = [], []

        x, y = [], [] 
        data_iter = test_inputs.as_numpy_iterator()

        data = next(data_iter)
        x, y = data
        batch = 0
        batch_len = None

        for i, idx in enumerate([test_dates.index(sd) for sd in start_dates]):
            while batch < int(idx / ds.batch_size):
                data = next(data_iter)
                x, y = data
                batch += 1
            arr_idx = idx % ds.batch_size
            logging.info("Processing batch {} - item {}".format(batch + 1, arr_idx))
            logging.debug(", ".join([str(v) for v in [idx, arr_idx, batch, type(arr_idx)]]))
            forecast_inputs.append(x[arr_idx, ...])
            gen_outputs.append(y[arr_idx, ...])

    if not network_folder:
        network_folder = os.path.join(".", "results", "networks",
                                      "{}.{}".format(network_name, seed))

    # FIXME: this is a mess as we have seed duplication in filename, sort it out
    dataset_name = dataset_name if dataset_name else ds.identifier
    network_path = os.path.join(network_folder,
                                "{}.{}.network_{}.{}.h5".
                                format(network_name, seed, dataset_name, seed))

    logging.info("Loading model from {}...".format(network_path))

    network = model_func(
        (*ds.shape, dl.num_channels),
        [],
        [],
        n_filters_factor=n_filters_factor,
        n_forecast_days=ds.n_forecast_days
    )
    network.load_weights(network_path)

    predictions = []

    for i, net_input in enumerate(forecast_inputs):
        logging.info("Running prediction {} - {}".format(i, start_dates[i]))
        pred = network(tf.convert_to_tensor([net_input]), training=False)
        predictions.append(pred)
    return predictions, gen_outputs


# TODO: better method via click via single 'icenet' entry point
def cli():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input_path")
    parser.add_argument("output_path")
    parser.add_argument("data_configuration")
    parser.add_argument("-n", "--network_path", default=None)
    # TODO: mechanism for dynamic lookup via importlib
    parser.add_argument("-m", "--model_func", default=models.unet_batchnorm)
    parser.add_argument("-s", "--seed", default=42, type=int)
    parser.add_argument("-f", "--n_forecast_days", default=93, type=int)
    # TODO: mechanism
    parser.add_argument("-d", "--start_dates", default=tuple([dt.datetime.now()
                                                             .date()]))
    args = parser.parse_args()

    logging.info("Prediction")
    predict_forecast(**vars(args))


if __name__ == "__main__":
    cli()
