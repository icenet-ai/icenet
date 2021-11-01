import argparse
import datetime as dt
import logging
import os
import re

import icenet2.model.models as models

from icenet2.data.dataset import IceNetDataSet

import numpy as np
import tensorflow as tf


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

        forecast_inputs, gen_outputs, _ = \
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

        for i, idx in enumerate([test_dates.index(sd) for sd in start_dates]):
            while batch < int(idx / ds.batch_size):
                data = next(data_iter)
                x, y = data
                batch += 1
            arr_idx = idx % ds.batch_size
            logging.info("Processing batch {} - item {}".format(
                batch + 1, arr_idx))
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


def date_arg(string):
    date_match = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", string)
    return dt.date(*[int(s) for s in date_match.groups()])


def get_args():
    # -b 1 -e 1 -w 1 -n 0.125
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", type=str)
    ap.add_argument("network_name", type=str)
    ap.add_argument("output_name", type=str)
    ap.add_argument("seed", type=int, default=42)
    ap.add_argument("datefile", type=argparse.FileType("r"))

    ap.add_argument("-n", "--n-filters-factor", type=float, default=1.)
    ap.add_argument("-o", "--skip-outputs", default=False, action="store_true")
    ap.add_argument("-t", "--testset", default=False, action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true", default=False)

    return ap.parse_args()


def main():
    args = get_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    dataset_config = \
        os.path.join(".", "dataset_config.{}.json".format(args.dataset))

    date_content = args.datefile.read()
    dates = [dt.date(*[int(v) for v in s.split("-")])
             for s in date_content.split()]
    args.datefile.close()

    output_dir = os.path.join(".", "results", "predict",
                              args.output_name,
                              "{}.{}".format(args.network_name, args.seed))

    forecasts, gen_outputs = predict_forecast(dataset_config,
                                              args.network_name,
                                              dataset_name=args.dataset,
                                              n_filters_factor=
                                              args.n_filters_factor,
                                              seed=args.seed,
                                              start_dates=dates,
                                              testset=args.testset)

    if os.path.exists(output_dir):
        raise RuntimeError("{} output already exists".format(output_dir))
    os.makedirs(output_dir)

    for date, forecast in zip(dates, forecasts):
        output_path = os.path.join(output_dir, date.strftime("%Y_%m_%d.npy"))

        logging.info("Saving {} - forecast output {}".
                     format(date, forecast.shape))
        np.save(output_path, forecast)

    if not args.skip_outputs:
        logging.info("Saving outputs generated for these inputs as well...")
        gen_dir = os.path.join(output_dir, "gen_outputs")
        os.makedirs(gen_dir)

        for date, output in zip(dates, gen_outputs):
            output_path = os.path.join(gen_dir, date.strftime("%Y_%m_%d.npy"))

            logging.info("Saving {} - generated output {}".
                         format(date, output.shape))
            np.save(output_path, output)

