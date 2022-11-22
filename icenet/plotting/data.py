import argparse
import json
import logging
import os
import subprocess

import numpy as np
import tensorflow as tf

from icenet.data.datasets.utils import get_decoder
from icenet.utils import setup_logging

import matplotlib.pyplot as plt


@setup_logging
def tfrecord_args():
    """

    :return:
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("configuration", type=argparse.FileType("r"))
    ap.add_argument("-i", "--index", default=1, type=int)
    ap.add_argument("-l", "--levels", default=100, type=int)
    ap.add_argument("-o", "--output", default="plot")

    return ap.parse_args()


def plot_tfrecord():
    args = tfrecord_args()

    ds = tf.data.TFRecordDataset([args.file])
    config = json.load(args.configuration)
    args.configuration.close()

    decoder = get_decoder(tuple(config['shape']),
                          config['num_channels'],
                          config['n_forecast_days'])

    ds = ds.map(decoder).batch(1)
    it = ds.as_numpy_iterator()

    for _ in range(0, args.index):
        data = next(it)

    x, y, sample_weights = data
    logging.debug("x {}".format(x.shape))
    logging.debug("y {}".format(y.shape))
    logging.debug("sample_weights {}".format(sample_weights.shape))

    output_dir = os.path.join(args.output, "plot_set")
    os.makedirs(output_dir, exist_ok=True)
    subprocess.run("rm -v {}/{}.*.png".format(
        output_dir, config["identifier"]), shell=True)

    for i, channel in enumerate(config['channels']):
        output_path = os.path.join(output_dir, "{}.{:03d}_{}.png".
                                   format(config["identifier"], i, channel))
        logging.info("Producing {}".format(output_path))

        fig, ax = plt.subplots()
        ax.contourf(x[0, ..., i], levels=args.levels)
        plt.savefig(output_path)
        plt.close()

    for i in range(config['n_forecast_days']):
        output_path = os.path.join(output_dir, "{}.y.{:03d}.png".
                                   format(config["identifier"], i + 1))
        y_out = y[0, ..., i, 0]

        logging.info("Producing {}".format(output_path))

        if len(y_out.flatten()) == np.isnan(y_out).sum():
            logging.warning("Skipping {} due to fully nan".format(output_path))
        else:
            fig, ax = plt.subplots()
            ax.contourf(y_out, levels=args.levels)
            plt.savefig(output_path)
            plt.close()
