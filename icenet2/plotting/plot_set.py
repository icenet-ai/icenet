import argparse
import json
import logging
import os
import subprocess

import numpy as np
import tensorflow as tf

from icenet2.data.dataset import get_decoder

import matplotlib.pyplot as plt


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("configuration", type=argparse.FileType("r"))
    ap.add_argument("-i", "--index", default=1, type=int)
    ap.add_argument("-o", "--output", default="plot")

    return ap.parse_args()


def main():
    args = parse_args()
    logging.getLogger().setLevel(logging.DEBUG)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

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
    subprocess.run("rm -v {}/*.png".format(output_dir), shell=True)

    for i, channel in enumerate(config['channels']):
        output_path = os.path.join(output_dir, "{:03d}_{}.png".
                                   format(i, channel))
        logging.info("Producing {}".format(output_path))

        fig, ax = plt.subplots()
        ax.contourf(x[0, ..., i])
        plt.savefig(output_path)
        plt.close()

    for i in range(config['n_forecast_days']):
        output_path = os.path.join(output_dir, "y.{:03d}.png".format(i))
        y_out = y[0, ..., i, 0]

        logging.info("Producing {}".format(output_path))

        if len(y_out.flatten()) == np.isnan(y_out).sum():
            logging.warning("Skipping {} due to fully nan".format(output_path))
        else:
            fig, ax = plt.subplots()
            ax.contourf(y_out)
            plt.savefig(output_path)
            plt.close()


if __name__ == "__main__":
    main()
