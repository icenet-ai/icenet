import argparse
import json
import logging
import os
import subprocess

import numpy as np
import tensorflow as tf

from math import ceil

from download_toolbox.cli import date_arg

from icenet.data.datasets.utils import get_decoder
from icenet.data.network_dataset import IceNetDataSet
from icenet.plotting.cli import get_sample_get_args, tfrecord_args

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_tfrecord():
    args = tfrecord_args()

    ds = tf.data.TFRecordDataset([args.file])
    config = json.load(args.configuration)
    args.configuration.close()

    decoder = get_decoder(tuple(config['shape']), config['num_channels'],
                          config['n_forecast_steps'])

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
    subprocess.run("rm -v {}/{}.*.png".format(output_dir, config["identifier"]),
                   shell=True)

    for i, channel in enumerate(config['channels']):
        output_path = os.path.join(
            output_dir, "{}.{:03d}_{}.png".format(config["identifier"], i,
                                                  channel))
        logging.info("Producing {}".format(output_path))

        fig, ax = plt.subplots()
        ax.contourf(x[0, ..., i], levels=args.levels)
        plt.savefig(output_path)
        plt.close()

    for i in range(config['n_forecast_steps']):
        output_path = os.path.join(
            output_dir, "{}.y.{:03d}.png".format(config["identifier"], i + 1))
        y_out = y[0, ..., i, 0]

        logging.info("Producing {}".format(output_path))

        if len(y_out.flatten()) == np.isnan(y_out).sum():
            logging.warning("Skipping {} due to fully nan".format(output_path))
        else:
            fig, ax = plt.subplots()
            ax.contourf(y_out, levels=args.levels)
            plt.savefig(output_path)
            plt.close()


def plot_sample_cli():
    """

    """
    args = get_sample_get_args()
    logging.debug("Opening dataset to get loader: {}".format(args.dataset))
    ds = IceNetDataSet(args.dataset)
    dl = ds.get_data_loader()

    logging.debug("Generating sample for {}".format(args.date))
    net_input, net_output, net_weight = dl.generate_sample(
        args.date, prediction=args.prediction)

    if args.weights:
        channel_data = net_weight.squeeze()
        channel_labels = [
            "weights{}".format(i) for i in range(channel_data.shape[-1])
        ]
        logging.info("Plotting {} weights from sample".format(
            len(channel_labels)))
    elif args.outputs:
        channel_data = net_output.squeeze()
        channel_labels = [
            "outputs{}".format(i) for i in range(channel_data.shape[-1])
        ]
        logging.info("Plotting {} outputs from sample".format(
            len(channel_labels)))
    else:
        logging.info("Plotting inputs from sample")
        channel_data = net_input
        channel_labels = dl.channel_names
        logging.info("Plotting {} inputs from sample".format(
            len(channel_labels)))

    plot_channel_data(channel_data,
                      channel_labels,
                      args.output_path,
                      cols=args.cols,
                      square_size=args.size)


def plot_channel_data(data: object,
                      var_names: list,
                      output_path: str = None,
                      cols: int = 4,
                      square_size: int = 4,
                      get_fig: bool = False):
    """

    :param data:
    :param var_names:
    :param output_path:
    :param cols:
    :param square_size:
    :param get_fig:
    """
    num_rows = int(len(var_names) / cols) + \
        ceil(len(var_names) / cols - int(len(var_names) / cols))

    logging.debug("Plot Rows {} Cols {} Channels {}".format(
        num_rows, cols, len(var_names)))
    fig = plt.figure(figsize=(cols * square_size, num_rows * square_size),
                     layout="tight",
                     dpi=150)

    for i, var in enumerate(var_names):
        ax1 = fig.add_subplot(num_rows, cols, i + 1)
        ax1.set_title("{}".format(var))

        im1 = ax1.imshow(data[..., i], interpolation='None')
        divider = make_axes_locatable(ax1)
        cax1 = divider.append_axes('right', size='3%', pad=square_size / 25)
        fig.colorbar(im1, cax=cax1, orientation='vertical')

    if not get_fig:
        logging.info("Saving to {}".format(output_path))
        plt.savefig(output_path)
        plt.close()
    else:
        return fig
