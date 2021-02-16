import argparse
import copy
import glob
import logging
import os
import pickle
import sys

import numpy as np
import tensorflow as tf

from icenet.architectures import unet_batchnorm

from icenet.utils import construct_custom_categorical_accuracy, categorical_focal_loss


def get_args():
    a = argparse.ArgumentParser()
    a.add_argument("-s", "--seed", default=42, type=int,
                   help='Random seed to use for training IceNet (defaults to 42).')
    a.add_argument("-sc", "--central-storage-strategy", name="strategy_cs",
                   help="Use CentralStorageStrategy", action="store_true", default=False)
    a.add_argument("data_loader", help="DataLoader file in filesystem")
    #TODO: a.add_argument("hyperparameters", help="YAML configuration", default=get_default_hyperparameters())
    return a.parse_args()


def init():
    np.random.seed(args.seed)
    tf.random.set_seed = args.seed


def config():
    # TODO: Source from YAML - can be generated for run experimentation via init
    return dict(
        learning_rate=5e-4,
        filter_size=3,
        n_filters_factor=2.,
        weight_decay=0.,
        batch_size=4,
        dropout_rate=0.5
    )


def train(hyperparameters):
    loss = categorical_focal_loss(gamma=2.)

    forecast_acc_mean = construct_custom_categorical_accuracy(use_all_forecast_months=True)
    forecast_acc_mean.__name__ = 'forecast_acc_mean'
    forecast_acc_1month = construct_custom_categorical_accuracy(use_all_forecast_months=False,
                                                                single_forecast_leadtime_idx=0)
    forecast_acc_1month.__name__ = 'forecast_acc_1month'
    forecast_acc_2month = construct_custom_categorical_accuracy(use_all_forecast_months=False,
                                                                single_forecast_leadtime_idx=1)
    forecast_acc_2month.__name__ = 'forecast_acc_2month'
    forecast_acc_3month = construct_custom_categorical_accuracy(use_all_forecast_months=False,
                                                                single_forecast_leadtime_idx=2)
    forecast_acc_3month.__name__ = 'forecast_acc_3month'
    forecast_acc_4month = construct_custom_categorical_accuracy(use_all_forecast_months=False,
                                                                single_forecast_leadtime_idx=3)
    forecast_acc_4month.__name__ = 'forecast_acc_4month'
    forecast_acc_5month = construct_custom_categorical_accuracy(use_all_forecast_months=False,
                                                                single_forecast_leadtime_idx=4)
    forecast_acc_5month.__name__ = 'forecast_acc_5month'
    forecast_acc_6month = construct_custom_categorical_accuracy(use_all_forecast_months=False,
                                                                single_forecast_leadtime_idx=5)
    forecast_acc_6month.__name__ = 'forecast_acc_6month'

    metrics = [forecast_acc_mean, forecast_acc_1month, forecast_acc_2month,
               forecast_acc_3month, forecast_acc_4month, forecast_acc_5month,
               forecast_acc_6month]

    network = unet_batchnorm(input_shape=input_shape,
                             loss=loss,
                             metrics=metrics,
                             learning_rate=wandb.config.learning_rate,
                             filter_size=wandb.config.filter_size,
                             n_filters_factor=wandb.config.n_filters_factor,
                             n_forecast_months=dataloader.n_forecast_months)


if __name__ == "__main__":
    args = get_args()
    init()
    train(config())
