import argparse
import datetime as dt
import json
import logging
import os
import random

from pprint import pformat

import numpy as np
import pandas as pd
import tensorflow as tf
import wandb

from tensorflow.keras.callbacks import \
    EarlyStopping, ModelCheckpoint, LearningRateScheduler
from wandb.keras import WandbCallback

import icenet2.model.losses as losses
import icenet2.model.metrics as metrics
from icenet2.model.utils import make_exp_decay_lr_schedule

from icenet2.data.dataset import IceNetDataSet
from icenet2.model.models import unet_batchnorm


def train_model(
        run_name,
        loader_config,
        batch_size=4,
        checkpoint_monitor='val_rmse',
        checkpoint_mode='min',
        dataset_class=IceNetDataSet,
        dataset_ratio=None,
        early_stopping_patience=30,
        epochs=2,
        filter_size=3,
        learning_rate=1e-4,
        lr_10e_decay_fac=1.0,
        lr_decay_start=10,
        lr_decay_end=30,
        max_queue_size=3,
        n_filters_factor=2,
        network_folder=None,
        network_save=True,
        pre_load_network=False,
        pre_load_path=None,
        seed=42,
        strategy=tf.distribute.get_strategy(),
        training_verbosity=1,
        workers=5,
        use_multiprocessing=True,
        use_tensorboard=True,
        use_wandb=True,
        wandb_offline=False):
    logging.info("Setting seed value {}".format(seed))
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

    lr_decay = -0.1 * np.log(lr_10e_decay_fac)
    wandb.init(
        project="icenet2",
        name=run_name,
        notes="{}: run at {}{}".format(run_name,
                                       dt.datetime.now().strftime("%D %T"),
                                       "" if
                                       not pre_load_network else
                                       " preload {}".format(pre_load_path)),
        entity="jambyr",
        config=dict(
            seed=seed,
            learning_rate=learning_rate,
            filter_size=filter_size,
            n_filters_factor=n_filters_factor,
            lr_10e_decay_fac=lr_10e_decay_fac,
            lr_decay=lr_decay,
            lr_decay_start=lr_decay_start,
            lr_decay_end=lr_decay_end,
            batch_size=batch_size,
        ),
        allow_val_change=True,
        mode='disabled' if not use_wandb else 'offline' if wandb_offline else 'online',
        settings=wandb.Settings(
            start_method="fork",
            _disable_stats=True,
        ),
    )

    logging.info("Hyperparameters: {}".format(pformat(wandb.config)))

    ds = dataset_class(loader_config, batch_size=batch_size)

    input_shape = (*ds.shape, ds.num_channels)
    train_ds, val_ds, test_ds = ds.get_split_datasets(ratio=dataset_ratio)

    if pre_load_network and not os.path.exists(pre_load_path):
        raise RuntimeError("{} is not available, so you cannot preload the "
                           "network with it!".format(pre_load_path))

    if not network_folder:
        network_folder = os.path.join(".", "results", "networks", run_name)

    if not os.path.exists(network_folder):
        logging.info("Creating network folder: {}".format(network_folder))
        os.makedirs(network_folder)

    network_path = os.path.join(network_folder,
                                "{}.network_{}.{}.h5".format(run_name,
                                                             ds.identifier,
                                                             seed))

    ratio = dataset_ratio if dataset_ratio else 1.0
    logging.info("# training samples: {}".format(ds.counts["train"] * ratio))
    logging.info("# validation samples: {}".format(ds.counts["val"] * ratio))
    logging.info("# input channels: {}".format(ds.num_channels))

    prev_best = None

    callbacks_list = list()

    # Checkpoint the model weights when a validation metric is improved
    callbacks_list.append(
        ModelCheckpoint(
            filepath=network_path,
            monitor=checkpoint_monitor,
            verbose=1,
            mode=checkpoint_mode,
            save_best_only=True
        ))

    # Abort training when validation performance stops improving
    callbacks_list.append(
        EarlyStopping(
            monitor=checkpoint_monitor,
            mode=checkpoint_mode,
            verbose=1,
            patience=early_stopping_patience,
            baseline=prev_best
        ))

    callbacks_list.append(
        LearningRateScheduler(
            make_exp_decay_lr_schedule(
                rate=lr_decay,
                start_epoch=lr_decay_start,
                end_epoch=lr_decay_end,
            )))

    if use_tensorboard:
        logging.info("Adding tensorboard callback")
        log_dir = "logs/" + dt.datetime.now().strftime("%d-%m-%y-%H%M%S")
        callbacks_list.append(tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                             histogram_freq=1))

    if use_wandb:
        # Log training metrics to wandb each epoch
        logging.info("Adding wandb callback")
        callbacks_list.append(
            WandbCallback(
                monitor=checkpoint_monitor,
                mode=checkpoint_mode,
                save_model=False,
                save_graph=False,
            ))

    ############################################################################
    #                              TRAINING MODEL
    ############################################################################

    with strategy.scope():
        loss = losses.WeightedMSE()
        metrics_list = [
            # metrics.weighted_MAE,
            metrics.WeightedMAE(),
            metrics.WeightedRMSE(),
            losses.WeightedMSE()
        ]

        network = unet_batchnorm(
            input_shape=input_shape,
            loss=loss,
            metrics=metrics_list,
            learning_rate=learning_rate,
            filter_size=filter_size,
            n_filters_factor=n_filters_factor,
            n_forecast_days=ds.n_forecast_days,
        )

    if pre_load_network:
        logging.info("Loading network weights from {}".
                     format(pre_load_path))
        network.load_weights(pre_load_path)

    network.summary()

    model_history = network.fit(
        train_ds,
        epochs=epochs,
        verbose=training_verbosity,
        callbacks=callbacks_list,
        validation_data=val_ds,
        max_queue_size=max_queue_size,
        workers=workers,
        use_multiprocessing=use_multiprocessing
    )

    if network_save:
        logging.info("Saving network to: {}".format(network_path))
        network.save_weights(network_path)

    return network_path, model_history


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", type=str)
    ap.add_argument("run_name", type=str)
    ap.add_argument("seed", type=int)

    ap.add_argument("-v", "--verbose", action="store_true", default=False)

    ap.add_argument("-b", "--batch-size", type=int, default=4)
    ap.add_argument("-e", "--epochs", type=int, default=4)
    ap.add_argument("-m", "--multiprocessing", action="store_true",
                    default=False)
    ap.add_argument("-n", "--n-filters-factor", type=float, default=1.)
    ap.add_argument("-nw", "--no-wandb", default=False, action="store_true")
    ap.add_argument("-p", "--preload", type=str)
    ap.add_argument("-qs", "--max-queue-size", default=10, type=int)
    ap.add_argument("-r", "--ratio", default=None, type=float)
    ap.add_argument("-s", "--strategy", default="default",
                    choices=("default", "mirrored", "central"))
    ap.add_argument("--gpus", default=None)
    ap.add_argument("-w", "--workers", type=int, default=4)
    ap.add_argument("-wo", "--wandb-offline", default=False, action="store_true")

    ap.add_argument("--lr", default=1e-4, type=float)
    ap.add_argument("--lr_10e_decay_fac", default=1.0, type=float,
                    help="Factor by which LR is multiplied by every 10 epochs "
                         "using exponential decay. E.g. 1 -> no decay (default)"
                         ", 0.5 -> halve every 10 epochs.")
    ap.add_argument('--lr_decay_start', default=10, type=int)
    ap.add_argument('--lr_decay_end', default=30, type=int)

    return ap.parse_args()


def main():
    args = get_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    dataset_config = \
        os.path.join(".", "dataset_config.{}.json".format(args.dataset))

    strategy = tf.distribute.MirroredStrategy() \
        if args.strategy == "mirrored" \
        else tf.distribute.experimental.CentralStorageStrategy() \
        if args.strategy == "central" \
        else tf.distribute.get_strategy()

    trained_path, history = \
        train_model(args.run_name,
                    dataset_config,
                    batch_size=args.batch_size,
                    dataset_ratio=args.ratio,
                    epochs=args.epochs,
                    learning_rate=args.lr,
                    lr_10e_decay_fac=args.lr_10e_decay_fac,
                    lr_decay_start=args.lr_decay_start,
                    lr_decay_end=args.lr_decay_end,
                    pre_load_network=args.preload is not None,
                    pre_load_path=args.preload,
                    max_queue_size=args.max_queue_size,
                    n_filters_factor=args.n_filters_factor,
                    seed=args.seed,
                    strategy=strategy,
                    training_verbosity=1 if args.verbose else 2,
                    use_multiprocessing=args.multiprocessing,
                    use_wandb=not args.no_wandb,
                    wandb_offline=args.wandb_offline,
                    workers=args.workers, )

    history_path = os.path.join(os.path.dirname(trained_path),
                                "{}_{}_history.json".
                                format(args.run_name, args.seed))
    with open(history_path, 'w') as fh:
        pd.DataFrame(history.history).to_json(fh)

    # TODO: we don't run through the test dates...

