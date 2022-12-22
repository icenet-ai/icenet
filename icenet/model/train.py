import argparse
import datetime as dt
import json
import logging
import os
import random
import time

from pprint import pformat

import numpy as np
import pandas as pd
import tensorflow as tf
import wandb

from tensorflow.keras.callbacks import \
    EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.models import load_model, save_model
from wandb.keras import WandbCallback

from icenet.data.dataset import IceNetDataSet, MergedIceNetDataSet
import icenet.model.losses as losses
import icenet.model.metrics as metrics
from icenet.model.utils import make_exp_decay_lr_schedule
import icenet.model.models as models
from icenet.utils import setup_logging


def train_model(
        run_name: object,
        dataset: object,
        batch_size: int = 4,
        checkpoint_monitor: str = 'val_rmse',
        checkpoint_mode: str = 'min',
        dataset_ratio: float = 1.0,
        early_stopping_patience: int = 30,
        epochs: int = 2,
        filter_size: float = 3,
        learning_rate: float = 1e-4,
        lr_10e_decay_fac: float = 1.0,
        lr_decay_start: float = 10,
        lr_decay_end: float = 30,
        max_queue_size: int = 3,
        model_func: object = models.unet_batchnorm,
        n_filters_factor: float = 2,
        network_folder: object = None,
        network_save: bool = True,
        pickup_weights: bool = False,
        pre_load_network: bool = False,
        pre_load_path: object = None,
        seed: int = 42,
        strategy: object = tf.distribute.get_strategy(),
        training_verbosity: int = 1,
        workers: int = 5,
        use_multiprocessing: bool = True,
        use_tensorboard: bool = True,
        use_wandb: bool = True,
        wandb_offline: bool = False,
        wandb_project: str = os.environ.get("ICENET_ENVIRONMENT"),
        wandb_user: str = os.environ.get("USER")) -> object:
    """

    :param run_name:
    :param dataset:
    :param batch_size:
    :param checkpoint_monitor:
    :param checkpoint_mode:
    :param dataset_ratio:
    :param early_stopping_patience:
    :param epochs:
    :param filter_size:
    :param learning_rate:
    :param lr_10e_decay_fac:
    :param lr_decay_start:
    :param lr_decay_end:
    :param max_queue_size:
    :param model_func:
    :param n_filters_factor:
    :param network_folder:
    :param network_save:
    :param pickup_weights:
    :param pre_load_network:
    :param pre_load_path:
    :param seed:
    :param strategy:
    :param training_verbosity:
    :param workers:
    :param use_multiprocessing:
    :param use_tensorboard:
    :param use_wandb:
    :param wandb_offline:
    :param wandb_project:
    :param wandb_user:
    :return:
    """

    lr_decay = -0.1 * np.log(lr_10e_decay_fac)
    wandb.init(
        project=wandb_project,
        name="{}.{}".format(run_name, seed),
        notes="{}: run at {}{}".format(run_name,
                                       dt.datetime.now().strftime("%D %T"),
                                       "" if
                                       not pre_load_network else
                                       " preload {}".format(pre_load_path)),
        entity=wandb_user,
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
        group=run_name,
    )

    logging.info("Hyperparameters: {}".format(pformat(wandb.config)))

    input_shape = (*dataset.shape, dataset.num_channels)

    if pre_load_network and not os.path.exists(pre_load_path):
        raise RuntimeError("{} is not available, so you cannot preload the "
                           "network with it!".format(pre_load_path))

    if not network_folder:
        network_folder = os.path.join(".", "results", "networks", run_name)

    if not os.path.exists(network_folder):
        logging.info("Creating network folder: {}".format(network_folder))
        os.makedirs(network_folder, exist_ok=True)

    weights_path = os.path.join(network_folder,
                                "{}.network_{}.{}.h5".format(run_name,
                                                             dataset.identifier,
                                                             seed))
    model_path = os.path.join(network_folder,
                              "{}.model_{}.{}".format(run_name,
                                                      dataset.identifier,
                                                      seed))

    history_path = os.path.join(network_folder,
                                "{}_{}_history.json".format(run_name, seed))

    prev_best = None
    callbacks_list = list()

    # Checkpoint the model weights when a validation metric is improved
    callbacks_list.append(
        ModelCheckpoint(
            filepath=weights_path,
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
            metrics.WeightedBinaryAccuracy(),
            metrics.WeightedMAE(),
            metrics.WeightedRMSE(),
            losses.WeightedMSE()
        ]

        network = model_func(
            input_shape=input_shape,
            loss=loss,
            metrics=metrics_list,
            learning_rate=learning_rate,
            filter_size=filter_size,
            n_filters_factor=n_filters_factor,
            n_forecast_days=dataset.n_forecast_days,
        )

    if pre_load_network:
        logging.info("Loading network weights from {}".format(pre_load_path))
        network.load_weights(pre_load_path)
    elif pickup_weights and os.path.exists(weights_path):
        logging.warning("Automagically loading network weights from {}".
                        format(weights_path))
        network.load_weights(weights_path)

    network.summary()

    ratio = dataset_ratio if dataset_ratio else 1.0
    train_ds, val_ds, test_ds = dataset.get_split_datasets(ratio=ratio)

    model_history = network.fit(
        train_ds,
        epochs=epochs,
        verbose=training_verbosity,
        callbacks=callbacks_list,
        validation_data=val_ds,
        max_queue_size=max_queue_size,
        # TODO: not useful for tf.data usage according to docs
        workers=workers,
        use_multiprocessing=use_multiprocessing
    )

    if network_save:
        logging.info("Saving network to: {}".format(weights_path))
        network.save_weights(weights_path)
        save_model(network, model_path)

        with open(history_path, 'w') as fh:
            pd.DataFrame(model_history.history).to_json(fh)

    return weights_path, model_path


def evaluate_model(model_path: object,
                   dataset: object,
                   dataset_ratio: float = 1.0,
                   max_queue_size: int = 3,
                   workers: int = 5,
                   use_multiprocessing: bool = True):
    """

    :param model_path:
    :param dataset:
    :param dataset_ratio:
    :param max_queue_size:
    :param workers:
    :param use_multiprocessing:
    """
    logging.info("Running evaluation against test set")
    network = load_model(model_path, compile=False)

    _, val_ds, test_ds = dataset.get_split_datasets(ratio=dataset_ratio)
    eval_data = val_ds

    if dataset.counts["test"] > 0:
        logging.warning("Using validation data source for evaluation, rather "
                        "than test set")
        eval_data = test_ds

    lead_times = list(range(1, dataset.n_forecast_days + 1))
    logging.info("Metric creation for lead time of {} days".
                 format(len(lead_times)))
    metric_names = ["binacc", "mae", "rmse"]
    metrics_classes = [
        metrics.WeightedBinaryAccuracy,
        metrics.WeightedMAE,
        metrics.WeightedRMSE,
    ]
    metrics_list = [cls(leadtime_idx=lt - 1)
                    for lt in lead_times
                    for cls in metrics_classes]

    network.compile(weighted_metrics=metrics_list)

    logging.info('Evaluating... ')
    tic = time.time()
    results = network.evaluate(
        eval_data,
        return_dict=True,
        verbose=0,
        max_queue_size=max_queue_size,
        workers=workers,
        use_multiprocessing=use_multiprocessing,
    )

    results_path = "{}.results.json".format(model_path)
    with open(results_path, "w") as fh:
        json.dump(results, fh)

    logging.debug(results)
    logging.info("Done in {:.1f}s".format(time.time() - tic))

    metric_vals = [[results[f'{name}{lt}']
                    for lt in lead_times] for name in metric_names]
    table_data = list(zip(lead_times, *metric_vals))
    table = wandb.Table(data=table_data, columns=['leadtime', *metric_names])

    # Log each metric vs. leadtime as a plot to wandb
    for name in metric_names:
        wandb.log(
            {f'{name}_plot': wandb.plot.line(table, x='leadtime', y=name)})


@setup_logging
def get_args():
    """

    :return:
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", type=str)
    ap.add_argument("run_name", type=str)
    ap.add_argument("seed", type=int)

    ap.add_argument("-b", "--batch-size", type=int, default=4)
    ap.add_argument("-ds", "--additional-dataset",
                    dest="additional", nargs="*", default=[])
    ap.add_argument("-e", "--epochs", type=int, default=4)
    ap.add_argument("--early-stopping", type=int, default=50)
    ap.add_argument("-m", "--multiprocessing",
                    action="store_true", default=False)
    ap.add_argument("-n", "--n-filters-factor", type=float, default=1.)
    ap.add_argument("-nw", "--no-wandb", default=False, action="store_true")
    ap.add_argument("-p", "--preload", type=str)
    ap.add_argument("-pw", "--pickup-weights",
                    action="store_true", default=False)
    ap.add_argument("-qs", "--max-queue-size", default=10, type=int)
    ap.add_argument("-r", "--ratio", default=1.0, type=float)
    ap.add_argument("-s", "--strategy", default="default",
                    choices=("default", "mirrored", "central"))
    ap.add_argument("--shuffle-train", default=False,
                    action="store_true", help="Shuffle the training set")
    ap.add_argument("--gpus", default=None)
    ap.add_argument("-v", "--verbose", action="store_true", default=False)
    ap.add_argument("-w", "--workers", type=int, default=4)
    ap.add_argument("-wo", "--wandb-offline", default=False, action="store_true")
    ap.add_argument("-wp", "--wandb-project",
                    default=os.environ.get("ICENET_ENVIRONMENT"), type=str)
    ap.add_argument("-wu", "--wandb-user",
                    default=os.environ.get("USER"), type=str)

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

    logging.warning("Setting seed for best attempt at determinism, value {}".
                    format(args.seed))
    # determinism is not guaranteed across different versions of TensorFlow.
    # determinism is not guaranteed across different hardware.
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    # numpy.random.default_rng ignores this, WARNING!
    np.random.seed(args.seed)
    random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.keras.utils.set_random_seed(args.seed)
    # See #8: tf.config.experimental.enable_op_determinism()

    # TODO: this should come from a factory in the future - not the only place
    #  that merged datasets are going to be available
    if len(args.additional) == 0:
        dataset = IceNetDataSet("dataset_config.{}.json".format(args.dataset),
                                batch_size=args.batch_size,
                                shuffling=args.shuffle_train)
    else:
        dataset = MergedIceNetDataSet([
            "dataset_config.{}.json".format(el) for el in [
                args.dataset, *args.additional
            ]
        ],
            batch_size=args.batch_size,
            shuffling=args.shuffle_train)

    strategy = tf.distribute.MirroredStrategy() \
        if args.strategy == "mirrored" \
        else tf.distribute.experimental.CentralStorageStrategy() \
        if args.strategy == "central" \
        else tf.distribute.get_strategy()

    weights_path, model_path = \
        train_model(args.run_name,
                    dataset,
                    batch_size=args.batch_size,
                    dataset_ratio=args.ratio,
                    early_stopping_patience=args.early_stopping,
                    epochs=args.epochs,
                    learning_rate=args.lr,
                    lr_10e_decay_fac=args.lr_10e_decay_fac,
                    lr_decay_start=args.lr_decay_start,
                    lr_decay_end=args.lr_decay_end,
                    pickup_weights=args.pickup_weights,
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
                    wandb_project=args.wandb_project,
                    wandb_user=args.wandb_user,
                    workers=args.workers)

    evaluate_model(model_path,
                   dataset,
                   dataset_ratio=args.ratio,
                   max_queue_size=args.max_queue_size,
                   use_multiprocessing=args.multiprocessing,
                   workers=args.workers)
