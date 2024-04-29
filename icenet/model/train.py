import datetime as dt
import json
import logging
import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.callbacks import \
    EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.models import load_model, save_model

from icenet.data.dataset import IceNetDataSet, MergedIceNetDataSet
from icenet.model.cli import train_args
import icenet.model.losses as losses
import icenet.model.metrics as metrics
from icenet.model.utils import attempt_seed_setup, make_exp_decay_lr_schedule
import icenet.model.models as models


def train_model(run_name: object,
                dataset: object,
                callback_objects: list = [],
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
                use_tensorboard: bool = True) -> object:
    """

    :param run_name:
    :param dataset:
    :param callback_objects:
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
    :return:
    """

    lr_decay = -0.1 * np.log(lr_10e_decay_fac)

    input_shape = (*dataset.shape, dataset.num_channels)

    if pre_load_network and not os.path.exists(pre_load_path):
        raise RuntimeError("{} is not available, so you cannot preload the "
                           "network with it!".format(pre_load_path))

    if not network_folder:
        network_folder = os.path.join(".", "results", "networks", run_name)

    if not os.path.exists(network_folder):
        logging.info("Creating network folder: {}".format(network_folder))
        os.makedirs(network_folder, exist_ok=True)

    weights_path = os.path.join(
        network_folder, "{}.network_{}.{}.h5".format(run_name,
                                                     dataset.identifier, seed))
    model_path = os.path.join(
        network_folder, "{}.model_{}.{}".format(run_name, dataset.identifier,
                                                seed))

    history_path = os.path.join(network_folder,
                                "{}_{}_history.json".format(run_name, seed))

    prev_best = None
    callbacks_list = list()

    # Checkpoint the model weights when a validation metric is improved
    callbacks_list.append(
        ModelCheckpoint(filepath=weights_path,
                        monitor=checkpoint_monitor,
                        verbose=1,
                        mode=checkpoint_mode,
                        save_best_only=True))

    # Abort training when validation performance stops improving
    callbacks_list.append(
        EarlyStopping(monitor=checkpoint_monitor,
                      mode=checkpoint_mode,
                      verbose=1,
                      patience=early_stopping_patience,
                      baseline=prev_best))

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
        callbacks_list.append(
            tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1))

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
        logging.warning("Automagically loading network weights from {}".format(
            weights_path))
        network.load_weights(weights_path)

    network.summary()

    ratio = dataset_ratio if dataset_ratio else 1.0
    train_ds, val_ds, test_ds = dataset.get_split_datasets(ratio=ratio)

    model_history = network.fit(
        train_ds,
        epochs=epochs,
        verbose=training_verbosity,
        callbacks=callbacks_list + callback_objects,
        validation_data=val_ds,
        max_queue_size=max_queue_size,
        # not useful for tf.data usage according to docs, but useful in dev
        workers=workers,
        use_multiprocessing=use_multiprocessing)

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
        eval_data = test_ds
        logging.info("Using test set for validation")
    else:
        logging.warning("Using validation data source for evaluation, rather "
                        "than test set")

    lead_times = list(range(1, dataset.n_forecast_days + 1))
    logging.info("Metric creation for lead time of {} days".format(
        len(lead_times)))
    metric_names = ["binacc", "mae", "rmse"]
    metrics_classes = [
        metrics.WeightedBinaryAccuracy,
        metrics.WeightedMAE,
        metrics.WeightedRMSE,
    ]
    metrics_list = [
        cls(leadtime_idx=lt - 1) for lt in lead_times
        for cls in metrics_classes
    ]

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

    return results, metric_names, lead_times


def main():
    args = train_args()
    attempt_seed_setup(args.seed)

    # TODO: this should come from a factory in the future - not the only place
    #  that merged datasets are going to be available
    if len(args.additional) == 0:
        dataset = IceNetDataSet("dataset_config.{}.json".format(args.dataset),
                                batch_size=args.batch_size,
                                shuffling=args.shuffle_train)
    else:
        dataset = MergedIceNetDataSet([
            "dataset_config.{}.json".format(el)
            for el in [args.dataset, *args.additional]
        ],
                                      batch_size=args.batch_size,
                                      shuffling=args.shuffle_train)

    strategy = tf.distribute.MirroredStrategy() \
        if args.strategy == "mirrored" \
        else tf.distribute.experimental.CentralStorageStrategy() \
        if args.strategy == "central" \
        else tf.distribute.get_strategy()

    # There is a better way of doing this by passing off to a dynamic factory
    # for other integrations, but for the moment I have no shame
    callback_objects = list()
    using_wandb = False
    run = None

    # TODO: this can and probably should be a decorator
    if not args.no_wandb:
        from icenet.model.handlers.wandb import init_wandb, finalise_wandb
        run, callback = init_wandb(args)

        if callback is not None:
            callback_objects.append(callback)
            using_wandb = True

    weights_path, model_path = \
        train_model(args.run_name,
                    dataset,
                    callback_objects=callback_objects,
                    checkpoint_mode=args.checkpoint_mode,
                    checkpoint_monitor=args.checkpoint_monitor,
                    dataset_ratio=args.ratio,
                    early_stopping_patience=args.early_stopping,
                    epochs=args.epochs,
                    filter_size=args.filter_size,
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
                    workers=args.workers)

    results, metric_names, leads = \
        evaluate_model(model_path,
                       dataset,
                       dataset_ratio=args.ratio,
                       max_queue_size=args.max_queue_size,
                       use_multiprocessing=args.multiprocessing,
                       workers=args.workers)

    if using_wandb:
        finalise_wandb(run, results, metric_names, leads)

