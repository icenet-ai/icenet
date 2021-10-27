import datetime as dt
import logging
import os

import numpy as np
import tensorflow as tf

from tensorflow.keras.callbacks import \
    EarlyStopping, ModelCheckpoint, LearningRateScheduler

import icenet2.model.losses as losses
import icenet2.model.metrics as metrics
from icenet2.model.utils import make_exp_decay_lr_schedule

from icenet2.data.loader import IceNetDataSet
from icenet2.model.models import unet_batchnorm


def train_model(
        run_name,
        loader_config,
        batch_size=4,
        checkpoint_monitor='val_weighted_MAE_corrected',
        checkpoint_mode='min',
        dataset_class=IceNetDataSet,
        early_stopping_patience=35,
        epochs=2,
        filter_size=3,
        learning_rate=1e-4,
        lr_10e_decay_fac=1.0,
        lr_decay_start=10,
        lr_decay_end=30,
        n_filters_factor=2,
        network_folder=None,
        network_save=True,
        pre_load_network=False,
        pre_load_path=None,
        seed=42,
        strategy=tf.distribute.get_strategy(),
        max_queue_size=3,
        workers=5,
        use_multiprocessing=True,
        use_tensorboard=True,
        training_verbosity=1,
        dataset_ratio=None,
    ):

    np.random.default_rng(seed)
    tf.random.set_seed(seed)

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

    logging.info("# training samples: {}".format(ds.counts["train"]))
    logging.info("# validation samples: {}".format(ds.counts["val"]))
    logging.info("# input channels: {}".format(ds.num_channels))

    prev_best = None

    loss = losses.WeightedMSE
    metrics_list = [
        # metrics.weighted_MAE,
        metrics.WeightedMAE,
        metrics.WeightedRMSE,
        losses.WeightedMSE
    ]

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

    lr_decay = -0.1 * np.log(lr_10e_decay_fac)
    callbacks_list.append(
        LearningRateScheduler(
            make_exp_decay_lr_schedule(
                rate=lr_decay,
                start_epoch=lr_decay_start,
                end_epoch=lr_decay_end,
            )))

    if use_tensorboard:
        log_dir = "logs/" + dt.datetime.now().strftime("%d-%m-%y-%H%M%S")
        callbacks_list.append(tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                             histogram_freq=1))

    ############################################################################
    #                              TRAINING MODEL
    ############################################################################

    with strategy.scope():
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

