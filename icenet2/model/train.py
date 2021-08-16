import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model

import icenet2.model.losses as losses
import icenet2.model.metrics as metrics

from icenet2.data.loader import IceNetDataSet
from icenet2.model.models import unet_batchnorm


def train_model(
        loader_config,
        batch_size=4,
        checkpoint_monitor='val_weighted_MAE_corrected',
        checkpoint_mode='min',
        dropout_rate=0.5,
        early_stopping_patience=35,
        epochs=2,
        filter_size=3,
        learning_rate=5e-4,
        n_filters_factor=2.,
        network_save=True,
        pre_load_network=False,
        pre_load_path=None,
        seed=42,
        weight_decay=0.,
        max_queue_size=3,
        workers=5,
        use_multiprocessing=True,
        training_verbosity=2,
        network_folder=os.path.join(".", "results", "networks")
    ):

    rs = RandomState(MT19937(SeedSequence(seed)))
    np.random.seed(seed=rs)
    tf.random.set_seed(seed)

    ds = IceNetDataSet(loader_config)

    input_shape = (*ds.shape, ds.num_channels)
    train_ds, val_ds, test_ds = ds.get_split_datasets(batch_size=batch_size)

    pre_load_network_fname = "preload_{}.{}.h5".format(ds.identifier, seed) \
        if not pre_load_path else pre_load_path

    if pre_load_network and not os.path.exists(pre_load_network_fname):
        raise RuntimeError("{} is not available, so you cannot preload the "
                           "network with it!".format(pre_load_network_fname))

    if not os.path.exists(network_folder):
        logging.info("Creating network folder: {}".format(network_folder))
        os.makedirs(network_folder)
    network_path = os.path.join(network_folder,
                                "network_{}.{}.h5".format(ds.identifier, seed))

    logging.info("# training samples: {}".format(ds.counts["train"]))
    logging.info("# validation samples: {}".format(ds.counts["val"]))
    logging.info("# input channels: {}".format(ds.num_channels))

    custom_objects = {
        'weighted_MAE_corrected': metrics.weighted_MAE_corrected,
        'weighted_RMSE_corrected': metrics.weighted_RMSE_corrected,
        'weighted_MSE_corrected': losses.weighted_MSE_corrected,
    }
    prev_best = None

    loss = losses.weighted_MSE_corrected
    metrics_list = [
        # metrics.weighted_MAE,
        metrics.weighted_MAE_corrected,
        metrics.weighted_RMSE_corrected,
        losses.weighted_MSE_corrected
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

    ############################################################################
    #                              TRAINING MODEL
    ############################################################################

    strategy = tf.distribute.experimental.CentralStorageStrategy()

    with strategy.scope():
        if pre_load_network:
            logging.info("Loading network from {}".
                format(pre_load_network_fname))
            network = load_model(pre_load_network_fname,
                                 custom_objects=custom_objects)
        else:
            network = unet_batchnorm(
                input_shape=input_shape,
                loss=loss,
                metrics=metrics_list,
                learning_rate=learning_rate,
                filter_size=filter_size,
                n_filters_factor=n_filters_factor,
                n_forecast_days=ds.n_forecast_days,
            )

        network.summary()

        logging.info("Compiling network")

        # TODO: Custom training for distributing loss calculations
        # TODO: Recode categorical_focal_loss(gamma=2.)
        # TODO: Recode construct_custom_categorical_accuracy(use_all_forecast_months=True|False. single_forecast_leadtime_idx=0)
        network.compile(
            optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
            loss=loss,  # 'MeanSquaredError',
            metrics=[])

        model_history = network.fit(
            train_ds,
            epochs=epochs,
            verbose=training_verbosity,
            callbacks=callbacks_list,
            steps_per_epoch=ds.counts["train"] / batch_size,
            validation_steps=ds.counts["val"] / batch_size,
            validation_data=val_ds,
            max_queue_size=max_queue_size,
            workers=workers,
            use_multiprocessing=use_multiprocessing
        )

    if network_save:
        logging.info("Saving network to: {}".format(network_path))
        network.save(network_path)

    # TODO: persist history for plotting?
    fig, ax = plt.subplots()
    ax.plot(model_history.history['val_weighted_MAE_corrected'], label='val')
    ax.plot(model_history.history['weighted_MAE_corrected'], label='train')
    ax.legend(loc='best')
    plt.savefig(os.path.join(network_folder, 'network_{}_history.png'.format(
        seed)))

