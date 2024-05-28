import datetime as dt
import logging
import os

from icenet.model.networks.base import BaseNetwork
from icenet.model.utils import make_exp_decay_lr_schedule

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.callbacks import \
    EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.layers import Conv2D, BatchNormalization, UpSampling2D, \
    concatenate, MaxPooling2D, Input
from tensorflow.keras.models import save_model, Model
from tensorflow.keras.optimizers import Adam


class TensorflowNetwork(BaseNetwork):
    def __init__(self,
                 *args,
                 checkpoint_mode: str = "min",
                 checkpoint_monitor: str = None,
                 early_stopping_patience: int = 0,
                 data_queue_size: int = 10,
                 lr_decay: tuple = (1.0, 0, 0),
                 pre_load_path: str = None,
                 strategy: str = None,
                 tensorboard_logdir: str = None,
                 verbose: bool = False,
                 **kwargs):
        self._checkpoint_mode = checkpoint_mode
        self._checkpoint_monitor = checkpoint_monitor
        self._early_stopping_patience = early_stopping_patience
        self._lr_decay = lr_decay
        self._tensorboard_logdir = tensorboard_logdir

        super().__init__(*args, **kwargs)

        self._weights_path = os.path.join(
            self.network_folder, "{}.network_{}.{}.h5".format(
                self.run_name, self.dataset.identifier, self.seed))

        if pre_load_path is not None and not os.path.exists(pre_load_path):
            raise RuntimeError("{} is not available, so you cannot preload the "
                               "network with it!".format(pre_load_path))
        self._pre_load_path = pre_load_path

        self._data_queue_size = data_queue_size
        self._strategy = strategy
        self._verbose = verbose

    def _attempt_seed_setup(self):
        super()._attempt_seed_setup()
        tf.random.set_seed(self._seed)
        tf.keras.utils.set_random_seed(self._seed)
        # See #8: tf.config.experimental.enable_op_determinism()

    def train(self,
              epochs: int,
              model_creator: callable,
              train_dataset: object,
              model_creator_kwargs: dict = None,
              save: bool = True,
              validation_dataset: object = None):

        strategy = tf.distribute.MirroredStrategy() \
            if self._strategy == "mirrored" \
            else tf.distribute.experimental.CentralStorageStrategy() \
            if self._strategy == "central" \
            else tf.distribute.get_strategy()

        history_path = os.path.join(self.network_folder,
                                    "{}_{}_history.json".format(
                                        self.run_name, self.seed))

        with strategy.scope():
            network = model_creator(**model_creator_kwargs)

        if self._pre_load_path and os.path.exists(self._pre_load_path):
            logging.warning("Automagically loading network weights from {}".format(
                self._pre_load_path))
            network.load_weights(self._pre_load_path)

        network.summary()

        model_history = network.fit(
            train_dataset,
            epochs=epochs,
            verbose=self._verbose,
            callbacks=self.callbacks,
            validation_data=validation_dataset,
            # TODO: pretty sure this is redundant for non-keras.utils.Sequence, legacy inclusion!
            max_queue_size=self._data_queue_size,
        )

        if save:
            logging.info("Saving network to: {}".format(self._weights_path))
            network.save_weights(self._weights_path)
            logging.info("Saving model to: {}".format(self.model_path))
            save_model(network, self.model_path)

            with open(history_path, 'w') as fh:
                pd.DataFrame(model_history.history).to_json(fh)

    def get_default_callbacks(self):
        callbacks_list = list()

        if self._checkpoint_monitor is not None:
            logging.info("Adding ModelCheckpoint callback")
            checkpoint_filestr = str(
                os.path.join(self.network_folder,
                             "checkpoint.{}.network_{}.{}.{}.h5".format(self.run_name, self.dataset.identifier, self.seed, "{epoch:03d}")))
            callbacks_list.append(
                ModelCheckpoint(filepath=checkpoint_filestr,
                                monitor=self._checkpoint_monitor,
                                verbose=1,
                                mode=self._checkpoint_mode,
                                save_best_only=True))

            if self._early_stopping_patience > 0:
                logging.info("Adding EarlyStopping callback")
                callbacks_list.append(
                    EarlyStopping(monitor=self._checkpoint_monitor,
                                  mode=self._checkpoint_mode,
                                  verbose=1,
                                  patience=self._early_stopping_patience,
                                  baseline=None))

        if self._lr_decay[0] != 1.0:
            logging.info("ADding LearningRateScheduler callback")
            lr_decay = -0.1 * np.log(self._lr_decay[0])

            callbacks_list.append(
                LearningRateScheduler(
                    make_exp_decay_lr_schedule(
                        rate=lr_decay,
                        start_epoch=self._lr_decay[1],
                        end_epoch=self._lr_decay[2],
                    )))

        if self._tensorboard_logdir is not None:
            logging.info("Adding tensorboard callback")
            log_dir = os.path.join(
                self._tensorboard_logdir,
                dt.datetime.now().strftime("%d-%m-%y-%H%M%S"))
            callbacks_list.append(
                tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                               histogram_freq=1))
        return callbacks_list


class HorovodNetwork(TensorflowNetwork):
    def train(self,
              epochs: int,
              model_creator: callable,
              train_dataset: object,
              model_creator_kwargs: dict = None,
              save: bool = True,
              validation_dataset: object = None):

        history_path = os.path.join(self.network_folder,
                                    "{}_{}_history.json".format(
                                        self.run_name, self.seed))

        import horovod.tensorflow.keras as hvd

        if hvd.is_initialized():
            logging.info("Horovod is initialized when we call train, with {} members".format(hvd.size()))
        else:
            raise RuntimeError("Horovod is not initialized")

        logging.debug("Calling {} to create our model".format(model_creator))
        network = model_creator(**model_creator_kwargs,
                                custom_optimizer=hvd.DistributedOptimizer(Adam(model_creator_kwargs["learning_rate"])),
                                experimental_run_tf_function=False)
        logging.debug("Created model for rank {}".format(hvd.rank()))
        
        if self._pre_load_path and os.path.exists(self._pre_load_path):
            logging.warning("Automagically loading network weights from {}".format(
                self._pre_load_path))
            network.load_weights(self._pre_load_path)

        if hvd.local_rank() == 0:
            network.summary()

        logging.debug("Calling training loop")
        model_history = network.fit(
            train_dataset.repeat(),
            epochs=epochs,
            verbose=1 if hvd.rank() == 0 and self._verbose else 0,
            callbacks=self.callbacks,
            validation_data=validation_dataset.repeat(),
            max_queue_size=self._data_queue_size,
            steps_per_epoch=self.dataset.counts["train"] // (self.dataset.batch_size * hvd.size()),
            validation_steps=self.dataset.counts["val"] // (self.dataset.batch_size * hvd.size()),
        )

        if save:
            logging.info("Saving network to: {}".format(self._weights_path))
            network.save_weights(self._weights_path)
            save_model(network, self.model_path)

            with open(history_path, 'w') as fh:
                pd.DataFrame(model_history.history).to_json(fh)


### Network architectures:
# --------------------------------------------------------------------
def unet_batchnorm(input_shape: object,
                   loss: object,
                   metrics: object,
                   learning_rate: float = 1e-4,
                   custom_optimizer: object = None,
                   experimental_run_tf_function: bool = True,
                   filter_size: float = 3,
                   n_filters_factor: float = 1,
                   n_forecast_days: int = 1,
                   legacy_rounding: bool = True) -> object:
    """

    :param input_shape:
    :param loss:
    :param metrics:
    :param learning_rate:
    :param custom_optimizer:
    :param experimental_run_tf_function:
    :param filter_size:
    :param n_filters_factor:
    :param n_forecast_days:
    :param legacy_rounding: Ensures filter number calculations are int()'d at the end of calculations
    :return:
    """
    inputs = Input(shape=input_shape)

    start_out_channels = 64
    reduced_channels = start_out_channels * n_filters_factor

    if not legacy_rounding:
        # We're assuming to just strip off any partial channels, rather than round
        reduced_channels = int(reduced_channels)

    channels = {
        start_out_channels * 2 ** pow:
            reduced_channels * 2 ** pow if not legacy_rounding else int(reduced_channels * 2 ** pow)
        for pow in range(4)
    }

    conv1 = Conv2D(channels[64],
                   filter_size,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(channels[64],
                   filter_size,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv1)
    bn1 = BatchNormalization(axis=-1)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(bn1)

    conv2 = Conv2D(channels[128],
                   filter_size,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(channels[128],
                   filter_size,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv2)
    bn2 = BatchNormalization(axis=-1)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(bn2)

    conv3 = Conv2D(channels[256],
                   filter_size,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(channels[256],
                   filter_size,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv3)
    bn3 = BatchNormalization(axis=-1)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(bn3)

    conv4 = Conv2D(channels[256],
                   filter_size,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(channels[256],
                   filter_size,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv4)
    bn4 = BatchNormalization(axis=-1)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(bn4)

    conv5 = Conv2D(channels[512],
                   filter_size,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(channels[512],
                   filter_size,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv5)
    bn5 = BatchNormalization(axis=-1)(conv5)

    up6 = Conv2D(channels[256],
                 2,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(
                     size=(2, 2), interpolation='nearest')(bn5))

    merge6 = concatenate([bn4, up6], axis=3)
    conv6 = Conv2D(channels[256],
                   filter_size,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(channels[256],
                   filter_size,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv6)
    bn6 = BatchNormalization(axis=-1)(conv6)

    up7 = Conv2D(channels[256],
                 2,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(
                     size=(2, 2), interpolation='nearest')(bn6))
    merge7 = concatenate([bn3, up7], axis=3)
    conv7 = Conv2D(channels[256],
                   filter_size,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(channels[256],
                   filter_size,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv7)
    bn7 = BatchNormalization(axis=-1)(conv7)

    up8 = Conv2D(channels[128],
                 2,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(
                     size=(2, 2), interpolation='nearest')(bn7))
    merge8 = concatenate([bn2, up8], axis=3)
    conv8 = Conv2D(channels[128],
                   filter_size,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(channels[128],
                   filter_size,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv8)
    bn8 = BatchNormalization(axis=-1)(conv8)

    up9 = Conv2D(channels[64],
                 2,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(
                     size=(2, 2), interpolation='nearest')(bn8))

    merge9 = concatenate([conv1, up9], axis=3)

    conv9 = Conv2D(channels[64],
                   filter_size,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(channels[64],
                   filter_size,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(channels[64],
                   filter_size,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv9)

    final_layer = Conv2D(n_forecast_days, kernel_size=1,
                         activation='sigmoid')(conv9)

    # Keras graph mode needs y_pred and y_true to have the same shape, so we
    #   we must pad an extra dimension onto the model output to train with
    #   an extra sample weight dimension in y_true.
    # final_layer = tf.expand_dims(final_layer, axis=-1)

    model = Model(inputs, final_layer)

    model.compile(optimizer=Adam(learning_rate=learning_rate)
                  if custom_optimizer is None else custom_optimizer,
                  loss=loss,
                  weighted_metrics=metrics,
                  experimental_run_tf_function=experimental_run_tf_function)

    return model
