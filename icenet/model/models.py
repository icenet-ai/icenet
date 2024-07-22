import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, UpSampling2D, \
    concatenate, MaxPooling2D, Input
from tensorflow.keras.optimizers import Adam
"""
Defines the Python-based sea ice forecasting models, such as the IceNet architecture
and the linear trend extrapolation model.
"""


@tf.keras.utils.register_keras_serializable()
class TemperatureScale(tf.keras.layers.Layer):
    """Temperature scaling layer

    Implements the temperature scaling layer for probability calibration,
    as introduced in Guo 2017 (http://proceedings.mlr.press/v70/guo17a.html).
    """

    def __init__(self, **kwargs):
        super(TemperatureScale, self).__init__(**kwargs)
        self.temp = tf.Variable(initial_value=1.0,
                                trainable=False,
                                dtype=tf.float32,
                                name='temp')

    def call(self, inputs: object, **kwargs):
        """ Divide the input logits by the T value.

        :param **kwargs:
        :param inputs:
        :return:
        """
        return tf.divide(inputs, self.temp)

    def get_config(self):
        """ For saving and loading networks with this custom layer.

        :return:
        """
        return {'temp': self.temp.numpy()}


### Network architectures:
# --------------------------------------------------------------------


def unet_batchnorm(input_shape: object,
                   loss: object,
                   metrics: object,
                   learning_rate: float = 1e-4,
                   filter_size: float = 3,
                   n_filters_factor: float = 1,
                   n_forecast_days: int = 1,
                   legacy_rounding: bool = False) -> object:
    """

    :param input_shape:
    :param loss:
    :param metrics:
    :param learning_rate:
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

    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss=loss,
                  weighted_metrics=metrics)

    return model


