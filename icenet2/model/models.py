import logging
import os

import numpy as np
import pandas as pd
import xarray as xr
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
        super(TemperatureScale, self).__init__()
        self.temp = tf.Variable(initial_value=1.0, trainable=False,
                                dtype=tf.float32, name='temp')

    def call(self, inputs):
        """ Divide the input logits by the T value. """
        return tf.divide(inputs, self.temp)

    def get_config(self):
        """ For saving and loading networks with this custom layer. """
        return {'temp': self.temp.numpy()}


### Network architectures:
# --------------------------------------------------------------------

def unet_batchnorm(input_shape, loss, metrics,
                   learning_rate=1e-4, filter_size=3,
                   n_filters_factor=1, n_forecast_days=1):
    inputs = Input(shape=input_shape)

    conv1 = Conv2D(np.int(64*n_filters_factor), filter_size, activation='relu',
                   padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(np.int(64*n_filters_factor), filter_size, activation='relu',
                   padding='same', kernel_initializer='he_normal')(conv1)
    bn1 = BatchNormalization(axis=-1)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(bn1)

    conv2 = Conv2D(np.int(128*n_filters_factor), filter_size, activation='relu',
                   padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(np.int(128*n_filters_factor), filter_size, activation='relu',
                   padding='same', kernel_initializer='he_normal')(conv2)
    bn2 = BatchNormalization(axis=-1)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(bn2)

    conv3 = Conv2D(np.int(256*n_filters_factor), filter_size, activation='relu',
                   padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(np.int(256*n_filters_factor), filter_size, activation='relu',
                   padding='same', kernel_initializer='he_normal')(conv3)
    bn3 = BatchNormalization(axis=-1)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(bn3)

    conv4 = Conv2D(np.int(256*n_filters_factor), filter_size, activation='relu',
                   padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(np.int(256*n_filters_factor), filter_size, activation='relu',
                   padding='same', kernel_initializer='he_normal')(conv4)
    bn4 = BatchNormalization(axis=-1)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(bn4)

    conv5 = Conv2D(np.int(512*n_filters_factor), filter_size, activation='relu',
                   padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(np.int(512*n_filters_factor), filter_size, activation='relu',
                   padding='same', kernel_initializer='he_normal')(conv5)
    bn5 = BatchNormalization(axis=-1)(conv5)

    up6 = Conv2D(np.int(256*n_filters_factor), 2, activation='relu',
                 padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2), interpolation='nearest')(bn5))

    merge6 = concatenate([bn4, up6], axis=3)
    conv6 = Conv2D(np.int(256*n_filters_factor), filter_size, activation='relu',
                   padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(np.int(256*n_filters_factor), filter_size, activation='relu',
                   padding='same', kernel_initializer='he_normal')(conv6)
    bn6 = BatchNormalization(axis=-1)(conv6)

    up7 = Conv2D(np.int(256*n_filters_factor), 2, activation='relu',
                 padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2), interpolation='nearest')(bn6))
    merge7 = concatenate([bn3,up7], axis=3)
    conv7 = Conv2D(np.int(256*n_filters_factor), filter_size, activation='relu',
                   padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(np.int(256*n_filters_factor), filter_size, activation='relu',
                   padding='same', kernel_initializer='he_normal')(conv7)
    bn7 = BatchNormalization(axis=-1)(conv7)

    up8 = Conv2D(np.int(128*n_filters_factor), 2, activation='relu',
                 padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2), interpolation='nearest')(bn7))
    merge8 = concatenate([bn2,up8], axis=3)
    conv8 = Conv2D(np.int(128*n_filters_factor), filter_size, activation='relu',
                   padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(np.int(128*n_filters_factor), filter_size, activation='relu',
                   padding='same', kernel_initializer='he_normal')(conv8)
    bn8 = BatchNormalization(axis=-1)(conv8)

    up9 = Conv2D(np.int(64*n_filters_factor), 2, activation='relu',
                 padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2), interpolation='nearest')(bn8))

    merge9 = concatenate([conv1, up9], axis=3)

    conv9 = Conv2D(np.int(64*n_filters_factor), filter_size, activation='relu',
                   padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(np.int(64*n_filters_factor), filter_size, activation='relu',
                   padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(np.int(64*n_filters_factor), filter_size, activation='relu',
                   padding='same', kernel_initializer='he_normal')(conv9)

    final_layer = Conv2D(n_forecast_days,
                         kernel_size=1, activation='sigmoid')(conv9)

    # Keras graph mode needs y_pred and y_true to have the same shape, so we
    #   we must pad an extra dimension onto the model output to train with
    #   an extra sample weight dimension in y_true.
    final_layer = tf.expand_dims(final_layer, axis=-1)

    model = Model(inputs, final_layer)

    model.compile(
        optimizer=Adam(lr=learning_rate),
        loss=loss,
        metrics=metrics)

    return model


def linear_trend_forecast(forecast_date, da, mask, n_linear_days,
                          missing_dates=(), shape=(432, 432)):
    """
    Returns a simple sea ice forecast based on a gridcell-wise linear
    extrapolation.

    Parameters:
    forecast_month (datetime.datetime): The month to forecast

    n_linear_years (int or str): Number of past years to use for linear trend
    extrapolation.

    da (xr.DataArray): xarray data array to use instead of observational

    Returns:
    output_map (np.ndarray): The output SIC map predicted
    by fitting a least squares linear trend to the past n_linear_years
    for the month being predicted.

    sie (np.float): The predicted sea ice extend (SIE).
    """

    valid_dates = [pd.Timestamp(date) for date in da.time.values]

    input_dates = [forecast_date - pd.DateOffset(days=1+lag)
                   for lag in range(n_linear_days)]

    # Do not use missing days in the linear trend projection
    input_dates = [date for date in input_dates
                   if pd.to_datetime(date).date() not in missing_dates]

    # Chop off input date from before data start
    input_dates = [date for date in input_dates if date in valid_dates]

    input_dates = sorted(input_dates)

    # The actual number of past years used
    actual_n_linear_days = len(input_dates)

    da = da.sel(time=input_dates)

    input_maps = np.array(da.data)

    if not actual_n_linear_days:
        actual_n_linear_days = 1
        input_maps = np.zeros((actual_n_linear_days, *shape))

    x = np.arange(actual_n_linear_days)
    y = input_maps.reshape(actual_n_linear_days, -1)

    # Fit the least squares linear coefficients
    r = np.linalg.lstsq(
        np.c_[x, np.ones_like(x)], y, rcond=None)[0]

    # y = mx + c
    output_map = np.matmul(np.array([actual_n_linear_days, 1]), r).\
        reshape(*shape)

    output_map[mask] = 0.

    output_map[output_map < 0] = 0.
    output_map[output_map > 1] = 1.

    sie = np.sum(output_map > 0.15) * 25**2

    return output_map, sie
