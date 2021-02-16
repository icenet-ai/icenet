import os

from datetime import datetime
from dateutil.relativedelta import relativedelta

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K


def categorical_focal_loss(gamma=2.):
    """
    SOURCE: https://github.com/umbertogriffo/focal-loss-keras/blob/master/losses.py

    Softmax version of focal loss.
           m
      FL = ∑  - (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
          c=1
      where m = number of classes, c = class and o = observation
    Parameters:
      gamma -- focusing parameter for modulating factor (1-p)
    Default value:
      gamma -- 2.0 as mentioned in the paper
    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
        https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy
    """
    def categorical_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """

        sample_weight = y_true[:, :, :, 0:1, :]
        y_true = y_true[:, :, :, 1:, :]

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * K.log(y_pred)

        # Calculate Focal Loss
        focal_loss = K.pow(1 - y_pred, gamma) * cross_entropy * sample_weight

        # Old code: loss is a tensor which is reduced implictly by TensorFlow
        # loss = K.mean(focal_loss, axis=1)

        # Mean pixelwise focal loss
        loss = K.mean(K.sum(focal_loss, axis=-2))

        return loss

    return categorical_focal_loss_fixed


def construct_custom_categorical_accuracy(use_all_forecast_months=True,
                                          single_forecast_leadtime_idx=None):

    '''
    Create the categorical accuracy with land/ocean/polar hole masked out.

    Parameters:
    use_all_forecast_months (bool): Whether to compute the masked accuracy
    over each forecast leadtime (True) or a specific forecast leadtime (False).

    single_forecast_leadtime_idx (int): If use_all_forecast_months is False, this selects
    which forecast leadtime index the masked accuracy should be computed over.
    '''

    def custom_categorical_accuracy(y_true, y_pred):
        metric = tf.keras.metrics.CategoricalAccuracy()

        if use_all_forecast_months:
            accs = []
            n_forecast_months = y_pred.shape[-1]
            for forecast_leadtime_idx in range(n_forecast_months):
                # Extract non-masked values in the loss mask
                sample_weight = y_true[:, :, :, 0:1, forecast_leadtime_idx] > 0
                accs.append(100 * metric(y_true[:, :, :, 1:, forecast_leadtime_idx],
                                         y_pred[:, :, :, :, forecast_leadtime_idx],
                                         sample_weight=sample_weight))
            acc_metric = np.mean(accs)

        elif not use_all_forecast_months:
            # Extract non-masked values in the loss mask
            sample_weight = y_true[:, :, :, 0:1, single_forecast_leadtime_idx] > 0
            acc_metric = 100 * metric(y_true[:, :, :, 1:, single_forecast_leadtime_idx],
                                      y_pred[:, :, :, :, single_forecast_leadtime_idx],
                                      sample_weight=sample_weight)

        return acc_metric

    return custom_categorical_accuracy


def filled_datetime_array(start_date, end_date):
    """
    Return a numpy array of datetimes, incrementing monthly, starting at start_date and
    going up to (but not including) end_date.
    """

    monthly_list = []
    date = start_date

    while date < end_date:
        monthly_list.append(date)
        date += relativedelta(months=1)

    return np.array(monthly_list)
