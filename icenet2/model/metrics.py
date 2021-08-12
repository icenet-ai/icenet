import tensorflow as tf
from tensorflow.keras import backend as K

"""
TensorFlow metrics.
"""


class ConstructLeadtimeAccuracy(tf.keras.metrics.CategoricalAccuracy):

    """ Computes the network's accuracy over the active grid cell region
    for either a) a specific lead time in months, or b) over all lead times
    at once. """

    def __init__(self,
                 name='construct_custom_categorical_accuracy',
                 use_all_forecast_months=True,
                 single_forecast_leadtime_idx=None,
                 **kwargs):
        super().__init__(name=name, **kwargs)

        self.use_all_forecast_months = use_all_forecast_months
        self.single_forecast_leadtime_idx = single_forecast_leadtime_idx

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.use_all_forecast_months:
            # Make class dimension final dimension for CategoricalAccuracy
            y_true = tf.transpose(y_true, [0, 1, 2, 4, 3])
            y_pred = tf.transpose(y_pred, [0, 1, 2, 4, 3])
            if sample_weight is not None:
                sample_weight = tf.transpose(sample_weight, [0, 1, 2, 4, 3])

            super().update_state(
                y_true, y_pred, sample_weight=sample_weight)

        elif not self.use_all_forecast_months:

            super().update_state(
                y_true[..., self.single_forecast_leadtime_idx],
                y_pred[..., self.single_forecast_leadtime_idx],
                sample_weight=sample_weight[..., self.single_forecast_leadtime_idx]>0)

    def result(self):
        return 100 * super().result()

    def get_config(self):
        """ For saving and loading networks with this custom metric. """
        return {
            'single_forecast_leadtime_idx': self.single_forecast_leadtime_idx,
            'use_all_forecast_months': self.use_all_forecast_months,
        }

    @classmethod
    def from_config(cls, config):
        """ For saving and loading networks with this custom metric. """
        return cls(**config)


@tf.function(input_signature=(
    tf.TensorSpec(shape=[None, 432, 432, None, 2], dtype=tf.float32),
    tf.TensorSpec(shape=[None, 432, 432, None, 1], dtype=tf.float32),
))
def weighted_MAE_corrected(y_true, y_pred):
    ''' Custom keras loss/metric for root mean squared error

    Parameters:
    y_true (ndarray): Ground truth outputs
    y_pred (ndarray): Network predictions

    Returns:
    Root mean squared error of SIC (%) (float)
    '''

    sample_weight = y_true[:, :, :, :, 1:]
    y_true = y_true[:, :, :, :, 0:1]

    abserr = 100*K.abs((y_true - y_pred))  # 432x432 abs errs in %
    weighted_MAE = K.mean(abserr*sample_weight)

    total_size = tf.size(sample_weight, out_type=tf.int32)
    mask_size = tf.math.reduce_sum(tf.cast(sample_weight>0, tf.int32))
    # correction = tf.cast(total_size / mask_size, tf.float32)
    correction = total_size / mask_size
    correction = tf.cast(correction, tf.float32)  # float64 by default

    corrected_MAE = weighted_MAE * correction

    return corrected_MAE


@tf.function(input_signature=(
    tf.TensorSpec(shape=[None, 432, 432, None, 2], dtype=tf.float32),
    tf.TensorSpec(shape=[None, 432, 432, None, 1], dtype=tf.float32),
))
def weighted_RMSE_corrected(y_true, y_pred):
    ''' Custom keras loss/metric for root mean squared error

    Parameters:
    y_true (ndarray): Ground truth outputs
    y_pred (ndarray): Network predictions

    Returns:
    Root mean squared error of SIC (%) (float)
    '''

    sample_weight = y_true[:, :, :, :, 1:]
    y_true = y_true[:, :, :, :, 0:1]

    err = 100*(y_true - y_pred)  # Convert to %
    err_squared = K.square(err)

    weighted_RMSE = K.sqrt(K.mean(err_squared*sample_weight))

    total_size = tf.size(sample_weight, out_type=tf.int32)
    mask_size = tf.math.reduce_sum(tf.cast(sample_weight>0, tf.int32))
    correction = (total_size / mask_size) ** 0.5
    correction = tf.cast(correction, tf.float32)  # float64 by default

    corrected_RMSE = weighted_RMSE * correction

    return corrected_RMSE


def weighted_RMSE(y_true, y_pred):
    ''' Custom keras loss/metric for root mean squared error

    Parameters:
    y_true (ndarray): Ground truth outputs
    y_pred (ndarray): Network predictions

    Returns:
    Root mean squared error of SIC (%) (float)
    '''

    sample_weight = y_true[:, :, :, :, 1]
    y_true = y_true[:, :, :, :, 0]

    err = 100*(y_true - y_pred)  # Convert to %

    return K.sqrt(K.mean(K.square(err)*sample_weight))


def weighted_MAE(y_true, y_pred):
    ''' Custom keras loss/metric for root mean squared error

    Parameters:
    y_true (ndarray): Ground truth outputs
    y_pred (ndarray): Network predictions

    Returns:
    Root mean squared error of SIC (%) (float)
    '''

    sample_weight = y_true[:, :, :, :, 1]
    y_true = y_true[:, :, :, :, 0]

    err = 100*(y_true - y_pred)  # Convert to %

    return K.mean(K.abs(err)*sample_weight)
