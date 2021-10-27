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


class WeightedBinaryAccuracy(tf.keras.metrics.BinaryAccuracy):
    """ Custom keras loss/metric for binary accuracy in classifying
    SIC > 15%

    Parameters:
    y_true (ndarray): Ground truth outputs
    y_pred (ndarray): Network predictions
    sample_weight (ndarray): Pixelwise mask weighting for metric summation

    Returns:
    Root mean squared error of SIC (%) (float)
    """

    def __init__(self, leadtime_idx=None, **kwargs):
        name = 'binacc'

        # Leadtime to compute metric over - leave as None to use all lead times
        if leadtime_idx is not None:
            name += str(leadtime_idx+1)
        self.leadtime_idx = leadtime_idx

        super().__init__(name=name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = y_true > 0.15
        y_pred = y_pred > 0.15

        if self.leadtime_idx is not None:
            y_true = y_true[..., self.leadtime_idx]
            y_pred = y_pred[..., self.leadtime_idx]
            if sample_weight is not None:
                sample_weight = sample_weight[..., self.leadtime_idx]

        # TF automatically reduces along final dimension - include dummy axis
        y_true = tf.expand_dims(y_true, axis=-1)
        y_pred = tf.expand_dims(y_pred, axis=-1)
        if sample_weight is not None:
            sample_weight = tf.expand_dims(sample_weight, axis=-1)

        return super().update_state(y_true, y_pred, sample_weight=sample_weight)

    def result(self):
        return 100 * super().result()

    def get_config(self):
        """ For saving and loading networks with this custom metric. """
        return {
            'leadtime_idx': self.leadtime_idx,
        }


class WeightedMAE(tf.keras.metrics.MeanAbsoluteError):
    """ Custom keras loss/metric for mean absolute error
    """

    def __init__(self, name='mae', leadtime_idx=None, **kwargs):
        # Leadtime to compute metric over - leave as None to use all lead times
        if leadtime_idx is not None:
            name += str(leadtime_idx+1)
        self.leadtime_idx = leadtime_idx

        super().__init__(name=name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):

        if self.leadtime_idx is not None:
            y_true = y_true[..., self.leadtime_idx]
            y_pred = y_pred[..., self.leadtime_idx]
            if sample_weight is not None:
                sample_weight = sample_weight[..., self.leadtime_idx]

        # TF automatically reduces along final dimension - include dummy axis
        y_true = tf.expand_dims(y_true, axis=-1)
        y_pred = tf.expand_dims(y_pred, axis=-1)

        return super().update_state(100*y_true, 100*y_pred, sample_weight)


class WeightedRMSE(tf.keras.metrics.RootMeanSquaredError):
    """ Custom keras loss/metric for root mean squared error
    """

    def __init__(self, leadtime_idx=None, name='rmse', **kwargs):
        # Leadtime to compute metric over - leave as None to use all lead times
        if leadtime_idx is not None:
            name += str(leadtime_idx+1)
        self.leadtime_idx = leadtime_idx

        super().__init__(name=name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):

        if self.leadtime_idx is not None:
            y_true = y_true[..., self.leadtime_idx]
            y_pred = y_pred[..., self.leadtime_idx]
            if sample_weight is not None:
                sample_weight = sample_weight[..., self.leadtime_idx]

        # TF automatically reduces along final dimension - include dummy axis
        y_true = tf.expand_dims(y_true, axis=-1)
        y_pred = tf.expand_dims(y_pred, axis=-1)

        return super().update_state(100*y_true, 100*y_pred, sample_weight)


class WeightedMSE(tf.keras.metrics.MeanSquaredError):
    """ Custom keras loss/metric for mean squared error
    """

    def __init__(self, leadtime_idx=None, **kwargs):
        name = 'mse'
        # Leadtime to compute metric over - leave as None to use all lead times
        if leadtime_idx is not None:
            name += str(leadtime_idx+1)
        self.leadtime_idx = leadtime_idx

        super().__init__(name=name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):

        if self.leadtime_idx is not None:
            y_true = y_true[..., self.leadtime_idx]
            y_pred = y_pred[..., self.leadtime_idx]
            if sample_weight is not None:
                sample_weight = sample_weight[..., self.leadtime_idx]

        # TF automatically reduces along final dimension - include dummy axis
        y_true = tf.expand_dims(y_true, axis=-1)
        y_pred = tf.expand_dims(y_pred, axis=-1)
        if sample_weight is not None:
            sample_weight = tf.expand_dims(sample_weight, axis=-1)

        return super().update_state(100*y_true, 100*y_pred, sample_weight)
