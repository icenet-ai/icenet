import tensorflow as tf


class WeightedMSE(tf.keras.losses.MeanSquaredError):
    """Custom keras loss/metric for mean squared error

    :param name:
    """

    def __init__(self,
                 name: str = 'mse', **kwargs):
        super().__init__(name=name, **kwargs)

    def __call__(self,
                 y_true: object,
                 y_pred: object,
                 sample_weight: object = None):
        """
        :param y_true: Ground truth outputs
        :param y_pred: Network predictions
        :param sample_weight: Pixelwise mask weighting for metric summation
        :return: Mean squared error of SIC (%) (float)
        """

        # TF automatically reduces along final dimension - include dummy axis
        y_true = tf.expand_dims(y_true, axis=-1)
        y_pred = tf.expand_dims(y_pred, axis=-1)

        #if sample_weight is not None:
        #    sample_weight = tf.expand_dims(sample_weight, axis=-1)

        return super().__call__(100*y_true, 100*y_pred, sample_weight=sample_weight)
