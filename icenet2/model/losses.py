import tensorflow as tf


class WeightedMSE(tf.keras.losses.MeanSquaredError):
    ''' Custom keras loss/metric for mean squared error
    '''

    def __init__(self, name='mse', **kwargs):
        super().__init__(name=name, **kwargs)

    def __call__(self, y_true, y_pred, sample_weight=None):
        """
        Parameters:
        y_true (ndarray): Ground truth outputs
        y_pred (ndarray): Network predictions
        sample_weight (ndarray): Pixelwise mask weighting for metric summation

        Returns:
        Mean squared error of SIC (%) (float)
        """

        # TF automatically reduces along final dimension - include dummy axis
        y_true = tf.expand_dims(y_true, axis=-1)
        y_pred = tf.expand_dims(y_pred, axis=-1)

        # Convert to SIC (%)
        return super().__call__(100*y_true, 100*y_pred, sample_weight)
