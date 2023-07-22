import tensorflow as tf


class WeightedMSE(tf.keras.losses.MeanSquaredError):
    """Custom keras loss/metric for mean squared error

    :param name:
    :param y_pred_distr: Whether or not y_pred is a TensorFlow Probability
    Distribution object. If so, the mean is taken before computation of the metric.
    :param loc: if True, use the loc of the TruncatedNormal rather than the mean.
    """

    def __init__(self,
                 name: str = 'mse',
                 y_pred_distr: bool = False,
                 loc: bool = False,
                 **kwargs):
        super().__init__(name=name, **kwargs)

        self._y_pred_distr = y_pred_distr
        self._loc = loc

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

        if self._y_pred_distr:
            if self._loc:
                y_pred = y_pred.distribution.loc
            else:
                y_pred = y_pred.mean()

        # TF automatically reduces along final dimension - include dummy axis
        y_true = tf.expand_dims(y_true, axis=-1)
        y_pred = tf.expand_dims(y_pred, axis=-1)

        # if sample_weight is not None:
        #    sample_weight = tf.expand_dims(sample_weight, axis=-1)

        return super().__call__(100*y_true, 100*y_pred, sample_weight=sample_weight)


class NLL(tf.keras.losses.Loss):
    ''' Custom keras loss/metric for negative log likelihood
    '''

    def __init__(self, name='nll', **kwargs):
        super().__init__(name=name, **kwargs)

    def __call__(self, y_true, y_pred, sample_weight=None):
        '''
        Parameters:
        y_true (ndarray): Ground truth outputs
        y_pred (ndarray): Network predictions
        sample_weight (ndarray): Pixelwise mask weighting for metric summation
        Returns:
        Negative log likelihood (float)
        '''

        return -y_pred.log_prob(y_true)


class WeightedNLL(tf.keras.losses.Loss):
    ''' Custom keras loss/metric for weighted negative log likelihood
    '''

    def __init__(self, name='nll', **kwargs):
        super().__init__(name=name, **kwargs)

    def __call__(self, y_true, y_pred, sample_weight=None):
        '''
        Parameters:
        y_true (ndarray): Ground truth outputs
        y_pred (ndarray): Network predictions
        sample_weight (ndarray): Pixelwise mask weighting for metric summation
        Returns:
        Negative log likelihood (float)
        '''
        # TODO: not update_state?
        return -y_pred.distribution.log_prob(y_true) * sample_weight
