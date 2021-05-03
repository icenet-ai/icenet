from tensorflow.keras import backend as K
import tensorflow as tf

###############################################################################
############### METRICS
###############################################################################

# TODO: binary accuracy


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
