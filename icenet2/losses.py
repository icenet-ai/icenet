from tensorflow.keras import backend as K
import tensorflow as tf

###############################################################################
############### LOSS FUNCTIONS
###############################################################################


@tf.function(input_signature=(
    tf.TensorSpec(shape=[None, 432, 432, None, 2], dtype=tf.float32),
    tf.TensorSpec(shape=[None, 432, 432, None, 1], dtype=tf.float32),
))
def weighted_MSE_corrected(y_true, y_pred):
    ''' Custom keras loss/metric for mean squared error

    Parameters:
    y_true (ndarray): Ground truth outputs
    y_pred (ndarray): Network predictions

    Returns:
    Mean squared error of SIC (%) (float)
    '''

    sample_weight = y_true[:, :, :, :, 1:]
    y_true = y_true[:, :, :, :, 0:1]

    err = 100*(y_true - y_pred)  # Convert to %

    err_squared = K.square(err)

    weighted_MSE = K.mean(err_squared*sample_weight)

    total_size = tf.size(sample_weight, out_type=tf.int32)
    mask_size = tf.math.reduce_sum(tf.cast(sample_weight>0, tf.int32))
    correction = total_size / mask_size
    correction = tf.cast(correction, tf.float32)  # float64 by default

    corrected_MSE = weighted_MSE * correction

    return corrected_MSE


@tf.function(input_signature=(
    tf.TensorSpec(shape=[None, 432, 432, None, 2], dtype=tf.float32),
    tf.TensorSpec(shape=[None, 432, 432, None, 1], dtype=tf.float32),
))
def weighted_MSE(y_true, y_pred):
    ''' Custom keras loss/metric for mean squared error

    Parameters:
    y_true (ndarray): Ground truth outputs
    y_pred (ndarray): Network predictions

    Returns:
    Mean squared error of SIC (%) (float)
    '''

    # tf.print(y_true.shape)
    sample_weight = y_true[..., 1:]
    # tf.print(sample_weight.shape)
    y_true = y_true[..., 0:1]
    # tf.print(y_true.shape)

    err = 100*(y_true - y_pred)  # Convert to %

    return K.mean(K.square(err)*sample_weight)
