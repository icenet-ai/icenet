from tensorflow.keras import backend as K

###############################################################################
############### METRICS
###############################################################################


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
