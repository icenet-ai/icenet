from tensorflow.keras import backend as K

###############################################################################
############### LOSS FUNCTIONS
###############################################################################


def weighted_MSE(y_true, y_pred):
    ''' Custom keras loss/metric for mean squared error

    Parameters:
    y_true (ndarray): Ground truth outputs
    y_pred (ndarray): Network predictions

    Returns:
    Mean squared error of SIC (%) (float)
    '''

    sample_weight = y_true[:, :, :, :, 1]
    y_true = y_true[:, :, :, :, 0]

    err = 100*(y_true - y_pred)  # Convert to %

    return K.mean(K.square(err)*sample_weight)
