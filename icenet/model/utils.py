import logging
import numpy as np
import matplotlib.pyplot as plt

################################################################################
# LEARNING RATE
################################################################################


def make_exp_decay_lr_schedule(rate: float,
                               start_epoch: int = 1,
                               end_epoch: object = np.inf) -> object:
    """ Returns an exponential learning rate function that multiplies by
    exp(-rate) each epoch after `start_epoch`.
    :param rate:
    :param start_epoch:
    :param end_epoch:
    :return:
    """

    def lr_scheduler_exp_decay(epoch, lr):
        """ Learning rate scheduler for fine tuning.
        Exponential decrease after start_epoch until end_epoch.
        :param epoch:
        :param lr:
        :return:
        """

        if start_epoch < epoch < end_epoch:
            lr = lr * np.math.exp(-rate)

        logging.info('\nSetting learning rate to: {}\n'.format(lr))

        return lr

    return lr_scheduler_exp_decay


###############################################################################
# PLOTTING
###############################################################################


def compute_heatmap(results_df: object,
                    model: object,
                    seed: object = 'NA',
                    metric: object = 'Binary accuracy') -> object:
    """
    Returns a binary accuracy heatmap of lead time vs. calendar month
    for a given model.
    :param results_df:
    :param model:
    :param seed:
    :param metric:
    :return:
    """

    month_names = np.array(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                            'Jul', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec'])

    # Mean over calendar month
    mean_df = results_df.loc[model, seed].reset_index().\
        groupby(['Calendar month', 'Leadtime']).mean()

    # Pivot
    heatmap_df = mean_df.reset_index().\
        pivot('Calendar month', 'Leadtime', metric).reindex(month_names)

    return heatmap_df


def arr_to_ice_edge_arr(arr: object,
                        thresh: object,
                        land_mask: object,
                        region_mask: object) -> object:

    """
    Compute a boolean mask with True over ice edge contour grid cells using
    matplotlib.pyplot.contour and an input threshold to define the ice edge
    (e.g. 0.15 for the 15% SIC ice edge or 0.5 for SIP forecasts). The contour
    along the coastline is removed using the region mask.
    :param arr:
    :param thresh:
    :param land_mask:
    :param region_mask:
    :return:
    """

    X, Y = np.meshgrid(np.arange(arr.shape[0]), np.arange(arr.shape[1]))
    X = X.T
    Y = Y.T

    cs = plt.contour(X, Y, arr, [thresh], alpha=0)  # Do not plot on any axes
    x = []
    y = []
    for p in cs.collections[0].get_paths():
        x_i, y_i = p.vertices.T
        x.extend(np.round(x_i))
        y.extend(np.round(y_i))
    x = np.array(x, int)
    y = np.array(y, int)
    ice_edge_arr = np.zeros(arr.shape, dtype=bool)
    ice_edge_arr[x, y] = True
    # Mask out ice edge contour that hugs the coastline
    ice_edge_arr[land_mask] = False
    ice_edge_arr[region_mask == 13] = False

    return ice_edge_arr


def arr_to_ice_edge_rgba_arr(arr: object,
                             thresh: object,
                             land_mask: object,
                             region_mask: object,
                             rgb: object) -> object:
    """

    :param arr:
    :param thresh:
    :param land_mask:
    :param region_mask:
    :param rgb:
    :return:
    """
    ice_edge_arr = arr_to_ice_edge_arr(arr, thresh, land_mask, region_mask)

    # Contour pixels -> alpha=1, alpha=0 elsewhere
    ice_edge_rgba_arr = np.zeros((*arr.shape, 4))
    ice_edge_rgba_arr[:, :, 3] = ice_edge_arr
    ice_edge_rgba_arr[:, :, :3] = rgb

    return ice_edge_rgba_arr
