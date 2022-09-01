import tensorflow as tf


"""

"""


class IceNetDataWarning(RuntimeWarning):
    pass


def write_tfrecord(writer: object,
                   x: object,
                   y: object,
                   sample_weights: object):
    """

    :param writer:
    :param x:
    :param y:
    :param sample_weights:
    :param data_check:
    """

    # FIXME: this will trigger eager computation of the dataset, should be
    #  optional but for the moment is commented out. It's potentially better
    #  situated elsewhere too
    #
    #    y_nans = da.isnan(y).sum()
    #    x_nans = da.isnan(x).sum()
    #    sw_nans = da.isnan(sample_weights).sum()

    #    if y_nans + x_nans + sw_nans > 0:
    #        logging.warning("NaNs detected {}: input = {}, "
    #                        "output = {}, weights = {}".
    #                        format(forecast_date, x_nans, y_nans, sw_nans))

    #        if data_check and sample_weights[da.isnan(y)].sum() > 0:
    #            raise IceNetDataWarning("NaNs in output with non-zero weights")

    #        if data_check and x_nans > 0:

    record_data = tf.train.Example(features=tf.train.Features(feature={
        "x": tf.train.Feature(
            float_list=tf.train.FloatList(value=x.reshape(-1))),
        "y": tf.train.Feature(
            float_list=tf.train.FloatList(value=y.reshape(-1))),
        "sample_weights": tf.train.Feature(
            float_list=tf.train.FloatList(value=sample_weights.reshape(-1))),
    })).SerializeToString()

    writer.write(record_data)
