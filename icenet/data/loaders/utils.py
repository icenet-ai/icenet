import tensorflow as tf
"""

"""


class IceNetDataWarning(RuntimeWarning):
    pass


def write_tfrecord(writer: object, x: object, y: object,
                   sample_weights: object):
    """

    :param writer:
    :param x:
    :param y:
    :param sample_weights:
    """

    record_data = tf.train.Example(features=tf.train.Features(
        feature={
            "x":
                tf.train.Feature(float_list=tf.train.FloatList(
                    value=x.reshape(-1))),
            "y":
                tf.train.Feature(float_list=tf.train.FloatList(
                    value=y.reshape(-1))),
            "sample_weights":
                tf.train.Feature(float_list=tf.train.FloatList(
                    value=sample_weights.reshape(-1))),
        })).SerializeToString()

    writer.write(record_data)
