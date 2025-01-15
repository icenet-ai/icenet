import tensorflow as tf


def get_decoder(shape: object,
                channels: object,
                forecasts: object,
                num_vars: int = 1,
                dtype: str = "float32") -> object:
    """Returns a decoder function used for parsing and decoding data from tfrecord protocol buffer.

    Args:
        shape: The shape of the input data.
        channels: The number of channels in the input data.
        forecasts: The number of days to forecast in prediction
        num_vars (optional): The number of variables in the input data. Defaults to 1.
        dtype (optional): The data type of the input data. Defaults to "float32".

    Returns:
        A function that can be used to parse and decode data. It takes in a protocol buffer
            (tfrecord) as input and returns the parsed and decoded data.
    """
    xf = tf.io.FixedLenFeature([*shape, channels], getattr(tf, dtype))
    yf = tf.io.FixedLenFeature([*shape, forecasts, num_vars],
                               getattr(tf, dtype))
    sf = tf.io.FixedLenFeature([*shape, forecasts, num_vars],
                               getattr(tf, dtype))

    @tf.function
    def decode_item(proto):
        features = {
            "x": xf,
            "y": yf,
            "sample_weights": sf,
        }

        item = tf.io.parse_example(proto, features)
        return item['x'], item['y'], item['sample_weights']

    return decode_item
