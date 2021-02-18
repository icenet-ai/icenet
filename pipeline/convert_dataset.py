import argparse
import glob
import logging
import os
import sys

import numpy as np
import tensorflow as tf


def get_args():
    a = argparse.ArgumentParser()
    a.add_argument("-s", "--batch-size", type=int, default=5)
    a.add_argument("-v", "--verbose", default=False, action="store_true")
    a.add_argument("input", help="Input directory")
    a.add_argument("output", help="Output directory")
    return a.parse_args()


def generate_data(input, output, batch_size=1, wildcard="*"):
    filelist = sorted(glob.glob(os.path.join(input, "{}.npz".format(wildcard))))

    train = round(len(filelist) * 0.8)
    test = round(len(filelist) * 0.9)
    val = len(filelist)
    options = tf.io.TFRecordOptions()
    batch_number = 0

    if not os.path.isdir(input):
        raise RuntimeError("Directory {} does not exist as an input".format(input))

    os.makedirs(output, exist_ok=True)

    def batch(files, num):
        i = 0
        while i < len(files):
            yield files[i:i+num]
            i += num

    for noms in batch(filelist, batch_size):
        tf_path = os.path.join(output, "{:05}.tfrecord".format(batch_number))

        with tf.io.TFRecordWriter(tf_path) as writer:
            logging.info("Processing batch file {}".format(tf_path))

            for nom in noms:
                logging.info("Processing input {}".format(nom))

                input_data = np.load(nom)

                (x, y) = (input_data['x'], input_data['y'])

                logging.debug("x shape {}, y shape {}".format(x.shape, y.shape))

                record_data = tf.train.Example(features=tf.train.Features(feature={
                    "x": tf.train.Feature(float_list=tf.train.FloatList(value=x.reshape(-1))),
                    "y": tf.train.Feature(float_list=tf.train.FloatList(value=y.reshape(-1))),
                })).SerializeToString()

                writer.write(record_data)

        batch_number += 1


if __name__ == "__main__":
    args = get_args()
    log_state = logging.DEBUG if args.verbose else logging.INFO
    logging.getLogger().setLevel(log_state)
    generate_data(args.input, args.output, batch_size=args.batch_size)
