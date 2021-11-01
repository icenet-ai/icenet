import argparse
import concurrent.futures
import datetime as dt
import glob
import json
import logging
import os

# https://stackoverflow.com/questions/55852831/
# tf-data-vs-keras-utils-sequence-performance

import numpy as np
import tensorflow as tf

from icenet2.data.loader import IceNetDataLoader
from icenet2.data.process import IceNetPreProcessor
from icenet2.data.producers import DataProducer


def get_decoder(shape, channels, forecasts, num_vars=1, dtype="float32"):
    xf = tf.io.FixedLenFeature(
        [*shape, channels], getattr(tf, dtype))
    yf = tf.io.FixedLenFeature(
        [*shape, forecasts, num_vars], getattr(tf, dtype))
    sf = tf.io.FixedLenFeature(
        [*shape, forecasts, num_vars], getattr(tf, dtype))

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


class IceNetDataSet(DataProducer):
    def __init__(self,
                 configuration_path,
                 *args,
                 batch_size=4,
                 path=os.path.join(".", "network_datasets"),
                 **kwargs):
        self._config = dict()
        self._configuration_path = configuration_path
        self._load_configuration(configuration_path)

        super().__init__(*args,
                         identifier=self._config["identifier"],
                         north=bool(self._config["north"]),
                         path=path,
                         south=bool(self._config["south"]),
                         **kwargs)

        self._batch_size = batch_size
        self._counts = self._config["counts"]
        self._dtype = getattr(np, self._config["dtype"])
        self._loader_config = self._config["loader_config"]
        self._n_forecast_days = self._config["n_forecast_days"]
        self._num_channels = self._config["num_channels"]
        self._shape = tuple(self._config["shape"])

        self._missing_dates = [
            dt.datetime.strptime(s, IceNetPreProcessor.DATE_FORMAT)
            for s in self._config["missing_dates"]]

    def _load_configuration(self, path):
        if os.path.exists(path):
            logging.info("Loading configuration {}".format(path))

            with open(path, "r") as fh:
                obj = json.load(fh)

                self._config.update(obj)
        else:
            raise OSError("{} not found".format(path))

    def get_split_datasets(self, ratio=None):
        train_fns = glob.glob("{}/*.tfrecord".format(
            self.get_data_var_folder("train"),
            missing_error=True))
        val_fns = glob.glob("{}/*.tfrecord".format(
            self.get_data_var_folder("val"),
            missing_error=True))
        test_fns = glob.glob("{}/*.tfrecord".format(
            self.get_data_var_folder("test"),
            missing_error=True))

        if not (len(train_fns) + len(val_fns) + len(test_fns)):
            raise RuntimeError("No files have been found, abandoning...")

        if ratio:
            if ratio > 1.0:
                raise RuntimeError("Ratio cannot be more than 1")

            logging.info("Reducing datasets to {} of total files".format(ratio))
            train_idx, val_idx, test_idx = \
                int(len(train_fns) * ratio), \
                int(len(val_fns) * ratio), \
                int(len(test_fns) * ratio)

            if train_idx > 0:
                train_fns = train_fns[:train_idx]
            if val_idx > 0:
                val_fns = val_fns[:val_idx]
            if test_idx > 0:
                test_fns = test_fns[:test_idx]

        train_ds, val_ds, test_ds = \
            tf.data.TFRecordDataset(train_fns,
                                    num_parallel_reads=self.batch_size), \
            tf.data.TFRecordDataset(val_fns,
                                    num_parallel_reads=self.batch_size), \
            tf.data.TFRecordDataset(test_fns,
                                    num_parallel_reads=self.batch_size),

        # TODO: Comparison/profiling runs
        # TODO: parallel for batch size while that's small
        # TODO: obj.decode_item might not work here - figure out runtime
        #  implementation based on wrapped function call that can be serialised
        decoder = get_decoder(self.shape,
                              self.num_channels,
                              self.n_forecast_days,
                              dtype=self._dtype.__name__)

        train_ds = train_ds.\
                shuffle(int(min(len(train_fns) / 4, 100)),
                        reshuffle_each_iteration=True).\
                map(decoder, num_parallel_calls=self.batch_size).\
                batch(self.batch_size)

        val_ds = val_ds.\
            map(decoder, num_parallel_calls=self.batch_size).\
            batch(self.batch_size)

        test_ds = test_ds.\
            map(decoder, num_parallel_calls=self.batch_size).\
            batch(self.batch_size)

        return train_ds.prefetch(tf.data.AUTOTUNE), \
               val_ds.prefetch(tf.data.AUTOTUNE), \
               test_ds.prefetch(tf.data.AUTOTUNE)

    def get_data_loader(self):
        loader = IceNetDataLoader(self.loader_config,
                                  self.identifier,
                                  self._config["var_lag"],
                                  dataset_config_path=os.path.dirname(
                                      self._configuration_path),
                                  loss_weight_days=self._config[
                                      "loss_weight_days"],
                                  north=self.north,
                                  output_batch_size=self._config[
                                      "output_batch_size"],
                                  path=self._config["loader_path"],
                                  south=self.south,
                                  var_lag_override=self._config[
                                      "var_lag_override"])
        return loader

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def channels(self):
        return self._config['channels']

    @property
    def counts(self):
        return self._counts

    @property
    def dtype(self):
        return self._dtype

    @property
    def loader_config(self):
        return self._loader_config

    @property
    def missing_days(self):
        return self._missing_dates

    @property
    def n_forecast_days(self):
        return self._n_forecast_days

    @property
    def num_channels(self):
        return self._num_channels

    @property
    def shape(self):
        return self._shape
