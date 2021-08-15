import collections
import datetime as dt
import json
import logging
import os
import sys

from concurrent.futures import ProcessPoolExecutor
from pprint import pformat

# https://stackoverflow.com/questions/55852831/
# tf-data-vs-keras-utils-sequence-performance

import numpy as np
import tensorflow as tf

from icenet2.data.sic.mask import Masks
from icenet2.data.process import IceNetPreProcessor
from icenet2.data.producers import Generator


# TODO: TFDatasetGenerator should be created, so we can also have an
#  alternate numpy based loader. Easily abstracted after implementation and
#  can also inherit from a new BatchGenerator - this family tree can be rich!
class IceNetDataLoader(Generator):
    """
    Custom data loader class for generating batches of input-output tensors for
    training IceNet. Inherits from  keras.utils.Sequence, which ensures each the
    network trains once on each  sample per epoch. Must implement a __len__
    method that returns the  number of batches and a __getitem__ method that
    returns a batch of data. The  on_epoch_end method is called after each
    epoch.
    See: https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence

    This inherits Generator, not a Processor, as it combines multiple Processors
    from the configuration
    """

    def __init__(self,
                 configuration_path,
                 identifier,
                 var_lag,
                 *args,
                 output_batch_size=32,
                 path=os.path.join(".", "network_datasets"),
                 seed=None,
                 var_lag_override=None,
                 **kwargs):
        super().__init__(*args,
                         identifier=identifier,
                         path=path,
                         **kwargs)

        self._channels = dict()
        self._channel_names = []
        self._config = dict()
        self._masks = Masks(north=self.north, south=self.south)
        self._output_batch_size = output_batch_size
        self._rng = np.random.default_rng(seed)
        self._seed = seed

        self._var_lag = var_lag
        self._var_lag_override = dict() \
            if not var_lag_override else var_lag_override

        self._load_configuration(configuration_path)
        self._determine_names_and_channels()

    def get_datasets(self):
        train_ds, val_ds, test_ds = None, None, None

        return train_ds, val_ds, test_ds

    def generate(self):

        # TODO: for each set, validate every variable has an appropriate file
        #  in the configuration arrays, otherwise drop the forecast date

        for dataset in ("train", "val", "test"):
            batch_number = 0
            forecast_dates = set([dt.datetime.strptime(s,
                                  IceNetPreProcessor.DATE_FORMAT).date()
                                  for identity in self._config.keys()
                                  for s in self._config[identity][dataset]])
            output_dir = self.get_data_var_folder(dataset)

            logging.info("{} {} dates in total, generating cache "
                         "data.".format(len(forecast_dates), dataset))

            def batch(dates, num):
                i = 0
                while i < len(dates):
                    yield dates[i:i + num]
                    i += num

            tasks = []

            for dates in batch(forecast_dates, self._output_batch_size):
                tf_path = os.path.join(output_dir,
                                       "{:08}.tfrecord".format(batch_number))

                batch_number += 1
                with tf.io.TFRecordWriter(tf_path) as writer:
                    for date in dates:
                        logging.debug("Generating date {}".
                                      format(date.strftime(
                                             IceNetPreProcessor.DATE_FORMAT)))
                        self._generate_tfrecord(writer, date)

    def _generate_tfrecord(self, writer, forecast_date):
        x, y = None, None

        logging.debug("x shape {}, y shape {}".format(x.shape, y.shape))

        record_data = tf.train.Example(features=tf.train.Features(feature={
            "x": tf.train.Feature(
                float_list=tf.train.FloatList(value=x.reshape(-1))),
            "y": tf.train.Feature(
                float_list=tf.train.FloatList(value=y.reshape(-1))),
        })).SerializeToString()

        writer.write(record_data)

    def _load_configuration(self, path):
        if os.path.exists(path):
            logging.info("Loading configuration {}".format(path))

            with open(path, "r") as fh:
                obj = json.load(fh)

                self._config.update(obj)
        else:
            raise OSError("{} not found".format(path))

    def _determine_names_and_channels(self):
        """
        Set up a list of strings for the names of each input variable (in the
        correct order) by looping over the `input_data` dictionary.
        """

        lag_vars = [(var, data_format)
                    for data_format in ("abs", "anom")
                    for identity in self._config.keys()
                    for var in self._config[identity][data_format]]

        for var_name, data_format in lag_vars:
            var_prefix = "{}_{}".format(var_name, data_format)
            var_lag = (self._var_lag
                       if var_name not in self._var_lag_override
                       else self._var_lag_override[var_name])

            self._channel_names += ["{}_{}".format(var_prefix, i)
                                    for i in np.arange(1, var_lag + 1)]
            self._channels[var_prefix] = int(var_lag)

        trend_names = [(var, self._config[identity]["linear_trend_days"])
                       for identity in self._config.keys()
                       for var in self._config[identity]["linear_trends"]]

        for (var_name, trend_days) in trend_names:
            var_prefix = "{}_linear_trend".format(var_name)

            self._channel_names += ["{}_{}".format(var_prefix, leadtime)
                                    for leadtime in
                                    np.arange(1, trend_days + 1)]
            self._channels[var_prefix] = int(trend_days)

        # Metadata input variables that don't span time
        meta_names = [var
                      for identity in self._config.keys()
                      for var in self._config[identity]["meta"]]

        for var_name in meta_names:
            self._channel_names.append(var_name)
            self._channels[var_name] = 1

        logging.debug("Variable names deduced:\n{}".format(
            pformat(self._channel_names)))
        logging.debug("Channel quantities deduced:\n{}\n\nTotal channels: {}".
            format(pformat(self._channels), self.num_channels))

    @property
    def num_channels(self):
        return sum(self._channels.values())


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    dl = IceNetDataLoader("loader.test1.json",
                          "test_forecast",
                          7,
                          north=True)
    #dl.prepare()

