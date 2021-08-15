import collections
import datetime as dt
import json
import logging
import os
import sys

from concurrent.futures import ProcessPoolExecutor
from dateutil.relativedelta import relativedelta
from pprint import pformat

# https://stackoverflow.com/questions/55852831/
# tf-data-vs-keras-utils-sequence-performance

import numpy as np
import pandas as pd
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
                 loss_weight_days=True,
                 n_forecast_days=93,
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
        self._channel_files = dict()
        self._config = dict()
        self._loss_weight_days = loss_weight_days
        self._masks = Masks(north=self.north, south=self.south)
        self._meta_channels = []
        self._missing_dates = []
        self._n_forecast_days = n_forecast_days
        self._output_batch_size = output_batch_size
        self._rng = np.random.default_rng(seed)
        self._seed = seed

        self._var_lag = var_lag
        self._var_lag_override = dict() \
            if not var_lag_override else var_lag_override

        self._load_configuration(configuration_path)
        self._construct_channels()

        self._dtype = getattr(np, self._config["dtype"])
        self._shape = tuple(self._config["shape"])
        self._missing_dates = [
            dt.datetime.strptime(s, IceNetPreProcessor.DATE_FORMAT)
            for s in self._config["missing_dates"]]

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
                                  for s in
                                  self._config["sources"][identity][dataset]])
            output_dir = self.get_data_var_folder(dataset)

            logging.info("{} {} dates in total, generating cache "
                         "data.".format(len(forecast_dates), dataset))

            def batch(dates, num):
                i = 0
                while i < len(dates):
                    yield dates[i:i + num]
                    i += num

            for dates in batch(forecast_dates, self._output_batch_size):
                tf_path = os.path.join(output_dir,
                                       "{:08}.tfrecord".format(batch_number))

                with tf.io.TFRecordWriter(tf_path) as writer:
                    for date in dates:
                        logging.debug("Generating date {}".
                                      format(date.strftime(
                                             IceNetPreProcessor.DATE_FORMAT)))
                        self._generate_tfrecord(writer, pd.Timestamp(date))

                logging.info("Finished batch {}".format(tf_path))
                batch_number += 1

    def _get_var_file(self, var_name, date,
                      date_format=IceNetPreProcessor.DATE_FORMAT,
                      filename_override=None):
        filename = "{}.npy".format(date.strftime(date_format)) if not \
            filename_override else filename_override
        source_path = os.path.join(var_name.split("_")[0],
                                   self.hemisphere_str[0],
                                   filename)

        files = [potential for potential in self._channel_files[var_name]
                 if source_path in potential]

        if len(files) > 1:
            logging.warning("Multiple files found for {}, only returning {}".
                            format(filename, files[0]))
        elif not len(files):
            logging.warning("No files in channel list for {}".format(filename))
            return None
        return files[0]

    def _generate_tfrecord(self, writer, forecast_date):
        x, y = None, None

        # OUTPUT SETUP - happens for any sample, even if only predicting
        # TODO: is there any benefit to changing this? Not likely

        # Build up the set of N_samps output SIC time-series
        #   (each n_forecast_months long in the time dimension)

        # To become array of shape (*raw_data_shape, n_forecast_months)
        sample_sic_list = []

        for leadtime_idx in range(self._n_forecast_days):
            forecast_day = forecast_date + pd.DateOffset(days=leadtime_idx)
            sic_filename = self._get_var_file("siconca_abs", forecast_day)

            if not sic_filename:
                # Output file does not exist - fill it with NaNs
                sample_sic_list.append(
                    np.full(self._shape, np.nan))

            else:
                sample_sic_list.append(np.load(sic_filename))

        sample_output = np.stack(sample_sic_list, axis=2)
        sample_output = np.moveaxis(sample_output, source=0, destination=2)

        y = np.zeros((*self._shape,
                      self._n_forecast_days,
                      2),
                     dtype=self._dtype)

        y[:, :, :, 0] = sample_output

        # Masked recomposition of output
        for leadtime_idx in range(self._n_forecast_days):
            forecast_day = forecast_date + pd.DateOffset(days=leadtime_idx)

            if any([forecast_day == missing_date
                    for missing_date in self._missing_dates]):
                sample_weight = np.zeros(self._shape,
                                         self._dtype)

            else:
                # Zero loss outside of 'active grid cells'
                sample_weight = self._masks.get_active_cell_mask(forecast_day)
                sample_weight = sample_weight.astype(self._dtype)

                # Scale the loss for each month s.t. March is
                #   scaled by 1 and Sept is scaled by 1.77
                if self._loss_weight_days:
                    sample_weight *= 33928. / np.sum(sample_weight)

            y[:, :, leadtime_idx, 1] = sample_weight

        # INPUT FEATURES

        X = np.zeros((
            *self._shape,
            self.num_channels
        ), dtype=self._dtype)

        present_date = forecast_date - relativedelta(days=1)
        v1, v2 = 0, 0

        for var_name, num_channels in self._channels.items():
            # FIXME: surely these are current date to previous n, changed
            if "linear_trend" not in var_name:
                input_days = [present_date - relativedelta(days=int(lag))
                              for lag in np.arange(0, num_channels)]
            else:
                input_days = [present_date + relativedelta(days=int(lead))
                              for lead in np.arange(1, num_channels + 1)]

            v2 += num_channels

            X[:, :, v1:v2] = \
                np.stack([np.load(self._get_var_file(var_name, date))
                          for date in input_days], axis=-1)

            v1 += num_channels

        for var_name in self._meta_channels:
            if self._channels[var_name] >= 1:
                raise RuntimeError("{} meta variable cannot have more than "
                                   "one channel".format(var_name))

            X[:, :, v1] = \
                np.stack([np.load(
                    self._get_var_file(var_name, forecast_date, "%j"))],
                    axis=-1)

            v1 += self._channels[var_name]

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

    def _add_channel_files(self, var_name, filelist):
        if var_name in self._channel_files:
            logging.warning("{} already has files, but more found, "
                            "this could be an unintentional merge of "
                            "sources".format(var_name))
        else:
            self._channel_files[var_name] = []

        logging.debug("Adding {} to {} channel".format(len(filelist), var_name))
        self._channel_files[var_name] += filelist

    def _construct_channels(self):
        lag_vars = [(identity, var, data_format)
                    for data_format in ("abs", "anom")
                    for identity in self._config["sources"].keys()
                    for var in self._config["sources"][identity][data_format]]

        for identity, var_name, data_format in lag_vars:
            var_prefix = "{}_{}".format(var_name, data_format)
            var_lag = (self._var_lag
                       if var_name not in self._var_lag_override
                       else self._var_lag_override[var_name])

            self._channel_names += ["{}_{}".format(var_prefix, i)
                                    for i in np.arange(1, var_lag + 1)]
            self._channels[var_prefix] = int(var_lag)
            self._add_channel_files(
                var_prefix,
                self._config["sources"][identity]["var_files"][var_name])

        trend_names = [(identity, var,
                        self._config["sources"][identity]["linear_trend_days"])
                       for identity in self._config["sources"].keys()
                       for var in
                       self._config["sources"][identity]["linear_trends"]]

        for identity, var_name, trend_days in trend_names:
            var_prefix = "{}_linear_trend".format(var_name)

            self._channel_names += ["{}_{}".format(var_prefix, leadtime)
                                    for leadtime in
                                    np.arange(1, trend_days + 1)]
            self._channels[var_prefix] = int(trend_days)
            self._add_channel_files(
                var_prefix,
                self._config["sources"][identity]["var_files"][var_name])

        # Metadata input variables that don't span time
        meta_names = [(identity, var)
                      for identity in self._config["sources"].keys()
                      for var in self._config["sources"][identity]["meta"]]

        for identity, var_name in meta_names:
            self._meta_channels.append(var_name)
            self._channels[var_name] = 1
            self._add_channel_files(
                var_name,
                self._config["sources"][identity]["var_files"][var_name])

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
    dl.prepare()

