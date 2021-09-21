import collections
import concurrent.futures
import datetime as dt
import glob
import json
import logging
import os
import sys

from concurrent.futures import ThreadPoolExecutor
from dateutil.relativedelta import relativedelta
from pprint import pformat

# https://stackoverflow.com/questions/55852831/
# tf-data-vs-keras-utils-sequence-performance

import numpy as np
import pandas as pd
import tensorflow as tf

from icenet2.data.sic.mask import Masks
from icenet2.data.process import IceNetPreProcessor
from icenet2.data.producers import DataProducer, Generator


def get_decoder(shape, channels, forecasts, num_vars=2, dtype="float32"):
    xf = tf.io.FixedLenFeature(
        [*shape, channels], getattr(tf, dtype))
    yf = tf.io.FixedLenFeature(
        [*shape, forecasts, num_vars], getattr(tf, dtype))

    @tf.function
    def decode_item(proto):
        features = {
            "x": xf,
            "y": yf,
        }

        item = tf.io.parse_single_example(proto, features)
        return item['x'], item['y']

    return decode_item


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
                 dataset_config_path=".",
                 generate_workers=8,
                 loss_weight_days=True,
                 n_forecast_days=93,
                 output_batch_size=32,
                 path=os.path.join(".", "network_datasets"),
                 var_lag_override=None,
                 **kwargs):
        super().__init__(*args,
                         identifier=identifier,
                         path=path,
                         **kwargs)

        self._channels = dict()
        self._channel_names = []
        self._channel_files = dict()
        self._configuration_path = configuration_path
        self._dataset_config_path = dataset_config_path
        self._config = dict()
        self._loss_weight_days = loss_weight_days
        self._masks = Masks(north=self.north, south=self.south)
        self._meta_channels = []
        self._missing_dates = []
        self._n_forecast_days = n_forecast_days
        self._output_batch_size = output_batch_size
        self._workers = 8

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

    def generate(self):
        # TODO: for each set, validate every variable has an appropriate file
        #  in the configuration arrays, otherwise drop the forecast date
        splits = ("train", "val", "test")
        counts = {el: 0 for el in splits}

        for dataset in splits:
            batch_number = 0

            forecast_dates = sorted(list(set([dt.datetime.strptime(s,
                                    IceNetPreProcessor.DATE_FORMAT).date()
                                    for identity in
                                              self._config["sources"].keys()
                                    for s in
                                    self._config["sources"][identity]
                                    ["dates"][dataset]])))
            output_dir = self.get_data_var_folder(dataset)

            logging.info("{} {} dates in total, generating cache "
                         "data.".format(len(forecast_dates), dataset))

            def batch(dates, num):
                i = 0
                while i < len(dates):
                    yield dates[i:i + num]
                    i += num

            # TODO: generate_sample for processpoolexecution/distribution,
            #  we want to max out write I/O for generating these sets and the
            #  GIL gets in the way
            def generate_and_write(dl, path, dates):
                with tf.io.TFRecordWriter(path) as writer:
                    # TODO: multiprocess
                    for date in dates:
                        logging.debug("Generating date {}".format(
                            date.strftime(IceNetPreProcessor.DATE_FORMAT)))
                        x, y = dl.generate_sample(date)
                        dl._write_tfrecord(writer, x, y)
                return path

            with ThreadPoolExecutor(max_workers=self._workers) as executor:
                tf_path = os.path.join(output_dir,
                                       "{:08}.tfrecord")

                futures = []

                for dates in batch(forecast_dates, self._output_batch_size):
                    futures.append(executor.submit(generate_and_write,
                                                   self,
                                                   tf_path.format(batch_number),
                                                   dates))
                    batch_number += 1
                    counts[dataset] += len(dates)

                for fut in concurrent.futures.as_completed(futures):
                    path = fut.result()
                    logging.info("Finished batch {}".format(path))

        self._write_dataset_config(counts)

    def generate_sample(self, forecast_date):
        # OUTPUT SETUP - happens for any sample, even if only predicting
        # TODO: is there any benefit to changing this? Not likely

        # Build up the set of N_samps output SIC time-series
        #   (each n_forecast_months long in the time dimension)

        # To become array of shape (*raw_data_shape, n_forecast_months)
        sample_sic_list = []

        for leadtime_idx in range(self._n_forecast_days):
            forecast_day = forecast_date + relativedelta(days=leadtime_idx)
            sic_filename = self._get_var_file("siconca_abs", forecast_day)

            if not sic_filename:
                # Output file does not exist - fill it with NaNs
                sample_sic_list.append(
                    np.full(self._shape, np.nan))

            else:
                channel_data = np.load(sic_filename)
                sample_sic_list.append(channel_data)

        sample_output = np.stack(sample_sic_list, axis=2)

        y = np.zeros((*self._shape,
                      self._n_forecast_days,
                      2),
                     dtype=self._dtype)

        y[:, :, :, 0] = sample_output

        # Masked recomposition of output
        for leadtime_idx in range(self._n_forecast_days):
            forecast_day = forecast_date + relativedelta(days=leadtime_idx)

            if any([forecast_day == missing_date
                    for missing_date in self._missing_dates]):
                sample_weight = np.zeros(self._shape,
                                         self._dtype)

            else:
                # Zero loss outside of 'active grid cells'
                sample_weight = self._masks.get_active_cell_mask(
                    forecast_day.month)
                sample_weight = sample_weight.astype(self._dtype)

                # Scale the loss for each month s.t. March is
                #   scaled by 1 and Sept is scaled by 1.77
                if self._loss_weight_days:
                    sample_weight *= 33928. / np.sum(sample_weight)

            y[:, :, leadtime_idx, 1] = sample_weight

        y[..., 0:1] = np.nan_to_num(y[..., 0:1])

        # INPUT FEATURES
        x = np.zeros((
            *self._shape,
            self.num_channels
        ), dtype=self._dtype)

        v1, v2 = 0, 0

        for var_name, num_channels in self._channels.items():
            if var_name in self._meta_channels:
                continue

            if "linear_trend" not in var_name:
                input_days = [forecast_date - relativedelta(days=int(lag))
                              for lag in np.arange(1, num_channels + 1)]
            else:
                input_days = [forecast_date + relativedelta(days=int(lead))
                              for lead in np.arange(1, num_channels + 1)]

            v2 += num_channels

            x[:, :, v1:v2] = \
                np.stack([np.load(self._get_var_file(var_name, date))
                          if self._get_var_file(var_name, date)
                          else np.zeros(self._shape)
                          for date in input_days], axis=-1)

            v1 += num_channels

        for var_name in self._meta_channels:
            if self._channels[var_name] > 1:
                raise RuntimeError("{} meta variable cannot have more than "
                                   "one channel".format(var_name))

            x[:, :, v1] = \
                np.load(self._get_var_file(var_name, forecast_date, "%j"))

            v1 += self._channels[var_name]

        logging.debug("x shape {}, y shape {}".format(x.shape, y.shape))

        return x, y

    def _add_channel_files(self, var_name, filelist):
        if var_name in self._channel_files:
            logging.warning("{} already has files, but more found, "
                            "this could be an unintentional merge of "
                            "sources".format(var_name))
        else:
            self._channel_files[var_name] = []

        logging.debug("Adding {} to {} channel".format(len(filelist), var_name))
        self._channel_files[var_name] += filelist

    # FIXME: there is a chance of reordering of channel names
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

        # Keep intuitive order from the start, which will allow easier
        # analysis of input data down the line
        self._channel_names = sorted(self._channel_names)
        self._channels = {var: self._channels[var] for var in
                          sorted(self._channels.keys())}
        self._meta_channels = sorted(self._meta_channels)

        logging.debug("Variable names deduced:\n{}".format(
            pformat(self._channel_names)))
        logging.debug("Channel quantities deduced:\n{}\n\nTotal channels: {}".
            format(pformat(self._channels), self.num_channels))

    def _get_var_file(self, var_name, date,
                      date_format=IceNetPreProcessor.DATE_FORMAT,
                      filename_override=None):
        filename = "{}.npy".format(date.strftime(date_format)) if not \
            filename_override else filename_override
        source_path = os.path.join(self.hemisphere_str[0],
                                   var_name.split("_")[0],
                                   filename)

        files = [potential for potential in self._channel_files[var_name]
                 if source_path in potential]

        if len(files) > 1:
            logging.warning("Multiple files found for {}, only returning {}".
                            format(filename, files[0]))
        elif not len(files):
            #logging.warning("No files in channel list for {}".format(filename))
            return None
        return files[0]

    def _load_configuration(self, path):
        if os.path.exists(path):
            logging.info("Loading configuration {}".format(path))

            with open(path, "r") as fh:
                obj = json.load(fh)

                self._config.update(obj)
        else:
            raise OSError("{} not found".format(path))

    def _write_dataset_config(self, counts):
        # TODO: move to utils for this and process
        def _serialize(x):
            if x is dt.date:
                return x.strftime(IceNetPreProcessor.DATE_FORMAT)
            return str(x)

        configuration = {
            "identifier":       self.identifier,
            "implementation":   self.__class__.__name__,
            "counts":           counts,
            "dtype":            self._dtype.__name__,
            "loader_config":    self._configuration_path,
            "missing_dates":    [date.strftime(
                IceNetPreProcessor.DATE_FORMAT) for date in
                self._missing_dates],
            "n_forecast_days":  self._n_forecast_days,
            "north":            self.north,
            "num_channels":     self.num_channels,
            # FIXME: this naming is inconsistent, sort it out!!! ;)
            "shape":            list(self._shape),
            "south":            self.south,

            # For recreating this dataloader
            # "dataset_config_path = ".",
            "loader_path":      self._path,
            "loss_weight_days": self._loss_weight_days,
            "output_batch_size": self._output_batch_size,
            "var_lag":          self._var_lag,
            "var_lag_override": self._var_lag_override,
        }

        output_path = os.path.join(self._dataset_config_path,
                                   "dataset_config.{}.json".format(
                                       self.identifier))

        logging.info("Writing configuration to {}".format(output_path))

        with open(output_path, "w") as fh:
            json.dump(configuration, fh, indent=4, default=_serialize)

    def _write_tfrecord(self, writer, x, y):
        record_data = tf.train.Example(features=tf.train.Features(feature={
            "x": tf.train.Feature(
                float_list=tf.train.FloatList(value=x.reshape(-1))),
            "y": tf.train.Feature(
                float_list=tf.train.FloatList(value=y.reshape(-1))),
        })).SerializeToString()

        writer.write(record_data)

    @property
    def num_channels(self):
        return sum(self._channels.values())


class IceNetDataSet(DataProducer):
    def __init__(self,
                 configuration_path,
                 *args,
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

    def get_split_datasets(self, batch_size=4, prefetch=4, ratio=None):
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
            tf.data.TFRecordDataset(train_fns), \
            tf.data.TFRecordDataset(val_fns), \
            tf.data.TFRecordDataset(test_fns),

        # TODO: Comparison/profiling runs
        # TODO: parallel for batch size while that's small
        # TODO: obj.decode_item might not work here - figure out runtime
        #  implementation based on wrapped function call that can be serialised
        decoder = get_decoder(self.shape,
                              self.num_channels,
                              self.n_forecast_days,
                              dtype=self._dtype.__name__)

        train_ds = train_ds.map(decoder, num_parallel_calls=batch_size).\
            batch(batch_size)  # .shuffle(batch_size)
        val_ds = val_ds.map(decoder, num_parallel_calls=batch_size).\
            batch(batch_size)
        test_ds = test_ds.map(decoder, num_parallel_calls=batch_size).\
            batch(batch_size)

        return train_ds.prefetch(prefetch), \
               val_ds.prefetch(prefetch), \
               test_ds.prefetch(prefetch)

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
    def counts(self):
        return self._counts

    @property
    def dtype(self):
        return self._dtype

    @property
    def loader_config(self):
        return self._loader_config

    @property
    def loss_weight_days(self):
        return self._loss_weight_days

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


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    dl = IceNetDataLoader("loader.test1.json",
                          "test_forecast",
                          7,
                          north=True)
    dl.generate()

    ds = IceNetDataSet(os.path.join(".", "dataset_config.test_forecast.json"))
    _, _, test = ds.get_split_datasets()
    x1, y1 = list(test.as_numpy_iterator())[0]
    print(x1.shape)
    print(y1.shape)

    other_dl = ds.get_data_loader()
    x2, y2 = other_dl.generate_sample(dt.date(2020, 1, 1))
    print(x2.shape)
    print(y2.shape)

