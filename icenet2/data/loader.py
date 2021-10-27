import concurrent.futures
import datetime as dt
import glob
import json
import logging
import os

from concurrent.futures import ProcessPoolExecutor
from dateutil.relativedelta import relativedelta
from pprint import pformat

# https://stackoverflow.com/questions/55852831/
# tf-data-vs-keras-utils-sequence-performance

import numpy as np
import tensorflow as tf

from icenet2.data.sic.mask import Masks
from icenet2.data.process import IceNetPreProcessor
from icenet2.data.producers import DataProducer, Generator


def generate_and_write(path, dates_args):
    with tf.io.TFRecordWriter(path) as writer:
        for date in dates_args.keys():
            x, y, sample_weights = generate_sample(date, *dates_args[date])
            write_tfrecord(writer, x, y, sample_weights)
    return path


def generate_sample(forecast_date,
                    channels,
                    dtype,
                    loss_weight_days,
                    masks,
                    meta_channels,
                    missing_dates,
                    n_forecast_days,
                    num_channels,
                    shape,
                    var_files,
                    output_files):
    # OUTPUT SETUP - happens for any sample, even if only predicting
    # TODO: is there any benefit to changing this? Not likely

    # Build up the set of N_samps output SIC time-series
    #   (each n_forecast_months long in the time dimension)

    # To become array of shape (*raw_data_shape, n_forecast_days)
    sample_sic_list = []

    for leadtime_idx in range(n_forecast_days):
        forecast_day = forecast_date + relativedelta(days=leadtime_idx)
        sic_filename = output_files[forecast_day] \
            if forecast_day in output_files else None

        if not sic_filename:
            # Output file does not exist - fill it with NaNs
            sample_sic_list.append(np.full(shape, np.nan))

        else:
            channel_data = np.load(sic_filename)
            sample_sic_list.append(channel_data)

    sample_output = np.stack(sample_sic_list, axis=2)

    y = np.zeros((*shape,
                  n_forecast_days,
                  1),
                 dtype=dtype)
    sample_weights = np.zeros((*shape,
                               n_forecast_days,
                               1),
                              dtype=dtype)

    y[:, :, :, 0] = sample_output

    # Masked recomposition of output
    for leadtime_idx in range(n_forecast_days):
        forecast_day = forecast_date + relativedelta(days=leadtime_idx)

        if any([forecast_day == missing_date
                for missing_date in missing_dates]):
            sample_weight = np.zeros(shape, dtype)
        else:
            # Zero loss outside of 'active grid cells'
            sample_weight = masks[forecast_day]
            sample_weight = sample_weight.astype(dtype)

            # Scale the loss for each month s.t. March is
            #   scaled by 1 and Sept is scaled by 1.77
            if loss_weight_days:
                sample_weight *= 33928. / np.sum(sample_weight)

        sample_weights[:, :, leadtime_idx, 0] = sample_weight

    # CHEAT: y[..., 0:1] = np.nan_to_num(y[..., 0:1])

    # INPUT FEATURES
    x = np.zeros((
        *shape,
        num_channels
    ), dtype=dtype)

    v1, v2 = 0, 0

    for var_name, num_channels in channels.items():
        if var_name in meta_channels:
            continue

        v2 += num_channels

        logging.debug("Var {}: files {}".format(var_name,
                                                var_files[var_name].values()))
        x[:, :, v1:v2] = \
            np.stack([np.load(filename)
                      if filename
                      else np.zeros(shape)
                      for filename in var_files[var_name].values()], axis=-1)

        v1 += num_channels

    for var_name in meta_channels:
        if channels[var_name] > 1:
            raise RuntimeError("{} meta variable cannot have more than "
                               "one channel".format(var_name))

        x[:, :, v1] = np.load(var_files[var_name])
        v1 += channels[var_name]

    logging.debug("x shape {}, y shape {}".format(x.shape, y.shape))

    return x, y, sample_weights


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
        return item['x'], item['y']

    return decode_item


def write_tfrecord(writer, x, y, sample_weights):
    record_data = tf.train.Example(features=tf.train.Features(feature={
        "x": tf.train.Feature(
            float_list=tf.train.FloatList(value=x.reshape(-1))),
        "y": tf.train.Feature(
            float_list=tf.train.FloatList(value=y.reshape(-1))),
        "sample_weights": tf.train.Feature(
            float_list=tf.train.FloatList(value=sample_weights.reshape(-1))),
    })).SerializeToString()

    writer.write(record_data)


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
        self._workers = generate_workers

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
        futures = []

        def batch(dates, num):
            i = 0
            while i < len(dates):
                yield dates[i:i + num]
                i += num

        # This was a quick and dirty beef-up of the implementation as it's
        # very I/O bursty work. It significantly reduces the overall time
        # taken to produce a full dataset at BAS so we can use this as a
        # paradigm moving forward (with a slightly cleaner implementation)
        with ProcessPoolExecutor(max_workers=self._workers) as executor:
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

                tf_path = os.path.join(output_dir,
                                       "{:08}.tfrecord")

                for dates in batch(forecast_dates, self._output_batch_size):
                    args = {}

                    for date in dates:
                        masks = {}
                        var_files = {}

                        for day in range(self._n_forecast_days):
                            forecast_day = date + relativedelta(days=day)

                            masks[forecast_day] = \
                                self._masks.get_active_cell_mask(
                                    forecast_day.month)

                        for var_name in self._meta_channels:
                            var_files[var_name] = \
                                self._get_var_file(var_name, date, "%j")

                        for var_name, num_channels in self._channels.items():
                            if var_name in self._meta_channels:
                                continue

                            if "linear_trend" not in var_name:
                                # Collect all lag channels + the forecast date
                                input_days = [
                                    date - relativedelta(days=int(lag))
                                    for lag in
                                    np.arange(1, num_channels + 1)]
                            else:
                                input_days = [
                                    date + relativedelta(days=int(lead))
                                    for lead in
                                    np.arange(1, num_channels + 1)]

                            var_files[var_name] = {
                                input_date: self._get_var_file(
                                    var_name, input_date)
                                for input_date in set(sorted(input_days))}

                        output_files = {
                            input_date:
                                self._get_var_file("siconca_abs", input_date)
                            for input_date in [
                                date + relativedelta(days=leadtime_idx)
                                for leadtime_idx in
                                range(self._n_forecast_days)]
                        }

                        # TODO: I don't like this, but I was trying to ensure
                        #  no deadlock to producing sets due to this object
                        #  not being serializable (even though I'm sure it
                        #  is). Refactor and clean this up!
                        args[date] = [
                            self._channels,
                            self._dtype,
                            self._loss_weight_days,
                            masks,
                            self._meta_channels,
                            self._missing_dates,
                            self._n_forecast_days,
                            self.num_channels,
                            self._shape,
                            var_files,
                            output_files,
                        ]

                    futures.append(executor.submit(generate_and_write,
                                                   tf_path.format(batch_number),
                                                   args))

                    logging.debug("Submitted {} dates as batch {}".format(
                        len(dates), batch_number))
                    batch_number += 1
                    counts[dataset] += len(dates)

                logging.info("{} tasks submitted".format(len(futures)))

            for fut in concurrent.futures.as_completed(futures):
                path = fut.result()
                logging.info("Finished output {}".format(path))

        self._write_dataset_config(counts)

    def generate_sample(self, date):
        # TODO: UGH this is repeating due to the need to reproduce process
        #  for DL
        masks = {}
        var_files = {}

        for day in range(self._n_forecast_days):
            forecast_day = date + relativedelta(days=day)

            masks[forecast_day] = \
                self._masks.get_active_cell_mask(
                    forecast_day.month)

        for var_name in self._meta_channels:
            var_files[var_name] = \
                self._get_var_file(var_name, date, "%j")

        for var_name, num_channels in self._channels.items():
            if var_name in self._meta_channels:
                continue

            if "linear_trend" not in var_name:
                # Collect all lag input channels + forecast date
                input_days = [
                    date - relativedelta(days=int(lag))
                    for lag in
                    np.arange(1, num_channels + 1)]
            else:
                input_days = [
                    date + relativedelta(days=int(lead))
                    for lead in
                    np.arange(1, num_channels + 1)]

            var_files[var_name] = {
                input_date: self._get_var_file(
                    var_name, input_date)
                for input_date in input_days}

        output_files = {
            input_date:
                self._get_var_file("siconca_abs", input_date)
            for input_date in [
                date + relativedelta(days=leadtime_idx)
                for leadtime_idx in
                range(self._n_forecast_days)]
        }

        return generate_sample(
            date,
            self._channels,
            self._dtype,
            self._loss_weight_days,
            masks,
            self._meta_channels,
            self._missing_dates,
            self._n_forecast_days,
            self.num_channels,
            self._shape,
            var_files,
            output_files)

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
        # As of Python 3.7 dict guarantees the order of keys based on
        # original insertion order, which is great for this method
        lag_vars = [(identity, var, data_format)
                    for data_format in ("abs", "anom")
                    for identity in
                    sorted(self._config["sources"].keys())
                    for var in
                    sorted(self._config["sources"][identity][data_format])]

        for identity, var_name, data_format in lag_vars:
            var_prefix = "{}_{}".format(var_name, data_format)
            var_lag = (self._var_lag
                       if var_name not in self._var_lag_override
                       else self._var_lag_override[var_name])

            self._channels[var_prefix] = int(var_lag)
            self._add_channel_files(
                var_prefix,
                self._config["sources"][identity]["var_files"][var_name])

        trend_names = [(identity, var,
                        self._config["sources"][identity]["linear_trend_days"])
                       for identity in
                       sorted(self._config["sources"].keys())
                       for var in
                       sorted(
                           self._config["sources"][identity]["linear_trends"])]

        for identity, var_name, trend_days in trend_names:
            var_prefix = "{}_linear_trend".format(var_name)

            self._channels[var_prefix] = int(trend_days)
            self._add_channel_files(
                var_prefix,
                self._config["sources"][identity]["var_files"][var_name])

        # Metadata input variables that don't span time
        meta_names = [(identity, var)
                      for identity in
                      sorted(self._config["sources"].keys())
                      for var in
                      sorted(self._config["sources"][identity]["meta"])]

        for identity, var_name in meta_names:
            self._meta_channels.append(var_name)
            self._channels[var_name] = 1
            self._add_channel_files(
                var_name,
                self._config["sources"][identity]["var_files"][var_name])

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
            # logging.warning("No files in channel list for {}".format(
            # filename))
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
            # This is only for convenience ;)
            "channels":         [
                "{}_{}".format(channel, i)
                for channel, s in
                self._channels.items()
                for i in range(1, s + 1)],
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

    @property
    def config(self):
        return self._config

    @property
    def num_channels(self):
        return sum(self._channels.values())


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
                shuffle(len(train_fns), reshuffle_each_iteration=True).\
                map(decoder, num_parallel_calls=self.batch_size).\
                batch(self.batch_size)

        val_ds = val_ds.\
            shuffle(len(train_fns)).\
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
    x2, y2, _ = other_dl.generate_sample(dt.date(2020, 1, 1))
    print(x2.shape)
    print(y2.shape)

