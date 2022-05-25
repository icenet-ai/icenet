import argparse
import concurrent.futures
import datetime as dt
import json
import logging
import os

from concurrent.futures import ProcessPoolExecutor
from dateutil.relativedelta import relativedelta
from pprint import pformat

import numpy as np
import tensorflow as tf

from icenet2.data.sic.mask import Masks
from icenet2.data.process import IceNetPreProcessor
from icenet2.data.producers import Generator
from icenet2.data.cli import add_date_args, process_date_args


def generate_and_write(path, dates_args):
    with tf.io.TFRecordWriter(path) as writer:
        for date in dates_args.keys():
            x, y, sample_weights = generate_sample(date, *dates_args[date])

            write_tfrecord(writer, x, y, sample_weights)
    return path


# FIXME: I want to get rid of the datetime calculations here, it's prone to
#  error and the ordering is already determined. Move the sample generation
#  to a purely list based affair, guaranteeing order and reducing duplication
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
    # logging.debug("Forecast date {}:\n{}\n{}".format(forecast_date,
    # pformat(var_files), pformat(output_files)))
    
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

    if np.isnan(y).any():
        nans = np.isnan(y)
        logging.debug("Fixing {} nans in output {}".
                      format(np.sum(nans), forecast_date))
        sample_weights[nans] = 0
        y[nans] = 0.

    # Check our output
    m = np.isnan(y)
    if np.sum(sample_weights[m]) > 0:
        np.save("{}".format(forecast_date.strftime("%Y_%m_%d.nan.npy")),
                            np.array([y, sample_weights]))
        raise RuntimeError("Forecast {} is a nanset".format(forecast_date))

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
        
        var_filenames = var_files[var_name].values()

        x[:, :, v1:v2] = \
            np.stack([np.load(filename)
                      if filename
                      else np.zeros(shape)
                      for filename in var_filenames], axis=-1)

        v1 += num_channels

    for var_name in meta_channels:
        if channels[var_name] > 1:
            raise RuntimeError("{} meta variable cannot have more than "
                               "one channel".format(var_name))

        x[:, :, v1] = np.load(var_files[var_name])
        v1 += channels[var_name]

    logging.debug("x shape {}, y shape {}".format(x.shape, y.shape))

    return x, y, sample_weights


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

    def write_dataset_config_only(self):
        splits = ("train", "val", "test")
        counts = {el: 0 for el in splits}

        logging.info("Writing dataset configuration without data generation")

        # FIXME: cloned mechanism from generate() - do we need to treat these as
        #  sets that might have missing data for fringe cases?
        for dataset in splits:
            forecast_dates = sorted(list(set(
                [dt.datetime.strptime(s,
                 IceNetPreProcessor.DATE_FORMAT).date()
                 for identity in
                 self._config["sources"].keys()
                 for s in
                 self._config["sources"][identity]
                 ["dates"][dataset]])))

            logging.info("{} {} dates in total, NOT generating cache "
                         "data.".format(len(forecast_dates), dataset))
            counts[dataset] += len(forecast_dates)

        self._write_dataset_config(counts, network_dataset=False)

    def generate(self, dates_override=None, pickup=False):
        # TODO: for each set, validate every variable has an appropriate file
        #  in the configuration arrays, otherwise drop the forecast date
        splits = ("train", "val", "test")

        if dates_override and type(dates_override) is dict:
            for split in splits:
                assert split in dates_override.keys() \
                       and type(dates_override[split]) is list, \
                       "{} needs to be list in dates_override".format(split)
        else:
            raise RuntimeError("dates_override needs to be a dict if supplied")

        counts = {el: 0 for el in splits}
        futures = []

        def batch(batch_dates, num):
            i = 0
            while i < len(batch_dates):
                yield batch_dates[i:i + num]
                i += num

        # This was a quick and dirty beef-up of the implementation as it's
        # very I/O bursty work. It significantly reduces the overall time
        # taken to produce a full dataset at BAS so we can use this as a
        # paradigm moving forward (with a slightly cleaner implementation)
        with ProcessPoolExecutor(max_workers=self._workers) as executor:
            for dataset in splits:
                batch_number = 0

                forecast_dates = set([dt.datetime.strptime(s,
                                      IceNetPreProcessor.DATE_FORMAT).date()
                                      for identity in
                                      self._config["sources"].keys()
                                      for s in
                                      self._config["sources"][identity]
                                      ["dates"][dataset]])

                if dates_override:
                    logging.info("{} available {} dates".
                                 format(len(forecast_dates), dataset))
                    forecast_dates = forecast_dates.intersection(
                        dates_override[dataset])
                forecast_dates = sorted(list(forecast_dates))

                output_dir = self.get_data_var_folder(dataset)
                tf_path = os.path.join(output_dir, "{:08}.tfrecord")

                logging.info("{} {} dates to process, generating cache "
                             "data.".format(len(forecast_dates), dataset))

                for dates in batch(forecast_dates, self._output_batch_size):
                    args = {}

                    for date in dates:
                        var_files, masks, output_files = \
                            self.get_sample_files(date)

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

                    if not os.path.exists(tf_path.format(batch_number)):
                        futures.append(executor.submit(
                            generate_and_write,
                            tf_path.format(batch_number),
                            args))

                        logging.debug("Submitted {} dates as batch {}".format(
                            len(dates), batch_number))
                    else:
                        logging.warning("Skipping {} on pickup run".
                                        format(tf_path.format(batch_number)))
                    batch_number += 1
                    counts[dataset] += len(dates)

                logging.info("{} tasks submitted".format(len(futures)))

            for fut in concurrent.futures.as_completed(futures):
                path = fut.result()

                logging.info("Finished output {}".format(path))

        self._write_dataset_config(counts)

    def get_sample_files(self, date):
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
                for input_date in set(sorted(input_days))}

        output_files = {
            input_date:
                self._get_var_file("siconca_abs", input_date)
            for input_date in [
                date + relativedelta(days=leadtime_idx)
                for leadtime_idx in
                range(self._n_forecast_days)]
        }

        return var_files, masks, output_files

    def generate_sample(self, date):
        var_files, masks, output_files = self.get_sample_files(date)

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
        filename = os.path.join(str(date.year),
                                "{}.npy".format(date.strftime(date_format))) \
            if not filename_override else filename_override
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

    def _write_dataset_config(self, counts, network_dataset=True):
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
            # FIXME: badly named, should really be dataset_path
            "loader_path":      self._path if network_dataset else False,
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


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", type=str)
    ap.add_argument("hemisphere", choices=("north", "south"))

    ap.add_argument("-v", "--verbose", action="store_true", default=False)

    ap.add_argument("-l", "--lag", type=int, default=2)

    ap.add_argument("-fn", "--forecast-name", dest="forecast_name",
                    default=None, type=str)
    ap.add_argument("-fd", "--forecast-days", dest="forecast_days",
                    default=93, type=int)

    ap.add_argument("-ob", "--output-batch-size", dest="batch_size", type=int,
                    default=8)
    ap.add_argument("-w", "--workers", help="Number of workers to use "
                                            "generating sets",
                    type=int, default=8)

    ap.add_argument("-p", "--pickup", help="Skip existing tfrecords",
                    default=False, action="store_true")
    ap.add_argument("-c", "--cfg-only", help="Do not generate data, "
                                             "only config", default=False,
                    action="store_true", dest="cfg")

    add_date_args(ap)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO
                        if not args.verbose else logging.DEBUG)
    return args


def main():
    args = get_args()
    dates = process_date_args(args)

    dl = IceNetDataLoader("loader.{}.json".format(args.name),
                          args.forecast_name
                          if args.forecast_name else args.name,
                          args.lag,
                          n_forecast_days=args.forecast_days,
                          north=args.hemisphere == "north",
                          south=args.hemisphere == "south",
                          output_batch_size=args.batch_size,
                          generate_workers=args.workers)
    if args.cfg:
        dl.write_dataset_config_only()
    else:
        dl.generate(dates_override=dates
                    if sum([len(v) for v in dates.values()]) > 0 else None,
                    pickup=args.pickup)
