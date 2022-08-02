import argparse
import datetime as dt
import json
import logging
import os
import sys
import time

from pprint import pprint, pformat

import dask
from dask.distributed import Client, LocalCluster
import numpy as np
import pandas as pd
import tensorflow as tf
import xarray as xr

from icenet2.data.sic.mask import Masks
from icenet2.data.process import IceNetPreProcessor
from icenet2.data.producers import Generator
from icenet2.data.cli import add_date_args, process_date_args

"""

"""


def generate_and_write(path: str,
                       dates_args: object,
                       dry: bool = False):
    """

    :param path:
    :param dates_args:
    :param dry:
    :return:
    """
    count = 0
    times = []

    with tf.io.TFRecordWriter(path) as writer:
        for date in dates_args.keys():
            start = time.time()

            try:
                x, y, sample_weights = generate_sample(date, *dates_args[date])
                if not dry:
                    write_tfrecord(writer, x, y, sample_weights)
                count += 1
            except IceNetDataWarning:
                continue

            end = time.time()
            times.append(end - start)
            logging.debug("Time taken to produce {}: {}".
                          format(date, times[-1]))
    return path, count, times


# FIXME: I want to get rid of the datetime calculations here, it's prone to
#  error and the ordering is already determined. Move the sample generation
#  to a purely list based affair, guaranteeing order and reducing duplication
def generate_sample(forecast_date: object,
                    channels: object,
                    dtype: object,
                    loss_weight_days: bool,
                    meta_channels: object,
                    missing_dates: object,
                    n_forecast_days: int,
                    num_channels: int,
                    shape: object,
                    var_ds: object,
                    trend_ds: object,
                    meta: object,
                    masks: object,
                    data_check: bool = True):
    """

    :param forecast_date:
    :param channels:
    :param dtype:
    :param loss_weight_days:
    :param meta_channels:
    :param missing_dates:
    :param n_forecast_days:
    :param num_channels:
    :param shape:
    :param var_ds:
    :param trend_ds:
    :param meta:
    :param masks:
    :param data_check:
    :return:
    """

    # To become array of shape (*raw_data_shape, n_forecast_days)
    forecast_dts = [forecast_date + dt.timedelta(days=n)
                    for n in range(n_forecast_days)]
    var_ds = var_ds.transpose("xc", "yc", "time")
    sample_output = var_ds.siconca_abs.sel(time=forecast_dts).to_numpy()

    y = np.zeros((*shape, n_forecast_days, 1), dtype=dtype)
    sample_weights = np.zeros((*shape, n_forecast_days, 1), dtype=dtype)

    y[:, :, :, 0] = sample_output

    # Masked recomposition of output
    for leadtime_idx in range(n_forecast_days):
        forecast_day = forecast_date + dt.timedelta(days=leadtime_idx)

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

    # INPUT FEATURES
    x = np.zeros((*shape, num_channels), dtype=dtype)
    v1, v2 = 0, 0

    for var_name, num_channels in channels.items():
        if var_name in meta_channels:
            continue

        v2 += num_channels

        channel_ds = trend_ds if var_name.endswith("linear_trend") else var_ds
        channel_dates = [pd.Timestamp(forecast_date - dt.timedelta(days=n))
                         for n in range(num_channels)]
        channel_data = []
        for cdate in channel_dates:
            try:
                channel_data.append(getattr(channel_ds, var_name).
                                    sel(time=cdate).to_numpy())
            except KeyError:
                channel_data.append(np.zeros(shape))

        x[:, :, v1:v2] = np.transpose(channel_data, [1, 2, 0])
        v1 += num_channels

    for var_name in meta_channels:
        if channels[var_name] > 1:
            raise RuntimeError("{} meta variable cannot have more than "
                               "one channel".format(var_name))

        if var_name in ["sin", "cos"]:
            ref_date = "2012-{}-{}".format(forecast_date.month,
                                           forecast_date.day)
            trig_val = meta[var_name].sel(time=ref_date).to_numpy()
            np.broadcast_to([trig_val], shape)
        else:
            x[:, :, v1] = meta[var_name]
        v1 += channels[var_name]

    y_nans = np.sum(np.isnan(y))
    x_nans = np.sum(np.isnan(x))
    sw_nans = np.sum(np.isnan(sample_weights))

    if y_nans + x_nans + sw_nans > 0:
        logging.warning("NaNs detected {}: input = {}, "
                        "output = {}, weights = {}".
                        format(forecast_date, x_nans, y_nans, sw_nans))

        if data_check and np.sum(sample_weights[np.isnan(y)]) > 0:
            raise IceNetDataWarning("NaNs in output with non-zero weights")

        if data_check and x_nans > 0:
            raise IceNetDataWarning("NaNs detected in data for {}".
                                    format(forecast_date))

    return x, y, sample_weights


def write_tfrecord(writer: object,
                   x: object,
                   y: object,
                   sample_weights: object):
    """

    :param writer:
    :param x:
    :param y:
    :param sample_weights:
    """
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

    :param configuration_path,
    :param identifier,
    :param var_lag,
    :param dataset_config_path: 
    :param generate_workers: 
    :param loss_weight_days: 
    :param n_forecast_days: 
    :param output_batch_size: 
    :param path: 
    :param var_lag_override: 
    """

    def __init__(self,
                 configuration_path: str,
                 identifier: str,
                 var_lag: int,
                 *args,
                 dataset_config_path: str = ".",
                 dry: bool = False,
                 generate_workers: int = 8,
                 loss_weight_days: bool = True,
                 n_forecast_days: int = 93,
                 output_batch_size: int = 32,
                 path: str = os.path.join(".", "network_datasets"),
                 var_lag_override: object = None,
                 **kwargs):
        super().__init__(*args,
                         identifier=identifier,
                         path=path,
                         **kwargs)

        self._channels = dict()
        self._channel_files = dict()
        self._channel_ds = None

        self._configuration_path = configuration_path
        self._dataset_config_path = dataset_config_path
        self._config = dict()
        self._dry = dry
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
        """

        """
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

    def generate(self,
                 dates_override: object = None,
                 pickup: bool = False):
        """

        :param dates_override:
        :param pickup:
        """
        # TODO: for each set, validate every variable has an appropriate file
        #  in the configuration arrays, otherwise drop the forecast date
        splits = ("train", "val", "test")

        if dates_override and type(dates_override) is dict:
            for split in splits:
                assert split in dates_override.keys() \
                       and type(dates_override[split]) is list, \
                       "{} needs to be list in dates_override".format(split)
        elif dates_override:
            raise RuntimeError("dates_override needs to be a dict if supplied")

        counts = {el: 0 for el in splits}
        exec_times = []

        def batch(batch_dates, num):
            i = 0
            while i < len(batch_dates):
                yield batch_dates[i:i + num]
                i += num

        # This was a quick and dirty beef-up of the implementation as it's
        # very I/O bursty work. It significantly reduces the overall time
        # taken to produce a full dataset at BAS so we can use this as a
        # paradigm moving forward (with a slightly cleaner implementation)
        #
        # EDIT: updated for xarray intermediary
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
                samples = 0

                if not pickup or \
                    (pickup and
                     not os.path.exists(tf_path.format(batch_number))):
                    for date in dates:
                        var_ds, trend_ds, meta, masks = \
                            self.get_sample_ds(date)

                        args[date] = [
                            self._channels,
                            self._dtype,
                            self._loss_weight_days,
                            self._meta_channels,
                            self._missing_dates,
                            self._n_forecast_days,
                            self.num_channels,
                            self._shape,
                            var_ds,
                            trend_ds,
                            meta,
                            masks
                        ]

                    tf_data, samples, times = generate_and_write(
                        tf_path.format(batch_number), args, dry=self._dry)
                    if samples > 0:
                        logging.info("Finished output {}".format(tf_data))
                        exec_times += times
                else:
                    logging.warning("Skipping {} on pickup run".
                                    format(tf_path.format(batch_number)))

                batch_number += 1 if samples > 0 else 0
                counts[dataset] += samples

        if len(exec_times) > 0:
            logging.info("Average sample generation time: {}".
                         format(np.average(exec_times)))
        self._write_dataset_config(counts)

    def generate_sample(self, date: object):
        """

        :param date:
        :return:
        """
        var_ds, trend_ds, meta, masks = self.get_sample_ds(date)

        return generate_sample(
            date,
            self._channels,
            self._dtype,
            self._loss_weight_days,
            self._meta_channels,
            self._missing_dates,
            self._n_forecast_days,
            self.num_channels,
            self._shape,
            var_ds,
            trend_ds,
            meta,
            masks,
            data_check=False)

    def get_sample_ds(self, date: object) -> object:
        """

        :param date:
        :return:
        """
        if not self._channel_ds:
            self._channel_ds = dict(
                linear_trends=None,
                masks={},
                meta=dict(sin=None, cos=None, land=None),
                vars=None,
            )

            for var_name in self._meta_channels:
                self._channel_ds["meta"][var_name] = \
                    xr.open_dataarray(self._get_var_file(var_name))

            var_files = []
            trend_files = []

            for var_name, num_channels in self._channels.items():
                if var_name in self._meta_channels:
                    continue

                var_file = self._get_var_file(var_name)

                if var_file:
                    if var_name.endswith("linear_trend"):
                        trend_files.append(var_file)
                    else:
                        var_files.append(var_file)

            kwargs = dict(
                chunks=dict(time=1),
                drop_variables=["month", "plev", "realization"],
                parallel=True,
            )
            self._channel_ds["linear_trends"] = \
                xr.open_mfdataset(trend_files, **kwargs)
            self._channel_ds["vars"] = \
                xr.open_mfdataset(var_files, **kwargs)

        self._channel_ds["masks"] = {}
        for day in range(self._n_forecast_days):
            forecast_day = date + dt.timedelta(days=day)

            self._channel_ds["masks"][forecast_day] = \
                self._masks.get_active_cell_mask(forecast_day.month)

        return self._channel_ds["vars"], \
            self._channel_ds["linear_trends"], \
            self._channel_ds["meta"], \
            self._channel_ds["masks"]

    def _add_channel_files(self,
                           var_name: str,
                           filelist: object):
        """

        :param var_name:
        :param filelist:
        """
        if var_name in self._channel_files:
            logging.warning("{} already has files, but more found, "
                            "this could be an unintentional merge of "
                            "sources".format(var_name))
        else:
            self._channel_files[var_name] = []

        logging.debug("Adding {} to {} channel".format(len(filelist), var_name))
        self._channel_files[var_name] += filelist

    def _construct_channels(self):
        """

        """
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
                [el for el in
                 self._config["sources"][identity]["var_files"][var_name]
                 if var_prefix in os.path.split(el)[1]])

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
            filelist = [el for el in
                        self._config["sources"][identity]["var_files"][var_name]
                        if "linear_trend" in os.path.split(el)[1]]

            self._add_channel_files(var_prefix, filelist)

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

    def _get_var_file(self, var_name: str):
        """

        :param var_name:
        :return:
        """

        filename = "{}.nc".format(var_name)
        files = self._channel_files[var_name]

        if len(self._channel_files[var_name]) > 1:
            logging.warning("Multiple files found for {}, only returning {}".
                            format(filename, files[0]))
        elif not len(files):
            logging.warning("No files in channel list for {}".format(filename))
            return None
        return files[0]

    def _load_configuration(self, path: str):
        """

        :param path:
        """
        if os.path.exists(path):
            logging.info("Loading configuration {}".format(path))

            with open(path, "r") as fh:
                obj = json.load(fh)

                self._config.update(obj)
        else:
            raise OSError("{} not found".format(path))

    def _write_dataset_config(self,
                              counts: object,
                              network_dataset: bool = True):
        """

        :param counts:
        :param network_dataset:
        :return:
        """
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

    ap.add_argument("-dp", "--dask-port", type=int, default=8888)
    ap.add_argument("-w", "--workers", help="Number of workers to use "
                                            "generating sets",
                    type=int, default=2)

    ap.add_argument("-c", "--cfg-only", help="Do not generate data, "
                                             "only config", default=False,
                    action="store_true", dest="cfg")
    ap.add_argument("-d", "--dry",
                    help="Don't output files, just generate data",
                    default=False, action="store_true")
    ap.add_argument("-p", "--pickup", help="Skip existing tfrecords",
                    default=False, action="store_true")

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
                          dry=args.dry,
                          n_forecast_days=args.forecast_days,
                          north=args.hemisphere == "north",
                          south=args.hemisphere == "south",
                          output_batch_size=args.batch_size,
                          generate_workers=args.workers)
    if args.cfg:
        dl.write_dataset_config_only()
    else:
        dashboard = "localhost:{}".format(args.dask_port)
        cluster = LocalCluster(
            n_workers=args.workers,
            scheduler_port=0,
            dashboard_address=dashboard,
        )
        logging.info("Dashboard at {}".format(dashboard))

        with Client(cluster) as client:
            logging.info("Using dask client {}".format(client))
            dl.generate(dates_override=dates
                        if sum([len(v) for v in dates.values()]) > 0 else None,
                        pickup=args.pickup)


class IceNetDataWarning(RuntimeWarning):
    pass
