import datetime as dt
import json
import logging
import os
from abc import abstractmethod

from pprint import pformat

import numpy as np

from icenet.data.process import IceNetPreProcessor
from icenet.data.producers import Generator

"""

"""


class IceNetBaseDataLoader(Generator):
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
                 dates_override: object = None,
                 dry: bool = False,
                 generate_workers: int = 8,
                 loss_weight_days: bool = True,
                 n_forecast_days: int = 93,
                 output_batch_size: int = 32,
                 path: str = os.path.join(".", "network_datasets"),
                 pickup: bool = False,
                 var_lag_override: object = None,
                 **kwargs):
        super().__init__(*args,
                         identifier=identifier,
                         path=path,
                         **kwargs)

        self._channels = dict()
        self._channel_files = dict()

        self._configuration_path = configuration_path
        self._dataset_config_path = dataset_config_path
        self._dates_override = dates_override
        self._config = dict()
        self._dry = dry
        self._loss_weight_days = loss_weight_days
        self._meta_channels = []
        self._missing_dates = []
        self._n_forecast_days = n_forecast_days
        self._output_batch_size = output_batch_size
        self._pickup = pickup
        self._trend_steps = dict()
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

    @abstractmethod
    def generate_sample(self,
                        date: object,
                        prediction: bool = False):
        """

        :param date:
        :param prediction:
        :return:
        """
        pass

    def get_sample_files(self) -> object:
        """

        :param date:
        :return:
        """
        # FIXME: is this not just the same as _channel_files now?
        # FIXME: still experimental code, move to multiple implementations
        # FIXME: CLEAN THIS ALL UP ONCE VERIFIED FOR local/shared STORAGE!
        var_files = dict()

        for var_name, num_channels in self._channels.items():
            var_file = self._get_var_file(var_name)

            if not var_file:
                raise RuntimeError("No file returned for {}".format(var_name))

            if var_name not in var_files:
                var_files[var_name] = var_file
            elif var_file != var_files[var_name]:
                raise RuntimeError("Differing files? {} {} vs {}".
                                   format(var_name,
                                          var_file,
                                          var_files[var_name]))

        return var_files

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
                        self._config["sources"][identity]["linear_trend_steps"])
                       for identity in
                       sorted(self._config["sources"].keys())
                       for var in
                       sorted(
                           self._config["sources"][identity]["linear_trends"])]

        for identity, var_name, trend_steps in trend_names:
            var_prefix = "{}_linear_trend".format(var_name)

            self._channels[var_prefix] = len(trend_steps)
            self._trend_steps[var_prefix] = trend_steps
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
    def dates_override(self):
        return self._dates_override

    @property
    def num_channels(self):
        return sum(self._channels.values())

    @property
    def pickup(self):
        return self._pickup

    @property
    def workers(self):
        return self._workers


