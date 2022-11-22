from abc import abstractmethod, ABCMeta
from pprint import pformat

import collections
import datetime as dt
import glob
import logging
import os
import re

import pandas as pd

from icenet.utils import Hemisphere, HemisphereMixin


class DataCollection(HemisphereMixin, metaclass=ABCMeta):
    """

    :param identifier:
    :param north:
    :param south:
    :param path:
    """

    @abstractmethod
    def __init__(self, *args,
                 identifier: object = None,
                 north: bool = True,
                 south: bool = False,
                 path: str = os.path.join(".", "data"),
                 **kwargs):
        self._identifier = identifier
        self._path = os.path.join(path, identifier)
        self._hemisphere = (Hemisphere.NORTH if north else Hemisphere.NONE) | \
                           (Hemisphere.SOUTH if south else Hemisphere.NONE)

        assert self._identifier, "No identifier supplied"
        assert self._hemisphere != Hemisphere.NONE, "No hemispheres selected"

    @property
    def base_path(self):
        return self._path

    @base_path.setter
    def base_path(self, path):
        self._path = path

    @property
    def identifier(self):
        return self._identifier


class DataProducer(DataCollection):
    """
    :param dry:
    :param overwrite:
    """
    def __init__(self, *args,
                 dry: bool = False,
                 overwrite: bool = False,
                 **kwargs):
        super(DataProducer, self).__init__(*args, **kwargs)

        self.dry = dry
        self.overwrite = overwrite

        if os.path.exists(self._path):
            logging.debug("{} already exists".format(self._path))
        else:
            if not os.path.islink(self._path):
                logging.info("Creating path: {}".format(self._path))
                os.makedirs(self._path, exist_ok=True)
            else:
                logging.info("Skipping creation for symlink: {}".format(
                    self._path))

        # NOTE: specific limitation for the DataProducers, they'll only do one
        # hemisphere per instance
        assert self._hemisphere != Hemisphere.BOTH, "Both hemispheres selected"

    def get_data_var_folder(self,
                            var: str,
                            append: object = None,
                            hemisphere: object = None,
                            missing_error: bool = False) -> str:
        """

        :param var:
        :param append:
        :param hemisphere:
        :param missing_error:
        :return:
        """
        if not append:
            append = []

        if not hemisphere:
            # We can make the assumption because this implementation is limited
            # to a single hemisphere
            hemisphere = self.hemisphere_str[0]

        data_var_path = os.path.join(
            self.base_path, *[hemisphere, var, *append]
        )

        if not os.path.exists(data_var_path):
            if not missing_error:
                os.makedirs(data_var_path, exist_ok=True)
            else:
                raise OSError("Directory {} is missing and this is "
                              "flagged as an error!".format(data_var_path))

        return data_var_path


class Downloader(DataProducer):
    """

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def download(self):
        """

        """
        raise NotImplementedError("{}.download is abstract".
                                  format(__class__.__name__))


class Generator(DataProducer):
    """
    
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def generate(self):
        """

        """
        raise NotImplementedError("{}.generate is abstract".
                                  format(__class__.__name__))


class Processor(DataProducer):
    """
    
    :param identifier: 
    :param source_data: 
    :param *args: 
    :param file_filters: 
    :param test_dates: 
    :param train_dates: 
    :param val_dates:
    """
    def __init__(self,
                 identifier: str,
                 source_data: object,
                 *args,
                 file_filters: object = (),
                 lead_time: int = 93,
                 test_dates: object = (),
                 train_dates: object = (),
                 val_dates: object = (),
                 **kwargs):
        super().__init__(*args,
                         identifier=identifier,
                         **kwargs)

        self._file_filters = list(file_filters)
        self._lead_time = lead_time
        self._source_data = os.path.join(source_data,
                                         identifier,
                                         self.hemisphere_str[0])
        self._var_files = dict()
        self._processed_files = dict()

        # TODO: better as a mixin?
        Dates = collections.namedtuple("Dates", ["train", "val", "test"])
        self._dates = Dates(train=list(train_dates),
                            val=list(val_dates),
                            test=list(test_dates))

    def init_source_data(self,
                         lag_days: object = None):
        """

        :param lag_days:
        """

        if not os.path.exists(self.source_data):
            raise OSError("Source data directory {} does not exist".
                          format(self.source_data))

        var_files = {}

        for date_category in ["train", "val", "test"]:
            dates = sorted(getattr(self._dates, date_category))

            if dates:
                logging.info("Processing {} dates for {} category".
                             format(len(dates), date_category))
            else:
                logging.info("No {} dates for this processor".
                             format(date_category))
                continue

            # TODO: ProcessPool for this (avoid the GIL for globbing)
            # FIXME: needs to deal with a lack of continuity in the date ranges
            if lag_days:
                logging.info("Including lag of {} days".format(lag_days))

                additional_lag_dates = []

                for date in dates:
                    for day in range(lag_days):
                        lag_date = date - dt.timedelta(days=day + 1)
                        if lag_date not in dates:
                            additional_lag_dates.append(lag_date)
                dates += list(set(additional_lag_dates))

            # FIXME: this is conveniently supplied for siconca_abs on
            #  training with OSISAF data, but are we exploiting the
            #  convenient usage of this data for linear trends?
            if self._lead_time:
                logging.info("Including lead of {} days".format(self._lead_time))

                additional_lead_dates = []

                for date in dates:
                    for day in range(self._lead_time):
                        lead_day = date + dt.timedelta(days=day + 1)
                        if lead_day not in dates:
                            additional_lead_dates.append(lead_day)
                dates += list(set(additional_lead_dates))

            globstr = "{}/**/[12]*.nc".format(self.source_data)

            logging.debug("Globbing {} from {}".format(date_category, globstr))
            dfs = glob.glob(globstr, recursive=True)
            logging.debug("Globbed {} files".format(len(dfs)))

            # FIXME: using hyphens broadly no?
            data_dates = [df.split(os.sep)[-1][:-3].replace("_", "-")
                          for df in dfs]
            dt_series = pd.Series(dfs, index=data_dates)

            logging.debug("Create structure of {} files".format(len(dt_series)))

            # Ensure we're ordered, it has repercussions for xarray
            for date in sorted(dates):
                try:
                    match_dfs = dt_series[date.strftime("%Y")]

                    if type(match_dfs) == str:
                        match_dfs = [match_dfs]
                except KeyError:
                    logging.info("No data found for {}, outside data boundary "
                                 "perhaps?".format(date.strftime("%Y-%m-%d")))
                    match_dfs = []

                for df in match_dfs:
                    if any([flt in os.path.split(df)[1]
                            for flt in self._file_filters]):
                        continue

                    path_comps = str(os.path.split(df)[0]).split(os.sep)
                    var = path_comps[-1]

                    # The year is in the path, fall back one further
                    if re.match(r'^\d{4}$', var):
                        var = path_comps[-2]

                    if var not in var_files.keys():
                        var_files[var] = list()

                    if df not in var_files[var]:
                        var_files[var].append(df)

        # TODO: allow option to ditch dates from train/val/test for missing
        #  var files
        self._var_files = {
            var: var_files[var] for var in sorted(var_files.keys())
        }
        for var in self._var_files.keys():
            logging.info("Got {} files for {}".format(
                len(self._var_files[var]), var))

    @abstractmethod
    def process(self):
        """

        """
        raise NotImplementedError("{}.process is abstract".
                                  format(__class__.__name__))

    def save_processed_file(self,
                            var_name: str,
                            name: str,
                            data: object, **kwargs):
        """

        :param var_name:
        :param name:
        :param data:
        :param kwargs:
        :return:
        """
        file_path = os.path.join(
            self.get_data_var_folder(var_name, **kwargs), name)
        data.to_netcdf(file_path)

        if var_name not in self._processed_files.keys():
            self._processed_files[var_name] = list()

        if file_path not in self._processed_files[var_name]:
            logging.debug("Adding {} file: {}".format(var_name, file_path))
            self._processed_files[var_name].append(file_path)
        else:
            logging.warning("{} already exists in {} processed list".
                            format(file_path, var_name))
        return file_path

    @property
    def dates(self):
        return self._dates

    @property
    def lead_time(self):
        return self._lead_time

    @property
    def processed_files(self):
        return self._processed_files

    @property
    def source_data(self):
        return self._source_data
