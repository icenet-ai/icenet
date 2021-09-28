from abc import abstractmethod

import collections
import datetime as dt
import glob
import logging
import os
import re

import numpy as np

from icenet2.utils import Hemisphere, HemisphereMixin


class DataProducer(HemisphereMixin):

    @abstractmethod
    def __init__(self, *args,
                 identifier=None,
                 dry=False,
                 overwrite=False,
                 north=True,
                 south=False,
                 path=os.path.join(".", "data"),
                 **kwargs):
        self.dry = dry
        self.overwrite = overwrite

        self._identifier = identifier
        self._path = os.path.join(path, identifier)
        self._hemisphere = (Hemisphere.NORTH if north else Hemisphere.NONE) | \
                           (Hemisphere.SOUTH if south else Hemisphere.NONE)

        if os.path.exists(self._path):
            logging.debug("{} already exists".format(self._path))
        else:
            logging.info("Creating path: {}".format(self._path))
            os.makedirs(self._path, exist_ok=True)

        assert self._identifier, "No identifier supplied"
        assert self._hemisphere != Hemisphere.NONE, "No hemispheres selected"
        # NOTE: specific limitation for the DataProducers, they'll only do one
        # hemisphere per instance
        assert self._hemisphere != Hemisphere.BOTH, "Both hemispheres selected"

    @property
    def base_path(self):
        return self._path

    @base_path.setter
    def base_path(self, path):
        self._path = path

    @property
    def identifier(self):
        return self._identifier

    def get_data_var_folder(self, var, hemisphere=None, missing_error=False):
        if not hemisphere:
            # We can make the assumption because this implementation is limited
            # to a single hemisphere
            hemisphere = self.hemisphere_str[0]

        hemi_path = os.path.join(self.base_path, hemisphere)
        if not os.path.exists(hemi_path):
            if not missing_error:
                logging.info("Creating hemisphere path: {}".format(hemi_path))
                os.mkdir(hemi_path)
            else:
                raise OSError("Hemisphere directory {} is missing and this is "
                              "flagged as an error!".format(hemi_path))

        var_path = os.path.join(self.base_path, hemisphere, var)
        if not os.path.exists(var_path):
            if not missing_error:
                logging.info("Creating var path: {}".format(var_path))
                os.mkdir(var_path)
            else:
                raise OSError("Variable directory {} is missing and this is "
                              "flagged as an error!".format(var_path))
        return var_path


class Downloader(DataProducer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def download(self):
        raise NotImplementedError("{}.download is abstract".
                                  format(__class__.__name__))


class Generator(DataProducer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def generate(self):
        raise NotImplementedError("{}.generate is abstract".
                                  format(__class__.__name__))


class Processor(DataProducer):
    def __init__(self,
                 identifier,
                 source_data,
                 *args,
                 file_filters=tuple(),
                 test_dates=tuple(),
                 train_dates=tuple(),
                 val_dates=tuple(),
                 **kwargs):
        super().__init__(*args,
                         identifier=identifier,
                         **kwargs)

        self._file_filters = list(file_filters)
        self._source_data = os.path.join(source_data, identifier)
        self._var_files = dict()
        self._processed_files = dict()

        # TODO: better as a mixin?
        Dates = collections.namedtuple("Dates", ["train", "val", "test"])
        self._dates = Dates(train=list(train_dates),
                            val=list(val_dates),
                            test=list(test_dates))

    def init_source_data(self, lag_days=None, lead_days=None):
        path_to_glob = os.path.join(self._source_data,
                                    *self.hemisphere_str)

        if not os.path.exists(path_to_glob):
            raise OSError("Source data directory {} does not exist".
                          format(path_to_glob))

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
                dates += [dates[0] - dt.timedelta(days=day + 1)
                          for day in range(lag_days)]

            if lead_days:
                logging.info("Including lead of {} days".format(lead_days))
                dates += [dates[-1] + dt.timedelta(days=day + 1)
                          for day in range(lead_days)]

            globstr = "{}/**/*.nc".format(
                path_to_glob)

            dfs = glob.glob(globstr, recursive=True)

            for date in dates:
                match_str = date.strftime("%Y_%m_%d")
                # TODO: test if endswith works better than re.search
                match_dfs = [
                    df for df in dfs
                    if df.endswith("{}.nc".format(match_str))
                ]

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

        # Ensure we're ordered, it has repercussions for xarray
        self._var_files = {
            var: sorted(var_files[var]) for var in sorted(var_files.keys())
        }
        for var in self._var_files.keys():
            logging.info("Got {} files for {}".format(
                len(self._var_files[var]), var))

    @abstractmethod
    def process(self):
        raise NotImplementedError("{}.process is abstract".
                                  format(__class__.__name__))

    def save_processed_file(self, var_name, name, data):
        path = os.path.join(self.get_data_var_folder(var_name), name)
        np.save(path, data)

        if var_name not in self._processed_files.keys():
            self._processed_files[var_name] = list()

        if path not in self._processed_files[var_name]:
            logging.debug("Adding {} file: {}".format(var_name, path))
            self._processed_files[var_name].append(path)
        else:
            logging.warning("{} already exists in {} processed list".
                            format(path, var_name))
        return path

    @property
    def dates(self):
        return self._dates

    @property
    def processed_files(self):
        return self._processed_files

    @property
    def source_data(self):
        return self._source_data
