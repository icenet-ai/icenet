import collections
import glob
import json
import logging
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

from icenet2.data.producers import Processor
from icenet2.data.sic.mask import Masks


class IceNetPreProcessor(Processor):
    def __init__(self,
                 abs_vars,
                 anom_vars,
                 name,
                 test_dates,
                 train_dates,
                 val_dates,
                 *args,
                 data_shape=(432, 432),
                 default_lag=14,
                 dtype=np.float32,
                 exclude_vars=(),
                 file_filters=tuple(["_latlon_"]),
                 identifier=None,
                 include_circday=True,
                 include_land=True,
                 lag_override=None,
                 linear_trend_years=None,
                 minmax=True,
                 no_normalise=tuple(["siconca"]),
                 path=os.path.join(".", "network_datasets"),
                 source_data=os.path.join(".", "data"),
                 **kwargs):
        super().__init__(*args,
                         identifier=identifier,
                         path=os.path.join(path, name),
                         **kwargs)

        self._abs_vars = abs_vars
        self._anom_vars = anom_vars

        self._name = name
        self._source_data = os.path.join(source_data, identifier)

        self._data_shape = data_shape
        #self._default_lag = default_lag
        self._dtype = dtype
        self._exclude_vars = exclude_vars
        self._file_filters = file_filters
        self._include_circday = include_circday
        self._include_land = include_land
        #self._lag_override = lag_override
        #self._linear_trend_years = linear_trend_years
        self._no_normalise = no_normalise
        self._normalise = self._normalise_array_mean \
            if not minmax else self._normalise_array_scaling
        self._var_files = dict()

        Dates = collections.namedtuple("Dates", ["train", "val", "test"])
        self._dates = Dates(train=train_dates, val=val_dates, test=test_dates)

    def init_source_data(self):
        path_to_glob = os.path.join(self._source_data, *self.hemisphere_str)

        for date_category in ["train", "val", "test"]:
            dates = getattr(self._dates, date_category)

            if dates:
                logging.info("Processing {} dates for {} category".
                             format(len(dates), date_category))
            else:
                logging.info("No {} dates for this processor".
                             format(date_category))
                continue

            for date in dates:
                globstr = "{}/**/*_{}.nc".format(
                    path_to_glob,
                    date.strftime("%Y%m%d"))

                for df in glob.glob(globstr, recursive=True):
                    if any([flt in os.path.split(df)[1]
                            for flt in self._file_filters]):
                        continue

                    var = os.path.split(df)[0].split(os.sep)[-1]
                    if var not in self._var_files.keys():
                        self._var_files[var] = list()
                    self._var_files[var].append(df)

    def process(self):
        for var_name in self._abs_vars + self._anom_vars:
            if var_name not in self._var_files.keys():
                logging.warning("{} does not exist".format(var_name))
                continue
            self._save_variable(var_name)

        self._save_circday()
        self._save_land()

    def pre_normalisation(self, var_name, da):
        logging.debug("No pre normalisation implemented for {}".
                      format(var_name))
        return da

    def post_normalisation(self, var_name, da):
        logging.debug("No post normalisation implemented for {}".
                      format(var_name))
        return da

    def _save_variable(self, var_name):
        da = self.open_dataarray_from_files(var_name)

        if var_name in self._anom_vars:
            clim_path = os.path.join(self.get_data_var_folder("params"),
                                     "climatology.{}".format(var_name))

            if not os.path.exists(clim_path):
                if self._dates.train:
                    climatology = da.sel(time=self._dates.train).\
                        groupby('time.dayofyear', restore_coord_dims=True).\
                        mean()

                    climatology.to_netcdf(clim_path)
                else:
                    raise RuntimeError("{} does not exist and no training "
                                       "data is supplied".format(clim_path))
            else:
                climatology = xr.open_dataarray(clim_path)

            da = da.groupby('time.dayofyear') - climatology

        da.data = np.asarray(da.data, dtype=self._dtype)

        da = self.pre_normalisation(var_name, da)

        if var_name in self._no_normalise:
            logging.info("No normalisation for {}".format(var_name))
        else:
            logging.info("Normalising {}".format(var_name))
            da = self._normalise(var_name, da)

        da.data[np.isnan(da.data)] = 0.

        da = self.post_normalisation(var_name, da)

        self._save_output(da, var_name)

    def _save_land(self):
        land_mask = Masks(north=self.north, south=self.south).get_land_mask()
        land_map = np.ones(self._data_shape, dtype=self._dtype)
        land_map[~land_mask] = -1.

        np.save(os.path.join(
            self.get_data_var_folder("meta"), 'land.npy'), land_map)

    # FIXME: will there be inaccuracies due to leap year offsets?
    def _save_circday(self):
        for date in pd.date_range(start='2012-1-1', end='2012-12-31'):
            if self.north:
                circday = date.dayofyear
            else:
                circday = date.dayofyear + 365.25 / 2

            cos_day = np.cos(2 * np.pi * circday / 366, dtype=self._dtype)
            sin_day = np.sin(2 * np.pi * circday / 366, dtype=self._dtype)

            np.save(os.path.join(self.get_data_var_folder("meta"),
                                 date.strftime('cos_%j.npy')), cos_day)
            np.save(os.path.join(self.get_data_var_folder("meta"),
                                 date.strftime('sin_%j.npy')), sin_day)

    def _save_output(self, da, var_name):

        """
        Saves an xarray DataArray as daily averaged .npy files using the
        self.paths data structure.
        Parameters:
        da (xarray.DataArray): The DataArray to save.
        dataset_type (str): Either 'obs' or 'transfer' (for CMIP6 data) - the type
        of dataset being saved.
        varname (str): Variable name being saved.
        data_format (str): Either 'abs' or 'anom' - the format of the data
        being saved.
        """

        for date in da.time.values:
            slice = da.sel(time=date).data
            date = pd.Timestamp(date)
            year_str = '{:04d}'.format(date.year)
            month_str = '{:02d}'.format(date.month)
            fname = '{}_{}.npy'.format(year_str, month_str)

            np.save(
                os.path.join(self.get_data_var_folder(var_name), fname), slice)

    def open_dataarray_from_files(self, var_name):

        """
        Open the yearly xarray files, accounting for some ERA5 variables that have
        erroneous 'unknown' NetCDF variable names which prevents concatentation.
        """

        logging.info("Opening files for {}".format(var_name))
        ds_list = [xr.open_dataset(path) for path in self._var_files[var_name]]

        # Set of variables names
        # var_set = set([next(iter(
        #    xr.open_dataset(path).data_vars.values())).name
        #               for path in self._var_files[var_name]])

        # For processing one file, we're going to assume a single non-lambert
        # variable exists at the start and rename all of them
        var_names = []
        for ds in ds_list:
            var_names += [name for name in list(ds.data_vars.keys())
                          if not name.startswith("lambert_")]

        var_names = set(var_names)
        logging.debug("Files have var names {} which will be renamed to {}".
                      format(", ".join(var_names), var_name))

        for i, ds in enumerate(ds_list):
            ds = ds.rename({k: var_name for k in var_names})
            ds_list[i] = ds

        ds = xr.combine_nested(ds_list,
                               concat_dim='time')

        da = getattr(ds, var_name)
        return da

    @staticmethod
    def mean_and_std(array):

        """
        Return the mean and standard deviation of an array-like object (intended
        use case is for normalising a raw satellite data array based on a list
        of samples used for training).
        """

        mean = np.nanmean(array)
        std = np.nanstd(array)

        logging.info("Mean: {:.3f}, std: {:.3f}".
                     format(mean.item(), std.item()))

        return mean, std

    def _normalise_array_mean(self, var_name, da):

        """
        Using the *training* data only, compute the mean and
        standard deviation of the input raw satellite DataArray (`da`)
        and return a normalised version. If minmax=True,
        instead normalise to lie between min and max of the elements of `array`.

        If min, max, mean, or std are given values other than None,
        those values are used rather than being computed from the training
        months.

        Returns:
        new_da (xarray.DataArray): Normalised array.
        mean, std (float): Pre-computed mean and standard deviation for the
        normalisation.
        min, max (float): Pre-computed min and max for the normalisation.
        """

        mean_path = os.path.join(
            self.get_data_var_folder("normalisation.mean"),
            "{}".format(var_name))

        if os.path.exists(mean_path):
            mean, std = tuple([self._dtype(el) for el in
                               open(mean_path, "r").read().split(",")])
        elif self._dates.train:
            training_samples = da.sel(time=self._dates.train).data
            training_samples = training_samples.ravel()

            mean, std = IceNetPreProcessor.mean_and_std(training_samples)
        else:
            raise RuntimeError("Either a normalisation file or training data "
                               "must be supplied")

        new_da = (da - mean) / std
        open(mean_path, "w").write(",".join([str(f) for f in
                                             [mean, std]]))
        return new_da

    def _normalise_array_scaling(self, var_name, da):
        scale_path = os.path.join(
            self.get_data_var_folder("normalisation.scale"),
            "{}".format(var_name))

        if os.path.exists(scale_path):
            minimum, maximum = tuple([self._dtype(el) for el in
                               open(scale_path, "r").read().split(",")])
        elif self._dates.train:
            training_samples = da.sel(time=self._dates.train).data
            training_samples = training_samples.ravel()

            minimum = np.nanmin(training_samples).astype(self._dtype)
            maximum = np.nanmax(training_samples).astype(self._dtype)
        else:
            raise RuntimeError("Either a normalisation file or training data "
                               "must be supplied")

        new_da = (da - minimum) / (maximum - minimum)
        open(scale_path, "w").write(",".join([str(f) for f in
                                              [minimum, maximum]]))
        return new_da
