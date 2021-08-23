import collections
import datetime as dt
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
from icenet2.model.models import linear_trend_forecast

"""

TODO: missing dates

"""


class IceNetPreProcessor(Processor):
    DATE_FORMAT = "%Y-%m-%d"

    def __init__(self,
                 abs_vars,
                 anom_vars,
                 name,
                 train_dates,
                 val_dates,
                 test_dates,
                 *args,
                 data_shape=(432, 432),
                 dtype=np.float32,
                 exclude_vars=(),
                 file_filters=tuple(["_latlon_"]),
                 identifier=None,
                 include_circday=True,
                 include_land=True,
                 linear_trends=tuple(["siconca"]),
                 linear_trend_days=7,
                 meta_vars=tuple(),
                 missing_dates=tuple(),
                 minmax=True,
                 no_normalise=tuple(["siconca"]),
                 path=os.path.join(".", "processed"),
                 source_data=os.path.join(".", "data"),
                 update_loader=True,
                 **kwargs):
        super().__init__(identifier,
                         source_data,
                         *args,
                         file_filters=file_filters,
                         path=os.path.join(path, name),
                         train_dates=train_dates,
                         val_dates=val_dates,
                         test_dates=test_dates,
                         **kwargs)

        self._abs_vars = abs_vars
        self._anom_vars = anom_vars
        self._meta_vars = list(meta_vars)

        self._name = name

        self._data_shape = data_shape
        self._dtype = dtype
        self._exclude_vars = exclude_vars
        self._include_circday = include_circday
        self._include_land = include_land
        self._linear_trends = linear_trends
        self._linear_trend_days = linear_trend_days
        self._missing_dates = list(missing_dates)
        self._no_normalise = no_normalise
        self._normalise = self._normalise_array_mean \
            if not minmax else self._normalise_array_scaling
        self._update_loader = os.path.join(".",
                                           "loader.{}.json".format(name)) \
            if update_loader else None

    def process(self):
        for var_name in self._abs_vars + self._anom_vars:
            if var_name not in self._var_files.keys():
                logging.warning("{} does not exist".format(var_name))
                continue
            self._save_variable(var_name)

        if self._include_circday:
            self._save_circday()

        if self._include_land:
            self._save_land()

        if self._update_loader:
            self.update_loader_config()

    def pre_normalisation(self, var_name, da):
        logging.debug("No pre normalisation implemented for {}".
                      format(var_name))
        return da

    def post_normalisation(self, var_name, da):
        logging.debug("No post normalisation implemented for {}".
                      format(var_name))
        return da

    # TODO: update this to store parameters, if appropriate
    def update_loader_config(self):
        def _serialize(x):
            if x is dt.date:
                return x.strftime(IceNetPreProcessor.DATE_FORMAT)
            return str(x)

        source = {
            "name":             self._name,
            "implementation":   self.__class__.__name__,
            "anom":             self._anom_vars,
            "abs":              self._abs_vars,
            "dates":            self._dates._asdict(),
            "linear_trends":    self._linear_trends,
            "linear_trend_days": self._linear_trend_days,
            "meta":             self._meta_vars,
            # TODO: intention should perhaps be to strip these from
            #  other date sets, this is just an indicative placeholder
            #  for the mo
            "var_files":        self._processed_files,
        }

        configuration = {
            "sources": {}
        }

        if os.path.exists(self._update_loader):
            logging.info("Loading configuration {}".format(self._update_loader))
            with open(self._update_loader, "r") as fh:
                obj = json.load(fh)
                configuration.update(obj)

        configuration["sources"][self.identifier] = source

        # Ideally should always be in together
        if "dtype" in configuration:
            assert configuration["dtype"] == self._dtype.__name__

        if "shape" in configuration:
            assert configuration["shape"] == list(self._data_shape)

        configuration["dtype"] = self._dtype.__name__
        configuration["shape"] = list(self._data_shape)

        if "missing_dates" not in configuration:
            configuration["missing_dates"] = []
        configuration["missing_dates"] += self._missing_dates

        logging.info("Writing configuration to {}".format(self._update_loader))

        with open(self._update_loader, "w") as fh:
            json.dump(configuration, fh, indent=4, default=_serialize)

    def _save_variable(self, var_name):
        da = self._open_dataarray_from_files(var_name)

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

        # TODO: Check, but I believe this should be before norm given source
        #  data usage
        if var_name in self._linear_trends:
            da = self._build_linear_trend_da(da)

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

        if "land" not in self._meta_vars:
            self._meta_vars.append("land")

        land_path = self.save_processed_file("land", "land.npy", land_map)

        for date in pd.date_range(start='2012-1-1', end='2012-12-31'):
            link_path = os.path.join(os.path.dirname(land_path),
                                     date.strftime('%j.npy'))

            if not os.path.islink(link_path):
                os.symlink(os.path.basename(land_path), link_path)
            self.processed_files["land"].append(link_path)

    def _save_circday(self):
        for date in pd.date_range(start='2012-1-1', end='2012-12-31'):
            if self.north:
                circday = date.dayofyear
            else:
                circday = date.dayofyear + 365.25 / 2

            cos_day = np.cos(2 * np.pi * circday / 366, dtype=self._dtype)
            sin_day = np.sin(2 * np.pi * circday / 366, dtype=self._dtype)

            self.save_processed_file("cos",
                                     date.strftime('%j.npy'),
                                     cos_day)
            self.save_processed_file("sin",
                                     date.strftime('%j.npy'),
                                     sin_day)

        for var_name in ["sin", "cos"]:
            if var_name not in self._meta_vars:
                self._meta_vars.append(var_name)

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
            fname = '{:04d}_{:02d}_{:02d}.npy'.\
                format(date.year, date.month, date.day)

            self.save_processed_file(var_name, fname, slice)

    def _open_dataarray_from_files(self, var_name):

        """
        Open the yearly xarray files, accounting for some ERA5 variables that
        have erroneous 'unknown' NetCDF variable names which prevents
        concatentation.
        """

        logging.info("Opening files for {}".format(var_name))
        ds = xr.open_mfdataset(self._var_files[var_name], concat_dim="time")

        # Set of variables names
        # var_set = set([next(iter(
        #    xr.open_dataset(path).data_vars.values())).name
        #               for path in self._var_files[var_name]])

        # For processing one file, we're going to assume a single non-lambert
        # variable exists at the start and rename all of them
        var_names = [name for name in list(ds.data_vars.keys())
                     if not name.startswith("lambert_")]

        var_names = set(var_names)
        logging.debug("Files have var names {} which will be renamed to {}".
                      format(", ".join(var_names), var_name))

        ds = ds.rename({k: var_name for k in var_names})
        da = getattr(ds, var_name)

        all_dates = self.dates.train + self.dates.val + self.dates.test
        da_dates = [pd.to_datetime(d).date() for d in da.time.values]
        search = [el for el in all_dates if el in da_dates]

        logging.info("Time dimension is {} units long".format(len(da.time)))
        da = da.sel(time=search)
        logging.info("Filtered to {} units long based on configuration "
                     "requirements".format(len(da.time)))

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

    def _build_linear_trend_da(self, input_da):
        """
        Construct a DataArray `linea_trend_da` containing the linear trend SIC
        forecasts based on the input DataArray `input_da`.
        `linear_trend_da` will be saved in monthly averages using
        the `save_xarray_in_monthly_averages` method.
        Parameters:
        `input_da` (xarray.DataArray): Input DataArray to produce linear SIC
        forecasts for.
        `dataset` (str): 'obs' or 'cmip6' (dictates whether to skip missing
        observational months in the linear trend extrapolation)
        Returns:
        `linear_trend_da` (xarray.DataArray): DataArray whose time slices
        correspond to the linear trend SIC projection for that month.
        """

        linear_trend_da = input_da.copy(data=np.zeros(input_da.shape,
                                                      dtype=self._dtype))

        # FIXME: change the trend dating to dailies. This is not going to
        #  work for simple preprocessing of small datasets
        forecast_dates = [pd.Timestamp(date) for date in
                          input_da.time.values][self._linear_trend_days:]
        last_period = forecast_dates[-self._linear_trend_days:]

        forecast_dates.extend([
            date + pd.DateOffset(days=self._linear_trend_days)
            for date in last_period])

        linear_trend_da = linear_trend_da.assign_coords(
            {'time': forecast_dates})
        land_mask = Masks(north=self.north, south=self.south).get_land_mask()

        for forecast_date in forecast_dates:
            linear_trend_da.loc[dict(time=forecast_date)] = \
                linear_trend_forecast(
                    forecast_date, input_da, land_mask,
                    self._linear_trend_days,
                    missing_dates=(),
                    shape=self._data_shape)[0]

        return linear_trend_da
