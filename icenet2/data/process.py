import datetime as dt
import json
import logging
import os
import pickle

import dask
import numpy as np
import pandas as pd
import xarray as xr

from icenet2.data.producers import Processor
from icenet2.data.sic.mask import Masks
from icenet2.model.models import linear_trend_forecast


class IceNetPreProcessor(Processor):
    DATE_FORMAT = "%Y_%m_%d"

    def __init__(self,
                 abs_vars,
                 anom_vars,
                 name,
                 # FIXME: the preprocessors don't need to have the concept of
                 #  train, test, val: they only need to output daily files
                 #  that either are, or are not, part of normalisation /
                 #  climatology calculations. Not a problem, just fix
                 train_dates,
                 val_dates,
                 test_dates,
                 *args,
                 data_shape=(432, 432),
                 dtype=np.float32,
                 exclude_vars=(),
                 file_filters=tuple(["latlon_"]),
                 identifier=None,
                 linear_trends=tuple(["siconca"]),
                 linear_trend_days=7,
                 meta_vars=tuple(),
                 missing_dates=tuple(),
                 minmax=True,
                 no_normalise=tuple(["siconca"]),
                 path=os.path.join(".", "processed"),
                 ref_procdir=None,
                 source_data=os.path.join(".", "data"),
                 update_key=None,
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
        # TODO: Ugh, this should not be here any longer
        self._meta_vars = list(meta_vars)

        self._name = name

        self._data_shape = data_shape
        self._dtype = dtype
        self._exclude_vars = exclude_vars
        self._linear_trends = linear_trends
        self._linear_trend_days = linear_trend_days
        self._missing_dates = list(missing_dates)
        self._no_normalise = no_normalise
        self._normalise = self._normalise_array_mean \
            if not minmax else self._normalise_array_scaling
        self._refdir = ref_procdir
        self._update_key = self.identifier if not update_key else update_key
        self._update_loader = os.path.join(".",
                                           "loader.{}.json".format(name)) \
            if update_loader else None

    def process(self):
        for var_name in self._abs_vars + self._anom_vars:
            if var_name not in self._var_files.keys():
                logging.warning("{} does not exist".format(var_name))
                continue
            self._save_variable(var_name)

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

        # We have to be explicit with "dates" as the properties will not be
        # caught by _serialize
        source = {
            "name":             self._name,
            "implementation":   self.__class__.__name__,
            "anom":             self._anom_vars,
            "abs":              self._abs_vars,
            "dates":            {
                "train":        [d.strftime(IceNetPreProcessor.DATE_FORMAT)
                                 for d in self._dates.train],
                "val":          [d.strftime(IceNetPreProcessor.DATE_FORMAT)
                                 for d in self._dates.val],
                "test":         [d.strftime(IceNetPreProcessor.DATE_FORMAT)
                                 for d in self._dates.test],
            },
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

        configuration["sources"][self._update_key] = source

        # Ideally should always be in together
        if "dtype" in configuration:
            assert configuration["dtype"] == self._dtype.__name__

        if "shape" in configuration:
            assert configuration["shape"] == list(self._data_shape)

        configuration["dtype"] = self._dtype.__name__
        configuration["shape"] = list(self._data_shape)

        if "missing_dates" not in configuration:
            configuration["missing_dates"] = []

        # Conversion required one way or another, so perhaps more efficient
        # than a union
        for d in sorted(self._missing_dates):
            date_str = d.strftime(IceNetPreProcessor.DATE_FORMAT)

            if date_str not in configuration["missing_dates"]:
                configuration["missing_dates"].append(date_str)

        logging.info("Writing configuration to {}".format(self._update_loader))

        with open(self._update_loader, "w") as fh:
            json.dump(configuration, fh, indent=4, default=_serialize)

    def _save_variable(self, var_name):
        with dask.config.set(**{'array.slicing.split_large_chunks': True}):
            da = self._open_dataarray_from_files(var_name)

            # FIXME: we should ideally store train dates against the
            #  normalisation and climatology, to ensure recalculation on
            #  reprocess. All this need be is in the path, to be honest

            if var_name in self._anom_vars:
                if self._refdir:
                    logging.info("Loading climatology from alternate "
                                 "directory: {}".format(self._refdir))
                    clim_path = os.path.join(self._refdir,
                                             "params",
                                             "climatology.{}".format(var_name))
                else:
                    clim_path = os.path.join(self.get_data_var_folder("params"),
                                             "climatology.{}".format(var_name))

                if not os.path.exists(clim_path):
                    logging.info("Generating climatology {}".format(clim_path))

                    if self._dates.train:
                        climatology = da.sel(time=self._dates.train).\
                            groupby('time.month', restore_coord_dims=True).\
                            mean()

                        climatology.to_netcdf(clim_path)
                    else:
                        raise RuntimeError("{} does not exist and no "
                                           "training data is supplied".
                                           format(clim_path))
                else:
                    logging.info("Reusing climatology {}".format(clim_path))
                    climatology = xr.open_dataarray(clim_path)

                if not set(da.groupby("time.month").all().month.values).\
                        issubset(set(climatology.month.values)):
                    logging.warning(
                        "We don't have a full climatology ({}) "
                        "compared with data ({})".format(
                            ",".join([str(i)
                                      for i in climatology.month.values]),
                            ",".join([str(i)
                                      for i in da.groupby("time.month").
                                     all().month.values])))
                    da = da - climatology.mean()
                else:
                    da = da.groupby("time.month") - climatology

            da.data = np.asarray(da.data, dtype=self._dtype)

            da = self.pre_normalisation(var_name, da)

            if var_name in self._linear_trends:
                da = self._build_linear_trend_da(da, var_name)

            if var_name in self._no_normalise:
                logging.info("No normalisation for {}".format(var_name))
            else:
                logging.info("Normalising {}".format(var_name))
                da = self._normalise(var_name, da)

            da.data[np.isnan(da.data)] = 0.

            da = self.post_normalisation(var_name, da)

            self._save_output(da, var_name)

    def _save_output(self, da, var_name):

        """
        Saves an xarray DataArray as daily averaged .npy files using the
        self.paths data structure.
        Parameters:
        da (xarray.DataArray): The DataArray to save.
        dataset_type (str): Either 'obs' or 'transfer' (for CMIP6 data) - the
        type of dataset being saved.
        varname (str): Variable name being saved.
        data_format (str): Either 'abs' or 'anom' - the format of the data
        being saved.
        """

        for date in da.time.values:
            slice = da.sel(time=date).data
            date = pd.Timestamp(date)
            fname = "{:04d}_{:02d}_{:02d}.npy".\
                format(date.year, date.month, date.day)

            self.save_processed_file(var_name, fname, slice,
                                     append=[str(date.year)])

    def _open_dataarray_from_files(self, var_name):

        """
        Open the yearly xarray files, accounting for some ERA5 variables that
        have erroneous 'unknown' NetCDF variable names which prevents
        concatentation.
        """

        logging.info("Opening files for {}".format(var_name))
        ds = xr.open_mfdataset(self._var_files[var_name],
                               # Solves issue with inheriting files without
                               # time dimension (only having coordinate)
                               combine="nested",
                               concat_dim="time",
                               coords="minimal",
                               compat="override",
                               drop_variables=("lat", "lon"),
                               # TODO: Wasteful on small sets, but much faster
                               #  on big sets: make optional
                               parallel=True)

        # For processing one file, we're going to assume a single non-lambert
        # variable exists at the start and rename all of them
        var_names = [name for name in list(ds.data_vars.keys())
                     if not name.startswith("lambert_")]

        var_names = set(var_names)
        logging.debug("Files have var names {} which will be renamed to {}".
                      format(", ".join(var_names), var_name))

        ds = ds.rename({k: var_name for k in var_names})
        da = getattr(ds, var_name)

        # all_dates = self.dates.train + self.dates.val + self.dates.test
        # logging.debug("{} dates in total".format(len(all_dates)))

        da_dates = [pd.to_datetime(d).date() for d in da.time.values]
        logging.debug("{} dates in da".format(len(da_dates)))

        # search = sorted(list(set([el for el in all_dates
        #                          if pd.to_datetime(el).date() in da_dates])))
        # logging.debug("Selecting {} dates from da".format(len(search)))

        # We no longer select on all_dates, as it destroys lag/lead processed
        # selections from the dataset
        try:
            da = da.sel(time=da_dates)
        except KeyError:
            # There is likely non-resampled data being used
            # TODO: we could use nearest neighbour on this coordinate,
            #  but this feels more reliable to dodgy input data when
            #  transferring
            logging.warning("Data selection failed, likely not daily sampled "
                            "data so will give that a try")
            da = da.resample(time="1D").mean().sel(time=da_dates).sortby("time")
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

        if self._refdir:
            logging.info("Using alternate processing directory {} for "
                         "mean".format(self._refdir))
            proc_dir = os.path.join(self._refdir, "normalisation.mean")
        else:
            proc_dir = self.get_data_var_folder("normalisation.mean")

        mean_path = os.path.join(proc_dir, "{}".format(var_name))

        if os.path.exists(mean_path):
            logging.debug("Loading norm-average mean-std from {}".
                          format(mean_path))
            mean, std = tuple([self._dtype(el) for el in
                               open(mean_path, "r").read().split(",")])
        elif self._dates.train:
            logging.debug("Generating norm-average mean-std from {} training "
                          "dates".format(len(self._dates.train)))
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
        if self._refdir:
            logging.info("Using alternate processing directory {} for "
                         "scaling".format(self._refdir))
            proc_dir = os.path.join(self._refdir, "normalisation.scale")
        else:
            proc_dir = self.get_data_var_folder("normalisation.scale")

        scale_path = os.path.join(proc_dir, "{}".format(var_name))

        if os.path.exists(scale_path):
            logging.debug("Loading norm-scaling min-max from {}".
                          format(scale_path))
            minimum, maximum = tuple([self._dtype(el) for el in
                               open(scale_path, "r").read().split(",")])
        elif self._dates.train:
            logging.debug("Generating norm-scaling min-max from {} training "
                          "dates".format(len(self._dates.train)))
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

    def _build_linear_trend_da(self, input_da, var_name, max_years=35):
        """
        Construct a DataArray `linear_trend_da` containing the linear trend SIC
        forecasts based on the input DataArray `input_da`.
        `input_da` (xarray.DataArray): Input DataArray to produce linear SIC
        forecasts for.
        """

        data_dates = sorted([pd.Timestamp(date)
                             for date in input_da.time.values])

        # the old method doesn't work with non-contiguous forecast ranges
        trend_dates = set()

        for dat_date in data_dates:
            trend_dates = trend_dates.union(
                [dat_date + pd.DateOffset(days=d)
                 for d in range(self._linear_trend_days)])

        trend_dates = list(sorted(trend_dates))
        logging.info("Generating {} trend dates".format(len(trend_dates)))

        linear_trend_da = \
            xr.broadcast(input_da, xr.DataArray(pd.date_range(
                data_dates[0],
                data_dates[-1] + pd.DateOffset(days=self._linear_trend_days)),
                    dims="time"))[0]
        linear_trend_da = linear_trend_da.sel(time=trend_dates)
        linear_trend_da.data = np.zeros(linear_trend_da.shape)

        land_mask = Masks(north=self.north, south=self.south).get_land_mask()

        # Could use shelve, but more likely we'll run into concurrency issues
        # pickleshare might be an option but a little over-engineery
        trend_cache_path = os.path.join(
            self.get_data_var_folder("linear_trends"),
            "{}.nc".format(var_name))
        trend_cache = linear_trend_da.copy()
        trend_cache.data = np.full_like(linear_trend_da.data, np.nan)

        if os.path.exists(trend_cache_path):
            trend_cache = xr.open_dataarray(trend_cache_path)
            logging.info("Loaded {} entries from {}".
                         format(len(trend_cache.time), trend_cache_path))

        def data_selector(da,
                          processing_date,
                          missing_dates=tuple()):
            target_date = pd.to_datetime(processing_date)

            date_da = da[(da.time['time.month'] == target_date.month) &
                         (da.time['time.day'] == target_date.day) &
                         (da.time <= target_date) &
                         ~da.time.isin(missing_dates)].\
                isel(time=slice(0, max_years))
            return date_da

        for forecast_date in sorted(trend_dates, reverse=True):
            if not trend_cache.sel(time=forecast_date).isnull().all():
                output_map = trend_cache.sel(time=forecast_date)
            else:
                output_map = linear_trend_forecast(
                    data_selector, forecast_date, input_da, land_mask,
                    missing_dates=self._missing_dates,
                    shape=self._data_shape)

            linear_trend_da.loc[dict(time=forecast_date)] = output_map

        logging.info("Writing new trend cache for {}".format(var_name))
        trend_cache.close()
        linear_trend_da.to_netcdf(trend_cache_path)

        return linear_trend_da

    @property
    def missing_dates(self):
        return self._missing_dates

    @missing_dates.setter
    def missing_dates(self, arr):
        self._missing_dates = arr
