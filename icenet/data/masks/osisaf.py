import datetime as dt
import logging
import os

import numpy as np
import pandas as pd
import xarray as xr

from download_toolbox.interface import Configuration, DatasetConfig
from download_toolbox.utils import run_command
from preprocess_toolbox.processor import Processor


class MaskDatasetConfig(DatasetConfig):
    def __init__(self,
                 downloaded_files: list = None,
                 identifier: str = "masks",
                 **kwargs):
        super().__init__(identifier=identifier,
                         levels=[None, None, None],
                         path_components=[],
                         var_names=["land", "active_grid_cell", "polarhole"],
                         **kwargs)

        if not self.location.north and not self.location.south:
            raise NotImplementedError("Location must be north or south, not custom")

        self._downloaded_files = [] if downloaded_files is None else downloaded_files
        self._year = 2000
        self._retrieve_cmd_template_osi450 = \
            "wget -O {} ftp://osisaf.met.no/reprocessed/ice/conc/v2p0/{:04d}/{:02d}/{}"
        self._filename_template_osi450 = \
            'ice_conc_{}_ease2-250_cdr-v2p0_{:04d}{:02d}021200.nc'
        self._hemi_str = "nh" if self.location.north else "sh"

    def _download_or_load(self, month):
        filename_osi450 = self._filename_template_osi450.format(self._hemi_str, self._year, month)
        month_path = os.path.join(self.path, filename_osi450)

        if not os.path.exists(month_path):
            run_command(self._retrieve_cmd_template_osi450.format(
                month_path, self._year, month, filename_osi450))

            if filename_osi450 not in self._downloaded_files:
                self._downloaded_files.append(filename_osi450)
        else:
            logging.info("siconca {} already exists".format(filename_osi450))

        logging.info("Opening {}".format(month_path))
        return xr.open_dataset(month_path)

    def _generate_active_grid_cell(self):
        agcm_files = list()

        for month in range(1, 13):
            mask_path = os.path.join(self.path,
                                     "active_grid_cell_mask_{}_{:02d}.npy".format(self._hemi_str, month))

            if not os.path.exists(mask_path):
                ds = self._download_or_load(month)
                shape = ds.isel(time=0).ice_conc.shape

                status_flag = ds['status_flag']
                status_flag = np.array(status_flag.data).astype(np.uint8)
                status_flag = status_flag.reshape(*shape)

                binary = np.unpackbits(status_flag, axis=1).reshape(*shape, 8)

                # TODO: Add source/explanation for these magic numbers (index slicing nos.).
                #  Mask out: land, lake, and 'outside max climatology' (open sea)
                max_extent_mask = np.sum(binary[:, :, [7, 6, 0]], axis=2).reshape(*shape) >= 1
                max_extent_mask = ~max_extent_mask

                if self.location.north:
                    # TODO: Add source/explanation for these indices.
                    max_extent_mask[325:386, 317:380] = False

                logging.info("Saving {}".format(mask_path))
                np.save(mask_path, max_extent_mask)
            agcm_files.append(mask_path)
        return agcm_files

    def _generate_land(self):
        land_mask_path = os.path.join(self.path,
                                      "land_mask.{}.npy".format(self._hemi_str))

        if not os.path.exists(land_mask_path):
            ds = self._download_or_load(1)
            shape = ds.isel(time=0).ice_conc.shape

            status_flag = ds['status_flag']
            status_flag = np.array(status_flag.data).astype(np.uint8)
            status_flag = status_flag.reshape(*shape)

            binary = np.unpackbits(status_flag, axis=1).reshape(*shape, 8)
            land_mask = np.sum(binary[:, :, [7, 6]], axis=2).reshape(*shape) >= 1

            logging.info("Saving {}".format(land_mask_path))
            np.save(land_mask_path, land_mask)
        return land_mask_path

    def _generate_polarhole(self):
        polarhole_files = list()
        # Generate the polar hole masks
        # TODO: south doesn't require any polar hole masks and ice_conc is hardcoded
        ds = self._download_or_load(1)
        shape = ds.isel(time=0).ice_conc.shape

        x = np.tile(np.arange(0, shape[1]).
                    reshape(shape[0], 1), (1, shape[1])) - 215.5
        y = np.tile(np.arange(0, shape[1]).
                    reshape(1, shape[1]), (shape[0], 1)) - 215.5
        squaresum = np.square(x) + np.square(y)

        for i, radius in enumerate([28, 11, 3]):
            polarhole_data = np.full(shape, False)
            polarhole_data[squaresum < radius ** 2] = True

            polarhole_path = os.path.join(
                self.path,
                "polarhole_mask_{}_{:02d}.npy".format(self._hemi_str, i + 1))
            logging.info("Saving polarhole {}".format(polarhole_path))
            np.save(polarhole_path, polarhole_data)
            polarhole_files.append(polarhole_path)
        return polarhole_files

    def save_data_for_config(self,
                             rename_var_list: dict = None,
                             source_ds: object = None,
                             source_files: list = None,
                             time_dim_values: list = None,
                             var_filter_list: list = None,
                             **kwargs):

        for var_config in self.variables:
            files = getattr(self, "_generate_{}".format(var_config.name))()
            self.var_files[var_config.name] = files

    def get_config(self,
                   config_funcs: dict = None,
                   strip_keys: list = None):
        return super().get_config(strip_keys=[
            #"_filename_template_osi450",
            #"_hemi_str",
            "_identifier",
            "_levels",
            "_path_components",
            #"_retrieve_cmd_template_osi450",
            "_var_names",
            #"_year",
        ])

    @property
    def config(self):
        if self._config is None:
            logging.debug("Creating dataset configuration with {}".format(self.location.name))
            self._config = Configuration(config_type=self.config_type,
                                         directory=self.root_path,
                                         identifier=self.location.name)
        return self._config


class Masks(Processor):
    def __init__(self,
                 dataset_config: DatasetConfig,
                 *args,
                 # TODO: we need to review, consuming for recreation is a bit narly no!?
                 #  Should these even be in the configuration file? ? ?
                 absolute_vars: list = None,
                 identifier: str = None,
                 **kwargs):
        mask_ds = MaskDatasetConfig(
            base_path=dataset_config.base_path,
            frequency=dataset_config.frequency,
            location=dataset_config.location,
        )
        mask_ds.save_data_for_config()
        self._dataset_config = mask_ds.save_config()
        self._hemi_str = "north" if dataset_config.location.north else "south"

        super().__init__(mask_ds,
                         absolute_vars=["active_grid_cell", "land", "land_map", "polarhole"],
                         dtype=np.dtype(bool),
                         identifier="masks.{}".format(self._hemi_str),
                         **kwargs)

        self._source_files = mask_ds.var_files.copy()
        self._region = (slice(None, None), slice(None, None))

    def get_config(self,
                   config_funcs: dict = None,
                   strip_keys: list = None):
        return {
            "implementation": "{}:{}".format(self.__module__, self.__class__.__name__),
            "absolute_vars": self.abs_vars,
            "dataset_config": self._dataset_config,
            "path": self.path,
            "processed_files": self._processed_files,
            "source_files": self._source_files,
        }

    def process(self):
        # Active grid cell mask preparation
        mask_files = self._source_files["active_grid_cell"]

        da = xr.DataArray(data=[np.load(acgm_month_file) for acgm_month_file in mask_files],
                          dims=["month", "yc", "xc"],
                          coords=dict(
                              month=range(1, 13),
                          ),
                          attrs=dict(description="IceNet active grid cell mask metadata"))

        self.save_processed_file("active_grid_cell",
                                 os.path.basename(self.active_grid_cell_filename),
                                 da,
                                 overwrite=False)

        # Land mask preparation
        land_mask = np.load(self._source_files["land"])

        da = xr.DataArray(data=land_mask,
                          dims=["yc", "xc"],
                          attrs=dict(description="IceNet land mask metadata"))

        self.save_processed_file("land", os.path.basename(self.land_filename), da, overwrite=False)

        land_map = np.ones(land_mask.shape, dtype=np.float32)
        land_map[~land_mask] = -1.
        da = xr.DataArray(data=land_map,
                          dims=["yc", "xc"],
                          attrs=dict(description="IceNet land map metadata"))
        self.save_processed_file("land_map", os.path.basename(self.land_map_filename), da, convert=False, overwrite=False)

        # Polar hole mask preparation
        mask_files = self._source_files["polarhole"]

        da = xr.DataArray(data=[np.load(polarhole_file) for polarhole_file in mask_files],
                          dims=["polarhole", "yc", "xc"],
                          coords=dict(
                              polarhole=[pd.Timestamp(el) for el in [
                                  dt.date(1987, 6, 1),
                                  dt.date(2005, 10, 1),
                                  dt.date(2015, 12, 1),
                              ]],
                          ),
                          attrs=dict(description="IceNet polar hole mask metadata"))

        self.save_processed_file("polarhole",
                                 os.path.basename(self.polarhole_filename),
                                 da,
                                 overwrite=False)

        self.save_config()

    def active_grid_cell(self, date=None, *args, **kwargs):
        """

        Args:
            date:
            *args:
            **kwargs:

        Returns:

        """
        da = xr.open_dataarray(self.active_grid_cell_filename)
        da = da.sel(month=pd.to_datetime(date).month)
        return da.data[self._region]

    def inactive_grid_cell(self, date=None, *args, **kwargs):
        """

        Args:
            date:
            *args:
            **kwargs:

        Returns:

        """
        return ~(self.active_grid_cell(date, *args, **kwargs))

    def land(self, *args, **kwargs):
        """

        Args:
            *args:
            **kwargs:

        Returns:

        """
        da = xr.open_dataarray(self.land_filename)
        return da.data[self._region]

    def polarhole(self, date, *args, **kwargs):
        """

        Args:
            date:
            *args:
            **kwargs:

        Returns:

        """
        da = xr.open_dataarray(self.polarhole_filename)
        polarhole_mask = np.full(da.isel(polarhole=0).shape, False)
        da = da[da.polarhole >= date]

        if len(da.polarhole) > 0:
            polarhole_mask = da.isel(polarhole=0)
            # logging.debug("Selecting mask {} for {}".format(polarhole_mask.polarhole, date))
            return polarhole_mask.data[self._region]
        else:
            return polarhole_mask[self._region]

    def get_active_cell_da(self, src_da: object) -> object:
        """Generate an xarray.DataArray object containing the active cell masks
         for each timestamp in a given source DataArray.

        Args:
            src_da: Source xarray.DataArray object containing time, xc, yc
                coordinates.

        Returns:
            An xarray.DataArray containing active cell masks for each time
                in source DataArray.
        """
        return xr.DataArray(
            [
                self.active_grid_cell(pd.to_datetime(date).month)
                for date in src_da.time.values
            ],
            dims=('time', 'yc', 'xc'),
            coords={
                'time': src_da.time.values,
                'yc': src_da.yc.values,
                'xc': src_da.xc.values,
            })

    def get_blank_mask(self) -> object:
        """Returns an empty mask.

        Returns:
            A numpy array of flags set to false for pre-defined `self._region`
                of shape `self._shape` (the `data_shape` instance initialisation
                value).
        """
        shape = self.land().shape
        return np.full(shape, False)[self._region]

    def __getitem__(self, item):
        """Sets slice of region wanted for masking, and allows method chaining.

        This might be a semantically dodgy thing to do, but it works for the mo

        Args:
            item: Index/slice to extract.
        """
        logging.info("Mask region set to: {}".format(item))
        self._region = item
        return self

    @property
    def region(self):
        return self._region

    @region.setter
    def region(self, value):
        self._region = value

    def reset_region(self):
        """Resets the mask region and logs a message indicating that the whole mask will be returned."""
        logging.info("Mask region reset, whole mask will be returned")
        self._region = (slice(None, None), slice(None, None))

    @property
    def active_grid_cell_filename(self):
        return os.path.join(self.path, "active_grid_cell.{}.nc".format(self._hemi_str))

    @property
    def land_filename(self):
        return os.path.join(self.path, "land.{}.nc".format(self._hemi_str))

    @property
    def land_map_filename(self):
        return os.path.join(self.path, "land_map.{}.nc".format(self._hemi_str))

    @property
    def polarhole_filename(self):
        return os.path.join(self.path, "polarhole.{}.nc".format(self._hemi_str))

