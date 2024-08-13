import datetime as dt
import logging
import os

logging.basicConfig(level=logging.DEBUG)

import numpy as np
import pandas as pd
import xarray as xr

from download_toolbox.interface import Configuration, DatasetConfig
from download_toolbox.utils import run_command
from preprocess_toolbox.processor import Processor, ProcessingError


class MaskDatasetConfig(DatasetConfig):
    def __init__(self,
                 identifier="masks",
                 **kwargs):
        super().__init__(identifier=identifier,
                         levels=[None, None, None],
                         path_components=[],
                         var_names=["land", "active_grid_cell", "polarhole"],
                         **kwargs)

        if not self.location.north and not self.location.south:
            raise NotImplementedError("Location must be north or south, not custom")

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

                # FIXME: Remove Caspian and Black seas - should we do this sh?
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
                             var_filter_list: list = None):

        for var_config in self.variables:
            files = getattr(self, "_generate_{}".format(var_config.name))()
            self.var_files[var_config.name] = files

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
                 **kwargs):
        mask_ds = MaskDatasetConfig(
            frequency=dataset_config.frequency,
            location=dataset_config.location,
        )
        mask_ds.save_data_for_config()
        mask_ds.save_config()

        super().__init__(mask_ds,
                         absolute_vars=["active_grid_cell", "land", "polarhole"],
                         identifier="masks",
                         **kwargs)

        self._source_files = mask_ds.var_files.copy()

    def get_config(self,
                   config_funcs: dict = None,
                   strip_keys: list = None):
        return {
            self.update_key: {
                "name":     self.identifier,
                "files":    {var_name: os.path.join(self.path, "{}.nc".format(var_name)) for var_name in self.abs_vars}
            }
        }

    def process(self):
        # Active grid cell mask preparation
        destination_filename = os.path.join(self.path, "active_grid_cell.nc")
        mask_files = self._source_files["active_grid_cell"]

        if not os.path.exists(destination_filename):
            da = xr.DataArray(data=[np.load(acgm_month_file) for acgm_month_file in mask_files],
                              dims=["month", "yc", "xc"],
                              coords=dict(
                                  month=range(1, 13),
                              ),
                              attrs=dict(description="IceNet active grid cell mask metadata"))
            self.save_processed_file("active_grid_cell", os.path.basename(destination_filename), da)

        # Land mask preparation
        filename = "land.nc"
        if not os.path.exists(os.path.join(self.path, filename)):
            land_mask = np.load(self._source_files["land"])
            land_map = np.ones(land_mask.shape, dtype=self.dtype)
            land_map[~land_mask] = -1.

            da = xr.DataArray(data=land_map,
                              dims=["yc", "xc"],
                              attrs=dict(description="IceNet land mask metadata"))
            self.save_processed_file("land", filename, da)

        # Polar hole mask preparation
        destination_filename = os.path.join(self.path, "polarhole.nc")
        mask_files = self._source_files["polarhole"]

        if not os.path.exists(destination_filename):
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
            self.save_processed_file("polarhole", os.path.basename(destination_filename), da)

        self.save_config()

    @property
    def active_grid_cell_filenames(self):
        return os.path.join(self.path, "active_grid_cell.nc")

    @property
    def land_filenames(self):
        return os.path.join(self.path, "land.nc")

    @property
    def polarhole_filenames(self):
        return os.path.join(self.path, "polarhole.nc")
