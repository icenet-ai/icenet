import logging
import os

import numpy as np
import xarray as xr

from icenet.utils import run_command

from download_toolbox.interface import Configuration, DatasetConfig
from preprocess_toolbox.processor import Processor

"""
NSIDC-0780_SeaIceRegions
ds = xr.open_dataset("scratch/NSIDC-0780_SeaIceRegions_EASE2-S12.5km_v1.0.nc")
sm = ds.sea_ice_region_RH_surface_mask for SOUTH
sea_ice_region_surface_mask for NORTH
pprint.pprint(ds.sea_ice_region_surface_mask.flag_meanings.split())
['land',
 'fresh_free_water',
 'ice_on_land',
 'floating_ice_shelf',
 'ocean_disconnected',
 'off_earth']
xr.where(sm < 30, 0, 1)
"""


class MaskDatasetConfig(DatasetConfig):
    def __init__(self,
                 downloaded_files: list = None,
                 identifier: str = "masks",
                 mask_variable: str = None,
                 resolution: float = 6.25,
                 **kwargs):
        super().__init__(identifier=identifier,
                         levels=[None],
                         path_components=[],
                         var_names=["land"],
                         **kwargs)

        if not self.location.north and not self.location.south:
            raise NotImplementedError("Location must be north or south, not custom")

        valid_resolutions = [3.125, 6.25, 12.5, 25]

        if resolution not in valid_resolutions:
            raise RuntimeError("Resolution for NSIDC masking should be one of {}".format(valid_resolutions))

        self._hemi_str = "N" if self.location.north else "S"
        self._resolution = resolution
        self._mask_filename = "NSIDC-0780_SeaIceRegions_EASE2-{}{}km_v1.0.nc".format(self._hemi_str, self._resolution)

        if mask_variable is None:
            self._mask_variable = "sea_ice_region_RH_surface_mask" if self.location.south else "sea_ice_region_surface_mask"
        self._downloaded_files = [] if downloaded_files is None else downloaded_files

        self._retrieve_cmd = "wget -O {} ftp://sidads.colorado.edu/pub/DATASETS/nsidc0780_seaice_masks_v1/netcdf/{}"

    def _download_or_load(self):
        dest_filename = os.path.join(self.path, self._mask_filename)
        if not os.path.exists(dest_filename):
            run_command(self._retrieve_cmd.format(dest_filename, self._mask_filename))

            if dest_filename not in self._downloaded_files:
                self._downloaded_files.append(dest_filename)
        else:
            logging.info("Mask {} already exists".format(dest_filename))

        logging.info("Opening {}".format(dest_filename))
        return xr.open_dataset(dest_filename)

    def _generate_land(self):
        land_mask_path = os.path.join(self.path, "land_mask.{}.npy".format(self._hemi_str))

        if not os.path.exists(land_mask_path):
            ds = self._download_or_load()
            sm = getattr(ds, self._mask_variable)
            land_mask = xr.where(sm < 30, 0, 1)
            # Boundary of our AMSR data
            land_mask = land_mask.sel(
                x=slice(-3.947e+06, 3.947e+06),
                y=slice(4.347e+06, -3.947e+06))
            logging.info("Saving {}".format(land_mask_path))
            np.save(land_mask_path, land_mask.data[::-1])
        return land_mask_path

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
            "_identifier",
            "_levels",
            "_mask_filename",
            "_mask_variable",
            "_path_components",
            "_resolution",
            "_retrieve_cmd",
            "_var_names",
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
                         absolute_vars=["land", "land_map"],
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
        # Land mask preparation
        land_mask = np.load(self._source_files["land"])

        da = xr.DataArray(data=land_mask,
                          dims=["yc", "xc"],
                          attrs=dict(description="IceNet land mask metadata"))

        self.save_processed_file("land", os.path.basename(self.land_filename), da, overwrite=False)

        land_map = np.ones(land_mask.shape, dtype=np.float32)
        land_map[~land_mask.astype(bool)] = -1.
        da = xr.DataArray(data=land_map,
                          dims=["yc", "xc"],
                          attrs=dict(description="IceNet land map metadata"))
        self.save_processed_file("land_map", os.path.basename(self.land_map_filename), da, convert=False, overwrite=False)

        self.save_config()

    def land(self, *args, **kwargs):
        """

        Args:
            *args:
            **kwargs:

        Returns:

        """
        da = xr.open_dataarray(self.land_filename)
        return da.data[self._region]

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
    def land_filename(self):
        return os.path.join(self.path, "land.{}.nc".format(self._hemi_str))

    @property
    def land_map_filename(self):
        return os.path.join(self.path, "land_map.{}.nc".format(self._hemi_str))
