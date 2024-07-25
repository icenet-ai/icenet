import datetime as dt
import logging
import os

import numpy as np
import xarray as xr

from icenet.utils import run_command

from download_toolbox.interface import DatasetConfig
from preprocess_toolbox.processor import Processor


# TODO: these stubs can be generalised into preprocess-toolbox and the Masks class made
#  the relevant import - definitely do this as there is a harsh interface of applications here
def land(ds_config: DatasetConfig, name: str, processed_path: str):
    return Masks(ds_config, [name, ], name, base_path=processed_path).get_land_mask_filenames()


def polarhole(ds_config: DatasetConfig, name: str, processed_path: str):
    return Masks(ds_config, [name, ], name, base_path=processed_path).get_polarhole_mask_filenames()


def active_grid_cell(ds_config: DatasetConfig, name: str, processed_path: str):
    return Masks(ds_config, [name, ], name, base_path=processed_path).get_active_grid_cell_mask_filenames()


class Masks(Processor):
    """Masking of regions to include/omit in dataset.

    NSIDC Land, Ocean, Coast, Ice, and Sea Ice Region Masks

    TODO: replace with NSIDC masks from:
     https://nsidc.org/data/user-resources/help-center/does-nsidc-have-tools-extract-and-geolocate-polar-stereographic-data#anchor-land-masks

    Citation
        Meier, Walter N., and J. Scott Stewart. (2023). NSIDC Land, Ocean, Coast, Ice, and Sea Ice Region Masks. NSIDC Special
        Report 25. Boulder CO, USA: National Snow and Ice Data Center.
        https://nsidc.org/sites/default/files/documents/technical-reference/nsidc-special-report-25.pdf.

    TODO: Add example usage.
    """

    def __init__(self,
                 dataset_config: DatasetConfig,
                 *args,
                 **kwargs):
        super().__init__(dataset_config, *args, **kwargs)

        self._land_mask_filename = "land_mask.npy"
        self._agcm_filename_tmpl = "active_grid_cell_mask_{:02d}.npy"

        self._polarhole_filename_tmpl = "polarhole{}_mask.npy"
        self._polarhole_radii = (28, 11, 3)
        self._polarhole_dates = (
            dt.date(1987, 6, 1),
            dt.date(2005, 10, 1),
            dt.date(2015, 12, 1),
        )

        self._north = dataset_config.location.north
        self._south = dataset_config.location.south

        # TODO: logic failure for the case of IceNet, or is it. Determine Location usage
        if not self._north and not self._south:
            raise RuntimeError("Location needs to be north or south hemisphere for IceNet Masks")

        self.generate()

    def generate(self):
        year = 2000

        retrieve_cmd_template_osi450 = \
            "wget -O {} ftp://osisaf.met.no/reprocessed/ice/conc/v2p0/{:04d}/{:02d}/{}"
        filename_template_osi450 = \
            'ice_conc_{}_ease2-250_cdr-v2p0_{:04d}{:02d}021200.nc'

        # Generate the land-lake-sea mask using the second day from each month
        # of the year 2000 (chosen arbitrarily as the mask is fixed within
        # month)

        hemi_str = "nh" if self._north else "sh"
        mask_folder = self.get_data_var_folder("masks")

        for month in range(1, 13):
            mask_path = os.path.join(mask_folder, self._agcm_filename_tmpl.format(month))

            if not os.path.exists(mask_path):
                # Download the data if not already downloaded
                filename_osi450 = filename_template_osi450.format(hemi_str, year, month)
                month_path = os.path.join(mask_folder, filename_osi450)

                if not os.path.exists(month_path):
                    run_command(retrieve_cmd_template_osi450.format(
                        month_path, year, month, filename_osi450))
                else:
                    logging.info("siconca {} already exists".format(filename_osi450))

                ds = xr.open_dataset(month_path)
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
                # if self.north:
                #     # TODO: Add source/explanation for these indices.
                #     max_extent_mask[325:386, 317:380] = False

                logging.info("Saving {}".format(mask_path))
                np.save(mask_path, max_extent_mask)

                land_mask_path = os.path.join(mask_folder,
                                              self._land_mask_filename)

                if not os.path.exists(land_mask_path) and month == 1:
                    land_mask = np.sum(binary[:, :, [7, 6]], axis=2).\
                                    reshape(*shape) >= 1

                    logging.info("Saving {}".format(land_mask_path))
                    np.save(land_mask_path, land_mask)

                if month == 1:
                    # Generate the polar hole masks
                    x = np.tile(np.arange(0, shape[1]).
                                reshape(shape[0], 1), (1, shape[1])).\
                        astype(self.dtype) - 215.5
                    y = np.tile(np.arange(0, shape[1]).
                                reshape(1, shape[1]), (shape[0], 1)).\
                        astype(self.dtype) - 215.5
                    squaresum = np.square(x) + np.square(y)

                    for i, radius in enumerate(self._polarhole_radii):
                        polarhole_data = np.full(shape, False)
                        polarhole_data[squaresum < radius ** 2] = True

                        polarhole_path = os.path.join(
                            mask_folder,
                            self._polarhole_filename_tmpl.format(i + 1))
                        logging.info("Saving polarhole {}".format(polarhole_path))
                        np.save(polarhole_path, polarhole)

            else:
                logging.info("Skipping {}, already exists".format(mask_path))

    def process(self):
        raise RuntimeError("Do not execute process on Masks, use generate() instead")

    def get_active_grid_cell_mask(self, month: object) -> object:
        """Loads an active grid cell mask from numpy file.

        Also, checks if a mask file exists for input month, and raises an error if it does not.

        Args:
            month: Month index representing the month for which the mask file is being checked.

        Returns:
            Active cell mask boolean(s) for corresponding month and pre-defined `self._region`.

        Raises:
            RuntimeError: If the mask file for the input month does not exist.
        """
        mask_path = os.path.join(
            self.get_data_var_folder("masks"),
            self._agcm_filename_tmpl.format(month))

        if not os.path.exists(mask_path):
            raise RuntimeError("Active cell masks have not been generated, "
                               "this is not done automatically so you might "
                               "want to address this!")

        logging.debug("Loading active cell mask {}".format(mask_path))
        return np.load(mask_path)

    def get_land_mask(self) -> object:
        """Generate an xarray.DataArray object containing the active cell masks
         for each timestamp in a given source DataArray.

        Returns:
            An numpy array of land mask flag(s) for corresponding month and
                pre-defined `self._region`.
        """
        mask_path = os.path.join(self.get_data_var_folder("masks"),
                                 self._land_mask_filename)

        if not os.path.exists(mask_path):
            raise RuntimeError("Land mask has not been generated, this is "
                               "not done automatically so you might want to "
                               "address this!")

        logging.debug("Loading land mask {}".format(mask_path))
        return np.load(mask_path)

    def get_polarhole_mask(self, date: object) -> object:
        """Get mask of polar hole region.

        """

        for i, r in enumerate(self._polarhole_radii):
            if date <= self._polarhole_dates[i]:
                polarhole_path = os.path.join(
                    self.get_data_var_folder("masks"),
                    self._polarhole_filename_tmpl.format(i + 1))
                logging.debug("Loading polarhole {}".format(polarhole_path))
                return np.load(polarhole_path)
        return None

    def get_active_grid_cell_mask_filenames(self):
        return [self._agcm_filename_tmpl.format(m) for m in range(1, 13)]

    def get_land_mask_filenames(self):
        return [self._land_mask_filename]

    def get_polarhole_mask_filenames(self):
        return [self._polarhole_filename_tmpl.format(i) for i in range(1, len(self._polarhole_radii) + 1)]