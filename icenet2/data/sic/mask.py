"""Data Masks
"""

import logging
import os
import shutil

import numpy as np
import xarray as xr

from icenet2.data.producers import Generator
from icenet2.utils import Hemisphere, run_command


class Masks(Generator):
    LAND_MASK_FILENAME = "land_mask.npy"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, identifier="masks", **kwargs)

    def generate(self,
                 year=2000,
                 save_land_mask=True,
                 save_polarhole_masks=False,
                 xy=(432, 432),
                 polarhole_radii=(28, 11, 3),
                 remove_temp_files=False):
        """Generate a set of data masks

        TODO: Abstract out hardcoded paths/URIs
        """

        siconca_folder = self.get_data_var_folder("siconca")
        mask_folder = self.get_data_var_folder("masks")

        # FIXME: cut-dirs can be nasty, better use -O, changed from 4 to 5
        retrieve_cmd_template_osi450 = \
            "wget -m -nH --cut-dirs=4 -P {} " \
            "ftp://osisaf.met.no/reprocessed/ice/conc/v2p0/{:04d}/{:02d}/{}"
        filename_template_osi450 = \
            'ice_conc_{}_ease2-250_cdr-v2p0_{:04d}{:02d}021200.nc'

        # Generate the land-lake-sea mask using the second day from each month
        # of the year 2000 (chosen arbitrarily as the mask is fixed within
        # month)
        for month in range(1, 13):
            # Download the data if not already downloaded
            filename_osi450 = filename_template_osi450.format(
                self.hemisphere_str[0], year, month)

            month_str = '{:02d}'.format(month)
            month_folder = os.path.join(siconca_folder, str(year), month_str)
            month_path = os.path.join(month_folder, filename_osi450)

            if not os.path.exists(month_path):
                run_command(retrieve_cmd_template_osi450.format(
                    siconca_folder, year, month, filename_osi450))
            else:
                logging.info("siconca {} already exists".
                             format(filename_osi450))

            with xr.open_dataset(month_path) as ds:
                status_flag = ds['status_flag']
                status_flag = np.array(status_flag.data).astype(np.uint8)
                status_flag = status_flag.reshape(*xy)

                binary = np.unpackbits(status_flag, axis=1).reshape(*xy, 8)

                # Mask out: land, lake, and 'outside max climatology' (open sea)
                max_extent_mask = np.sum(
                    binary[:, :, [7, 6, 0]], axis=2).reshape(*xy) >= 1
                max_extent_mask = ~max_extent_mask
                # FIXME: Remove Caspian and Black seas - should we do this sh?
                max_extent_mask[325:386, 317:380] = False

            mask_path = os.path.join(mask_folder,
                                     "active_grid_cell_mask_{:02d}.npy".
                                     format(month))
            logging.info("Saving {}".format(mask_path))
            np.save(mask_path, max_extent_mask)

            land_mask_path = os.path.join(mask_folder,
                                          Masks.LAND_MASK_FILENAME)

            if save_land_mask and \
                    not os.path.exists(land_mask_path) \
                    and month == 1:
                land_mask = np.sum(binary[:, :, [7, 6]], axis=2).\
                                reshape(*xy) >= 1

                logging.info("Saving {}".format(land_mask_path))
                np.save(land_mask_path, land_mask)

        # Delete the data/siconca/2000 folder holding the temporary daily files
        if remove_temp_files:
            logging.info("Removing {}".format(siconca_folder))
            shutil.rmtree(siconca_folder)

        if save_polarhole_masks:
            # Generate the polar hole masks
            x = np.tile(np.arange(0, xy[1]).reshape(xy[0], 1), (1, xy[1])).astype(
                np.float32) - 215.5
            y = np.tile(np.arange(0, xy[1]).reshape(1, xy[1]), (xy[0], 1)).astype(
                np.float32) - 215.5
            squaresum = np.square(x) + np.square(y)

            for i, radius in enumerate(polarhole_radii):
                polarhole = np.full(xy, False)
                polarhole[squaresum < radius**2] = True

                polarhole_path = os.path.join(self.get_data_var_folder("masks"),
                                              "polarhole{}_mask.npy".
                                              format(i+1))
                logging.info("Saving polarhole {}".format(polarhole_path))
                np.save(polarhole_path, polarhole)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    sh = Masks(north=False, south=True)
    sh.generate(save_polarhole_masks=True)
