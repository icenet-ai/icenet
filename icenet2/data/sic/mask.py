"""Data Masks
"""

import logging
import os
import shutil

import numpy as np
import xarray as xr

from icenet2.data.producers import Generator
from icenet2.utils import Hemisphere, HemisphereMixin, run_command


class Masks(Generator, HemisphereMixin):
    def __init__(self, *args, hemisphere, **kwargs):
        super().__init__(*args, **kwargs)
        self._hemisphere = hemisphere

    def generate(self,
                 year=2000,
                 save_land_mask=True,
                 save_polarhole_masks=False,
                 xy=(432, 432),
                 polarhole_radii=(28, 11, 3),
                 remove_temp_files=True):
        """Generate a set of data masks

        TODO: Abstract out hardcoded paths/URIs
        """

        if not self._hemisphere | Hemisphere.BOTH:
            raise ValueError("No hemisphere provided")

        siconca_folder = get_folder("data", HEMISPHERE_STRINGS[hemisphere],
                                    'siconca', '{:04d}'.format(year))
        mask_folder = get_folder("data", HEMISPHERE_STRINGS[hemisphere], 'masks')

        retrieve_cmd_template_osi450 = \
            "wget -m -nH --cut-dirs=4 -P {} " \
            "ftp://osisaf.met.no/reprocessed/ice/conc/v2p0/{:04d}/{:02d}/{}"
        filename_template_osi450 = \
            'ice_conc_{}_ease2-250_cdr-v2p0_{:04d}{:02d}021200.nc'

        # Generate the land-lake-sea mask using the second day from each month of
        # the year 2000 (chosen arbitrarily as the mask is fixed within month)
        for month in range(1, 13):
            # Download the data if not already downloaded
            filename_osi450 = filename_template_osi450.format(
                hemisphere, year, month)

            if not os.path.exists(os.path.join(siconca_folder, filename_osi450)):
                run_command(retrieve_cmd_template_osi450.format(
                    siconca_folder, year, month, filename_osi450))
            else:
                logging.info("siconca {} already exists".format(filename_osi450))

            month_str = '{:02d}'.format(month)
            month_folder = os.path.join(siconca_folder, month_str)

            day_path = os.path.join(month_folder, filename_osi450)

            with xr.open_dataset(day_path) as ds:
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
                                     ACTIVE_GRID_CELL_MASK_FORMAT.format(month))
            logging.info("Saving {}".format(mask_path))
            np.save(mask_path, max_extent_mask)

            land_mask_path = os.path.join(mask_folder, FILENAMES['land_mask'])

            if save_land_mask and not os.path.exists(land_mask_path) and month == 1:
                land_mask = np.sum(binary[:, :, [7, 6]], axis=2).reshape(*xy) >= 1

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

                polarhole_path = os.path.join(FOLDERS['masks'],
                                              "polarhole{}_mask.npy".format(i+1))
                logging.info("Saving polarhole {}".format(polarhole_path))
                np.save(polarhole_path, polarhole)

    def plot():
        raise NotImplementedError("Mask plot not implemented yet")
