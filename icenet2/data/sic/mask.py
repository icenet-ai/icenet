"""Data Masks
"""

import datetime as dt
import logging
import os
import shutil

import numpy as np
import xarray as xr

from icenet2.data.producers import Generator
from icenet2.utils import Hemisphere, run_command


class Masks(Generator):
    LAND_MASK_FILENAME = "land_mask.npy"
    # FIXME: nh/sh?
    POLARHOLE_RADII = (28, 11, 3)
    POLARHOLE_DATES = (
        dt.date(1987, 6, 1),
        dt.date(2005, 10, 1),
        dt.date(2015, 12, 1),
    )

    def __init__(self, *args,
                 polarhole_dates=POLARHOLE_DATES,
                 polarhole_radii=POLARHOLE_RADII,
                 **kwargs):
        super().__init__(*args, identifier="masks", **kwargs)

        self._polarhole_dates = polarhole_dates
        self._polarhole_radii = polarhole_radii
        self.init_params()

    def init_params(self):
        params_path = os.path.join(
            self.get_data_var_folder("masks"),
            "masks.params"
        )

        if not os.path.exists(params_path):
            with open(params_path, "w") as fh:
                for i, polarhole in enumerate(self._polarhole_radii):
                    fh.write("{}\n".format(
                        ",".join([str(polarhole),
                                  self._polarhole_dates[i].strftime("%Y%m%d")]
                                 )))
        else:
            lines = [l.strip().split(",")
                     for l in open(params_path, "r").readlines()]
            radii, dates = zip(*lines)
            self._polarhole_dates = [dt.datetime.strptime(el, "%Y%m%d").date()
                                     for el in dates]
            self._polarhole_radii = [int(r) for r in radii]

    def generate(self,
                 year=2000,
                 save_land_mask=True,
                 save_polarhole_masks=True,
                 xy=(432, 432),
                 remove_temp_files=False):
        """Generate a set of data masks

        TODO: need to review this, probably not appropriate for dual hemisphere
        """

        siconca_folder = self.get_data_var_folder("siconca")

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

            mask_path = os.path.join(self.get_data_var_folder("masks"),
                                     "active_grid_cell_mask_{:02d}.npy".
                                     format(month))
            logging.info("Saving {}".format(mask_path))
            np.save(mask_path, max_extent_mask)

            land_mask_path = os.path.join(self.get_data_var_folder("masks"),
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
            x = np.tile(np.arange(0, xy[1]).reshape(xy[0], 1), (1, xy[1])).\
                    astype(np.float32) - 215.5
            y = np.tile(np.arange(0, xy[1]).reshape(1, xy[1]), (xy[0], 1)).\
                    astype(np.float32) - 215.5
            squaresum = np.square(x) + np.square(y)

            for i, radius in enumerate(self._polarhole_radii):
                polarhole = np.full(xy, False)
                polarhole[squaresum < radius**2] = True

                polarhole_path = os.path.join(self.get_data_var_folder("masks"),
                                              "polarhole{}_mask.npy".
                                              format(i+1))
                logging.info("Saving polarhole {}".format(polarhole_path))
                np.save(polarhole_path, polarhole)

    def get_active_cell_mask(self, month):
        mask_path = os.path.join(self.get_data_var_folder("masks"),
                                 "active_grid_cell_mask_{:02d}.npy".
                                 format(month))

        if not os.path.exists(mask_path):
            raise RuntimeError("Active cell masks have not been generated, "
                               "this is not done automatically so you might "
                               "want to address this!")

        logging.debug("Loading active cell mask {}".format(mask_path))
        return np.load(mask_path)

    def get_land_mask(self, land_mask_filename=LAND_MASK_FILENAME):
        mask_path = os.path.join(self.get_data_var_folder("masks"),
                                 land_mask_filename)

        if not os.path.exists(mask_path):
            raise RuntimeError("Land mask has not been generated, this is "
                               "not done automatically so you might want to "
                               "address this!")

        logging.debug("Loading land mask {}".format(mask_path))
        return np.load(mask_path)

    def get_polarhole_mask(self, date):
        for i, r in enumerate(self._polarhole_radii):
            if date <= self._polarhole_dates[i]:
                polarhole_path = os.path.join(self.get_data_var_folder("masks"),
                                              "polarhole{}_mask.npy".
                                              format(i + 1))
                logging.info("Loading polarhole {}".format(polarhole_path))
                return np.load(polarhole_path)
        return None


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    sh = Masks(north=False, south=True)
    sh.generate(save_polarhole_masks=True)
    nh = Masks(north=True, south=False)
    nh.generate(save_polarhole_masks=True)
