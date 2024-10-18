import datetime as dt
import logging
import os
import shutil

import numpy as np
import pandas as pd
import xarray as xr

from icenet.data.cli import download_args
from icenet.data.producers import Generator
from icenet.utils import run_command
from icenet.data.sic.utils import SIC_HEMI_STR
"""Sea Ice Masks

"""


class Masks(Generator):
    """Masking of regions to include/omit in dataset.

    TODO: Add example usage.
    """

    LAND_MASK_FILENAME = "land_mask.npy"
    # FIXME: nh/sh?
    POLARHOLE_RADII = (28, 11, 3)
    POLARHOLE_DATES = (
        dt.date(1987, 6, 1),
        dt.date(2005, 10, 1),
        dt.date(2015, 12, 1),
    )

    def __init__(self,
                 *args,
                 polarhole_dates: object = POLARHOLE_DATES,
                 polarhole_radii: object = POLARHOLE_RADII,
                 data_shape: object = (432, 432),
                 dtype: object = np.float32,
                 longitudes = None,
                 latitudes = None,
                 **kwargs):
        """Initialises Masks across specified hemispheres.

        Args:
            polarhole_dates: Dates for polar hole (missing data) in data.
            polarhole_radii: Radii of polar hole.
            data_shape: Shape of input dataset.
            dtype: Store mask as this type.
        """
        super().__init__(*args, identifier="masks", **kwargs)

        self._polarhole_dates = polarhole_dates
        self._polarhole_radii = polarhole_radii
        self._dtype = dtype
        self._shape = data_shape
        self.longitudes = longitudes
        self.latitudes = latitudes
        self._region = (slice(None, None), slice(None, None))
        self._region_geo_mask = None

        self.init_params()

    def init_params(self):
        """Initialises the parameters of the Masks class.

        This method will create a `masks.params` file if it does not exist.
        And, stores the polar_radii and polar_dates instance variables into it.
        If it already exists, it will read and store the values to the instance
        variables
        """

        params_path = os.path.join(self.get_data_var_folder("masks"),
                                   "masks.params")

        if not os.path.exists(params_path):
            with open(params_path, "w") as fh:
                for i, polarhole in enumerate(self._polarhole_radii):
                    fh.write("{}\n".format(",".join([
                        str(polarhole),
                        self._polarhole_dates[i].strftime("%Y%m%d")
                    ])))
        else:
            lines = [
                el.strip().split(",")
                for el in open(params_path, "r").readlines()
            ]
            radii, dates = zip(*lines)
            self._polarhole_dates = [
                dt.datetime.strptime(el, "%Y%m%d").date() for el in dates
            ]
            self._polarhole_radii = [int(r) for r in radii]

    def generate(self,
                 year: int = 2000,
                 save_land_mask: bool = True,
                 save_polarhole_masks: bool = True,
                 remove_temp_files: bool = False):
        """Generate a set of data masks.

        Args:
            year (optional): Which year to use for generate masks from.
                Defaults to 2000.
            save_land_mask (optional): Whether to output land mask.
                Defaults to True.
            save_polarhole_masks (optional):  Whether to output polar hole masks.
                Defaults to True.
            remove_temp_files (optional): Whether to remove temporary directory.
                Defaults to False.
        """
        siconca_folder = self.get_data_var_folder("siconca")

        # FIXME: cut-dirs can be change intolerant, better use -O, changed
        #  from 4 to 5
        retrieve_cmd_template_osi450 = \
            "wget -m -nH --cut-dirs=4 -P {} " \
            "ftp://osisaf.met.no/reprocessed/ice/conc/v2p0/{:04d}/{:02d}/{}"
        filename_template_osi450 = \
            'ice_conc_{}_ease2-250_cdr-v2p0_{:04d}{:02d}021200.nc'

        # Generate the land-lake-sea mask using the second day from each month
        # of the year 2000 (chosen arbitrarily as the mask is fixed within
        # month)
        for month in range(1, 13):
            mask_path = os.path.join(
                self.get_data_var_folder("masks"),
                "active_grid_cell_mask_{:02d}.npy".format(month))
            if not os.path.exists(mask_path):
                # Download the data if not already downloaded
                filename_osi450 = filename_template_osi450.format(
                    SIC_HEMI_STR[self.hemisphere_str[0]], year, month)

                month_str = '{:02d}'.format(month)
                month_folder = os.path.join(siconca_folder, str(year), month_str)
                month_path = os.path.join(month_folder, filename_osi450)

                if not os.path.exists(month_path):
                    run_command(
                        retrieve_cmd_template_osi450.format(
                            siconca_folder, year, month, filename_osi450))
                else:
                    logging.info(
                        "siconca {} already exists".format(filename_osi450))

                with xr.open_dataset(month_path) as ds:
                    status_flag = ds['status_flag']
                    status_flag = np.array(status_flag.data).astype(np.uint8)
                    status_flag = status_flag.reshape(*self._shape)

                    binary = np.unpackbits(status_flag, axis=1).\
                        reshape(*self._shape, 8)

                    # TODO: Add source/explanation for these magic numbers (index slicing nos.).
                    # Mask out: land, lake, and 'outside max climatology' (open sea)
                    max_extent_mask = np.sum(binary[:, :, [7, 6, 0]],
                                             axis=2).reshape(*self._shape) >= 1
                    max_extent_mask = ~max_extent_mask

                    # FIXME: Remove Caspian and Black seas - should we do this sh?
                    if self.north:
                        # TODO: Add source/explanation for these indices.
                        max_extent_mask[325:386, 317:380] = False

                logging.info("Saving {}".format(mask_path))

                np.save(mask_path, max_extent_mask)

                land_mask_path = os.path.join(self.get_data_var_folder("masks"),
                                              Masks.LAND_MASK_FILENAME)

                if save_land_mask and \
                        not os.path.exists(land_mask_path) \
                        and month == 1:
                    land_mask = np.sum(binary[:, :, [7, 6]], axis=2).\
                                    reshape(*self._shape) >= 1

                    logging.info("Saving {}".format(land_mask_path))
                    np.save(land_mask_path, land_mask)
            else:
                logging.info("Skipping {}, already exists".format(mask_path))

        # Delete the data/siconca/2000 folder holding the temporary daily files
        if remove_temp_files:
            logging.info("Removing {}".format(siconca_folder))
            shutil.rmtree(siconca_folder)

        if save_polarhole_masks and not self.south:
            # Generate the polar hole masks
            x = np.tile(np.arange(0, self._shape[1]).
                        reshape(self._shape[0], 1), (1, self._shape[1])).\
                astype(self._dtype) - 215.5
            y = np.tile(np.arange(0, self._shape[1]).
                        reshape(1, self._shape[1]), (self._shape[0], 1)).\
                astype(self._dtype) - 215.5
            squaresum = np.square(x) + np.square(y)

            for i, radius in enumerate(self._polarhole_radii):
                polarhole = np.full(self._shape, False)
                polarhole[squaresum < radius**2] = True

                polarhole_path = os.path.join(
                    self.get_data_var_folder("masks"),
                    "polarhole{}_mask.npy".format(i + 1))
                logging.info("Saving polarhole {}".format(polarhole_path))
                np.save(polarhole_path, polarhole)

    def get_region_data(self, data):
        """
        Get either a lat/lon region or a pixel bounded region via slicing.

        If setting region via lat/lon, coordinates must be passed by calling
        `self.set_region_by_lonlat` method first.
        """
        if self._region_geo_mask is not None:
            if self.longitudes is None or self.latitudes is None:
                raise ValueError(f"Call {self.__name__}.set_region_by_lonlat first," +
                                 "to pass in latitude and longitude coordinates.")
            array = xr.DataArray(
                data,
                dims=('yc', 'xc'),
                coords={
                    'yc': self.yc,
                    'xc': self.xc,
                })
            array["lon"] = (("yc", "xc"), self.longitudes.data)
            array["lat"] = (("yc", "xc"), self.latitudes.data)
            array = array.where(self._region_geo_mask.compute(), drop=True).values

            # When used as weights for xarray.DataArray.weighted(), it shouldn't have
            # nan's in grid (i.e., outside of lat/lon bounds), so set these areas to 0.
            array = np.nan_to_num(array)

            return array
        else:
            return data[self._region]

    def get_active_cell_mask(self, month: object) -> object:
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
            "active_grid_cell_mask_{:02d}.npy".format(month))

        if not os.path.exists(mask_path):
            raise RuntimeError("Active cell masks have not been generated, "
                               "this is not done automatically so you might "
                               "want to address this!")
        # logging.debug("Loading active cell mask {}".format(mask_path))
        data = np.load(mask_path)

        return self.get_region_data(data)


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
        active_cell_mask = [
                self.get_active_cell_mask(pd.to_datetime(date).month)
                for date in src_da.time.values
            ]

        active_cell_mask_da = xr.DataArray(
            active_cell_mask,
            dims=('time', 'yc', 'xc'),
            coords={
                'time': src_da.time.values,
                'yc': src_da.yc.values,
                'xc': src_da.xc.values,
            })

        return active_cell_mask_da

    def get_land_mask(self,
                      land_mask_filename: str = LAND_MASK_FILENAME) -> object:
        """Generate an xarray.DataArray object containing the active cell masks
         for each timestamp in a given source DataArray.

        Args:
            land_mask_filename (optional): Land mask output filename.
                Defaults to `Masks.LAND_MASK_FILENAME`.

        Returns:
            An numpy array of land mask flag(s) for corresponding month and
                pre-defined `self._region`.
        """
        mask_path = os.path.join(self.get_data_var_folder("masks"),
                                 land_mask_filename)

        if not os.path.exists(mask_path):
            raise RuntimeError("Land mask has not been generated, this is "
                               "not done automatically so you might want to "
                               "address this!")

        # logging.debug("Loading land mask {}".format(mask_path))
        data = np.load(mask_path)
        return self.get_region_data(data)

    def get_polarhole_mask(self, date: object) -> object:
        """Get mask of polar hole region.

        TODO:
            Explain date literals as class instance for POLARHOLE_DATES
            and POLARHOLE_RADII
        """
        if self.south:
            return None

        for i, r in enumerate(self._polarhole_radii):
            if date <= self._polarhole_dates[i]:
                polarhole_path = os.path.join(
                    self.get_data_var_folder("masks"),
                    "polarhole{}_mask.npy".format(i + 1))
                # logging.debug("Loading polarhole {}".format(polarhole_path))
                data = np.load(polarhole_path)
                return self.get_region_data(data)
        return None

    def get_blank_mask(self) -> object:
        """Returns an empty mask.

        Returns:
            A numpy array of flags set to false for pre-defined `self._region`
                of shape `self._shape` (the `data_shape` instance initialisation
                value).
        """
        data = np.full(self._shape, False)
        return self.get_region_data(data)

    def set_region_by_lonlat(self, xc, yc, lon, lat, region):
        """
        Sets the region based on longitude and latitude bounds by converting
        them into index slices based on the provided xarray DataArray.

        Alternative to __getitem__ if not using slicing.

        Args:
            xc:
            yc:
            lon:
            lat:
            region: lat/lon region bounds to get masks for, [lon_min, lat_min, lon_max, lat_max]
            src_da: An xarray.DataArray that contains longitude and latitude coordinates.
        """
        self.xc = xc
        self.yc = yc
        self.longitudes = lon
        self.latitudes = lat
        self.region_geographic = region

        lon_min, lat_min, lon_max, lat_max = region

        lon_mask = (lon >= lon_min) & (lon <= lon_max)
        lat_mask = (lat >= lat_min) & (lat <= lat_max)

        lat_lon_mask = lat_mask & lon_mask

        rows, cols = np.where(lat_lon_mask)

        row_min, row_max = rows.min(), rows.max()
        col_min, col_max = cols.min(), cols.max()

        # Specify min/max latlon bounds via slicing
        # (sideffect of slicing rectangular area aligned with `xc, yc`,
        # instead of actual lon/lat)
        # self._region = (slice(row_min, row_max+1), slice(col_min, col_max+1))

        # Instead, when this is set, masks based on actual lon/lat bounds.
        self._region_geo_mask = lat_lon_mask

    def __getitem__(self, item):
        """Sets slice of region wanted for masking, and allows method chaining.

        This might be a semantically dodgy thing to do, but it works for the mo

        Args:
            item: Index/slice to extract.
        """
        logging.info("Mask region set to: {}".format(item))
        self._region = item
        return self

    def reset_region(self):
        """Resets the mask region and logs a message indicating that the whole mask will be returned."""
        logging.info("Mask region reset, whole mask will be returned")
        self._region = (slice(None, None), slice(None, None))
        self._region_geo_mask = None


def main():
    """Entry point of Masks class - used to create executable that calls it."""
    args = download_args(dates=False, var_specs=False)

    north = args.hemisphere == "north"
    south = args.hemisphere == "south"

    Masks(north=north, south=south).\
        generate(save_polarhole_masks=north)
