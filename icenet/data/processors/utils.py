import argparse
import glob
import logging
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

from icenet.utils import Hemisphere
from icenet.data.producers import DataProducer

from scipy import interpolate
from scipy.spatial.qhull import QhullError

"""

"""


def sic_interpolate(da: object,
                    masks: object) -> object:
    """

    :param da:
    :param masks:
    :return:
    """
    for date in da.time.values:
        polarhole_mask = masks.get_polarhole_mask(
            pd.to_datetime(date).date())

        da_day = da.sel(time=date)
        xx, yy = np.meshgrid(np.arange(432), np.arange(432))

        # Grid cells outside of polar hole or NaN regions
        valid = ~np.isnan(da_day.data)

        # Interpolate polar hole
        if type(polarhole_mask) is np.ndarray:
            valid = valid & ~polarhole_mask

        # Interpolate if there is more than one missing grid cell
        if np.sum(~valid) >= 1:
            # Find grid cell locations surrounding NaN regions for bilinear
            # interpolation
            nan_mask = np.ma.masked_array(np.full((432, 432), 0.))
            nan_mask[~valid] = np.ma.masked

            nan_neighbour_arrs = {}
            for order in 'C', 'F':
                # starts and ends indexes of masked element chunks
                slice_ends = np.ma.clump_masked(nan_mask.ravel(order=order))

                nan_neighbour_idxs = []
                nan_neighbour_idxs.extend([s.start for s in slice_ends])
                nan_neighbour_idxs.extend([s.stop - 1 for s in slice_ends])

                nan_neighbour_arr_i = np.array(np.full((432, 432), False),
                                               order=order)
                nan_neighbour_arr_i.ravel(order=order)[nan_neighbour_idxs] \
                    = True
                nan_neighbour_arrs[order] = nan_neighbour_arr_i

            nan_neighbour_arr = nan_neighbour_arrs['C'] + \
                nan_neighbour_arrs['F']
            # Remove artefacts along edge of the grid
            nan_neighbour_arr[:, 0] = \
                nan_neighbour_arr[0, :] = \
                nan_neighbour_arr[:, -1] = \
                nan_neighbour_arr[-1, :] = False

            if np.sum(nan_neighbour_arr) == 1:
                res = np.where(np.array(nan_neighbour_arr) == True)
                logging.warning("Not enough nans for interpolation, extending {}".format(res))
                x_idx, y_idx = res[0][0], res[1][0]
                nan_neighbour_arr[x_idx-1:x_idx+2, y_idx] = True
                nan_neighbour_arr[x_idx, y_idx-1:y_idx+2] = True
                logging.debug(np.where(np.array(nan_neighbour_arr) == True))

            # Perform bilinear interpolation
            x_valid = xx[nan_neighbour_arr]
            y_valid = yy[nan_neighbour_arr]
            values = da_day.data[nan_neighbour_arr]

            x_interp = xx[~valid]
            y_interp = yy[~valid]

            try:
                if len(x_valid) or len(y_valid):
                    interp_vals = interpolate.griddata((x_valid, y_valid),
                                                       values,
                                                       (x_interp, y_interp),
                                                       method='linear')
                    da.sel(time=date).data[~valid] = interp_vals
                else:
                    logging.warning("No valid values to interpolate with on "
                                    "{}".format(date))
            except QhullError:
                logging.exception("Geometrical degeneracy from QHull, interpolation failed")

    return da


def condense_main():
    ap = argparse.ArgumentParser()
    ap.add_argument("identifier")
    ap.add_argument("hemisphere", choices=("north", "south"))
    ap.add_argument("variable")

    ap.add_argument("-n", "--numpy", action="store_true", default=False)
    ap.add_argument("-v", "--verbose", action="store_true", default=False)
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    condense_data(args.identifier, args.hemisphere, args.variable)


def condense_data(identifier: str,
                  hemisphere: str,
                  variable: str):
    """Takes existing daily files and creates yearly files

    Previous early versions of the pipeline were storing files day by day, which
    is pretty wasteful. This allows us to create the yearly files and avoid
    all that nasty re-downloading business

    :param identifier:
    :param hemisphere:
    :param variable:
    """
    logging.info("Condensing data into singular file")

    dp = DataProducer(identifier=identifier,
                      north=getattr(Hemisphere,
                                    hemisphere.upper()) == Hemisphere.NORTH,
                      south=getattr(Hemisphere,
                                    hemisphere.upper()) == Hemisphere.SOUTH)

    data_path = dp.get_data_var_folder(variable, missing_error=True)

    logging.debug("Collecting files from {}".format(data_path))
    dfs = glob.glob(os.path.join(data_path, "**", "*.nc"))

    def year_batch(filenames):
        df_years = set([os.path.split(os.path.dirname(f_year))[-1]
                        for f_year in filenames])

        for year_el in df_years:
            year_dfs = [el for el in filenames
                        if os.path.split(os.path.dirname(el))[-1] == year_el
                        and not os.path.split(el)[1].startswith("latlon")]
            logging.debug("{} has {} files".format(year_el, len(year_dfs)))
            yield year_el, year_dfs

    if len(dfs):
        logging.debug("Got {} files, collecting to {}...".format(len(dfs),
                                                                 data_path))

        for year, year_files in year_batch(dfs):
            year_path = os.path.join(data_path, "{}.nc".format(year))

            if not os.path.exists(year_path):
                logging.info("Loading {}".format(year))
                ds = xr.open_mfdataset(year_files, parallel=True)
                years, datasets = zip(*ds.groupby("time.year"))
                if len(years) > 1:
                    raise RuntimeError("Too many years in one file {}".
                                       format(years))
                logging.info("Saving to {}".format(year_path))
                xr.save_mfdataset(datasets, [year_path])
    else:
        logging.info("No valid files found.")
