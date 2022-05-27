import logging

import numpy as np
import pandas as pd

from scipy import interpolate
from scipy.spatial.qhull import QhullError


class SICInterpolation:
    @staticmethod
    def interpolate(da, masks):
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
