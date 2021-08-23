import datetime as dt
import logging

import numpy as np
import pandas as pd

from scipy import interpolate

from icenet2.data.process import IceNetPreProcessor
from icenet2.data.sic.mask import Masks


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

                # Perform bilinear interpolation
                x_valid = xx[nan_neighbour_arr]
                y_valid = yy[nan_neighbour_arr]
                values = da_day.data[nan_neighbour_arr]

                x_interp = xx[~valid]
                y_interp = yy[~valid]

                if len(x_valid) or len(y_valid):
                    da.sel(time=date).data[~valid] = \
                        interpolate.griddata((x_valid, y_valid),
                                             values,
                                             (x_interp, y_interp),
                                             method='linear')
                else:
                    logging.warning("No valid values to interpolate with on "
                                    "{}".format(date))

        return da


# TODO: pick up identifiers from the interfaces
class IceNetCMIPPreProcessor(IceNetPreProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, identifier="cmip", **kwargs)

    def pre_normalisation(self, var_name, da):
        """
        Convert the cmip6 xarray time dimension to use day=1, hour=0 convention
        used in the rest of the project.
        """

        standardised_dates = []
        for datetime64 in da.time.values:
            date = pd.Timestamp(datetime64, unit='s')
            date = date.replace(day=1, hour=0)
            standardised_dates.append(date)
        da = da.assign_coords({'time': standardised_dates})

        if var_name == "siconca":
            masks = Masks(north=self.north, south=self.south)
            return SICInterpolation.interpolate(da, masks)

        return da


class IceNetERA5PreProcessor(IceNetPreProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args,
                         file_filters=["latlon_"],
                         identifier="era5",
                         **kwargs)


class IceNetOSIPreProcessor(IceNetPreProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, identifier="osisaf", **kwargs)

    def pre_normalisation(self, var_name, da):
        if var_name != "siconca":
            raise RuntimeError("OSISAF SIC implementation should be dealing "
                               "with siconca, ")
        else:
            masks = Masks(north=self.north, south=self.south)
            return SICInterpolation.interpolate(da, masks)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)

    osi = IceNetOSIPreProcessor(
        ["siconca"],
        [],
        "test1",
        list([
            pd.to_datetime(date).date() for date in
            pd.date_range(dt.date(1989, 1, 1), dt.date(1989, 1, 6), freq='D')
        ]),
        [],
        [],
        include_circday=False,
        include_land=False,
        linear_trends=["siconca"],
        linear_trend_days=3,
    )
    osi.init_source_data()
    osi.process()

