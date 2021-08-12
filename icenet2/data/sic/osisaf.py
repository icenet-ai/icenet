import copy
import logging
import os
import sys
import time

import datetime as dt
from pprint import pformat

import numpy as np
import pandas as pd
import xarray as xr

from scipy import interpolate

from icenet2.data.producers import Downloader
from icenet2.data.sic.mask import Masks


class SICDownloader(Downloader):
    """
    Downloads OSI-SAF SIC data from 1979-present using OpenDAP.
    The dataset comprises OSI-450 (1979-2015) and OSI-430-b (2016-ownards)
    Monthly averages are-computed on the server-side.
    This script can take about an hour to run.

    The query URLs were obtained from the following sites:
        - OSI-450 (1979-2016): https://thredds.met.no/thredds/dodsC/osisaf/
            met.no/reprocessed/ice/conc_v2p0_nh_agg.html
        - OSI-430-b (2016-present): https://thredds.met.no/thredds/dodsC/osisaf/
            met.no/reprocessed/ice/conc_crb_nh_agg.html
    """

    def __init__(self,
                 *args,
                 dates=(),
                 **kwargs):
        super().__init__(*args, identifier="osisaf", **kwargs)

        self._dates = dates
        self._masks = Masks(north=self.north, south=self.south)

        self._mask_dict = {
            month: self._masks.get_active_cell_mask(month)
            for month in np.arange(1, 12+1)
        }

    def download(self):
        hs = self.hemisphere_str[0]

        ref_date = dt.date(1978, 1, 1)
        osi450_start = dt.date(1979, 1, 1)
        osi430b_start = dt.date(2016, 1, 1)

        # FIXME: do ranged request optimsisation
        dt_arr = list(sorted(copy.copy(self._dates)))

        while len(dt_arr):
            el = dt_arr.pop()

            date_str = el.strftime("%Y%m%d")
            fpath = os.path.join(self.get_data_var_folder("siconca"),
                                 "siconca_{}.nc".format(date_str))
            if os.path.exists(fpath):
                logging.info("{} already exists, skipping".format(fpath))
                continue

            if el < osi450_start:
                raise NotImplementedError("No available date {}".
                                          format(osi450_start))
            elif el < osi430b_start:
                da_osi450 = xr.open_dataarray(
                    "https://thredds.met.no/thredds/dodsC/osisaf/met.no/"
                    "reprocessed/ice/conc_v2p0_{0}_agg?"
                    "xc[0:1:431],"
                    "yc[0:1:431],"
                    "lat[0:1:431][0:1:431],"
                    "lon[0:1:431][0:1:431],"
                    "time[{1}:1:{1}],"
                    "ice_conc[{1}:1:{1}][0:1:431][0:1:431]".format(
                        hs,
                        (el - osi450_start).days,
                    ))
                da = da_osi450.resample(time='1D').mean()
            else:
                da_osi430b = xr.open_dataarray(
                    "https://thredds.met.no/thredds/dodsC/osisaf/met.no/"
                    "reprocessed/ice/conc_crb_{0}_agg?"
                    "xc[0:1:431],"
                    "yc[0:1:431],"
                    "lat[0:1:431][0:1:431],"
                    "lon[0:1:431][0:1:431],"
                    "time[{1}:1:{1}],"
                    "ice_conc[{1}:1:{1}][0:1:431][0:1:431]".format(
                        hs,
                        (el - osi430b_start).days,
                    ))
                da = da_osi430b.resample(time='1D').mean()
                #da_osi430b.lat.values = da_osi450.lat.values
                #da = xr.concat([da_osi450, da_osi430b], dim='time')

            da /= 100.  # Convert from SIC % to fraction

            dates = [pd.Timestamp(date) for date in da.time.values]
            if len(dates) > 1:
                logging.warning("Multiple dates, but not right: {}".
                                format(pformat(dates)))
                break

            for date in dates:
                # Grab mask
                mask = self._mask_dict[date.month]

                # Set outside mask to zero
                da.loc[date].data[~mask] = 0.

                # Grab polar hole
                polarhole_mask = self._masks.get_polarhole_mask(date)
                skip_interp = True if not polarhole_mask else False

                # Interpolate polar hole
                if not skip_interp:
                    xx, yy = np.meshgrid(np.arange(432), np.arange(432))

                    valid = ~polarhole_mask

                    x = xx[valid]
                    y = yy[valid]

                    x_interp = xx[polarhole_mask]
                    y_interp = yy[polarhole_mask]

                    values = da.loc[date].data[valid]

                    interp_vals = interpolate.griddata(
                        (x, y), values, (x_interp, y_interp), method='linear')
                    interpolated_array = da.loc[date].data.copy()
                    interpolated_array[polarhole_mask] = interp_vals
                    da.loc[date].data = interpolated_array


                da.to_netcdf(fpath)


if __name__ == "__main__":
    sic = SICDownloader(
        dates=dt.date(2020, 1, 1)
    )
    sic.download()
