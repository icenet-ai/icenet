import copy
import fnmatch
import ftplib
import logging
import os

import datetime as dt
from ftplib import FTP

import numpy as np
import pandas as pd
import xarray as xr

from icenet.data.cli import download_args
from icenet.data.producers import Downloader
from icenet.data.sic.mask import Masks
from icenet.utils import Hemisphere
from icenet.data.sic.utils import SIC_HEMI_STR

"""

"""

invalid_sic_days = {
    Hemisphere.NORTH: [
        *[d.date() for d in
          pd.date_range(dt.date(1979, 5, 21), dt.date(1979, 6, 4))],
        *[d.date() for d in
          pd.date_range(dt.date(1979, 6, 10), dt.date(1979, 6, 26))],
        dt.date(1979, 7, 1),
        *[d.date() for d in
          pd.date_range(dt.date(1979, 7, 24), dt.date(1979, 7, 28))],
        *[d.date() for d in
          pd.date_range(dt.date(1980, 1, 4), dt.date(1980, 1, 10))],
        *[d.date() for d in
          pd.date_range(dt.date(1980, 2, 27), dt.date(1980, 3, 4))],
        *[d.date() for d in
          pd.date_range(dt.date(1980, 3, 16), dt.date(1980, 3, 22))],
        *[d.date() for d in
          pd.date_range(dt.date(1980, 4, 9), dt.date(1980, 4, 15))],
        *[d.date() for d in
          pd.date_range(dt.date(1981, 2, 27), dt.date(1981, 3, 5))],
        *[d.date() for d in
          pd.date_range(dt.date(1984, 8, 12), dt.date(1984, 8, 24))],
        dt.date(1984, 9, 14),
        *[d.date() for d in
          pd.date_range(dt.date(1985, 9, 22), dt.date(1985, 9, 28))],
        *[d.date() for d in
          pd.date_range(dt.date(1986, 3, 29), dt.date(1986, 7, 1))],
        *[d.date() for d in
          pd.date_range(dt.date(1987, 1, 3), dt.date(1987, 1, 19))],
        *[d.date() for d in
          pd.date_range(dt.date(1987, 1, 29), dt.date(1987, 2, 2))],
        dt.date(1987, 2, 23),
        *[d.date() for d in
          pd.date_range(dt.date(1987, 2, 26), dt.date(1987, 3, 2))],
        dt.date(1987, 3, 13),
        *[d.date() for d in
          pd.date_range(dt.date(1987, 3, 22), dt.date(1987, 3, 26))],
        *[d.date() for d in
          pd.date_range(dt.date(1987, 4, 3), dt.date(1987, 4, 17))],
        *[d.date() for d in
          pd.date_range(dt.date(1987, 12, 1), dt.date(1988, 1, 12))],
        dt.date(1989, 1, 3),
        *[d.date() for d in
          pd.date_range(dt.date(1990, 12, 21), dt.date(1990, 12, 26))],
        dt.date(1979, 5, 28),
        dt.date(1979, 5, 30),
        dt.date(1979, 6, 1),
        dt.date(1979, 6, 3),
        dt.date(1979, 6, 11),
        dt.date(1979, 6, 13),
        dt.date(1979, 6, 15),
        dt.date(1979, 6, 17),
        dt.date(1979, 6, 19),
        dt.date(1979, 6, 21),
        dt.date(1979, 6, 23),
        dt.date(1979, 6, 25),
        dt.date(1979, 7, 1),
        dt.date(1979, 7, 25),
        dt.date(1979, 7, 27),
        dt.date(1984, 9, 14),
        dt.date(1987, 1, 16),
        dt.date(1987, 1, 18),
        dt.date(1987, 1, 30),
        dt.date(1987, 2, 1),
        dt.date(1987, 2, 23),
        dt.date(1987, 2, 27),
        dt.date(1987, 3, 1),
        dt.date(1987, 3, 13),
        dt.date(1987, 3, 23),
        dt.date(1987, 3, 25),
        dt.date(1987, 4, 4),
        dt.date(1987, 4, 6),
        dt.date(1987, 4, 10),
        dt.date(1987, 4, 12),
        dt.date(1987, 4, 14),
        dt.date(1987, 4, 16),
        dt.date(1987, 4, 4),
        dt.date(1990, 1, 26)
    ],
    Hemisphere.SOUTH: [
        dt.date(1979, 2, 5),
        dt.date(1979, 2, 25),
        dt.date(1979, 3, 23),
        *[d.date() for d in
          pd.date_range(dt.date(1979, 3, 26), dt.date(1979, 3, 30))],
        dt.date(1979, 4, 12),
        dt.date(1979, 5, 16),
        *[d.date() for d in
          pd.date_range(dt.date(1979, 5, 21), dt.date(1979, 5, 27))],
        *[d.date() for d in
          pd.date_range(dt.date(1979, 7, 10), dt.date(1979, 7, 18))],
        dt.date(1979, 8, 10),
        dt.date(1979, 9, 3),
        *[d.date() for d in
          pd.date_range(dt.date(1980, 1, 4), dt.date(1980, 1, 10))],
        dt.date(1980, 2, 16),
        *[d.date() for d in
          pd.date_range(dt.date(1980, 2, 27), dt.date(1980, 3, 4))],
        *[d.date() for d in
          pd.date_range(dt.date(1980, 3, 14), dt.date(1980, 3, 22))],
        dt.date(1980, 3, 31),
        *[d.date() for d in
          pd.date_range(dt.date(1980, 4, 9), dt.date(1980, 4, 15))],
        dt.date(1980, 4, 22),
        *[d.date() for d in
          pd.date_range(dt.date(1981, 2, 27), dt.date(1981, 3, 5))],
        dt.date(1981, 6, 10),
        *[d.date() for d in
          pd.date_range(dt.date(1981, 8, 3), dt.date(1982, 8, 9))],
        dt.date(1982, 8, 6),
        *[d.date() for d in
          pd.date_range(dt.date(1983, 7, 7), dt.date(1983, 7, 11))],
        dt.date(1983, 7, 22),
        dt.date(1984, 6, 12),
        *[d.date() for d in
          pd.date_range(dt.date(1984, 8, 12), dt.date(1984, 8, 24))],
        *[d.date() for d in
          pd.date_range(dt.date(1984, 9, 13), dt.date(1984, 9, 17))],
        *[d.date() for d in
          pd.date_range(dt.date(1984, 10, 3), dt.date(1984, 10, 9))],
        *[d.date() for d in
          pd.date_range(dt.date(1984, 11, 18), dt.date(1984, 11, 22))],
        dt.date(1985, 7, 23),
        *[d.date() for d in
          pd.date_range(dt.date(1985, 9, 22), dt.date(1985, 9, 28))],
        *[d.date() for d in
          pd.date_range(dt.date(1986, 3, 29), dt.date(1986, 11, 2))],
        *[d.date() for d in
          pd.date_range(dt.date(1987, 1, 3), dt.date(1987, 1, 15))],
        *[d.date() for d in
          pd.date_range(dt.date(1987, 12, 1), dt.date(1988, 1, 12))],
        dt.date(1990, 8, 14),
        dt.date(1990, 8, 15),
        dt.date(1990, 8, 24),
        *[d.date() for d in
          pd.date_range(dt.date(1990, 12, 22), dt.date(1990, 12, 26))],
        dt.date(1979, 2, 5),
        dt.date(1979, 2, 25),
        dt.date(1979, 3, 23),
        dt.date(1979, 3, 27),
        dt.date(1979, 3, 29),
        dt.date(1979, 4, 12),
        dt.date(1979, 5, 16),
        dt.date(1979, 7, 11),
        dt.date(1979, 7, 13),
        dt.date(1979, 7, 15),
        dt.date(1979, 7, 17),
        dt.date(1979, 8, 10),
        dt.date(1979, 9, 3),
        dt.date(1980, 2, 16),
        dt.date(1980, 3, 15),
        dt.date(1980, 3, 31),
        dt.date(1980, 4, 22),
        dt.date(1981, 6, 10),
        dt.date(1982, 8, 6),
        dt.date(1983, 7, 8),
        dt.date(1983, 7, 10),
        dt.date(1983, 7, 22),
        dt.date(1984, 6, 12),
        dt.date(1984, 9, 14),
        dt.date(1984, 9, 16),
        dt.date(1984, 10, 4),
        dt.date(1984, 10, 6),
        dt.date(1984, 10, 8),
        dt.date(1984, 11, 19),
        dt.date(1984, 11, 21),
        dt.date(1985, 7, 23),
        *pd.date_range(dt.date(1986, 7, 2), dt.date(1986, 11, 1)),
        dt.date(1990, 8, 14),
        dt.date(1990, 8, 15),
        dt.date(1990, 8, 24)
    ]
}

var_remove_list = ['time_bnds', 'raw_ice_conc_values', 'total_standard_error',
                   'smearing_standard_error', 'algorithm_standard_error',
                   'status_flag', 'Lambert_Azimuthal_Grid']


class SICDownloader(Downloader):
    """Downloads OSI-SAF SIC data from 1979-present using OpenDAP.

    The dataset comprises OSI-450 (1979-2015) and OSI-430-b (2016-ownards)
    Monthly averages are-computed on the server-side.
    This script can take about an hour to run.

    The query URLs were obtained from the following sites:
        - OSI-450 (1979-2016): https://thredds.met.no/thredds/dodsC/osisaf/
            met.no/reprocessed/ice/conc_v2p0_nh_agg.html
        - OSI-430-b (2016-present): https://thredds.met.no/thredds/dodsC/osisaf/
            met.no/reprocessed/ice/conc_crb_nh_agg.html

    :param additional_invalid_dates:
    :param dates:
    :param delete_tempfiles:
    :param download:
    :param dtype:
    """
    def __init__(self,
                 *args,
                 additional_invalid_dates: object = (),
                 dates: object = (),
                 delete_tempfiles: bool = True,
                 download: bool = True,
                 dtype: object = np.float32,
                 **kwargs):
        super().__init__(*args, identifier="osisaf", **kwargs)

        self._dates = dates
        self._delete = delete_tempfiles
        self._download = download
        self._dtype=dtype
        self._invalid_dates = invalid_sic_days[self.hemisphere] + \
            list(additional_invalid_dates)
        self._masks = Masks(north=self.north, south=self.south)

        self._mask_dict = {
            month: self._masks.get_active_cell_mask(month)
            for month in np.arange(1, 12+1)
        }

    def download(self):
        """

        """
        hs = SIC_HEMI_STR[self.hemisphere_str[0]]

        logging.info(
            "Not downloading SIC files, (re)processing NC files in "
            "existence already" if not self._download else
            "Downloading SIC datafiles to .temp intermediates...")

        ftp_osi450 = "/reprocessed/ice/conc/v2p0/{:04d}/{:02d}/"
        ftp_osi430b = "/reprocessed/ice/conc-cont-reproc/v2p0/{:04d}/{:02d}/"
        cache = {}
        osi430b_start = dt.date(2016, 1, 1)

        # TODO: filter based on existing data
        dt_arr = list(reversed(sorted(copy.copy(self._dates))))
        data_files = []
        ftp = None
        var = "siconca"

        while len(dt_arr):
            el = dt_arr.pop()

            if el in self._invalid_dates:
                logging.warning("Date {} is in invalid list".format(el))
                continue

            date_str = el.strftime("%Y_%m_%d")
            temp_path = os.path.join(
                self.get_data_var_folder(var, append=[str(el.year)]),
                "{}.temp".format(date_str))
            nc_path = os.path.join(
                self.get_data_var_folder(var, append=[str(el.year)]),
                "{}.nc".format(date_str))

            if not self._download:
                if os.path.exists(nc_path):
                    reproc_path = os.path.join(
                        self.get_data_var_folder(var,
                                                 append=[str(el.year)]),
                        "{}.reproc.nc".format(date_str))

                    logging.debug("{} exists, becoming {}".
                                  format(nc_path, reproc_path))
                    os.rename(nc_path, reproc_path)
                    data_files.append(reproc_path)
                else:
                    if os.path.exists(temp_path):
                        logging.info("Using existing {}".format(temp_path))
                        data_files.append(temp_path)
                    else:
                        logging.debug("{} does not exist".format(nc_path))
                continue
            else:
                if not os.path.isdir(os.path.dirname(temp_path)):
                    os.makedirs(os.path.dirname(temp_path), exist_ok=True)

                if os.path.exists(temp_path) or os.path.exists(nc_path):
                    logging.debug("{} file exists, skipping".format(date_str))
                    if not os.path.exists(nc_path):
                        data_files.append(temp_path)
                    continue

                if not ftp:
                    logging.info("FTP opening")
                    ftp = FTP('osisaf.met.no')
                    ftp.login()

                chdir_path = ftp_osi450 if el < osi430b_start else ftp_osi430b
                chdir_path = chdir_path.format(el.year, el.month)

                try:
                    ftp.cwd(chdir_path)

                    if chdir_path not in cache:
                        cache[chdir_path] = ftp.nlst()

                    cache_match = "ice_conc_{}_ease*_{:04d}{:02d}{:02d}*.nc".\
                        format(hs, el.year, el.month, el.day)
                    ftp_files = [el for el in cache[chdir_path]
                                 if fnmatch.fnmatch(el, cache_match)]

                    if len(ftp_files) > 1:
                        raise ValueError("More than a single file found: {}".
                                         format(ftp_files))
                    elif not len(ftp_files):
                        logging.warning("File is not available: {}".
                                        format(cache_match))
                        continue
                except ftplib.error_perm:
                    logging.warning("FTP error, possibly missing month chdir "
                                    "for {}".format(date_str))
                    continue

                with open(temp_path, "wb") as fh:
                    ftp.retrbinary("RETR {}".format(ftp_files[0]), fh.write)

                logging.debug("Downloaded {}".format(temp_path))
                data_files.append(temp_path)

        if ftp:
            ftp.quit()

        logging.debug("Files being processed: {}".format(data_files))

        if len(data_files):
            ds = xr.open_mfdataset(data_files,
                                   combine="nested",
                                   concat_dim="time",
                                   data_vars=["ice_conc"],
                                   drop_variables=var_remove_list,
                                   engine="netcdf4")

            logging.debug("Processing out extraneous data")

            ds = ds.drop_vars(var_remove_list, errors="ignore")
            da = ds.resample(time="1D").mean().ice_conc

            da = da.where(da < 9.9e+36, 0.)  # Missing values
            da /= 100.  # Convert from SIC % to fraction

            if 'lat' not in da.coords:
                raise RuntimeError("latitude missing, fix required that has "
                                   "been removed in this version")
                # TODO: ref another file if this is missing, but hopefully the
                #  coordinates will be projected from mfdataset
                # logging.warning("Adding lat vals to coords, as missing in "
                #                "this set: {}".format(file))
                # da.coords['lat'] = lat_vals

            # In experimenting, I don't think this is actually required
            for month, mask in self._mask_dict.items():
                da.loc[dict(time=(da['time.month'] == month))].values[:, ~mask] = 0.

            for date in da.time.values:
                day_da = da.sel(time=slice(date, date))

                if np.sum(np.isnan(day_da.data)) > 0:
                    logging.warning("NaNs detected, adding to invalid "
                                    "list: {}".format(date))
                    self._invalid_dates.append(pd.to_datetime(date))

            var_folder = self.get_data_var_folder(var)
            group_by = "time.year"

            # xr.concat([ds, ds2], dim="time").sortby("time")
            for year, year_da in da.groupby(group_by):
                req_date = pd.to_datetime(year_da.time.values[0])
                year_path = os.path.join(
                    var_folder, "{}.nc".format(
                        getattr(req_date, "year")))

                logging.info("Saving {}".format(year_path))
                year_da.compute()
                year_da.to_netcdf(year_path)

        self.missing_dates()

        if self._delete:
            for fpath in data_files:
                os.unlink(fpath)

    def missing_dates(self):
        """

        :return:
        """
        filenames = set([os.path.join(
            self.get_data_var_folder("siconca"),
            "{}.nc".format(el.strftime("%Y")))
            for el in self._dates])
        filenames = [f for f in filenames if os.path.exists(f)]

        logging.info("Opening for interpolation: {}".format(filenames))
        ds = xr.open_mfdataset(filenames,
                               combine="nested",
                               concat_dim="time",
                               parallel=True)
        return self._missing_dates(ds.ice_conc)

    def _missing_dates(self, da: object) -> object:
        """

        :param da:
        :return:
        """
        if pd.Timestamp(1979, 1, 2) in da.time.values \
                and dt.date(1979, 1, 1) in self._dates\
                and pd.Timestamp(1979, 1, 1) not in da.time.values:
            da_1979_01_01 = da.sel(
                time=[pd.Timestamp(1979, 1, 2)]).copy().assign_coords(
                {'time': [pd.Timestamp(1979, 1, 1)]})
            da = xr.concat([da, da_1979_01_01], dim='time')
            da = da.sortby('time')

        dates_obs = [pd.to_datetime(date).date() for date in da.time.values]
        dates_all = [pd.to_datetime(date).date() for date in
                     pd.date_range(min(self._dates), max(self._dates))]

        # Weirdly, we were getting future warnings for timestamps, but unsure
        # where from
        invalid_dates = [pd.to_datetime(d).date() for d in self._invalid_dates]
        missing_dates = [date for date in dates_all
                         if date not in dates_obs
                         or date in invalid_dates]

        logging.info("Processing {} missing dates".format(len(missing_dates)))

        missing_dates_path = os.path.join(
            self.get_data_var_folder("siconca"), "missing_days.csv")

        with open(missing_dates_path, "a") as fh:
            for date in missing_dates:
                # FIXME: slightly unusual format for Ymd dates
                fh.write(date.strftime("%Y,%m,%d\n"))

        logging.debug("Interpolating {} missing dates".
                      format(len(missing_dates)))

        for date in missing_dates:
            # TODO: test, but overcomes issue with reprocessing
            if pd.Timestamp(date) not in da.time.values:
                logging.info("Interpolating {}".format(date))
                da = xr.concat([da,
                                da.interp(time=pd.to_datetime(date))],
                               dim='time')

        logging.debug("Finished interpolation")

        da = da.sortby('time')
        da.data = np.array(da.data, dtype=self._dtype)

        for date in missing_dates:
            date_str = pd.to_datetime(date).strftime("%Y_%m_%d")
            fpath = os.path.join(
                self.get_data_var_folder(
                    "siconca", append=[str(pd.to_datetime(date).year)]),
                "missing.{}.nc".format(date_str))

            if not os.path.exists(fpath):
                day_da = da.sel(time=slice(date, date))
                mask = self._mask_dict[pd.to_datetime(date).month]

                day_da.data[0][~mask] = 0.

                logging.info("Writing missing date file {}".format(fpath))
                day_da.to_netcdf(fpath)

        return da


def main():
    args = download_args(var_specs=False)

    logging.info("OSASIF-SIC Data Downloading")
    sic = SICDownloader(
        dates=[pd.to_datetime(date).date() for date in
               pd.date_range(args.start_date, args.end_date, freq="D")],
        delete_tempfiles=args.delete,
        north=args.hemisphere == "north",
        south=args.hemisphere == "south",
    )
    sic.download()
