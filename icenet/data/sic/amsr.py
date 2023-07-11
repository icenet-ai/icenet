import copy
import logging
import os

import datetime as dt
import numpy as np
import pandas as pd
import xarray as xr

from icenet.data.cli import download_args
from icenet.data.producers import Downloader
from icenet.data.sic.mask import Masks
from icenet.utils import Hemisphere, run_command
from icenet.data.sic.utils import SIC_HEMI_STR, DaskWrapper


var_remove_list = ["polar_stereographic"]


class AMSRDownloader(Downloader):
    """Downloads AMSR2 SIC data from 2012-present using HTTP.

    The data can come from yearly zips, or individual files

    The query URLs were obtained from the following site:
        - https://seaice.uni-bremen.de/data/amsr2/asi_daygrid_swath/
        - n3125/ or n6250/ for lower or higher resolutions respectively

    :param chunk_size:
    :param dates:
    :param delete_tempfiles:
    :param download:
    :param dtype:
    """
    def __init__(self,
                 *args,
                 chunk_size: int = 10,
                 dates: object = (),
                 delete_tempfiles: bool = True,
                 download: bool = True,
                 dtype: object = np.float32,
                 **kwargs):
        super().__init__(*args, identifier="amsr2", **kwargs)

        self._chunk_size = chunk_size
        self._dates = dates
        self._delete = delete_tempfiles
        self._download = download
        self._dtype = dtype
        self._masks = Masks(north=self.north, south=self.south)

        self._mask_dict = {
            month: self._masks.get_active_cell_mask(month)
            for month in np.arange(1, 12+1)
        }

    def download(self):
        """

        """
        hs = SIC_HEMI_STR[self.hemisphere_str[0]][0]
        data_files = []
        var = "siconca"

        logging.info(
            "Not downloading AMSR SIC files, (re)processing NC files in "
            "existence already" if not self._download else
            "Downloading SIC datafiles to .temp intermediates...")

        cache = {}
        amsr2_start = dt.date(2012, 7, 2)

        dt_arr = list(reversed(sorted(copy.copy(self._dates))))

        # Filtering dates based on existing data (SAME AS OSISAF)
        filter_years = sorted(set([d.year for d in dt_arr]))
        extant_paths = [
            os.path.join(self.get_data_var_folder(var),
                         "{}.nc".format(filter_ds))
            for filter_ds in filter_years
        ]
        extant_paths = [df for df in extant_paths if os.path.exists(df)]

        if len(extant_paths) > 0:
            extant_ds = xr.open_mfdataset(extant_paths)
            exclude_dates = pd.to_datetime(extant_ds.time.values)
            logging.info("Excluding {} dates already existing from {} dates "
                         "requested.".format(len(exclude_dates), len(dt_arr)))

            dt_arr = sorted(list(set(dt_arr).difference(exclude_dates)))
            dt_arr.reverse()

            # We won't hold onto an active dataset during network I/O
            extant_ds.close()
        # End filtering

        while len(dt_arr):
            el = dt_arr.pop()

            date_str = el.strftime("%Y_%m_%d")
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
                    logging.debug("{} does not exist".format(nc_path))
                continue
            else:
                if not os.path.isdir(os.path.dirname(nc_path)):
                    os.makedirs(os.path.dirname(nc_path), exist_ok=True)

                if os.path.exists(nc_path):
                    logging.debug("{} file exists, skipping".format(date_str))
                    continue

                # TODO: check for file existence
                # TODO: download nc_file

                logging.debug("Downloaded {}".format(nc_file))
                data_files.append(nc_file)

        logging.debug("Files being processed: {}".format(data_files))

        if len(data_files):
            ds = xr.open_mfdataset(data_files,
                                   combine="nested",
                                   concat_dim="time",
                                   data_vars=["ice_conc"],
                                   drop_variables=var_remove_list,
                                   engine="netcdf4",
                                   chunks=dict(time=self._chunk_size,),
                                   parallel=True)

            logging.debug("Processing out extraneous data")

            da = ds.resample(time="1D").mean().z

            da /= 100.  # Convert from SIC % to fraction

            # TODO: validate, are we to be applying the OSISAF mask?
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

            for year, year_da in da.groupby(group_by):
                req_date = pd.to_datetime(year_da.time.values[0])

                year_path = os.path.join(
                    var_folder, "{}.nc".format(getattr(req_date, "year")))
                old_year_path = os.path.join(
                    var_folder, "old.{}.nc".format(getattr(req_date, "year")))

                if os.path.exists(year_path):
                    logging.info("Existing file needs concatenating: {} -> {}".
                                 format(year_path, old_year_path))
                    os.rename(year_path, old_year_path)
                    old_da = xr.open_dataarray(old_year_path)
                    year_da = year_da.drop_sel(time=old_da.time,
                                               errors="ignore")
                    year_da = xr.concat([old_da, year_da],
                                        dim="time").sortby("time")
                    old_da.close()
                    os.unlink(old_year_path)

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
                               chunks=dict(time=self._chunk_size, ),
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
    args = download_args(var_specs=False,
                         workers=True,
                         extra_args=[
                            (("-u", "--use-dask"),
                             dict(action="store_true", default=False)),
                            (("-c", "--sic-chunking-size"),
                             dict(type=int, default=10)),
                            (("-dt", "--dask-timeouts"),
                             dict(type=int, default=120)),
                            (("-dp", "--dask-port"),
                             dict(type=int, default=8888))
                         ])

    logging.info("AMSR-SIC Data Downloading")
    sic = AMSRDownloader(
        chunk_size=args.sic_chunking_size,
        dates=[pd.to_datetime(date).date() for date in
               pd.date_range(args.start_date, args.end_date, freq="D")],
        delete_tempfiles=args.delete,
        north=args.hemisphere == "north",
        south=args.hemisphere == "south",
    )
    if args.use_dask:
        logging.warning("Attempting to use dask client for SIC processing")
        dw = DaskWrapper(workers=args.workers)
        dw.dask_process(method=sic.download)
    else:
        sic.download()
