import datetime
import logging
import os

from itertools import product

import ecmwfapi
import numpy as np
import pandas as pd
import xarray as xr

from icenet2.data.cli import download_args
from icenet2.data.interfaces.downloader import ClimateDownloader
from icenet2.data.interfaces.utils import \
    batch_requested_dates, get_daily_filenames


class HRESDownloader(ClimateDownloader):
    PARAM_TABLE = 128
    HRES_PARAMS = {
        "siconca":      (31, "siconc"),     # sea_ice_area_fraction
        "tos":          (34, "sst"),    # sea surface temperature (actually
                                        # sst?)
        "zg":           (129, "z"),     # geopotential
        "ta":           (130, "t"),     # air_temperature (t)
        "hus":          (133, "q"),     # specific_humidity
        "psl":          (134, "sp"),    # surface_pressure
        "uas":          (165, "u10"),   # 10m_u_component_of_wind
        "vas":          (166, "v10"),   # 10m_v_component_of_wind
        "tas":          (167, "t2m"),   # 2m_temperature (t2m)
        # TODO: Zero at forecast init, we set the step to 12 for values to
        #  process currently, though this needs review.
        "rlds":         (175, "strd"),
        "rsds":         (169, "ssrd"),

        # plev  129.128 / 130.128 / 133.128
        # sfc   31.128 / 34.128 / 134.128 /
        #       165.128 / 166.128 / 167.128 / 169.128 / 177.128
    }

    MARS_TEMPLATE = """
retrieve,
  class=od,
  date={date},
  expver=1,
  levtype={levtype},
  {levlist}param={params},
  step=0/6/12/18,
  stream=oper,
  time=00:00:00,
  type=fc,
  area={area},
  grid=0.25/0.25,
  target="{target}",
  format=netcdf
    """

    def __init__(self,
                 *args,
                 identifier="mars.hres",
                 **kwargs):
        super().__init__(*args,
                         identifier=identifier,
                         **kwargs)

        self._server = ecmwfapi.ECMWFService("mars")

    def _get_dates_for_request(self):
        return batch_requested_dates(self._dates, attribute="month")

    def _single_download(self, var_names, pressures, req_dates):
        levtype = "plev" if pressures else "sfc"

        for dt in req_dates:
            assert dt.year == req_dates[0].year
            assert dt.month == req_dates[0].month

        request_month = req_dates[0].strftime("%Y%m")
        request_target = "{}.{}.{}.nc".format(
            self.hemisphere_str[0], levtype, request_month)
        partial_request_target = "partial.{}".format(request_target)

        download_dates = []

        for var_name, pressure in product(var_names, pressures.split('/')
                                          if pressures else [None]):
            var = var_name if not pressure else \
                "{}{}".format(var_name, pressure)
            var_folder = os.path.join(self.get_data_var_folder(var),
                                      str(req_dates[0].year))

            for destination_date in req_dates:
                daily_path, regridded_name = get_daily_filenames(
                    var_folder, var, destination_date.strftime("%Y_%m_%d"))

                if not os.path.exists(daily_path) \
                        and not os.path.exists(regridded_name):
                    if destination_date not in download_dates:
                        download_dates.append(destination_date)
                elif not os.path.exists(regridded_name):
                    self._files_downloaded.append(daily_path)

        download_dates = sorted(list(set(download_dates)))

        if not len(download_dates):
            logging.info("We have all the files we need from MARS API")
            return

        downloaded_files = []

        if download_dates[-1].date() - datetime.datetime.now().date() == \
            datetime.timedelta(days=-1):
            partial_request = HRESDownloader.MARS_TEMPLATE.format(
                area="/".join([str(s) for s in self.hemisphere_loc]),
                date=download_dates[-1].strftime("%Y%m%d"),
                levtype=levtype,
                levlist="levelist={},\n  ".format(pressures) if pressures else "",
                params="/".join(
                    ["{}.{}".format(
                        HRESDownloader.HRES_PARAMS[v][0],
                        HRESDownloader.PARAM_TABLE)
                        for v in var_names]),
                target=partial_request_target,
                # We are only allowed date prior to -24 hours ago, dynamically
                # retrieve if date is today
                step="/".join([str(i) for i in
                               range(datetime.datetime.now().hour)]),
            )

            logging.debug("PART STEP MARS REQUEST: \n{}\n".
                          format(partial_request))

            if not os.path.exists(partial_request_target):
                self._server.execute(partial_request, partial_request_target)
                downloaded_files.append(partial_request_target)
                partial_datetime = download_dates.pop()
                logging.warning("Removed partial date {}".
                                format(partial_datetime.strftime("%Y%m%d")))

        if len(download_dates) > 0:
            request = HRESDownloader.MARS_TEMPLATE.format(
                area="/".join([str(s) for s in self.hemisphere_loc]),
                date="/".join([el.strftime("%Y%m%d") for el in download_dates]),
                levtype=levtype,
                levlist="levelist={},\n  ".format(pressures) if pressures else "",
                params="/".join(
                    ["{}.{}".format(
                        HRESDownloader.HRES_PARAMS[v][0],
                        HRESDownloader.PARAM_TABLE)
                     for v in var_names]),
                target=request_target,
                # We are only allowed date prior to -24 hours ago, dynamically
                # retrieve if date is today
                step="/".join([str(i) for i in range(24)]),
            )

            logging.debug("MARS REQUEST: \n{}\n".format(request))

            if not os.path.exists(request_target):
                self._server.execute(request, request_target)
                downloaded_files.append(request_target)

        ds = xr.open_mfdataset(downloaded_files)

        ds = ds.resample(time='1D').reduce(np.mean)

        for day in ds.time.values:
            date_str = pd.to_datetime(day).strftime("%Y_%m_%d")

            for var_name, pressure in product(var_names, pressures.split('/')
                                              if pressures else [None]):
                var = var_name if not pressure else \
                    "{}{}".format(var_name, pressure)
                var_folder = os.path.join(self.get_data_var_folder(var),
                                          str(pd.to_datetime(day).year))

                # For the year component - 365 * 50 is a lot of files ;)
                os.makedirs(var_folder, exist_ok=True)

                daily_path, _ = get_daily_filenames(var_folder, var, date_str)

                da = getattr(ds,
                             HRESDownloader.HRES_PARAMS[var_name][1])

                if pressure:
                    da = da.sel(level=int(pressure))

                # Just to make sure
                da_daily = da.sel(time=slice(
                    pd.to_datetime(day), pd.to_datetime(day)))

                logging.info("Saving new daily file: {}".format(daily_path))
                da_daily.to_netcdf(daily_path)

                if daily_path not in self._files_downloaded:
                    self._files_downloaded.append(daily_path)

        ds.close()

        if self.delete:
            for downloaded_file in [request_target, partial_request_target]:
                if os.path.exists(downloaded_file):
                    logging.info("Removing {}".format(downloaded_file))
                    os.unlink(downloaded_file)
            
    def download(self):
        logging.info("Building request(s), downloading and daily averaging "
                     "from {} API".format(self.identifier.upper()))

        sfc_vars = [var for idx, var in enumerate(self.var_names)
                    if not self.pressure_levels[idx]]
        plev_vars = [var for idx, var in enumerate(self.var_names)
                     if self.pressure_levels[idx]]
        pressures = "/".join([str(s) for s in sorted(set(
            [p for ps in self.pressure_levels if ps for p in ps]))])

        dates_per_request = self._get_dates_for_request()

        for req_batch in dates_per_request:
            self._single_download(sfc_vars, None, req_batch)
            self._single_download(plev_vars, pressures, req_batch)

        logging.info("{} daily files downloaded".
                     format(len(self._files_downloaded)))

    def additional_regrid_processing(self, datafile, cube_ease):
        (datafile_path, datafile_name) = os.path.split(datafile)
        var_name = datafile_path.split(os.sep)[-2]

        if var_name == 'tos':
            # Overwrite maksed values with zeros
            logging.debug("MARS additional regrid: {}".format(var_name))
            cube_ease.data[cube_ease.data.mask] = 0.
            cube_ease.data[:, self._masks.get_land_mask()] = 0.
            cube_ease.data = cube_ease.data.data


def main():
    args = download_args()

    logging.info("ERA5 HRES Data Downloading")
    hres = HRESDownloader(
        var_names=["tas", "ta", "tos", "psl", "zg", "hus", "rlds",
                   "rsds", "uas", "vas", "siconca"],
        pressure_levels=[None, [500], None, None, [250, 500], [1000],
                         None, None, None, None, None],
        dates=[pd.to_datetime(date).date() for date in
               pd.date_range(args.start_date, args.end_date,
                             freq="D")],
        delete_tempfiles=args.delete,
        north=args.hemisphere == "north",
        south=args.hemisphere == "south"
    )
    hres.download()
    hres.regrid()
    hres.rotate_wind_data()


if __name__ == "__main__":
    main()
