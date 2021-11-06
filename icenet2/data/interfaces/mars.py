import datetime as dt
import logging
import os
import tempfile

from itertools import product

import ecmwfapi
import pandas as pd
import xarray as xr

from icenet2.data.cli import download_args
from icenet2.data.interfaces.downloader import ClimateDownloader


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
        "rsds":         (169, "ssrd"),  # surface_downwelling_shortwave_flux_in_
                                        # air
        "rlds":         (177, "str"),   # surface_net_upward_longwave_flux

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
  step=0,
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
        return self.dates

    def _single_download(self, var_names, pressures, req_date):
        levtype = "plev" if pressures else "sfc"

        request_date = req_date.strftime("%Y%m%d")
        request_target = "nh.{}.{}.nc".format(levtype, request_date)

        request = HRESDownloader.MARS_TEMPLATE.format(
            area="/".join([str(s) for s in self.hemisphere_loc]),
            date=request_date,
            levtype=levtype,
            levlist="levelist={},\n  ".format(pressures) if pressures else "",
            params="/".join(
                ["{}.{}".format(
                    HRESDownloader.HRES_PARAMS[v][0],
                    HRESDownloader.PARAM_TABLE)
                 for v in var_names]),
            target=request_target,
        )

        logging.debug("MARS REQUEST: \n{}\n".format(request))

        # FIXME: Hateful duplication to avoid unnecessary requests
        missing = False
        for var_name, pressure in product(var_names, pressures.split('/')
                if pressures else [None]):
            # TODO: refactor, this is common pattern CDS as well,
            #  slightly cleaned up in this implementation
            var = var_name if not pressure else \
                "{}{}".format(var_name, pressure)
            var_folder = os.path.join(self.get_data_var_folder(var),
                                      str(req_date.year))

            date_str = req_date.strftime("%Y_%m_%d")
            regridded_name = os.path.join(var_folder,
                                          "{}_{}.nc".
                                          format(var, date_str))

            if not os.path.exists(regridded_name):
                missing = True
                break

        if not missing:
            logging.info("We have all the files we need for {}".
                         format(req_date))
            return

        with tempfile.TemporaryDirectory(dir=".") as tmpdir:
            tmpfile = os.path.join(tmpdir, "{}.nc".format(levtype))

            if not os.path.exists(tmpfile):
                self._server.execute(request, tmpfile)

            ds = xr.open_dataset(tmpfile)

            for var_name, pressure in product(var_names, pressures.split('/')
                                              if pressures else [None]):
                # TODO: refactor, this is common pattern CDS as well,
                #  slightly cleaned up in this implementation
                var = var_name if not pressure else \
                    "{}{}".format(var_name, pressure)
                var_folder = os.path.join(self.get_data_var_folder(var),
                                          str(req_date.year))

                # For the year component - 365 * 50 is a lot of files ;)
                os.makedirs(var_folder, exist_ok=True)

                date_str = req_date.strftime("%Y_%m_%d")
                daily_path = os.path.join(var_folder,
                                          "latlon_{}_{}.nc".
                                          format(var, date_str))
                regridded_name = os.path.join(var_folder,
                                              "{}_{}.nc".
                                              format(var, date_str))

                if not os.path.exists(regridded_name):
                    if not os.path.exists(daily_path):
                        da = getattr(ds,
                                     HRESDownloader.HRES_PARAMS[var_name][1])

                        if pressure:
                            da = da.sel(level=int(pressure))

                        # Just to make sure
                        da_daily = da.sel(time=slice(pd.to_datetime(req_date)))

                        logging.debug("Saving new daily file: {}".
                                      format(daily_path))
                        da_daily.to_netcdf(daily_path)

                    self._files_downloaded.append(daily_path)
                else:
                    logging.info("{} already exists".format(regridded_name))

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

        for req_date in dates_per_request:
            self._single_download(sfc_vars, None, req_date)
            self._single_download(plev_vars, pressures, req_date)

        logging.info("{} daily files downloaded".
                     format(len(self._files_downloaded)))

    def additional_regrid_processing(self, datafile, cube_ease):
        pass


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
        north=args.hemisphere == "north",
        south=args.hemisphere == "south"
    )
    hres.download()
    hres.regrid()
    hres.rotate_wind_data()


if __name__ == "__main__":
    main()
