import datetime
import logging
import os
import sys

from itertools import product

import ecmwfapi
import numpy as np
import pandas as pd
import xarray as xr

from icenet.data.cli import download_args
from icenet.data.interfaces.downloader import ClimateDownloader
from icenet.data.interfaces.utils import batch_requested_dates

"""

"""


class HRESDownloader(ClimateDownloader):
    """Climate downloader to provide CMIP6 reanalysis data from ESGF APIs

    :param identifier: how to identify this dataset

    """

    PARAM_TABLE = 128

    # Background on the use of forecast and observational data
    # https://confluence.ecmwf.int/pages/viewpage.action?pageId=85402030
    # https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation#ERA5:datadocumentation-Dateandtimespecification
    HRES_PARAMS = {
        "siconca":      (31, "siconc"),     # sea_ice_area_fraction (SEAS5)
        "tos":          (34, "sst"),    # sea surface temperature (actually
                                        # sst?)
        "zg":           (129, "z"),     # geopotential
        "ta":           (130, "t"),     # air_temperature (t)
        "hus":          (133, "q"),     # specific_humidity
        "psl":          (134, "sp"),    # surface_pressure
        "uas":          (165, "u10"),   # 10m_u_component_of_wind
        "vas":          (166, "v10"),   # 10m_v_component_of_wind
        "tas":          (167, "t2m"),   # 2m_temperature (t2m)
        # https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation#ERA5:datadocumentation-Meanrates/fluxesandaccumulations
        # https://apps.ecmwf.int/codes/grib/param-db/?id=175
        # https://confluence.ecmwf.int/pages/viewpage.action?pageId=197702790
        #
        # Mean rate/flux parameters in ERA5 (e.g. Table 4 for surface and
        # single levels) provide similar information to accumulations (e.g.
        # Table 3 for surface and single levels), except they are expressed as
        # temporal means, over the same processing periods, and so have units
        # of "per second".
        "rlds":         (175, "strd"),
        "rsds":         (169, "ssrd"),

        # plev  129.128 / 130.128 / 133.128
        # sfc   31.128 / 34.128 / 134.128 /
        #       165.128 / 166.128 / 167.128 / 169.128 / 177.128

        # ORAS5 variables in param-db (need to consider depth)
        #"thetao":       (151129, "thetao"),
        #"so":           (151130, "so"),
        # Better matches than equivalent X / Y parameters in param-db
        #"uo":           (151131, "uo"),
        #"vo":           (151132, "vo"),
    }

    # https://confluence.ecmwf.int/display/UDOC/Keywords+in+MARS+and+Dissemination+requests
    MARS_TEMPLATE = """
retrieve,
  class=od,
  date={date},
  expver=1,
  levtype={levtype},
  {levlist}param={params},
  step={step},
  stream=oper,
  time=12:00:00,
  type=fc,
  area={area},
  grid=0.25/0.25,
  target="{target}",
  format=netcdf
    """

    def __init__(self,
                 *args,
                 identifier: str = "mars.hres",
                 **kwargs):
        super().__init__(*args,
                         identifier=identifier,
                         **kwargs)

        self._server = ecmwfapi.ECMWFService("mars")

    def _single_download(self,
                         var_names: object,
                         pressures: object,
                         req_dates: object):
        """

        :param var_names:
        :param pressures:
        :param req_dates:
        :return:
        """

        for dt in req_dates:
            assert dt.year == req_dates[0].year

        downloads = []
        levtype = "plev" if pressures else "sfc"

        for req_batch in batch_requested_dates(req_dates, attribute="month"):
            req_batch = sorted(req_batch)
            request_month = req_batch[0].strftime("%Y%m")

            logging.info("Downloading month file {}".format(request_month))

            if req_batch[-1] - datetime.datetime.utcnow().date() == \
                    datetime.timedelta(days=-1):
                logging.warning("Not allowing partial requests at present, "
                                "removing {}".format(req_batch[-1]))
                req_batch = req_batch[:-1]

            request_target = "{}.{}.{}.nc".format(
                self.hemisphere_str[0], levtype, request_month)

            request = HRESDownloader.MARS_TEMPLATE.format(
                area="/".join([str(s) for s in self.hemisphere_loc]),
                date="/".join([el.strftime("%Y%m%d") for el in req_batch]),
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
                # TODO: too big - step="/".join([str(i) for i in range(24)]),
                step=0,
            )

            if not os.path.exists(request_target):
                logging.debug("MARS REQUEST: \n{}\n".format(request))

                try:
                    self._server.execute(request, request_target)
                except ecmwfapi.api.APIException:
                    logging.exception("Could not complete ECMWF request: {}")
                else:
                    downloads.append(request_target)
            else:
                logging.debug("Already have {}".format(request_target))
                downloads.append(request_target)

        logging.debug("Files downloaded: {}".format(downloads))

        ds = xr.open_mfdataset(downloads)
        ds = ds.resample(time='1D').mean()

        for var_name, pressure in product(var_names, pressures.split('/')
                                          if pressures else [None]):
            var = var_name if not pressure else \
                "{}{}".format(var_name, pressure)

            da = getattr(ds,
                         HRESDownloader.HRES_PARAMS[var_name][1])

            if pressure:
                da = da.sel(level=int(pressure))

            self.save_temporal_files(var, da)

        ds.close()

        if self.delete:
            for downloaded_file in downloads:
                if os.path.exists(downloaded_file):
                    logging.info("Removing {}".format(downloaded_file))
                    os.unlink(downloaded_file)

    def download(self):
        """

        """
        logging.info("Building request(s), downloading and daily averaging "
                     "from {} API".format(self.identifier.upper()))

        sfc_vars = [var for idx, var in enumerate(self.var_names)
                    if not self.pressure_levels[idx]]
        plev_vars = [var for idx, var in enumerate(self.var_names)
                     if self.pressure_levels[idx]]
        pressures = "/".join([str(s) for s in sorted(set(
            [p for ps in self.pressure_levels if ps for p in ps]))])

        # req_dates = self.filter_dates_on_data()

        dates_per_request = \
            batch_requested_dates(self._dates,
                                  attribute=self._group_dates_by)

        for req_batch in dates_per_request:
            if len(sfc_vars) > 0:
                self._single_download(sfc_vars, None, req_batch)

            if len(plev_vars) > 0:
                self._single_download(plev_vars, pressures, req_batch)

        logging.info("{} daily files downloaded".
                     format(len(self._files_downloaded)))

    def additional_regrid_processing(self,
                                     datafile: str,
                                     cube_ease: object):
        """

        :param datafile:
        :param cube_ease:
        """
        (datafile_path, datafile_name) = os.path.split(datafile)
        var_name = datafile_path.split(os.sep)[self._var_name_idx]

        if var_name == 'tos':
            # Overwrite maksed values with zeros
            logging.debug("MARS additional regrid: {}".format(var_name))
            cube_ease.data[cube_ease.data.mask] = 0.
            cube_ease.data[:, self._masks.get_land_mask()] = 0.
            cube_ease.data = cube_ease.data.data

        if var_name in ['rlds', 'rsds']:
            # FIXME: We're taking the mean across the hourly samples for the
            #  day in fc which needs to be comparative with the analysis product
            #  from ERA5. My interpretation is that this should be /24, but of
            #  course it doesn't work like that thanks to orbital rotation.
            #  We need to verify the exact mechanism for converting forecast
            #  values to reanalysis equivalents, but this rudimentary divisor
            #  should work in the meantime
            #
            #  FIXME FIXME FIXME
            cube_ease /= 12.

        if var_name.startswith("zg"):
            # https://apps.ecmwf.int/codes/grib/param-db/?id=129
            #
            # We want the geopotential height as per ERA5
            cube_ease /= 9.80665


def main():
    args = download_args()

    logging.info("ERA5 HRES Data Downloading")
    hres = HRESDownloader(
        var_names=args.vars,
        pressure_levels=args.levels,
        dates=[pd.to_datetime(date).date() for date in
               pd.date_range(args.start_date, args.end_date, freq="D")],
        delete_tempfiles=args.delete,
        north=args.hemisphere == "north",
        south=args.hemisphere == "south"
    )
    hres.download()
    hres.regrid()
    hres.rotate_wind_data()

