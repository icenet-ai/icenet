import configparser
import logging
import os
import time

import numpy as np
import pandas as pd
import xarray as xr

from icenet.data.cli import download_args
from icenet.data.interfaces.downloader import ClimateDownloader
from icenet.utils import run_command

"""
DATASET: global-reanalysis-phy-001-031-grepv2-daily
FTP ENDPOINT: ftp://my.cmems-du.eu/Core/GLOBAL_REANALYSIS_PHY_001_031/global-reanalysis-phy-001-031-grepv2-daily/1993/01/

"""


class ORAS5Downloader(ClimateDownloader):
    """Climate downloader to provide ORAS5 reanalysis data from CMEMS API

    These aren't available for CMIP training at daily frequencies

    :param identifier: how to identify this dataset
    :param var_map: override the default ERA5Downloader.CDI_MAP variable map
    """
    ENDPOINTS = {
        # TODO: See #49 - not yet used
        "cas":  "https://cmems-cas.cls.fr/cas/login",
        "dap":  "https://my.cmems-du.eu/thredds/dodsC/{dataset}",
        "motu": "https://my.cmems-du.eu/motu-web/Motu",
    }

    VAR_MAP = {
        "thetao": "thetao_oras",    # sea_water_potential_temperature
        "so": "so_oras",            # sea_water_salinity
        "uo": "uo_oras",            # eastward_sea_water_velocity
        "vo": "vo_oras",            # northward_sea_water_velocity
        "zos": "zos_oras",          # sea_surface_height_above_geoid
        "mlotst": "mlotst_oras",    # ocean_mixed_layer_thickness_defined_by_sigma_theta
    }

    def __init__(self,
                 *args,
                 cred_file: str = os.path.expandvars("$HOME/.cmems.creds"),
                 dataset: str = "global-reanalysis-phy-001-031-grepv2-daily",
                 identifier: str = "oras5",
                 max_failures: int = 3,
                 service: str = "GLOBAL_REANALYSIS_PHY_001_031-TDS",
                 var_map: object = None,
                 **kwargs):
        super().__init__(*args,
                         identifier=identifier,
                         **kwargs)

        cp = configparser.ConfigParser(default_section="auth")
        cp.read(cred_file)
        self._creds = dict(cp["auth"])
        self._dataset = dataset
        self._max_failures = max_failures
        self._service = service
        self._var_map = var_map if var_map else ORAS5Downloader.VAR_MAP

        assert self._max_threads <= 8, "Too many request threads for ORAS5 " \
                                       "(max.8)"

        for var_name in self._var_names:
            assert var_name in self._var_map.keys(), \
                "{} not in ORAS5 var map".format(var_name)

        self.download_method = self._single_motu_download

    def postprocess(self,
                    var: str,
                    download_path: object):
        """

        :param var:
        :param download_path:
        """
        logging.info("Postprocessing {} to {}".format(var, download_path))
        ds = xr.open_dataset(download_path)

        da = getattr(ds, self._var_map[var]).rename(var)
        da = da.mean("depth").compute()
        da.to_netcdf(download_path)

    def _single_motu_download(self,
                              var: str,
                              level: object,
                              req_dates: int,
                              download_path: object):
        """Implements a single download from ... server
        :param var:
        :param level:
        :param req_dates:
        :param download_path:
        :return:

        """
        attempts = 1
        success = False

        cmd = \
            """motuclient --quiet --motu {} \
                    --service-id {} \
                    --product-id {} \
                    --longitude-min -180 \
                    --longitude-max 179.75 \
                    --latitude-min {} \
                    --latitude-max {} \
                    --date-min "{} 00:00:00" \
                    --date-max "{} 00:00:00" \
                    --depth-min 0.5056 \
                    --depth-max 0.5059 \
                    --variable {} \
                    --out-dir {} \
                    --out-name {} \
                    --user {} \
                    --pwd '{}' \
            """.format(self.ENDPOINTS['motu'],
                       self._service,
                       self._dataset,
                       self.hemisphere_loc[2],
                       self.hemisphere_loc[0],
                       req_dates[0].strftime("%Y-%m-%d"),
                       req_dates[-1].strftime("%Y-%m-%d"),
                       self._var_map[var],
                       os.path.split(download_path)[0],
                       os.path.split(download_path)[1],
                       self._creds['username'],
                       self._creds['password'])

        tic = time.time()
        while not success:
            logging.debug("Attempt {}".format(attempts))

            ret = run_command(cmd)
            if ret.returncode != 0 or not os.path.exists(download_path):
                attempts += 1
                if attempts > self._max_failures:
                    logging.error("Couldn't download {} between {} and {}".
                                  format(var, req_dates[0], req_dates[-1]))
                    break
                time.sleep(30)
            else:
                success = True

        if success:
            dur = time.time() - tic
            logging.debug("Done in {}m:{:.0f}s. ".format(np.floor(dur / 60),
                                                         dur % 60))
        return success

    def additional_regrid_processing(self,
                                     datafile: object,
                                     cube_ease: object) -> object:
        """

        :param datafile:
        :param cube_ease:
        :return:
        """
        cube_ease.data = np.ma.filled(cube_ease.data, fill_value=0.)
        return cube_ease


def main():
    args = download_args(workers=True, extra_args=(
        (("-n", "--do-not-download"),
         dict(dest="download", action="store_false", default=True)),
        (("-p", "--do-not-postprocess"),
         dict(dest="postprocess", action="store_false", default=True))))

    logging.info("ORAS5 Data Downloading")
    oras5 = ORAS5Downloader(
        var_names=args.vars,
        # TODO: currently hardcoded
        pressure_levels=[None for _ in args.vars],
        dates=[pd.to_datetime(date).date() for date in
               pd.date_range(args.start_date, args.end_date, freq="D")],
        delete_tempfiles=args.delete,
        download=args.delete,
        max_threads=args.workers,
        postprocess=args.postprocess,
        north=args.hemisphere == "north",
        south=args.hemisphere == "south",
    )
    oras5.download()
    oras5.regrid()

