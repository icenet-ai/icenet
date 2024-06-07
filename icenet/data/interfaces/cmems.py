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
from icenet.exceptions import CredentialsNotFoundError
"""
DATASET: cmems_mod_glo_phy-all_my_0.25deg_P1D-m
PRODUCT ID: GLOBAL_MULTIYEAR_PHY_ENS_001_031
DESCRIPTION: https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_ENS_001_031/description

"""


class ORAS5Downloader(ClimateDownloader):
    """Climate downloader to provide ORAS5 reanalysis data from CMEMS API

    These aren't available for CMIP training at daily frequencies

    :param identifier: how to identify this dataset
    :param var_map: override the default ERA5Downloader.CDI_MAP variable map
    """
    VAR_MAP = {
        "thetao": "thetao_oras",  # sea_water_potential_temperature
        "so": "so_oras",  # sea_water_salinity
        "uo": "uo_oras",  # eastward_sea_water_velocity
        "vo": "vo_oras",  # northward_sea_water_velocity
        "zos": "zos_oras",  # sea_surface_height_above_geoid
        "mlotst":
            "mlotst_oras",  # ocean_mixed_layer_thickness_defined_by_sigma_theta
    }

    def __init__(self,
                 *args,
                 cred_file: str = os.path.expandvars("$HOME/.cmems.creds"),
                 dataset: str = "cmems_mod_glo_phy-all_my_0.25deg_P1D-m",
                 identifier: str = "oras5",
                 max_failures: int = 3,
                 var_map: object = None,
                 **kwargs):
        super().__init__(*args,
                         drop_vars=["lambert_azimuthal_equal_area"],
                         identifier=identifier,
                         **kwargs)

        env_username_var = "COPERNICUSMARINE_SERVICE_USERNAME"
        env_password_var = "COPERNICUSMARINE_SERVICE_PASSWORD"
        check_env_variables = set([env_username_var, env_password_var]).issubset(os.environ)
        if os.path.exists(cred_file):
            cp = configparser.ConfigParser(default_section="auth")
            cp.read(cred_file)
            self._creds = dict(cp["auth"])
        elif check_env_variables:
            self._creds = dict({
                                  "username": os.environ[env_username_var],
                                  "password": os.environ[env_password_var],
                              })
        else:
            error_message = """Copernicus Marine credentials not found here: `{}`. \
                              Please either add user details to the file, \
                              or set environment variables \
                              `{}` and \
                              `{}`
                           """.format("$HOME/.cmems.creds",
                                      env_username_var,
                                      env_password_var
                                     )
            raise CredentialsNotFoundError(" ".join(error_message.split()))

        self._dataset = dataset
        self._max_failures = max_failures
        self._var_map = var_map if var_map else ORAS5Downloader.VAR_MAP

        assert self._max_threads <= 8, "Too many request threads for ORAS5 " \
                                       "(max.8)"

        for var_name in self._var_names:
            assert var_name in self._var_map.keys(), \
                "{} not in ORAS5 var map".format(var_name)

        self.download_method = self._single_motu_download

    def postprocess(self, var: str, download_path: object):
        """

        :param var:
        :param download_path:
        """
        logging.info(
            "Postprocessing Copernicus Marine data for variable `{}` at {}"
              .format(var, download_path)
        )

        temp_path = "{}.bak{}".format(*os.path.splitext(download_path))
        logging.debug("Moving to {}".format(temp_path))
        os.rename(download_path, temp_path)

        ds = xr.open_dataset(temp_path)

        da = getattr(ds, self._var_map[var]).rename(var)
        if "depth" in list(ds.coords):
            da = da.mean("depth").compute()
        da.to_netcdf(download_path)

    def _single_motu_download(self, var: str, level: object, req_dates: int,
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
            """copernicusmarine subset \
                    -i {} \
                    -x -180 \
                    -X 179.75 \
                    -y {} \
                    -Y {} \
                    -z 0.5056 \
                    -Z 0.5059 \
                    -t '{}T00:00:00' \
                    -T '{}T00:00:00' \
                    -v {} \
                    -o {} \
                    -f {} \
                    --username {} \
                    --password '{}' \
                    --no-metadata-cache \
                    --force-download \
            """.format(self._dataset,
                       self.hemisphere_loc[2],
                       self.hemisphere_loc[0],
                       req_dates[0].strftime("%Y-%m-%d"),
                       req_dates[-1].strftime("%Y-%m-%d"),
                       self._var_map[var],
                       os.path.split(download_path)[0],
                       os.path.split(download_path)[1],
                       self._creds['username'],
                       self._creds['password']
                      )

        cmd = " ".join(cmd.split())

        tic = time.time()
        while not success:
            logging.debug("Attempt {}".format(attempts))

            ret = run_command(cmd)
            if ret.returncode != 0 or not os.path.exists(download_path + ".nc"):
                attempts += 1
                if attempts > self._max_failures:
                    logging.error(
                        "Couldn't download {} between {} and {}".format(
                            var, req_dates[0], req_dates[-1]))
                    break
                time.sleep(30)
            else:
                success = True

        if success:
            # Copernicus Marine toolbox outputs with ".nc" extension
            os.rename(download_path + ".nc", download_path)
            dur = time.time() - tic
            logging.debug("Done in {}m:{:.0f}s. ".format(
                np.floor(dur / 60), dur % 60))
        return success

    def additional_regrid_processing(self, datafile: object,
                                     cube_ease: object) -> object:
        """

        :param datafile:
        :param cube_ease:
        :return:
        """
        cube_ease.data = np.ma.filled(cube_ease.data, fill_value=0.)
        return cube_ease


def main():
    args = download_args(workers=True,
                         extra_args=((("-n", "--do-not-download"),
                                      dict(dest="download",
                                           action="store_false",
                                           default=True)),
                                     (("-p", "--do-not-postprocess"),
                                      dict(dest="postprocess",
                                           action="store_false",
                                           default=True))))

    logging.info("ORAS5 Data Downloading")
    oras5 = ORAS5Downloader(
        var_names=args.vars,
        # TODO: currently hardcoded
        dates=[
            pd.to_datetime(date).date()
            for date in pd.date_range(args.start_date, args.end_date, freq="D")
        ],
        delete_tempfiles=args.delete,
        download=args.delete,
        levels=[None for _ in args.vars],
        max_threads=args.workers,
        postprocess=args.postprocess,
        north=args.hemisphere == "north",
        south=args.hemisphere == "south",
    )
    oras5.download()
    oras5.regrid()
