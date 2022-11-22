import logging
import os
import requests
import requests.adapters

from pprint import pformat

import cdsapi as cds
import numpy as np
import pandas as pd
import xarray as xr

from icenet.data.cli import download_args
from icenet.data.interfaces.downloader import ClimateDownloader
from icenet.data.interfaces.utils import batch_requested_dates

"""
Module to download hourly ERA5 reanalysis latitude-longitude maps,
compute daily averages, regrid them to the same EASE grid as the OSI-SAF sea
ice, data, and save as daily NetCDFs.
"""


class ERA5Downloader(ClimateDownloader):
    """Climate downloader to provide ERA5 reanalysis data from CDS API

    :param identifier: how to identify this dataset
    :param cdi_map: override the default ERA5Downloader.CDI_MAP variable map
    :param use_toolbox: whether to use CDS toolbox for remote aggregation
    :param show_progress: whether to show download progress
    """

    CDI_MAP = {
        'tas': '2m_temperature',
        'ta': 'temperature',  # 500
        'tos': 'sea_surface_temperature',
        'psl': 'surface_pressure',
        'zg': 'geopotential',  # 250 and 500
        'hus': 'specific_humidity',  # 1000
        'rlds': 'surface_thermal_radiation_downwards',
        'rsds': 'surface_solar_radiation_downwards',
        'uas': '10m_u_component_of_wind',
        'vas': '10m_v_component_of_wind',
    }

    def __init__(self,
                 *args,
                 identifier: str = "era5",
                 cdi_map: object = CDI_MAP,
                 use_toolbox: bool = False,
                 show_progress: bool = False,
                 **kwargs):
        super().__init__(*args,
                         identifier=identifier,
                         **kwargs)
        self.client = cds.Client(progress=show_progress)
        self._cdi_map = cdi_map

        self._use_toolbox = use_toolbox
        self.download_method = self._single_api_download

        if use_toolbox:
            self.download_method = self._single_toolbox_download

        if self._max_threads > 10:
            logging.info("Upping connection limit for max_threads > 10")
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=self._max_threads,
                pool_maxsize=self._max_threads
            )
            self.client.session.mount("https://", adapter)

    def _single_toolbox_download(self,
                                 var: object,
                                 level: object,
                                 req_dates: object,
                                 download_path: object):
        """Implements a single download from CDS Toolbox API

        :param var:
        :param level: the pressure level to download
        :param req_dates: the request dates
        :param download_path:
        """

        logging.debug("Processing {} dates".format(len(req_dates)))
        var_prefix = var[0:-(len(str(level)))] if level else var

        params_dict = {
            "realm":    "c3s",
            "project":  "app-c3s-daily-era5-statistics",
            "version":  "master",
            "workflow_name": "application",
            "kwargs": {
                "dataset": "reanalysis-era5-single-levels",
                "product_type": "reanalysis",
                "variable": self._cdi_map[var_prefix],
                "pressure_level": "-",
                "statistic": "daily_mean",
                "year": req_dates[0].year,
                "month": sorted(list(set([r.month for r in req_dates]))),
                "frequency": "1-hourly",
                "time_zone": "UTC+00:00",
                "grid": "0.25/0.25",
                "area": {
                    "lat": [min([self.hemisphere_loc[0],
                                 self.hemisphere_loc[2]]),
                            max([self.hemisphere_loc[0],
                                 self.hemisphere_loc[2]])],
                    "lon": [min([self.hemisphere_loc[1],
                                 self.hemisphere_loc[3]]),
                            max([self.hemisphere_loc[1],
                                 self.hemisphere_loc[3]])],
                },
            },
        }

        if level:
            params_dict["kwargs"]["dataset"] = \
                "reanalysis-era5-pressure-levels"
            params_dict["kwargs"]["pressure_level"] = level

        logging.debug("params_dict: {}".format(pformat(params_dict)))
        result = self.client.service(
            "tool.toolbox.orchestrator.workflow",
            params=params_dict)

        try:
            logging.info("Downloading data for {}...".format(var))
            logging.debug("Result: {}".format(result))
                
            location = result[0]['location']
            res = requests.get(location, stream=True)

            logging.info("Writing data to {}".format(download_path))

            with open(download_path, 'wb') as fh:
                for r in res.iter_content(chunk_size=1024):
                    fh.write(r)

            logging.info("Download completed: {}".format(download_path))

        except Exception as e:
            logging.exception("{} not deleted, look at the "
                              "problem".format(download_path))
            raise RuntimeError(e)

    def _single_api_download(self,
                             var: str,
                             level: object,
                             req_dates: object,
                             download_path: object):
        """Implements a single download from CDS API

        :param var:
        :param level: the pressure level to download
        :param req_dates: the request date
        :param download_path:
        """

        logging.debug("Processing {} dates".format(len(req_dates)))
        var_prefix = var[0:-(len(str(level)))] if level else var

        retrieve_dict = {
            "product_type": "reanalysis",
            "variable": self._cdi_map[var_prefix],
            "year": req_dates[0].year,
            "month": list(set(["{:02d}".format(rd.month)
                               for rd in sorted(req_dates)])),
            "day": ["{:02d}".format(d) for d in range(1, 32)],
            "time": ["{:02d}:00".format(h) for h in range(0, 24)],
            "format": "netcdf",
            "area": self.hemisphere_loc,
        }

        dataset = "reanalysis-era5-single-levels"
        if level:
            dataset = "reanalysis-era5-pressure-levels"
            retrieve_dict["pressure_level"] = level

        try:
            logging.info("Downloading data for {}...".format(var))

            self.client.retrieve(dataset, retrieve_dict, download_path)
            logging.info("Download completed: {}".format(download_path))

        except Exception as e:
            logging.exception("{} not deleted, look at the "
                              "problem".format(download_path))
            raise RuntimeError(e)

    def postprocess(self,
                    var: str,
                    download_path: object):
        """Processing of CDS downloaded files

        If we've not used the toolbox to download the files, we have a lot of
        hourly data to average out, which is taken care of here

        :param var:
        :param download_path:
        """
        # if not self._use_toolbox:
        logging.info("Postprocessing CDS API data at {}".format(download_path))
        da = xr.open_dataarray(download_path)

        if 'expver' in da.coords:
            raise RuntimeError("fix_near_real_time_era5_coords no longer "
                               "exists in the codebase for expver "
                               "in coordinates")

        da = da.resample(time='1D').mean().compute()
        da.to_netcdf(download_path)

    def _get_dates_for_request(self) -> object:
        """Appropriate monthly batching of dates for CDS requests

        :return:

        """
        return batch_requested_dates(self._dates, attribute="month")

    def additional_regrid_processing(self,
                                     datafile: str,
                                     cube_ease: object):
        """

        :param datafile:
        :param cube_ease:
        """
        (datafile_path, datafile_name) = os.path.split(datafile)
        var_name = datafile_path.split(os.sep)[self._var_name_idx]

        # FIXME: are these here or preproc?
        # if var_name == 'zg500' or var_name == 'zg250':
        #   da_daily = da_daily / 9.80665

        # if var_name == 'tos':
        #     # Replace every value outside of SST < 1000 with
        #    zeros (the ERA5 masked values)
        #     da_daily = da_daily.where(da_daily < 1000., 0)
        
        if var_name == 'tos':
            # Overwrite maksed values with zeros
            logging.debug("ERA5 regrid postprocess: {}".format(var_name))
            cube_ease.data[cube_ease.data.mask] = 0.
            cube_ease.data[:, self._masks.get_land_mask()] = 0.
            cube_ease.data = cube_ease.data.data
            cube_ease.data = np.where(np.isnan(cube_ease.data), 0., cube_ease.data)
        elif var_name in ['zg500', 'zg250']:
            # Convert from geopotential to geopotential height
            logging.debug("ERA5 additional regrid: {}".format(var_name))
            cube_ease /= 9.80665


def main():
    args = download_args(choices=["cdsapi", "toolbox"],
                         workers=True, extra_args=(
        (("-n", "--do-not-download"),
         dict(dest="download", action="store_false", default=True)),
        (("-p", "--do-not-postprocess"),
         dict(dest="postprocess", action="store_false", default=True))))

    logging.info("ERA5 Data Downloading")
    era5 = ERA5Downloader(
        var_names=args.vars,
        pressure_levels=args.levels,
        dates=[pd.to_datetime(date).date() for date in
               pd.date_range(args.start_date, args.end_date, freq="D")],
        delete_tempfiles=args.delete,
        download=args.download,
        max_threads=args.workers,
        postprocess=args.postprocess,
        north=args.hemisphere == "north",
        south=args.hemisphere == "south",
        use_toolbox=args.choice == "toolbox"
    )
    era5.download()
    era5.regrid()
    era5.rotate_wind_data()
