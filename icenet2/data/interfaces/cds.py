import logging
import os
import requests
import requests.adapters

import cdsapi as cds
import numpy as np
import pandas as pd
import xarray as xr

from icenet2.data.cli import download_args
from icenet2.data.interfaces.downloader import ClimateDownloader
from icenet2.data.interfaces.utils import \
    batch_requested_dates, get_daily_filenames

"""
Module to download hourly ERA5 reanalysis latitude-longitude maps,
compute daily averages, regrid them to the same EASE grid as the OSI-SAF sea
ice, data, and save as daily NetCDFs.
"""


class ERA5Downloader(ClimateDownloader):
    """Climate downloader to provide ERA5 reanalysis data from CDS API

    Args:
        identifier (string): how to identify this dataset
        cdi_map (map): override the default ERA5Downloader.CDI_MAP variable map
        use_toolbox (boolean): whether to use CDS toolbox for remote aggregation
        show_progress (boolean): whether to show download progress
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
                 identifier="era5",
                 cdi_map=CDI_MAP,
                 use_toolbox=True,
                 show_progress=False,
                 **kwargs):
        super().__init__(*args,
                         identifier=identifier,
                         **kwargs)
        self.client = cds.Client(progress=show_progress)
        self._cdi_map = cdi_map
        self._toolbox = use_toolbox

        if self._max_threads > 10:
            logging.info("Upping connection limit for max_threads > 10")
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=self._max_threads,
                pool_maxsize=self._max_threads
            )
            self.client.session.mount("https://", adapter)

    def _single_download(self, var_prefix, pressure, req_date):
        """Implements a single download from CDS API

        Args:
            var_prefix (string): the icenet variable name
            pressure (int): the pressure level to download
            req_date (datetime): the request date
        """
        # FIXME: confirmed, but the year start year end naming is a bit weird,
        #  hang up from the icenet port but we might want to consider relevance,
        #  it remains purely for compatibility with existing data

        # TODO: This is download and average for dailies, but could be easily
        #  abstracted for different temporal averaging

        for dt in req_date:
            assert dt.year == req_date[0].year
            assert dt.month == req_date[0].month

        var = var_prefix if not pressure else \
            "{}{}".format(var_prefix, pressure)
        var_folder = os.path.join(self.get_data_var_folder(var),
                                  str(req_date[0].year))

        # For the year component - 365 * 50 is a lot of files ;)
        os.makedirs(var_folder, exist_ok=True)

        downloads = []
        for destination_date in req_date:
            daily_path, regridded_name = get_daily_filenames(var_folder,
                                                   var,
                                                   destination_date.
                                                   strftime("%Y_%m_%d"))

            if not os.path.exists(daily_path) \
                    and not os.path.exists(regridded_name):
                downloads.append(destination_date)
            elif not os.path.exists(regridded_name):
                self._files_downloaded.append(daily_path)

        if self._toolbox:
            return self._single_toolbox_download(var_prefix,
                                                 var,
                                                 var_folder,
                                                 pressure,
                                                 req_date,
                                                 downloads)
        return self._single_api_download(var_prefix,
                                         var,
                                         var_folder,
                                         pressure,
                                         req_date,
                                         downloads)

    def _single_toolbox_download(self,
                                 var_prefix,
                                 var,
                                 var_folder,
                                 pressure,
                                 req_date,
                                 downloads):
        if len(downloads) > 0:
            logging.debug("Processing {} dates".format(len(downloads)))

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
                    "year": req_date[0].year,
                    "month": req_date[0].month,
                    "frequency": "1-hourly",
                    "time_zone": "UTC+00:00",
                    "grid": "0.25/0.25",
                    "area": self.hemisphere_loc,
                },
            }

            if pressure:
                params_dict["kwargs"]["dataset"] = \
                    "reanalysis-era5-pressure-levels"
                params_dict["kwargs"]["pressure_level"] = pressure

            result = self.client.service(
                "tool.toolbox.orchestrator.workflow",
                params=params_dict)

            temp_download_path = os.path.join(var_folder,
                                              "download_{}_{}_{}.nc".
                                              format(
                                                  var,
                                                  req_date[0].month,
                                                  req_date[0].year))

            try:
                logging.info("Downloading data for {}...".format(var))
                logging.debug("Result: {}".format(result))
                
                location = result[0]['location']
                res = requests.get(location, stream=True)

                logging.info("Writing data to " + temp_download_path)
                logging.getLogger("requests").setLevel(logging.WARNING)

                with open(temp_download_path, 'wb') as fh:
                    for r in res.iter_content(chunk_size=1024):
                        fh.write(r)

                logging.info("Download completed: {}".
                             format(temp_download_path))

                self._cds_file_process(temp_download_path,
                                       var,
                                       var_folder,
                                       resample=False)
            except Exception as e:
                logging.exception("{} not deleted, look at the "
                                  "problem".format(temp_download_path))
                raise RuntimeError(e)

            if self.delete:
                logging.debug("Remove {}".format(temp_download_path))
                os.unlink(temp_download_path)

    def _single_api_download(self,
                             var_prefix,
                             var,
                             var_folder,
                             pressure,
                             req_date,
                             downloads):
        if len(downloads) > 0:
            logging.debug("Processing {} dates".format(len(downloads)))

            retrieve_dict = {
                "product_type": "reanalysis",
                "variable": self._cdi_map[var_prefix],
                "year": req_date[0].year,
                "month": req_date[0].month,
                "day": ["{:02d}".format(d.day) for d in downloads],
                "time": ["{:02d}:00".format(h) for h in range(0, 24)],
                "format": "netcdf",
                "area": self.hemisphere_loc,
            }

            dataset = "reanalysis-era5-single-levels"

            if pressure:
                dataset = "reanalysis-era5-pressure-levels"
                retrieve_dict["pressure_level"] = pressure

            temp_download_path = os.path.join(var_folder,
                                              "download_{}_{}_{}.nc".
                                              format(
                                                  var,
                                                  req_date[0].month,
                                                  req_date[0].year))
            try:
                logging.info("Downloading data for {}...".format(var))

                self.client.retrieve(dataset, retrieve_dict, temp_download_path)
                logging.info("Download completed: {}".
                             format(temp_download_path))

                self._cds_file_process(temp_download_path, var, var_folder)
            except Exception as e:
                logging.exception("{} not deleted, look at the "
                                  "problem".format(temp_download_path))
                raise RuntimeError(e)

            if self.delete:
                logging.debug("Remove {}".format(temp_download_path))
                os.unlink(temp_download_path)
        else:
            logging.info("No dates needing downloading for {}".format(var))

    def _cds_file_process(self,
                          temp_download_path,
                          var,
                          var_folder,
                          resample=True):
        da = xr.open_dataarray(temp_download_path)

        if 'expver' in da.coords:
            raise RuntimeError("fix_near_real_time_era5_coords "
                               "no longer exists in the "
                               "codebase for expver in "
                               "coordinates")

        da_daily = da.resample(time='1D').reduce(np.mean)

        for day in da.time.values:
            date_str = pd.to_datetime(day).strftime("%Y_%m_%d")
            logging.debug(
                "Processing var {} for {}".format(var, date_str))

            daily_path, regridded_name = get_daily_filenames(
                var_folder, var, date_str)

            if not os.path.exists(daily_path):
                logging.debug(
                    "Saving new daily file: {}".format(daily_path))
                da_daily.sel(time=slice(day, day)).to_netcdf(daily_path)
                self._files_downloaded.append(daily_path)

    def _get_dates_for_request(self):
        return batch_requested_dates(self._dates, attribute="month")

    def additional_regrid_processing(self, datafile, cube_ease):
        (datafile_path, datafile_name) = os.path.split(datafile)
        var_name = datafile_path.split(os.sep)[-2]

        # FIXME: are these here or preproc?
        # if var_name == 'zg500' or var_name == 'zg250':
        #   da_daily = da_daily / 9.80665

        # if var_name == 'tos':
        #     # Replace every value outside of SST < 1000 with
        #    zeros (the ERA5 masked values)
        #     da_daily = da_daily.where(da_daily < 1000., 0)
        
        if var_name == 'tos':
            # Overwrite maksed values with zeros
            logging.debug("ERA5 additional regrid: {}".format(var_name))
            cube_ease.data[cube_ease.data.mask] = 0.
            cube_ease.data[:, self._masks.get_land_mask()] = 0.
            cube_ease.data = cube_ease.data.data
        elif var_name in ['zg500', 'zg250']:
            # Convert from geopotential to geopotential height
            logging.debug("ERA5 additional regrid: {}".format(var_name))
            cube_ease /= 9.80665


def main():
    args = download_args(choices=["toolbox", "cdsapi"], workers=True)

    logging.info("ERA5 Data Downloading")
    era5 = ERA5Downloader(
        var_names=["tas", "ta", "tos", "psl", "zg", "hus", "rlds", "rsds",
                   "uas", "vas"],
        pressure_levels=[None, [500], None, None, [250, 500], [1000], None,
                         None, None, None],
        dates=[pd.to_datetime(date).date() for date in
               pd.date_range(args.start_date, args.end_date,
                             freq="D")],
        delete_tempfiles=args.delete,
        max_threads=args.workers,
        north=args.hemisphere == "north",
        south=args.hemisphere == "south",
        use_toolbox=args.choice == "toolbox"
    )
    era5.download()
    era5.regrid()
    era5.rotate_wind_data()
