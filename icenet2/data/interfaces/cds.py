"""
Module to download hourly ERA5 reanalysis latitude-longitude maps,
compute daily averages, regrid them to the same EASE grid as the OSI-SAF sea
ice, data, and save as yearly NetCDFs.

The `variables` dictionary controls which NetCDF variables are downloaded/
regridded, as well their paths/filenames.

Only 120,000 hours of ERA5 data can be downloaded in a single Climate
Data Store request, so this script downloads and processes data in yearly
chunks.

A command line input dictates which variable is downloaded and allows this
script to be run in parallel for different variables.
"""

import collections
import logging
import os

import cdsapi as cds
import numpy as np
import pandas as pd
import xarray as xr

from icenet2.data.interfaces.downloader import ClimateDownloader


def get_names(var_folder, var, date_str):
    daily_path = os.path.join(var_folder,
                              "latlon_{}_{}.nc".
                              format(var, date_str))
    regridded_name = os.path.join(var_folder,
                                  "{}_{}.nc".
                                  format(var, date_str))
    return daily_path, regridded_name


class ERA5Downloader(ClimateDownloader):
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
                 **kwargs):
        super().__init__(*args,
                         identifier=identifier,
                         **kwargs)
        self.client = cds.Client()
        self._cdi_map = cdi_map

    def _single_download(self, var_prefix, pressure, req_date):
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
            daily_path, regridded_name = get_names(var_folder,
                                                   var,
                                                   destination_date.
                                                   strftime("%Y_%m_%d"))

            if not os.path.exists(daily_path):
                downloads.append(destination_date)

            if not os.path.exists(regridded_name):
                self._files_downloaded.append(daily_path)

        if len(downloads) > 0:
            logging.info("Processing dates: {}".format(downloads))

            retrieve_dict = {
                'product_type': 'reanalysis',
                'variable': self._cdi_map[var_prefix],
                'year': req_date[0].year,
                'month': req_date[0].month,
                'day': ["{:02d}".format(d.day) for d in downloads],
                'time': ["{:02d}:00".format(h) for h in range(0, 24)],
                'format': 'netcdf',
                'area': self.hemisphere_loc
            }

            dataset = 'reanalysis-era5-single-levels'

            if pressure:
                dataset = 'reanalysis-era5-pressure-levels'
                retrieve_dict['pressure_level'] = pressure

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

                    daily_path, regridded_name = get_names(var_folder,
                                                           var,
                                                           date_str)

                    if not os.path.exists(daily_path):
                        logging.debug(
                            "Saving new daily file: {}".format(daily_path))
                        da_daily.sel(time=slice(day, day)).to_netcdf(daily_path)
                        self._files_downloaded.append(daily_path)
            except Exception as e:
                logging.exception("{} not deleted, look at the "
                                  "problem".format(temp_download_path))
                raise RuntimeError(e)

            if self.delete:
                logging.debug("Remove {}".format(temp_download_path))
                os.unlink(temp_download_path)
        else:
            logging.info("No dates needing downloading for {}".format(var))

    def _get_dates_for_request(self):
        dates = collections.deque(sorted(self._dates))

        batched_dates = []
        batch = []

        while len(dates):
            if not len(batch):
                batch.append(dates.popleft())
            else:
                if batch[-1].month == dates[0].month:
                    batch.append(dates.popleft())
                else:
                    batched_dates.append(batch)
                    batch = []

        if len(batch):
            batched_dates.append(batch)

        if len(dates) > 0:
            raise RuntimeError("Batching didn't work!")

        return batched_dates

    def additional_regrid_processing(self, datafile, cube_ease):
        (datafile_path, datafile_name) = os.path.split(datafile)
        var_name = datafile_path.split(os.sep)[-1]

        # FIXME: are these here or preproc?
        # if var_name == 'zg500' or var_name == 'zg250':
        #   da_daily = da_daily / 9.80665

        # if var_name == 'tos':
        #     # Replace every value outside of SST < 1000 with
        #    zeros (the ERA5 masked values)
        #     da_daily = da_daily.where(da_daily < 1000., 0)

        if var_name == 'tos':
            # Overwrite maksed values with zeros
            cube_ease.data[cube_ease.data > 500.] = 0.
            cube_ease.data[cube_ease.data < 0.] = 0.

            cube_ease.data[:, self._masks.get_land_mask()] = 0.

            # Remove mask from masked array
            cube_ease.data = cube_ease.data.data
        elif var_name in ['zg500', 'zg250']:
            # Convert from geopotential to geopotential height
            cube_ease /= 9.80665

