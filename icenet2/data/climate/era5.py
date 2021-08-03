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

import logging
import os
import re

from itertools import product

import cdsapi as cds
import iris
import numpy as np
import pandas as pd
import xarray as xr

from icenet2.constants import *
from icenet2.data.climate.downloader import ClimateDownloader
from icenet2.data.utils import assign_lat_lon_coord_system
from icenet2.utils import run_command


class ERA5(ClimateDownloader):
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

    def __init__(self, *args, **kwargs):
        super(ClimateDownloader, self).__init__(*args,
                                                identifier="era5", **kwargs)
        self.client = cds.Client()

    def download(self):
        # FIXME: confirmed, but the year start year end naming is a bit weird,
        #  hang up from the icenet port but we might want to consider relevance,
        #  it remains purely for compatibility with existing data

        # TODO: This is download and average for dailies, but could be easily
        #  abstracted for different temporal averaging
        logging.info("Building request(s), downloading and daily averaging "
                     "from CDS API")

        dailies = []

        for idx, var_name in enumerate(self._var_names):
            pressures = [None] if not self._pressure_levels[idx] else \
                self._pressure_levels[idx]

            dates = sorted(self._dates)
            date_range = pd.date_range(dates[0], dates[-1])
            years = date_range.year.unique()

            for var_prefix, pressure, year in \
                    product(var_name, pressures, years):

                var = var_prefix if not pressure else \
                    "{}{}".format(var_prefix, pressure)
                var_folder = self.get_climate_var_folder(var)

                logging.debug("Processing var {} for year {}".format(
                    var, year))

                download_path = os.path.join(var_folder,
                                             "{}_latlon_download_{}.nc".
                                             format(var_name, year))
                daily_path = os.path.join(var_folder,
                                          "{}_latlon_{}.nc".
                                          format(var_name, year))
                regridded_name = re.sub(r'_latlon_', '_', daily_path)

                # FIXME: at the moment we just download all the data for each
                #  year, the export of the configuration applies the _dates
                retrieve_dict = {
                    'product_type': 'reanalysis',
                    'variable': ERA5.CDI_MAP[var_prefix],
                    'year': year,
                    'month': ["{:02d}".format(m) for m in range(0, 13)],
                    'day': ["{:02d}".format(m) for m in range(0, 32)],
                    'time': ["{:02d}:00".format(h) for h in range(0, 24)],
                    'format': 'netcdf',
                    'area': self.hemisphere_loc
                }

                dataset = 'reanalysis-era5-single-levels'

                if pressure:
                    dataset = 'reanalysis-era5-pressure-levels'
                    retrieve_dict['pressure_level'] = pressure

                if not os.path.exists(regridded_name):
                    logging.info("Downloading data for {}...".format(var_name))

                    cds.retrieve(dataset, retrieve_dict, download_path)
                    logging.debug('Download completed.')

                    logging.debug('Computing daily averages...')
                    da = xr.open_dataarray(download_path)

                    if 'expver' in da.coords:
                        raise RuntimeError("fix_near_real_time_era5_coords no "
                                           "longer exists in the codebase for "
                                           "expver in coordinates")

                    da_daily = da.resample(time='1D').reduce(np.mean)

                    # FIXME: are these here or preproc?
                    # if var_name == 'zg500' or var_name == 'zg250':
                    #   da_daily = da_daily / 9.80665

                    # if var_name == 'tos':
                    #     # Replace every value outside of SST < 1000 with
                    #    zeros (the ERA5 masked values)
                    #     da_daily = da_daily.where(da_daily < 1000., 0)

                    logging.debug("Saving new daily file")
                    da_daily.to_netcdf(daily_path)
                    self._files_downloaded.append(daily_path)

                    os.remove(download_path)

        logging.info("{} daily files downloaded".
                     format(len(self._files_downloaded)))

