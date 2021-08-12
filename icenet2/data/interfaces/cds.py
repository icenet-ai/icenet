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

from datetime import date

import cdsapi as cds
import numpy as np
import pandas as pd
import xarray as xr

from icenet2.data.interfaces.downloader import ClimateDownloader


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
        var = var_prefix if not pressure else \
            "{}{}".format(var_prefix, pressure)
        var_folder = self.get_data_var_folder(var)

        date_str = req_date.strftime("%Y%m%d")

        logging.debug("Processing var {} for year {}".format(var,
                                                             date_str))

        download_path = os.path.join(var_folder,
                                     "{}_latlon_download_{}.nc".
                                     format(var, date_str))
        daily_path = os.path.join(var_folder,
                                  "{}_latlon_{}.nc".
                                  format(var, date_str))
        regridded_name = re.sub(r'_latlon_', '_', daily_path)

        retrieve_dict = {
            'product_type': 'reanalysis',
            'variable': self._cdi_map[var_prefix],
            'year': req_date.year,
            'month': req_date.month,
            'day': req_date.day,
            'time': ["{:02d}:00".format(h) for h in range(0, 24)],
            'format': 'netcdf',
            'area': self.hemisphere_loc
        }

        dataset = 'reanalysis-era5-single-levels'

        if pressure:
            dataset = 'reanalysis-era5-pressure-levels'
            retrieve_dict['pressure_level'] = pressure

        if not os.path.exists(regridded_name) and \
                not os.path.exists(daily_path):
            logging.info("Downloading data for {}...".format(var))

            if self.dry:
                logging.info("DRY RUN: skipping CDS request: "
                             "{}".format(retrieve_dict))
            else:
                self.client.retrieve(dataset, retrieve_dict,
                                     download_path)
                logging.debug('Download completed.')

                logging.debug('Computing daily averages...')
                da = xr.open_dataarray(download_path)

                if 'expver' in da.coords:
                    raise RuntimeError("fix_near_real_time_era5_coords "
                                       "no longer exists in the "
                                       "codebase for expver in "
                                       "coordinates")

                da_daily = da.resample(time='1D').reduce(np.mean)

                logging.debug("Saving new daily file")
                da_daily.to_netcdf(daily_path)
            self._files_downloaded.append(daily_path)

            if not self.dry:
                os.remove(download_path)
        # TODO: check this is a reliable method for picking up
        #  ungridded files
        elif os.path.exists(daily_path):
            self._files_downloaded.append(daily_path)

    def _get_dates_for_request(self):
        # TODO: Stick some additional controls for batching downloads more
        #  easily
        dates = sorted(self._dates)
        #date_range = pd.date_range(dates[0], dates[-1])
        #years = date_range.year.unique()
        return dates

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


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    logging.info("ERA5 Downloader - direct module run")

    era5 = ERA5Downloader(
        var_names=["tas", "ta", "tos", "psl", "zg", "hus", "rlds", "rsds",
                   "uas", "vas"],
        pressure_levels=[None, [500], None, None, [250, 500], [1000], None,
                         None, None, None],
        dates=[date(2021, 1, 1)]
    )
    era5.download()
    era5.regrid()
    era5.rotate_wind_data()
