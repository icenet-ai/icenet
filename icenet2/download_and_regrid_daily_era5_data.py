import sys
import os
import cdsapi
import xarray as xr
import iris
import time
# import plotly.express as px
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
import numpy as np
sys.path.insert(0, os.path.join(os.getcwd(), 'icenet2'))  # if using jupyter kernel
import utils
import argparse

"""
Script to download hourly ERA5 reanalysis latitude-longitude maps,
compute daily averages, regrid them to the same EASE grid as the OSI-SAF sea ice,
data, and save as yearly NetCDFs.

The `variables` dictionary controls which NetCDF variables are downloaded/
regridded, as well their paths/filenames.

Only 120,000 hours of ERA5 data can be downloaded in a single Climate
Data Store request, so this script downloads and processes data in yearly chunks.

A command line input dictates which variable is downloaded and allows this script
to be run in parallel for different variables.
"""

################################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--var', default='tas')
# OPTIONS:
#   'nh': Northern Hemisphere (Arctic) only
#   'sh': Southern Hemisphere (Antarctic) only
#   'nh_sh': both Arctic and Antarctic
parser.add_argument('--hemisphere', default='nh')
args = parser.parse_args()

if args.hemisphere == 'nh' or args.hemisphere == 'sh':
    hemispheres = [args.hemisphere]
elif args.hemisphere == 'nh_sh':
    hemispheres = ['nh', 'sh']

################################################################################

overwrite = False  # Whether to skip yearly files already downloaded/regridded

################################################################################

# Latitude/longitude boundaries to download
if args.hemisphere == 'nh':
    area = [90, -180, 0, 180]
elif args.hemisphere == 'sh':
    area = [-90, -180, 0, 180]
elif args.hemisphere == 'nh_sh':
    area = [-90, -180, 90, 180]

# Which years to download
years = [
    '1979', '1980', '1981', '1982', '1983',
    '1984', '1985', '1986', '1987', '1988',
    '1989', '1990', '1991', '1992', '1993',
    '1994', '1995', '1996', '1997', '1998',
    '1999', '2000', '2001', '2002', '2003',
    '2004', '2005', '2006', '2007', '2008',
    '2009', '2010', '2011', '2012', '2013',
    '2014', '2015', '2016', '2017', '2018',
    '2019', '2020'
]

years = [[year] for year in years]

months = [
    '01', '02', '03', '04', '05', '06',
    '07', '08', '09', '10', '11', '12'
]

days = [
    '01', '02', '03',
    '04', '05', '06',
    '07', '08', '09',
    '10', '11', '12',
    '13', '14', '15',
    '16', '17', '18',
    '19', '20', '21',
    '22', '23', '24',
    '25', '26', '27',
    '28', '29', '30',
    '31',
]

times = [
    '00:00', '01:00', '02:00',
    '03:00', '04:00', '05:00',
    '06:00', '07:00', '08:00',
    '09:00', '10:00', '11:00',
    '12:00', '13:00', '14:00',
    '15:00', '16:00', '17:00',
    '18:00', '19:00', '20:00',
    '21:00', '22:00', '23:00',
]

variables = {
    'tas': {
        'cdi_name': '2m_temperature',
    },
    'ta500': {
        'plevel': '500',
        'cdi_name': 'temperature',
    },
    'tos': {
        'cdi_name': 'sea_surface_temperature',
    },
    'psl': {
        'cdi_name': 'surface_pressure',
    },
    'zg250': {
        'plevel': '250',
        'cdi_name': 'geopotential',
    },
    'zg500': {
        'plevel': '500',
        'cdi_name': 'geopotential',
    },
    'hus1000': {
        'plevel': '1000',
        'cdi_name': 'specific_humidity',
    },
    'rlds': {
        'cdi_name': 'surface_thermal_radiation_downwards',
    },
    'rsds': {
        'cdi_name': 'surface_solar_radiation_downwards',
    },
    'uas': {
        'cdi_name': '10m_u_component_of_wind',
    },
    'vas': {
        'cdi_name': '10m_v_component_of_wind',
    },
    # NOTE: uas and vas must both be downloaded in lat/lon form before regridding,
    #   because they must be regridded together (due to the rotation needed).
    'uas_and_vas': {
        'cdi_name': '10m_v_component_of_wind',
    },
}

var_dict = variables[args.var]

# ---------------- Download

cds = cdsapi.Client()

for year_chunk in years:

    year_start = year_chunk[0]
    year_end = year_chunk[-1]

    print('\n\n' + year_start + '-' + year_end + '\n\n')

    skipyear = False
    var_folder = os.path.join('data', args.hemisphere, args.var)
    if not os.path.exists(var_folder):
        os.makedirs(var_folder)

    download_path = os.path.join(var_folder, '{}_latlon_hourly_{}_{}.nc'.format(args.var, year_start, year_end))
    daily_fpath = os.path.join(var_folder, '{}_latlon_{}_{}.nc'.format(args.var, year_start, year_end))

    daily_ease_fpath = {}
    for hemisphere in hemispheres:
        var_hem_folder = os.path.join('data', hemisphere, args.var)
        if not os.path.exists(var_hem_folder):
            os.makedirs(var_hem_folder)

        daily_ease_fpath[hemisphere] = os.path.join(
            var_hem_folder, '{}_{}_{}.nc'.format(args.var, year_start, year_end))

        if os.path.exists(daily_ease_fpath[hemisphere]) and not overwrite:
            print('Skipping this year chunk due to existing file at: {}'.format(daily_ease_fpath[hemisphere]))
            skipyear = True

    if skipyear is True:
        continue

    retrieve_dict = {
        'product_type': 'reanalysis',
        'variable': var_dict['cdi_name'],
        'year': year_chunk,
        'month': months,
        'day': days,
        'time': times,
        'format': 'netcdf',
        'area': area
    }

    if 'plevel' not in var_dict.keys():
        dataset_str = 'reanalysis-era5-single-levels'

    elif 'plevel' in var_dict.keys():
        dataset_str = 'reanalysis-era5-pressure-levels'
        retrieve_dict['pressure_level'] = var_dict['plevel']

    # ---------------- Download hourly data and compute daily average

    # If daily data file already exists, skip downloading & averaging
    if not os.path.exists(daily_fpath):

        print("\nDownloading data for {}...\n".format(args.var))

        if os.path.exists(download_path):
            print("Removing pre-existing NetCDF file at {}". format(download_path))
            os.remove(download_path)

        cds.retrieve(dataset_str, retrieve_dict, download_path)
        print('\n\nDownload completed.')

        print('\n\nComputing daily averages... ', end='', flush=True)
        da = xr.open_dataarray(download_path)

        if 'expver' in da.coords:
            da = utils.fix_near_real_time_era5_coords(da)

        da_daily = da.resample(time='1D').reduce(np.mean)

        if args.var == 'zg500' or args.var == 'zg250':
            da_daily = da_daily / 9.80665

        if args.var == 'tos':
            # Replace every value outside of SST < 1000 with zeros (the ERA5 masked values)
            da_daily = da_daily.where(da_daily < 1000., 0)

        print(np.unique(da_daily.data)[[0, -1]])  # TEMP
        print('saving new daily year file... ', end='', flush=True)
        da_daily.to_netcdf(daily_fpath)
        os.remove(download_path)
        print('Done.')

    # ---------------- Regrid & save

    # For wind data, use a separate script for rotating and regridding
    if args.var != 'uas' and args.var != 'vas':

        for hemisphere in hemispheres:

            sic_day_folder = os.path.join('data', 'siconca', hemisphere)
            sic_day_fname = 'ice_conc_{}_ease2-250_cdr-v2p0_197901021200.nc'.format(hemisphere)
            sic_day_fpath = os.path.join(sic_day_folder, sic_day_fname)

            if not os.path.exists(sic_day_fpath):
                print("\nDownloading single daily SIC netCDF file for regridding ERA5 data to EASE grid...\n")

                # Ignore "Missing CF-netCDF ancially data variable 'status_flag'" warning
                warnings.simplefilter("ignore", UserWarning)

                retrieve_sic_day_cmd = 'wget -m -nH --cut-dirs=6 -P {} ' \
                    'ftp://osisaf.met.no/reprocessed/ice/conc/v2p0/1979/01/{}'
                os.system(retrieve_sic_day_cmd.format(sic_day_folder, sic_day_fname))

                print('Done.')

            # Load a single SIC map to obtain the EASE grid for regridding ERA data
            sic_EASE_cube = iris.load_cube(sic_day_fpath, 'sea_ice_area_fraction')

            # Convert EASE coord units to metres for regridding
            sic_EASE_cube.coord('projection_x_coordinate').convert_units('meters')
            sic_EASE_cube.coord('projection_y_coordinate').convert_units('meters')

            print("\nRegridding and saving {} {} reanalysis data... ".format(hemisphere, args.var), end='', flush=True)
            tic = time.time()

            cube = iris.load_cube(daily_fpath)
            cube = utils.assignLatLonCoordSystem(cube)

            # regrid onto the EASE grid
            cube_ease = cube.regrid(sic_EASE_cube, iris.analysis.Linear())

            toc = time.time()
            print("Done in {:.3f}s.".format(toc - tic))

            print("Saving {} regridded data... ".format(hemisphere), end='', flush=True)
            if os.path.exists(daily_ease_fpath[hemisphere]) and overwrite:
                os.remove(daily_ease_fpath[hemisphere])
            iris.save(cube_ease, daily_ease_fpath[hemisphere])
            print("Done.")

        os.remove(daily_fpath)
