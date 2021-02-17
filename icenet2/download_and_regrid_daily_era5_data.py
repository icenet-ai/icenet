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
import config
import icenet2_utils
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
args = parser.parse_args()

variable = args.var

# variable = 'zg500'

################################################################################


def assignLatLonCoordSystem(cube):
    ''' Assign coordinate system to iris cube to allow regridding. '''

    cube.coord('latitude').coord_system = iris.coord_systems.GeogCS(6367470.0)
    cube.coord('longitude').coord_system = iris.coord_systems.GeogCS(6367470.0)

    return cube


################################################################################

# Whether to skip variables that have already been downloaded or regridded
overwrite = True

area = [90, -180, 0, 180]  # Latitude/longitude boundaries to download

# Which years to download
years = [
    ['1979', '1980', '1981', '1982', '1983',
     '1984', '1985', '1986', '1987', '1988'],
    ['1989', '1990', '1991', '1992', '1993',
     '1994', '1995', '1996', '1997', '1998'],
    ['1999', '2000', '2001', '2002', '2003',
     '2004', '2005', '2006', '2007', '2008'],
    ['2009', '2010', '2011', '2012', '2013',
     '2014', '2015', '2016', '2017', '2018',
     '2019', '2020']
]

# TEMP yearwise download
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

# Download contains ERA5T with 'expver' coord -- remove 'expver' dim and concatenate into one array
# fix_near_real_time_era5_coords = True
#
# plot_wind_before_and_after_rot = True  # Save quiver plot of before and after wind rotation to EASE
# verify_wind_magnitude = True  # Check wind magnitude before and after is the same

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
}

var_dict = variables[variable]

# ---------------- Download

cds = cdsapi.Client()

for year_chunk in years:

    year_start = year_chunk[0]
    year_end = year_chunk[-1]

    print('\n\n' + year_start + '-' + year_end + '\n\n')

    var_folder = os.path.join(config.folders['data'], variable)
    if not os.path.exists(var_folder):
        os.makedirs(var_folder)

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

    download_path = os.path.join(var_folder,
                                 '{}_latlon_hourly_{}_{}.nc'.format(variable, year_start, year_end))

    print("\nDownloading data for {}...\n".format(variable))

    if os.path.exists(download_path):
        if not overwrite:
            print("Skipping due to existing file: {}". format(download_path))
            continue
        else:
            print("Removing pre-existing NetCDF file at {}". format(download_path))
            os.remove(download_path)

    # ---------------- Download

    cds.retrieve(dataset_str, retrieve_dict, download_path)
    print('\n\nDownload completed.')

    # ---------------- Compute daily average

    # TODO: augment filename by year_chunk
    print('\n\nComputing daily averages... ', end='', flush=True)
    da = xr.open_dataarray(download_path)
    da_daily = da.resample(time='1D').reduce(np.mean)

    print('saving new daily year file... ', end='', flush=True)
    daily_fpath = os.path.join(var_folder, '{}_latlon_{}_{}.nc'.format(variable, year_start, year_end))
    da_daily.to_netcdf(daily_fpath)
    os.remove(download_path)
    print('Done.')

    # ---------------- Regrid & save

    sic_day_fpath = os.path.join(config.folders['data'], 'ice_conc_nh_ease2-250_cdr-v2p0_197901021200.nc')

    if not os.path.exists(sic_day_fpath):
        print("\nDownloading single daily SIC netCDF file for regridding ERA5 data to EASE grid...\n")

        # Ignore "Missing CF-netCDF ancially data variable 'status_flag'" warning
        warnings.simplefilter("ignore", UserWarning)

        retrieve_sic_day_cmd = 'wget -m -nH --cut-dirs=6 -P {} ' \
            'ftp://osisaf.met.no/reprocessed/ice/conc/v2p0/1979/01/ice_conc_nh_ease2-250_cdr-v2p0_197901021200.nc'
        os.system(retrieve_sic_day_cmd.format(config.folders['data']))

        print('Done.')

    # Load a single SIC map to obtain the EASE grid for regridding ERA data
    sic_EASE_cube = iris.load_cube(sic_day_fpath, 'sea_ice_area_fraction')

    # Convert EASE coord units to metres for regridding
    sic_EASE_cube.coord('projection_x_coordinate').convert_units('meters')
    sic_EASE_cube.coord('projection_y_coordinate').convert_units('meters')

    print("\nRegridding and saving {} reanalysis data... ".format(variable), end='', flush=True)
    tic = time.time()

    cube = iris.load_cube(daily_fpath)
    cube = assignLatLonCoordSystem(cube)

    # regrid onto the EASE grid
    cube_ease = cube.regrid(sic_EASE_cube, iris.analysis.Linear())

    if variable == 'zg500' or variable == 'zg250':
        print('Converting geopotential to geopotential height... ', end='', flush=True)
        cube_ease = cube_ease / 9.80665

    toc = time.time()
    print("Done in {:.3f}s.".format(toc - tic))

    print("Saving regridded data... ", end='', flush=True)
    daily_ease_fpath = os.path.join(var_folder, '{}_{}_{}.nc'.format(variable, year_start, year_end))
    iris.save(cube_ease, daily_ease_fpath)
    os.remove(daily_fpath)
    print("Done.")
