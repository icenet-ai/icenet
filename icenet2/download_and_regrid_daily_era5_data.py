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

# land_mask = np.load(os.path.join(config.folders['masks'], config.fnames['land_mask']))
# vars = ['tas', 'tos', 'psl', 'ta500', 'zg250', 'zg500']
# cmaps = ['Reds', 'Reds', 'bone', 'Reds', 'bone', 'bone']
# for var, cmap in zip(vars, cmaps):
#     # var='tos'
#     print('\n{}:'.format(var))
#     # var='ta500'
#     ds = xr.open_mfdataset([os.path.join('data/{}'.format(var), f) for f in sorted(os.listdir('data/{}'.format(var)))], combine='by_coords')
#     da = next(iter(ds.data_vars.values()))
#     # da.time.values[15305]
#     # px.imshow(da.isel(time=15060))
#     # px.imshow(da.isel(time=15061))
#     # px.imshow(da.isel(time=15305))
#     # np.unique(np.where(da > 1e35)[0])
#     climatology = da.groupby('time.dayofyear', restore_coord_dims=True).mean()
#     # anom = (da.groupby('time.dayofyear') - climatology).compute()
#     # cmap = 'seismic'
#     # anom = anom.sel(time=slice('2012-01-01', '2012-12-31'))
#
#     # vpath = 'videos/era5/{}_2012.mp4'.format(var)
#     # vpath = 'videos/era5/{}.mp4'.format(var)
#     # utils.xarray_to_video(anom, vpath, 15, land_mask, data_type='anom', cmap=cmap, figsize=15)
#
#     vpath = 'videos/era5/{}_climatology.mp4'.format(var)
#     # vpath = 'videos/era5/{}.mp4'.format(var)
#     climatology = climatology.assign_coords({'dayofyear': utils.filled_daily_dates(datetime(2012,1,1), datetime(2013,1,1))})
#     climatology = climatology.rename({'dayofyear': 'time'})
#     climatology.data = np.array(climatology.data)
#     uniq = np.unique(climatology.data)
#     clim = [uniq[1], uniq[-1]]
#     utils.xarray_to_video(climatology, vpath, 15, land_mask, clim=clim, data_type='abs', cmap=cmap, figsize=15)
#
# raise ValueError('breaking here to stop running')

################################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--var', default='tas')
parser.add_argument('--hemisphere', default='nh')
args = parser.parse_args()

################################################################################

overwrite = False  # Whether to skip yearly files already downloaded/regridded

################################################################################


def assignLatLonCoordSystem(cube):
    ''' Assign coordinate system to iris cube to allow regridding. '''

    cube.coord('latitude').coord_system = iris.coord_systems.GeogCS(6367470.0)
    cube.coord('longitude').coord_system = iris.coord_systems.GeogCS(6367470.0)

    return cube


def fix_near_real_time_era5_coords(da):

    '''
    ERA5 data within several months of the present date is considered as a
    separate system, ERA5T. Downloads that contain both ERA5 and ERA5T data
    produce datasets with a length-2 'expver' dimension along axis 1, taking a value
    of 1 for ERA5 and a value of 5 for ERA5. This results in all-NaN values
    along latitude & longitude outside of the valid expver time span. This function
    finds the ERA5 and ERA5T time indexes and removes the expver dimension
    by concatenating the sub-arrays where the data is not NaN.
    '''

    if 'expver' in da.coords:
        # Find invalid time indexes in expver == 1 (ERA5) dataset
        arr = da.sel(expver=1).data
        arr = arr.reshape(arr.shape[0], -1)
        arr = np.sort(arr, axis=1)
        era5t_time_idxs = (arr[:, 1:] != arr[:, :-1]).sum(axis=1)+1 == 1
        era5t_time_idxs = (era5t_time_idxs) | (np.isnan(arr[:, 0]))

        era5_time_idxs = ~era5t_time_idxs

        da = xr.concat((da[era5_time_idxs, 0, :], da[era5t_time_idxs, 1, :]), dim='time')

        da = da.reset_coords('expver', drop=True)

        return da

    else:
        raise ValueError("'expver' not found in dataset.")


################################################################################

# Latitude/longitude boundaries to download
if args.hemisphere == 'nh':
    area = [90, -180, 0, 180]
elif args.hemisphere == 'sh':
    area = [-90, -180, 0, 180]

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

var_dict = variables[args.var]

# ---------------- Download

cds = cdsapi.Client()

for year_chunk in years:

    year_start = year_chunk[0]
    year_end = year_chunk[-1]

    print('\n\n' + year_start + '-' + year_end + '\n\n')

    var_folder = os.path.join('data', args.hemisphere, args.var)
    if not os.path.exists(var_folder):
        os.makedirs(var_folder)

    download_path = os.path.join(var_folder, '{}_latlon_hourly_{}_{}.nc'.format(args.var, year_start, year_end))
    daily_fpath = os.path.join(var_folder, '{}_latlon_{}_{}.nc'.format(args.var, year_start, year_end))
    daily_ease_fpath = os.path.join(var_folder, '{}_{}_{}.nc'.format(args.var, year_start, year_end))

    if os.path.exists(daily_ease_fpath) and not overwrite:
        print('Skipping this year chunk due to existing file at: {}'.format(daily_ease_fpath))
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

    print("\nDownloading data for {}...\n".format(args.var))

    if os.path.exists(download_path):
        print("Removing pre-existing NetCDF file at {}". format(download_path))
        os.remove(download_path)

    # ---------------- Download

    cds.retrieve(dataset_str, retrieve_dict, download_path)
    print('\n\nDownload completed.')

    # ---------------- Compute daily average

    print('\n\nComputing daily averages... ', end='', flush=True)
    da = xr.open_dataarray(download_path)

    if 'expver' in da.coords:
        da = fix_near_real_time_era5_coords(da)

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

    sic_day_folder = os.path.join('data', 'siconca', args.hemisphere)
    sic_day_fname = 'ice_conc_{}_ease2-250_cdr-v2p0_197901021200.nc'.format(args.hemisphere)
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

    print("\nRegridding and saving {} reanalysis data... ".format(args.var), end='', flush=True)
    tic = time.time()

    cube = iris.load_cube(daily_fpath)
    cube = assignLatLonCoordSystem(cube)

    # regrid onto the EASE grid
    cube_ease = cube.regrid(sic_EASE_cube, iris.analysis.Linear())

    toc = time.time()
    print("Done in {:.3f}s.".format(toc - tic))

    print("Saving regridded data... ", end='', flush=True)
    if os.path.exists(daily_ease_fpath) and overwrite:
        os.remove(daily_ease_fpath)
    iris.save(cube_ease, daily_ease_fpath)
    os.remove(daily_fpath)
    print("Done.")
