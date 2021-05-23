import sys
import os
# sys.path.insert(0, os.path.abspath(os.path.join(__file__, os.pardir)))
sys.path.insert(0, os.path.join(os.getcwd(), 'icenet2'))  # if using jupyter kernel
import xarray as xr
import iris
# import iris_grib
import plotly.express as px
import warnings
import time
from ecmwfapi import ECMWFService
import numpy as np
from utils import assignLatLonCoordSystem
import argparse

# COMMAND LINE INPUT
################################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--init_date', default='2012-01-01')
args = parser.parse_args()

init_date = args.init_date

# USER INPUT SECTION
################################################################################

download_folder = os.path.join('data', 'forecasts', 'seas5', 'latlon')
if not os.path.exists(download_folder):
    os.makedirs(download_folder)

EASE_folder = os.path.join('data', 'forecasts', 'seas5', 'EASE')
if not os.path.exists(EASE_folder):
    os.makedirs(EASE_folder)

do_download = True  # Download the ECMWF C-3S historical SIC forecasts
do_regrid = True  # Convert from GRIB to NetCDF and regrid from lat/lon to 19km NH EASE
do_video = True  # Produce video of forecast

overwrite = False

print('\n\nDownloading init date {}\n\n'.format(init_date))

# ECMWF API FOR HIGH RES DATA
################################################################################

server = ECMWFService("mars", url="https://api.ecmwf.int/v1",
                      key="16715f1aecf7dd0d56d7b9d95ebf0e97",
                      email="tomand@bas.ac.uk")

request_dict = {
    'class': 'od',
    'date': init_date,
    # 'date': '2012-09-01/2020-09-01',
    # 'date': '/'.join(['2012-{:02d}-01'.format(month) for month in np.arange(1,13)]),
    'expver': 1,
    'levtype': 'sfc',
    'method': 1,
    'number': list(map(int, np.arange(0, 25))),
    'origin': 'ecmf',
    'param': '31.128',
    'step': list(map(int, np.arange(0, 24 * 93 + 1, 24))),  # Max 93-day lead time
    'grid': '0.25/0.25',
    'format': 'netcdf',
    'stream': 'mmsf',
    'system': 5,
    'time': '00:00:00',
    'type': 'fc',
    'target': "output"
}

################################################################################

if not os.path.exists(download_folder):
    os.makedirs(download_folder)

download_filename = 'seas5_{}'.format(init_date.replace('-', '_'))

download_path = os.path.join(download_folder, '{}.nc'.format('latlon_' + download_filename))
EASE_path = os.path.join(EASE_folder, '{}.nc'.format('ease_' + download_filename))

if do_download:
    print('Beginning download...', end='', flush=True)
    tic = time.time()

    if os.path.exists(download_path) and not overwrite:
        print('File exists - skipping.')

    elif os.path.exists(download_path) and overwrite:
        print('Deleting existing file.')
        os.remove(download_path)

    # File doesn't exist or was just deleted
    if not os.path.exists(download_path):
        server.execute(request_dict, download_path)

        dur = time.time() - tic
        print("Done in {}m:{:.0f}s.\n\n ".format(np.floor(dur / 60), dur % 60))

if do_regrid:

    print("Regridding to EASE... ", end='', flush=True)

    if os.path.exists(EASE_path) and not overwrite:
        print('File exists - skipping.')

    elif os.path.exists(EASE_path) and overwrite:
        print('Deleting existing file.')
        os.remove(EASE_path)

    # File doesn't exist or was just deleted
    if not os.path.exists(EASE_path):

        ### Get EASE data
        #####################################################################

        hemisphere = 'nh'

        sic_day_folder = os.path.join('data', hemisphere, 'siconca')
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
        sic_EASE_cube = iris.load_cube(sic_day_fpath, 'sea_ice_area_fraction')[0]

        sic_EASE_cube.ancillary_variables()[0].rename('foo')
        sic_EASE_cube.ancillary_variables()[1].rename('foo')

        sic_EASE_cube.remove_coord('time')

        # Convert EASE coord units to metres for regridding
        sic_EASE_cube.coord('projection_x_coordinate').convert_units('meters')
        sic_EASE_cube.coord('projection_y_coordinate').convert_units('meters')

        ### Regrid forecast data
        #####################################################################

        cube = iris.load_cube(download_path)
        cube = assignLatLonCoordSystem(cube)

        # Compute ensemble mean
        cube = cube.collapsed('ensemble_member', iris.analysis.MEAN)

        cube = cube.regrid(sic_EASE_cube, iris.analysis.Linear())

        # Save the regridded cube in order to open in Xarray
        if os.path.exists(EASE_path):
            os.remove(EASE_path)
        iris.save(cube, EASE_path)

        # TODO: convert hours to days?

        print("Done")

        if do_video:

            print("Generating forecast video... ", end='', flush=True)

            video_folder = os.path.join('videos', 'forecast_videos', 'seas5')
            if not os.path.exists(video_folder):
                os.makedirs(video_folder)

            video_fname = download_filename
            video_fpath = os.path.join(video_folder, video_fname + '.mp4')
            land_mask = np.load('data/nh/masks/land_mask.npy')

            ds = xr.open_dataset(EASE_path)
            da = next(iter(ds.data_vars.values()))

            from misc import xarray_to_video
            xarray_to_video(da, video_fpath, mask=land_mask, cmap='Blues_r', fps=5)

            print("Done")

print('Done.')
