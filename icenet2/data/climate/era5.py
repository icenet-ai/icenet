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

import cdsapi as cds
import iris
import numpy as np
import xarray as xr

from icenet2.constants import *
from icenet2.data.utils import assign_lat_lon_coord_system
from icenet2.utils import get_folder, run_command


def download(var_name,
             cdi_name,
             plevel=None,
             hemispheres=[NORTH],
             years=[],
             months=[],
             days=[],
             times=[],
             overwrite=False,
             da_preavg_process=None,
             regrid_method=None):
    # FIXME: confirmed, but the year start year end naming is a bit weird,
    #  hang up from the icenet port but we might want to consider relevance,
    #  it remains purely for compatibility with existing data

    # TODO: This is download and average for dailies, but could be easily
    #  abstracted for different temporal averaging
    logging.info("Building request(s), downloading and daily averaging from "
                 "CDS API")

    dailies = []

    for hemi in hemispheres:
        for year in years:
            logging.debug("Processing data from {} for {}".
                      format(year, HEMISPHERE_STRINGS[hemi]))

            var_hem_folder = get_folder(
                'data', HEMISPHERE_STRINGS[hemi], var_name)

            download_path = os.path.join(var_hem_folder,
                                         '{}_latlon_hourly_{}_{}.nc'.format(
                                             var_name, year, year))
            daily_fpath = os.path.join(var_hem_folder, '{}_latlon_{}_{}.nc'.
                                       format(var_name, year, year))

            retrieve_dict = {
                'product_type': 'reanalysis',
                'variable': cdi_name,
                'year': year,
                'month': months,
                'day': days,
                'time': times,
                'format': 'netcdf',
                'area': HEMISPHERE_LOCATIONS[hemi]
            }

            dataset = 'reanalysis-era5-single-levels'

            if plevel:
                dataset = 'reanalysis-era5-pressure-levels'
                retrieve_dict['pressure_level'] = plevel

            # ---------------- Download hourly data and compute daily average

            # If daily data file already exists, skip downloading & averaging
            if not os.path.exists(daily_fpath):
                logging.info("Downloading data for {}...".format(var_name))

                # TODO: This probably isn't required, unless failure (see below)
                if os.path.exists(download_path):
                    logging.info("Removing pre-existing NetCDF file at {}".
                                 format(download_path))
                    os.remove(download_path)

                cds.retrieve(dataset, retrieve_dict, download_path)
                logging.debug('Download completed.')

                logging.info('Computing daily averages...')
                da = xr.open_dataarray(download_path)

                if 'expver' in da.coords:
                    raise RuntimeError("fix_near_real_time_era5_coords no "
                                       "longer exists in the codebase for "
                                       "expver in coordinates")

                da_daily = da.resample(time='1D').reduce(np.mean)

                # if var_name == 'zg500' or var_name == 'zg250':
                #   da_daily = da_daily / 9.80665

                # if var_name == 'tos':
                #     # Replace every value outside of SST < 1000 with
                #    zeros (the ERA5 masked values)
                #     da_daily = da_daily.where(da_daily < 1000., 0)
                if da_preavg_process:
                    da_daily = da_preavg_process(da_daily)

                logging.debug("Saving new daily year file...")
                da_daily.to_netcdf(daily_fpath)
                dailies.append(daily_fpath)
                # TODO: See previous TODO
                os.remove(download_path)

    logging.info("{} daily files produced")
    return dailies


def regrid_data(files,
                remove_original=False):
    # TODO: this is a bit messy to account for compatibility with existing
    #  data, so on fresh run from start we'll refine it all
    sic_ease_cubes = dict()

    for datafile in files:
        (datafile_path, datafile_name) = os.path.split(datafile)
        hemisphere_path = datafile_path.split(os.sep)[:-1]
        hemisphere = datafile_path.split(os.sep)[1]

        sic_day_folder = os.path.join(*hemisphere_path, "siconca")
        sic_day_fname = 'ice_conc_{}_ease2-250_cdr-v2p0_197901021200.nc'.\
            format(hemisphere)
        sic_day_path = os.path.join(sic_day_folder, sic_day_fname)

        if not os.path.exists(sic_day_path):
            logging.info("Downloading single daily SIC netCDF file for "
                         "regridding ERA5 data to EASE grid...")

            retrieve_sic_day_cmd = 'wget -m -nH --cut-dirs=6 -P {} ' \
                'ftp://osisaf.met.no/reprocessed/ice/conc/v2p0/1979/01/{}'
            run_command(retrieve_sic_day_cmd.
                        format(sic_day_folder, sic_day_fname))

        if hemisphere not in sic_ease_cubes:
            # Load a single SIC map to obtain the EASE grid for
            # regridding ERA data
            sic_ease_cubes[hemisphere] = iris.load_cube(sic_day_path,
                                                        'sea_ice_area_fraction')

            # Convert EASE coord units to metres for regridding
            sic_ease_cubes[hemisphere].coord(
                'projection_x_coordinate').convert_units('meters')
            sic_ease_cubes[hemisphere].coord(
                'projection_y_coordinate').convert_units('meters')

        logging.info("Regridding {}".format(datafile))
        cube = assign_lat_lon_coord_system(iris.load_cube(datafile))

        # regrid onto the EASE grid
        cube_ease = cube.regrid(sic_ease_cubes[hemisphere],
                                iris.analysis.Linear())

        new_datafile = os.path.join(datafile_path,
                                    re.sub(r'_latlon_', '_', datafile_name))
        logging.info("Saving regridded data to {}... ".format(new_datafile))
        iris.save(cube_ease, new_datafile)

        if remove_original:
            logging.info("Removing {}".format(datafile))
            os.remove(datafile)


def regrid_wind_data(files,
                     remove_original=False):
    pass


# TODO: refactor
def rotate_wind_data(files,
                     remove_original=False):
    sic_day_fpath = os.path.join(config.obs_data_folder,
                                 'ice_conc_nh_ease2-250_cdr-v2p0_197901021200.nc')

    if not os.path.exists(sic_day_fpath):
        print(
            "Downloading single daily SIC netCDF file for regridding ERA5 data to EASE grid...\n\n")

        # Ignore "Missing CF-netCDF ancially data variable 'status_flag'" warning
        warnings.simplefilter("ignore", UserWarning)

        retrieve_sic_day_cmd = 'wget -m -nH --cut-dirs=6 -P {} ' \
                               'ftp://osisaf.met.no/reprocessed/ice/conc/v2p0/1979/01/ice_conc_nh_ease2-250_cdr-v2p0_197901021200.nc'
        os.system(retrieve_sic_day_cmd.format(config.obs_data_folder))

        print('Done.')

    # Load a single SIC map to obtain the EASE grid for regridding ERA data
    sic_EASE_cube = iris.load_cube(sic_day_fpath, 'sea_ice_area_fraction')

    # Convert EASE coord units to metres for regridding
    sic_EASE_cube.coord('projection_x_coordinate').convert_units('meters')
    sic_EASE_cube.coord('projection_y_coordinate').convert_units('meters')

    land_mask = np.load(
        os.path.join(config.mask_data_folder, config.land_mask_filename))

    # get the gridcell angles
    angles = utils.gridcell_angles_from_dim_coords(sic_EASE_cube)

    # invert the angles
    utils.invert_gridcell_angles(angles)

    # Rotate, regrid, and save
    ################################################################################

    tic = time.time()

    print(f'\nRotating wind data in {wind_data_folder}')
    wind_cubes = {}
    for var in ['uas', 'vas']:
        EASE_path = os.path.join(wind_data_folder, f'{var}{fname_suffix}')
        wind_cubes[var] = iris.load_cube(EASE_path)

    # rotate the winds using the angles
    wind_cubes_r = {}
    wind_cubes_r['uas'], wind_cubes_r['vas'] = utils.rotate_grid_vectors(
        wind_cubes['uas'], wind_cubes['vas'], angles)

    # save the new cube
    for var, cube_ease_r in wind_cubes_r.items():
        EASE_path = os.path.join(wind_data_folder, f'{var}{fname_suffix}')

        if os.path.exists(EASE_path) and overwrite:
            print("Removing existing file: {}".format(EASE_path))
            os.remove(EASE_path)
        elif os.path.exists(EASE_path) and not overwrite:
            print("Skipping due to existing file: {}".format(EASE_path))
            sys.exit()

        iris.save(cube_ease_r, EASE_path)

    if gen_video:
        for var in ['uas', 'vas']:
            print(f'generating video for {var}')
            EASE_path = os.path.join(wind_data_folder, f'{var}{fname_suffix}')

            video_folder = os.path.join('videos', 'wind')
            if not os.path.exists(video_folder):
                os.makedirs(video_folder)
            fname = '{}_{}.mp4'.format(wind_data_folder.replace('/', '_'), var)
            video_path = os.path.join(video_folder, fname)

            utils.xarray_to_video(
                da=next(iter(xr.open_dataset(EASE_path).data_vars.values())),
                video_path=video_path,
                fps=6,
                mask=land_mask,
                figsize=7,
                dpi=100,
            )

    toc = time.time()
    print("Done in {:.3f}s.".format(toc - tic))
