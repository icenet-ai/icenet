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
        cube = iris.load_cube(datafile)
        cube.coord('latitude').coord_system = \
            iris.coord_systems.GeogCS(6367470.0)
        cube.coord('longitude').coord_system = \
            iris.coord_systems.GeogCS(6367470.0)

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
    # TODO: Refactor - check this is the right version
    def gridcell_angles_from_dim_coords(cube):
        """
        Wrapper for :func:`~iris.analysis.cartography.gridcell_angles`
        that derives the 2D X and Y lon/lat coordinates from 1D X and Y
        coordinates identifiable as 'x' and 'y' axes

        The provided cube must have a coordinate system so that its
        X and Y coordinate bounds (which are derived if necessary)
        can be converted to lons and lats
        """

        # get the X and Y dimension coordinates for the cube
        x_coord = cube.coord(axis='x', dim_coords=True)
        y_coord = cube.coord(axis='y', dim_coords=True)

        # add bounds if necessary
        if not x_coord.has_bounds():
            x_coord = x_coord.copy()
            x_coord.guess_bounds()
        if not y_coord.has_bounds():
            y_coord = y_coord.copy()
            y_coord.guess_bounds()

        # get the grid cell bounds
        x_bounds = x_coord.bounds
        y_bounds = y_coord.bounds
        nx = x_bounds.shape[0]
        ny = y_bounds.shape[0]

        # make arrays to hold the ordered X and Y bound coordinates
        x = np.zeros((ny, nx, 4))
        y = np.zeros((ny, nx, 4))

        # iterate over the bounds (in order BL, BR, TL, TR), mesh them and
        # put them into the X and Y bound coordinates (in order BL, BR, TR, TL)
        c = [0, 1, 3, 2]
        cind = 0
        for yi in [0, 1]:
            for xi in [0, 1]:
                xy = np.meshgrid(x_bounds[:, xi], y_bounds[:, yi])
                x[:, :, c[cind]] = xy[0]
                y[:, :, c[cind]] = xy[1]
                cind += 1

        # convert the X and Y coordinates to longitudes and latitudes
        source_crs = cube.coord_system().as_cartopy_crs()
        target_crs = ccrs.PlateCarree()
        pts = target_crs.transform_points(source_crs, x.flatten(), y.flatten())
        lons = pts[:, 0].reshape(x.shape)
        lats = pts[:, 1].reshape(x.shape)

        # get the angles
        angles = iris.analysis.cartography.gridcell_angles(lons, lats)

        # add the X and Y dimension coordinates from the cube to the angles cube
        angles.add_dim_coord(y_coord, 0)
        angles.add_dim_coord(x_coord, 1)

        # if the cube's X dimension preceeds its Y dimension
        # transpose the angles to match
        if cube.coord_dims(x_coord)[0] < cube.coord_dims(y_coord)[0]:
            angles.transpose()

        return angles

    def invert_gridcell_angles(angles):
        """
        Negate a cube of gridcell angles in place, transforming
        gridcell_angle_from_true_east <--> true_east_from_gridcell_angle
        """
        angles.data *= -1

        names = ['true_east_from_gridcell_angle',
                 'gridcell_angle_from_true_east']
        name = angles.name()
        if name in names:
            angles.rename(names[1 - names.index(name)])

    def rotate_grid_vectors(u_cube, v_cube, angles):
        """
        Wrapper for :func:`~iris.analysis.cartography.rotate_grid_vectors`
        that can rotate multiple masked spatial fields in one go by iterating
        over the horizontal spatial axes in slices
        """
        # lists to hold slices of rotated vectors
        u_r_all = iris.cube.CubeList()
        v_r_all = iris.cube.CubeList()

        # get the X and Y dimension coordinates for each source cube
        u_xy_coords = [u_cube.coord(axis='x', dim_coords=True),
                       u_cube.coord(axis='y', dim_coords=True)]
        v_xy_coords = [v_cube.coord(axis='x', dim_coords=True),
                       v_cube.coord(axis='y', dim_coords=True)]

        # iterate over X, Y slices of the source cubes, rotating each in turn
        for u, v in zip(u_cube.slices(u_xy_coords, ordered=False),
                        v_cube.slices(v_xy_coords, ordered=False)):
            u_r, v_r = iris.analysis.cartography.rotate_grid_vectors(u, v,
                                                                     angles)
            u_r_all.append(u_r)
            v_r_all.append(v_r)

        # return the slices, merged back together into a pair of cubes
        return (u_r_all.merge_cube(), v_r_all.merge_cube())

    # read ERA5 UV10 for the NH
    u10 = iris.load_cube(config.u10_latlon_path)
    v10 = iris.load_cube(config.v10_latlon_path)

    # read EASE-grid sea ice
    # sic = iris.load_cube('ice_conc_nh_ease2-250_icdr_v2p0_202002.nc',
    #                      iris.Constraint(cube_func=lambda cube:cube.name() == 'sea_ice_area_fraction'))

    sic = iris.load_cube(
        os.path.join(config.ice_data_folder, 'avg_sic_1979_01.nc'),
        'sea_ice_area_fraction')
    for coord in ['x', 'y']:
        sic.coord('projection_{}_coordinate'.format(coord)).convert_units(
            'm')

    for cube in [u10, v10]:
        cube.coord('latitude').coord_system = iris.coord_systems.GeogCS(
            6367470.0)
        cube.coord('longitude').coord_system = iris.coord_systems.GeogCS(
            6367470.0)

    # regrid the winds onto the EASE grid
    u10_ease = u10.regrid(sic, iris.analysis.Linear())
    v10_ease = v10.regrid(sic, iris.analysis.Linear())

    # get the gridcell angles
    angles = gridcell_angles_from_dim_coords(sic)

    # invert the angles
    invert_gridcell_angles(angles)

    # rotate the winds using the angles
    u10_ease_r, v10_ease_r = rotate_grid_vectors(u10_ease, v10_ease, angles)

    # save the regridded winds
    iris.save(u10_ease, 'era5_monthly_u10_2018_ease_grid_unrotated.nc')
    iris.save(v10_ease, 'era5_monthly_v10_2018_ease_grid_unrotated.nc')

    # save the rotated winds
    iris.save(u10_ease_r, 'era5_monthly_u10_2018_ease_grid.nc')
    iris.save(v10_ease_r, 'era5_monthly_v10_2018_ease_grid.nc')

    # are the wind speeds the same after rotation?
    print('Are the wind speeds the same after rotation?: {}'.format(
        np.all(np.isclose(ws_ease.data, ws_ease_r.data))))

    # project idealised data
    u10 = u10[0]
    v10 = v10[0]

    u10.data.fill(1)
    v10.data.fill(0)
    u10_ease = u10.regrid(sic, iris.analysis.Linear())
    v10_ease = v10.regrid(sic, iris.analysis.Linear())
    u10_ease_r, v10_ease_r = iris.analysis.cartography.rotate_grid_vectors(
        u10_ease, v10_ease, angles)
    iris.save(u10_ease_r, 'era5_monthly_u10_ease_grid_u_only.nc')
    iris.save(v10_ease_r, 'era5_monthly_v10_ease_grid_u_only.nc')

    u10.data.fill(0)
    v10.data.fill(1)
    u10_ease = u10.regrid(sic, iris.analysis.Linear())
    v10_ease = v10.regrid(sic, iris.analysis.Linear())
    u10_ease_r, v10_ease_r = iris.analysis.cartography.rotate_grid_vectors(
        u10_ease, v10_ease, angles)
    iris.save(u10_ease_r, 'era5_monthly_u10_ease_grid_v_only.nc')
    iris.save(v10_ease_r, 'era5_monthly_v10_ease_grid_v_only.nc')

    u10.data.fill(1)
    v10.data.fill(1)
    u10_ease = u10.regrid(sic, iris.analysis.Linear())
    v10_ease = v10.regrid(sic, iris.analysis.Linear())
    u10_ease_r, v10_ease_r = iris.analysis.cartography.rotate_grid_vectors(
        u10_ease, v10_ease, angles)
    iris.save(u10_ease_r, 'era5_monthly_u10_ease_grid_uv.nc')
    iris.save(v10_ease_r, 'era5_monthly_v10_ease_grid_uv.nc')
