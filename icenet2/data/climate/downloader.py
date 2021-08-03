import logging
import os
import re

from abc import abstractmethod

from icenet2.data.producers import Downloader
from icenet2.data.utils import assign_lat_lon_coord_system
from icenet2.utils import Hemisphere, run_command

import iris
import numpy as np


class ClimateDownloader(Downloader):

    @abstractmethod
    def __init__(self, *args,
                 var_names=(),
                 pressure_levels=(),
                 dates=(),
                 **kwargs):
        super(Downloader, self).__init__(*args, **kwargs)

        self._sic_ease_cubes = dict()
        self._files_downloaded = []


        self._var_names = list(var_names)
        self._pressure_levels = list(pressure_levels)
        self._dates = list(dates)

        assert len(self._var_names), "No variables requested"
        assert len(self._pressure_levels) != len(self._var_names), \
            "# of pressures must match # vars"
        assert len(self._dates), "Need to download at least one days worth"

        self._validate_config()

    def _validate_config(self):
        if self.hemisphere_str in os.path.split(self.base_path):
            raise RuntimeError("Don't include hemisphere string {} in "
                               "base path".format(self.hemisphere_str))

    # TODO: refactor
    def get_sic_ease_cube(self, hemisphere):
        if hemisphere not in self._sic_ease_cubes:
            sic_day_folder = os.path.join(, "siconca")
            sic_day_fname = 'ice_conc_{}_ease2-250_cdr-v2p0_197901021200.nc'. \
                format(self.hemisphere_str)
            sic_day_path = os.path.join(sic_day_folder, sic_day_fname)

            if not os.path.exists(sic_day_path):
                logging.info("Downloading single daily SIC netCDF file for "
                             "regridding ERA5 data to EASE grid...")

                retrieve_sic_day_cmd = 'wget -m -nH --cut-dirs=6 -P {} ' \
                                       'ftp://osisaf.met.no/reprocessed/ice/' \
                                       'conc/v2p0/1979/01/{}'.format(
                                        sic_day_path, sic_day_fname)
                run_command(retrieve_sic_day_cmd.
                            format(sic_day_folder, sic_day_fname))

            # Load a single SIC map to obtain the EASE grid for
            # regridding ERA data
            self._sic_ease_cubes[hemisphere] = \
                iris.load_cube(sic_day_path, 'sea_ice_area_fraction')

            # Convert EASE coord units to metres for regridding
            self._sic_ease_cubes[hemisphere].coord(
                'projection_x_coordinate').convert_units('meters')
            self._sic_ease_cubes[hemisphere].coord(
                'projection_y_coordinate').convert_units('meters')
        return self._sic_ease_cubes[hemisphere]

    # TODO: refactor
    def regrid(self,
               remove_original=False):
        # TODO: this is a bit messy to account for compatibility with existing
        #  data, so on fresh run from start we'll refine it all
        for datafile in self._files_downloaded:
            (datafile_path, datafile_name) = os.path.split(datafile)
            hemisphere, var_name = datafile_path.split(os.sep)[-2:]

            sic_ease_cube = self.get_sic_ease_cube(hemisphere)

            logging.debug("Regridding {}".format(datafile))
            cube = assign_lat_lon_coord_system(iris.load_cube(datafile))
            cube_ease = cube.regrid(sic_ease_cube, iris.analysis.Linear())

            new_datafile = os.path.join(datafile_path,
                                        re.sub(r'_latlon_', '_', datafile_name))
            logging.info("Saving regridded data to {}... ".format(new_datafile))
            iris.save(cube_ease, new_datafile)

            if remove_original:
                logging.info("Removing {}".format(datafile))
                os.remove(datafile)

    # TODO: refactor
    def rotate_wind_data(self,
                         apply_to=("uas", "vas"),
                         remove_original=False):

        sic_ease_cube = self.get_sic_ease_cube(self.hemisphere_str)

        land_mask = np.load(self.get_data_var_folder("masks"),
                            config.land_mask_filename))

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

        print("Done in {:.3f}s.".format(toc - tic))


