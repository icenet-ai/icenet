import concurrent
import datetime as dt
import logging
import os
import re
import tempfile

from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor
from itertools import product

from icenet2.data.sic.mask import Masks
from icenet2.data.producers import Downloader
from icenet2.data.utils import assign_lat_lon_coord_system, \
    gridcell_angles_from_dim_coords, \
    invert_gridcell_angles, \
    rotate_grid_vectors
from icenet2.utils import run_command

import iris


class ClimateDownloader(Downloader):

    def __init__(self, *args,
                 dates=(),
                 delete_tempfiles=True,
                 max_threads=1,
                 pressure_levels=(),
                 var_names=(),
                 **kwargs):
        super().__init__(*args, **kwargs)

        self._sic_ease_cubes = dict()
        self._files_downloaded = []

        self._dates = list(dates)
        self._masks = Masks(north=self.north, south=self.south)
        self._max_threads = max_threads
        self._pressure_levels = list(pressure_levels)
        self._var_names = list(var_names)

        self._delete = delete_tempfiles

        assert len(self._var_names), "No variables requested"
        assert len(self._pressure_levels) == len(self._var_names), \
            "# of pressures must match # vars"

        self._validate_config()

    def _validate_config(self):
        if self.hemisphere_str in os.path.split(self.base_path):
            raise RuntimeError("Don't include hemisphere string {} in "
                               "base path".format(self.hemisphere_str))

    # TODO: add native subprocessing for parallelism
    def download(self):
        logging.info("Building request(s), downloading and daily averaging "
                     "from {} API".format(self.identifier.upper()))

        with ThreadPoolExecutor(max_workers=self._max_threads) \
                as executor:
            futures = []
            for idx, var_name in enumerate(self.var_names):
                pressures = [None] if not self.pressure_levels[idx] else \
                    self._pressure_levels[idx]

                dates_per_request = self._get_dates_for_request()

                for var_prefix, pressure, req_date in \
                        product([var_name], pressures, dates_per_request):

                    future = executor.submit(self._single_download,
                                             var_prefix,
                                             pressure,
                                             req_date)
                    futures.append(future)

            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logging.error(e)

        logging.info("{} daily files downloaded".
                     format(len(self._files_downloaded)))

    @abstractmethod
    def _get_dates_for_request(self):
        raise NotImplementedError("Missing {} implementation".format(__name__))

    @abstractmethod
    def _single_download(self, var_prefix, pressure, req_date):
        raise NotImplementedError("Missing {} implementation".format(__name__))

    @property
    def sic_ease_cube(self):
        if self._hemisphere not in self._sic_ease_cubes:
            sic_day_fname = 'ice_conc_{}_ease2-250_cdr-v2p0_197901021200.nc'. \
                format(self.hemisphere_str[0])
            sic_day_path = os.path.join(self.get_data_var_folder("siconca"),
                                 sic_day_fname)
            if not os.path.exists(sic_day_path):
                logging.info("Downloading single daily SIC netCDF file for "
                             "regridding ERA5 data to EASE grid...")

                retrieve_sic_day_cmd = 'wget -m -nH --cut-dirs=6 -P {} ' \
                                       'ftp://osisaf.met.no/reprocessed/ice/' \
                                       'conc/v2p0/1979/01/{}'.\
                    format(self.get_data_var_folder("siconca"), sic_day_fname)

                run_command(retrieve_sic_day_cmd)

            # Load a single SIC map to obtain the EASE grid for
            # regridding ERA data
            self._sic_ease_cubes[self._hemisphere] = \
                iris.load_cube(sic_day_path, 'sea_ice_area_fraction')

            # Convert EASE coord units to metres for regridding
            self._sic_ease_cubes[self._hemisphere].coord(
                'projection_x_coordinate').convert_units('meters')
            self._sic_ease_cubes[self._hemisphere].coord(
                'projection_y_coordinate').convert_units('meters')
        return self._sic_ease_cubes[self._hemisphere]

    def regrid(self,
               files=None):
        for datafile in self._files_downloaded if not files else files:
            (datafile_path, datafile_name) = os.path.split(datafile)
            new_datafile = os.path.join(datafile_path,
                                        re.sub(r'^latlon_', '', datafile_name))

            if os.path.exists(new_datafile):
                logging.debug("Skipping {} as {} already exists".
                    format(datafile, os.path.basename(new_datafile)))
                continue

            logging.debug("Regridding {}".format(datafile))

            try:
                cube = iris.load_cube(datafile)
                cube = assign_lat_lon_coord_system(cube)
                cube_ease = cube.regrid(
                    self.sic_ease_cube, iris.analysis.Linear())
            except iris.exceptions.CoordinateNotFoundError:
                logging.warning("{} has no coordinates...".
                                format(datafile_name))
                if self.delete:
                    logging.debug("Deleting failed file {}...".
                                  format(datafile_name))
                    os.unlink(datafile)
                continue

            self.additional_regrid_processing(datafile, cube_ease)

            # TODO: filename chain can be handled better for sharing between
            #  methods
            logging.info("Saving regridded data to {}... ".format(new_datafile))
            iris.save(cube_ease, new_datafile)

            if self.delete:
                logging.info("Removing {}".format(datafile))
                os.remove(datafile)

    @abstractmethod
    def additional_regrid_processing(self, datafile, cube_ease):
        pass

    def rotate_wind_data(self,
                         apply_to=("uas", "vas"),
                         manual_files=None):
        assert len(apply_to) == 2, "Too many wind variables supplied: {}, " \
                                   "there should only be two.".\
            format(", ".join(apply_to))

        angles = gridcell_angles_from_dim_coords(self.sic_ease_cube)
        invert_gridcell_angles(angles)

        logging.info("Rotating wind data in {}".format(
            " ".join([self.get_data_var_folder(v) for v in apply_to])))

        wind_files = {}

        for var in apply_to:
            source = self.get_data_var_folder(var)

            file_source = self._files_downloaded \
                if not manual_files else manual_files

            latlon_files = [df for df in file_source if source in df]
            wind_files[var] = sorted([
                re.sub(r'latlon_', '', df) for df in latlon_files
                if os.path.dirname(df).split(os.sep)[-2] == var],
                key=lambda x: dt.date(*[int(el) for el in
                                        re.search(
                                            r'^(?:\w+_)(\d+)_(\d+)_(\d+).nc',
                                      os.path.basename(x)).groups()])
            )
            logging.info("{} files for {}".format(len(wind_files[var]), var))

        # NOTE: we're relying on apply_to having equal datasets
        assert len(wind_files[apply_to[0]]) == len(wind_files[apply_to[1]]), \
            "The wind file datasets are unequal in length"

        # validation
        for idx, wind_file_0 in enumerate(wind_files[apply_to[0]]):
            wind_file_1 = wind_files[apply_to[1]][idx]

            wd0 = re.sub(r'^{}_'.format(apply_to[0]), '',
                         os.path.basename(wind_file_0))

            if not wind_file_1.endswith(wd0):
                logging.error("Wind file array is not valid:".format(
                    zip(wind_files)))
                raise RuntimeError("{} is not at the end of {}, something is "
                                   "wrong".format(wd0, wind_file_1))

        for idx, wind_file_0 in enumerate(wind_files[apply_to[0]]):
            wind_file_1 = wind_files[apply_to[1]][idx]

            logging.info("Rotating {} and {}".format(wind_file_0, wind_file_1))

            wind_cubes = dict()
            wind_cubes_r = dict()

            wind_cubes[apply_to[0]] = iris.load_cube(wind_file_0)
            wind_cubes[apply_to[1]] = iris.load_cube(wind_file_1)

            wind_cubes_r[apply_to[0]], wind_cubes_r[apply_to[1]] = \
                rotate_grid_vectors(
                    wind_cubes[apply_to[0]], wind_cubes[apply_to[1]], angles,
                )

            # Original implementation is in danger of lost updates
            # due to potential lazy loading
            for i, name in enumerate([wind_file_0, wind_file_1]):
                # NOTE: implementation with tempfile caused problems on NFS
                # mounted filesystem, so avoiding in place of letting iris do it
                temp_name = os.path.join(os.path.split(name)[0],
                                         "temp.{}".format(
                                             os.path.basename(name)))
                logging.debug("Writing {}".format(temp_name))

                iris.save(wind_cubes_r[apply_to[i]], temp_name)
                os.replace(temp_name, name)

    @property
    def delete(self):
        return self._delete

    @property
    def dates(self):
        return self._dates

    @property
    def var_names(self):
        return self._var_names

    @property
    def pressure_levels(self):
        return self._pressure_levels
