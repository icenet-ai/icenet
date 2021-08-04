import logging
import os
import re
import tempfile

from abc import abstractmethod
from itertools import product

from icenet2.data.sic.mask import Masks
from icenet2.data.producers import Downloader
from icenet2.data.utils import assign_lat_lon_coord_system, \
    gridcell_angles_from_dim_coords, invert_gridcell_angles, rotate_grid_vectors
from icenet2.utils import run_command

import iris


class ClimateDownloader(Downloader):

    def __init__(self, *args,
                 var_names=(),
                 pressure_levels=(),
                 dates=(),
                 **kwargs):
        super().__init__(*args, **kwargs)

        self._sic_ease_cubes = dict()
        self._files_downloaded = []

        self._var_names = list(var_names)
        self._pressure_levels = list(pressure_levels)
        self._dates = list(dates)
        self._masks = Masks(north=self.north, south=self.south)

        assert len(self._var_names), "No variables requested"
        assert len(self._pressure_levels) == len(self._var_names), \
            "# of pressures must match # vars"

        self._validate_config()

    def _validate_config(self):
        if self.hemisphere_str in os.path.split(self.base_path):
            raise RuntimeError("Don't include hemisphere string {} in "
                               "base path".format(self.hemisphere_str))

    def download(self):
        logging.info("Building request(s), downloading and daily averaging "
                     "from {} API".format(self.identifier.upper()))

        for idx, var_name in enumerate(self.var_names):
            pressures = [None] if not self.pressure_levels[idx] else \
                self._pressure_levels[idx]

            dates_per_request = self._get_dates_for_request()

            for var_prefix, pressure, req_date in \
                    product([var_name], pressures, dates_per_request):
                self._single_download(var_prefix, pressure, req_date)

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
               files=None,
               remove_original=False):
        for datafile in self._files_downloaded if not files else files:
            (datafile_path, datafile_name) = os.path.split(datafile)

            logging.debug("Regridding {}".format(datafile))
            cube = assign_lat_lon_coord_system(iris.load_cube(datafile))
            cube_ease = cube.regrid(
                self.sic_ease_cube, iris.analysis.Linear())

            self.additional_regrid_processing(datafile, cube_ease)

            # TODO: filename chain can be handled better for sharing between
            #  methods
            new_datafile = os.path.join(datafile_path,
                                        re.sub(r'_latlon_', '_', datafile_name))
            logging.info("Saving regridded data to {}... ".format(new_datafile))
            iris.save(cube_ease, new_datafile)

            if remove_original:
                logging.info("Removing {}".format(datafile))
                os.remove(datafile)

    @abstractmethod
    def additional_regrid_processing(self, datafile, cube_ease):
        pass

    def rotate_wind_data(self,
                         apply_to=("uas", "vas"),
                         remove_original=False):
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
            wind_files[var] = [df for df in self._files_downloaded
                               if os.path.split(df)[0] in source]

        # NOTE: we're relying on apply_to having equal datasets
        assert len(wind_files[apply_to[0]]) == len(wind_files[apply_to[1]]), \
            "The wind file datasets are unequal in length"

        # a nicer manner of doing this, no doubt
        for idx, wind_file_0 in enumerate(wind_files[apply_to[0]]):
            wind_file_1 = wind_files[apply_to[1]][idx]

            logging.info("Rotating {} and {}".format(wind_file_0, wind_file_1))
            wind_cubes = dict()
            wind_cubes_r = dict()

            wind_cubes[apply_to[0]] = iris.load_cube(wind_file_0)
            wind_cubes[apply_to[1]] = iris.load_cube(wind_file_1)

            wind_cubes_r[apply_to[0]], wind_cubes_r[apply_to[1]] = \
                rotate_grid_vectors(
                    wind_cubes[apply_to[0]], wind_cubes[apply_to[1]], angles)

            # Original implementation is in danger of lost updates
            # due to potential lazy loading
            for i, name in enumerate(wind_file_0, wind_file_1):
                tmp_fh, tmp_name = tempfile.mktemp(dir=os.path.split(name)[0])
                tmp_fh.close()
                iris.save(wind_cubes_r[apply_to[i]], tmp_name)
                os.replace(tmp_name, name)

    @property
    def var_names(self):
        return self._var_names

    @property
    def pressure_levels(self):
        return self._pressure_levels
