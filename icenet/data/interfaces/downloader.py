import concurrent
import logging
import os
import re

from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor
from itertools import product

import numpy as np

from icenet.data.sic.mask import Masks
from icenet.data.sic.utils import SIC_HEMI_STR
from icenet.data.producers import Downloader
from icenet.data.utils import assign_lat_lon_coord_system, \
    gridcell_angles_from_dim_coords, \
    invert_gridcell_angles, \
    rotate_grid_vectors
from icenet.data.interfaces.utils import batch_requested_dates
from icenet.utils import run_command

import iris
import iris.exceptions
import pandas as pd

"""

"""


class ClimateDownloader(Downloader):
    """Climate downloader base class

    :param dates:
    :param delete_tempfiles:
    :param download:
    :param group_dates_by:
    :param max_threads:
    :param postprocess:
    :param pregrid_prefix:
    :param pressure_levels:
    :param var_name_idx:
    :param var_names:
    """

    def __init__(self, *args,
                 dates: object = (),
                 delete_tempfiles: bool = True,
                 download: bool = True,
                 group_dates_by: str = "year",
                 max_threads: int = 1,
                 postprocess: bool = True,
                 pregrid_prefix: str = "latlon_",
                 pressure_levels: object = (),
                 var_name_idx: int = -1,
                 var_names: object = (),
                 **kwargs):
        super().__init__(*args, **kwargs)

        self._dates = list(dates)
        self._delete = delete_tempfiles
        self._download = download
        self._files_downloaded = []
        self._group_dates_by = group_dates_by
        self._masks = Masks(north=self.north, south=self.south)
        self._max_threads = max_threads
        self._postprocess = postprocess
        self._pregrid_prefix = pregrid_prefix
        self._pressure_levels = list(pressure_levels)
        self._sic_ease_cubes = dict()
        self._var_name_idx = var_name_idx
        self._var_names = list(var_names)

        assert len(self._var_names), "No variables requested"
        assert len(self._pressure_levels) == len(self._var_names), \
            "# of pressures must match # vars"

        self._download_method = None

        self._validate_config()

    def _validate_config(self):
        """

        """
        if self.hemisphere_str in os.path.split(self.base_path):
            raise RuntimeError("Don't include hemisphere string {} in "
                               "base path".format(self.hemisphere_str))

    def download(self):
        """

        """

        logging.info("Building request(s), downloading and daily averaging "
                     "from {} API".format(self.identifier.upper()))

        requests = list()

        for idx, var_name in enumerate(self.var_names):
            pressures = [None] if not self.pressure_levels[idx] else \
                self._pressure_levels[idx]

            dates_per_request = \
                batch_requested_dates(self._dates,
                                      attribute=self._group_dates_by)

            for var_prefix, pressure, req_date in \
                    product([var_name], pressures, dates_per_request):
                requests.append((var_prefix, pressure, req_date))

        with ThreadPoolExecutor(max_workers=
                                min(len(requests), self._max_threads)) \
                as executor:
            futures = []

            for var_prefix, pressure, req_date in requests:
                future = executor.submit(self._single_download,
                                         var_prefix,
                                         pressure,
                                         req_date)
                futures.append(future)

            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logging.exception("Thread failure: {}".format(e))

        logging.info("{} daily files downloaded".
                     format(len(self._files_downloaded)))

    def filter_dates_on_data(self,
                             var_prefix: str,
                             level: object,
                             req_dates: object):
        """

        :param var_prefix:
        :param level:
        :param req_dates:
        :return: req_dates(list), merge_files(list)
        """
        logging.warning("NOT IMPLEMENTED YET, WE'LL JUST DOWNLOAD ANYWAY")
        # TODO: check existing yearly file for req_dates if already in place
        return req_dates, []

    # FIXME: this can be migrated for Dev#45 and we register in the subclass
    #  the potential implementations (e.g. CDS has toolbox and API, CMEMS has
    #  up to four implementations)
    def _single_download(self,
                         var_prefix: str,
                         level: object,
                         req_dates: object):
        """Implements a single download from CMEMS

        :param var_prefix: the icenet variable name
        :param level: the height to download
        :param req_dates: the request date
        """

        logging.info("Processing single download for {} @ {} with {} dates".
                     format(var_prefix, level, len(req_dates)))
        var = var_prefix if not level else \
            "{}{}".format(var_prefix, level)
        var_folder = self.get_data_var_folder(var)

        latlon_path, regridded_name = \
            self.get_req_filenames(var_folder, req_dates[0])

        req_dates, merge_files = \
            self.filter_dates_on_data(var_prefix, level, req_dates)

        if self.download and not os.path.exists(latlon_path):
            self.download_method(var,
                                 level,
                                 req_dates,
                                 latlon_path)

            logging.info("Downloaded to {}".format(latlon_path))
        else:
            logging.info("Skipping actual download")

        if self._postprocess:
            self.postprocess(var, latlon_path)

        if not os.path.exists(regridded_name):
            self._files_downloaded.append(latlon_path)

    def postprocess(self, var, download_path):
        logging.debug("No postprocessing in place for {}: {}".
                      format(var, download_path))

    def save_temporal_files(self, var, da,
                            date_format=None,
                            freq=None):
        """

        :param var:
        :param da:
        :param date_format:
        :param freq:
        """
        var_folder = self.get_data_var_folder(var)
        group_by = "time.{}".format(self._group_dates_by) if not freq else freq

        for dt, dt_da in da.groupby(group_by):
            req_date = pd.to_datetime(dt_da.time.values[0])
            latlon_path, regridded_name = \
                self.get_req_filenames(var_folder,
                                       req_date,
                                       date_format=date_format)

            logging.info("Retrieving and saving {}".format(latlon_path))
            dt_da.compute()
            dt_da.to_netcdf(latlon_path)

            if not os.path.exists(regridded_name):
                self._files_downloaded.append(latlon_path)

    @property
    def sic_ease_cube(self):
        """

        :return sic_cube:
        """
        if self._hemisphere not in self._sic_ease_cubes:
            sic_day_fname = 'ice_conc_{}_ease2-250_cdr-v2p0_197901021200.nc'. \
                format(SIC_HEMI_STR[self.hemisphere_str[0]])
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
               files: object = None):
        """

        :param files:
        """
        filelist = self._files_downloaded if not files else files
        batches = [filelist[b:b + 1000] for b in range(0, len(filelist), 1000)]

        max_workers = min(len(batches), self._max_threads)

        if max_workers > 0:
            with ThreadPoolExecutor(max_workers=max_workers) \
                    as executor:
                futures = []

                for files in batches:
                    future = executor.submit(self._batch_regrid, files)
                    futures.append(future)

                for future in concurrent.futures.as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        logging.exception("Thread failure: {}".format(e))
        else:
            logging.info("No regrid batches to processing, moving on...")

    def _batch_regrid(self,
                      files: object):
        """

        :param files:
        """
        for datafile in files:
            (datafile_path, datafile_name) = os.path.split(datafile)
            # TODO: mmmm, need to keep consistent with get_daily_filenames
            new_datafile = os.path.join(datafile_path,
                                        re.sub(
                                            r'^{}'.format(self.pregrid_prefix),
                                            '', datafile_name))

            if os.path.exists(new_datafile):
                logging.debug("Skipping {} as {} already exists".
                    format(datafile, os.path.basename(new_datafile)))
                continue

            logging.debug("Regridding {}".format(datafile))

            try:
                cube = iris.load_cube(datafile)
                cube = self.convert_cube(cube)

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
            iris.save(cube_ease, new_datafile, fill_value=np.nan)

            if self.delete:
                logging.info("Removing {}".format(datafile))
                os.remove(datafile)

    def convert_cube(self, cube: object):
        """Converts Iris cube to be fit for regrid

        :param cube: the cube requiring alteration
        :return cube: the altered cube
        """

        cube = assign_lat_lon_coord_system(cube)
        return cube

    @abstractmethod
    def additional_regrid_processing(self,
                                     datafile: str,
                                     cube_ease: object):
        """

        :param datafile:
        :param cube_ease:
        """
        pass

    def rotate_wind_data(self,
                         apply_to: object = ("uas", "vas"),
                         manual_files: object = None):
        """

        :param apply_to:
        :param manual_files:
        """
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
                re.sub(r'{}'.format(self.pregrid_prefix), '', df)
                for df in latlon_files
                if os.path.dirname(df).split(os.sep)
                [self._var_name_idx] == var],
                key=lambda x: int(re.search(r'^(?:\w+_)?(\d+).nc',
                                  os.path.basename(x)).group(1))
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

            try:
                wind_cubes_r[apply_to[0]], wind_cubes_r[apply_to[1]] = \
                    rotate_grid_vectors(
                        wind_cubes[apply_to[0]],
                        wind_cubes[apply_to[1]],
                        angles,
                    )
            except iris.exceptions.CoordinateNotFoundError:
                logging.exception("Failure to rotate due to coordinate issues. "
                                  "moving onto next file")
                continue

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

    def get_req_filenames(self,
                          var_folder: str,
                          req_date: object,
                          date_format: str = None):
        """

        :param var_folder:
        :param req_date:
        :param date_format:
        :return:
        """

        filename_date = getattr(req_date, self._group_dates_by) \
            if not date_format else req_date.strftime(date_format)

        latlon_path = os.path.join(
            var_folder, "{}{}.nc".format(self.pregrid_prefix, filename_date))
        regridded_name = os.path.join(
            var_folder, "{}.nc".format(filename_date))

        logging.debug("Got {} filenames: {} and {}".format(
            self._group_dates_by, latlon_path, regridded_name
        ))

        return latlon_path, regridded_name

    @property
    def dates(self):
        return self._dates

    @property
    def delete(self):
        return self._delete

    @property
    def download_method(self) -> callable:
        if not self._download_method:
            raise RuntimeError("Downloader has no method set, "
                               "implementation error")
        return self._download_method

    @download_method.setter
    def download_method(self, method: callable):
        self._download_method = method

    @property
    def group_dates_by(self):
        return self._group_dates_by

    @property
    def pregrid_prefix(self):
        return self._pregrid_prefix

    @property
    def pressure_levels(self):
        return self._pressure_levels

    @property
    def var_names(self):
        return self._var_names

