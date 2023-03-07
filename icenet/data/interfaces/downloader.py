import concurrent
import logging
import os
import re
import shutil
import tempfile

from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor
from itertools import product

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
import iris.analysis
import iris.cube
import iris.exceptions
import numpy as np
import pandas as pd
import xarray as xr

"""

"""


def filter_dates_on_data(latlon_path: str,
                         regridded_name: str,
                         req_dates: object,
                         check_latlon: bool = True,
                         check_regridded: bool = True,
                         drop_vars: list = None):
    """Reduces request dates and target files based on existing data

    To avoid what is potentially significant resource expense downloading
    extant data, downloaders should call this method to reduce the request
    dates only to that data not already present. This is a fairly naive
    implementation, in that if the data is present in either the latlon
    intermediate file OR the target regridded file, we'll not bother
    downloading again. This can be overridden via the method arguments.

    :param latlon_path:
    :param regridded_name:
    :param req_dates:
    :param check_latlon:
    :param check_regridded:
    :param drop_vars:
    :return: req_dates(list)
    """

    latlon_dates = list()
    regridded_dates = list()
    drop_vars = list() if drop_vars is None else drop_vars

    # Latlon files should in theory be aggregated and singular arrays
    # meaning we can naively open and interrogate the dates
    if check_latlon and os.path.exists(latlon_path):
        try:
            latlon_dates = xr.open_dataarray(
                latlon_path,
                drop_variables=drop_vars).time.values
            logging.debug("{} latlon dates already available in {}".format(
                len(latlon_dates), latlon_path
            ))
        except ValueError:
            logging.warning("Latlon {} dates not readable, ignoring file")

    if check_regridded and os.path.exists(regridded_name):
        regridded_dates = xr.open_dataarray(
            regridded_name,
            drop_variables=drop_vars).time.values
        logging.debug("{} regridded dates already available in {}".format(
            len(regridded_dates), regridded_name
        ))

    exclude_dates = list(set(latlon_dates).union(set(regridded_dates)))
    logging.debug("Excluding {} dates already existing from {} dates "
                  "requested.".format(len(exclude_dates), len(req_dates)))

    return sorted(list(pd.to_datetime(req_dates).
                       difference(pd.to_datetime(exclude_dates))))


def merge_files(new_datafile: str,
                other_datafile: str,
                drop_variables: object = None):
    """

    :param new_datafile:
    :param other_datafile:
    :param drop_variables:
    """
    drop_variables = list() if drop_variables is None else drop_variables

    if other_datafile is not None:
        (datafile_path, new_filename) = os.path.split(new_datafile)
        moved_new_datafile = \
            os.path.join(datafile_path, "new.{}".format(new_filename))
        os.rename(new_datafile, moved_new_datafile)
        d1 = xr.open_dataset(moved_new_datafile,
                             drop_variables=drop_variables)

        logging.info("Concatenating with previous data {}".format(
            other_datafile
        ))
        d2 = xr.open_dataset(other_datafile,
                             drop_variables=drop_variables)
        new_ds = xr.concat([d1, d2], dim="time").sortby("time")

        logging.info("Saving merged data to {}... ".
                     format(new_datafile))
        new_ds.to_netcdf(new_datafile)
        os.unlink(other_datafile)
        os.unlink(moved_new_datafile)


class ClimateDownloader(Downloader):
    """Climate downloader base class

    :param dates:
    :param delete_tempfiles:
    :param download:
    :param group_dates_by:
    :param max_threads:
    :param postprocess:
    :param pregrid_prefix:
    :param levels:
    :param var_name_idx:
    :param var_names:
    """

    def __init__(self, *args,
                 dates: object = (),
                 delete_tempfiles: bool = True,
                 download: bool = True,
                 drop_vars: list = None,
                 group_dates_by: str = "year",
                 levels: object = (),
                 max_threads: int = 1,
                 postprocess: bool = True,
                 pregrid_prefix: str = "latlon_",
                 var_name_idx: int = -1,
                 var_names: object = (),
                 **kwargs):
        super().__init__(*args, **kwargs)

        self._dates = list(dates)
        self._delete = delete_tempfiles
        self._download = download
        self._drop_vars = list() if drop_vars is None else drop_vars
        self._files_downloaded = []
        self._group_dates_by = group_dates_by
        self._levels = list(levels)
        self._masks = Masks(north=self.north, south=self.south)
        self._max_threads = max_threads
        self._postprocess = postprocess
        self._pregrid_prefix = pregrid_prefix
        self._rotatable_files = []
        self._sic_ease_cubes = dict()
        self._var_name_idx = var_name_idx
        self._var_names = list(var_names)

        assert len(self._var_names), "No variables requested"
        assert len(self._levels) == len(self._var_names), \
            "# of levels must match # vars"

        if not self._delete:
            logging.warning("!!! Deletions of temp files are switched off: be "
                            "careful with this, you need to manage your "
                            "files manually")
        self._download_method = None

        self._validate_config()

    def _validate_config(self):
        """

        """
        if self.hemisphere_str in os.path.split(self.base_path):
            raise RuntimeError("Don't include hemisphere string {} in "
                               "base path".format(self.hemisphere_str))

    def download(self):
        """Handles concurrent (threaded) downloading for variables

        This takes dates, variables and levels as configured, batches them into
        requests and submits those via a ThreadPoolExecutor for concurrent
        downloading. Returns nothing, relies on _single_download to implement
        appropriate updates to this object to record state changes arising from
        downloading.
        """

        logging.info("Building request(s), downloading and daily averaging "
                     "from {} API".format(self.identifier.upper()))

        requests = list()

        for idx, var_name in enumerate(self.var_names):
            levels = [None] if not self.levels[idx] else self.levels[idx]

            dates_per_request = \
                batch_requested_dates(self._dates,
                                      attribute=self._group_dates_by)

            for var_prefix, level, req_date in \
                    product([var_name], levels, dates_per_request):
                requests.append((var_prefix, level, req_date))

        with ThreadPoolExecutor(max_workers=min(len(requests),
                                                self._max_threads)) \
                as executor:
            futures = []

            for var_prefix, level, req_date in requests:
                future = executor.submit(self._single_download,
                                         var_prefix,
                                         level,
                                         req_date)
                futures.append(future)

            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logging.exception("Thread failure: {}".format(e))

        logging.info("{} daily files downloaded".
                     format(len(self._files_downloaded)))

    def _single_download(self,
                         var_prefix: str,
                         level: object,
                         req_dates: object):
        """Implements a single download based on configured download_method

        This allows delegation of downloading logic in a consistent manner to
        the configured download_method, ensuring a guarantee of adherence to
        naming and processing flow within ClimateDownloader implementations.

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

        req_dates = filter_dates_on_data(latlon_path, regridded_name, req_dates)

        if len(req_dates):
            if self._download:
                with tempfile.TemporaryDirectory() as tmpdir:
                    tmp_latlon_path = os.path.join(tmpdir, os.path.basename("{}.download".format(latlon_path)))

                    self.download_method(var,
                                         level,
                                         req_dates,
                                         tmp_latlon_path)

                    if os.path.exists(latlon_path):
                        (ll_path, ll_file) = os.path.split(latlon_path)
                        rename_latlon_path = os.path.join(
                            ll_path, "{}_old{}".format(
                                *os.path.splitext(ll_file)))
                        os.rename(latlon_path, rename_latlon_path)
                        old_da = xr.open_dataarray(rename_latlon_path,
                                                   drop_variables=self._drop_vars)
                        tmp_da = xr.open_dataarray(tmp_latlon_path,
                                                   drop_variables=self._drop_vars)

                        logging.debug("Input (old): \n{}".format(old_da))
                        logging.debug("Input (dl): \n{}".format(tmp_da))

                        da = xr.concat([old_da, tmp_da], dim="time")
                        logging.debug("Output: \n{}".format(da))

                        da.to_netcdf(latlon_path)
                        old_da.close()
                        tmp_da.close()
                        os.unlink(rename_latlon_path)
                    else:
                        shutil.move(tmp_latlon_path, latlon_path)

                logging.info("Downloaded to {}".format(latlon_path))
            else:
                logging.info("Skipping actual download to {}".
                             format(latlon_path))
        else:
            logging.info("No requested dates remain, likely already present")

        if self._postprocess and os.path.exists(latlon_path):
            self.postprocess(var, latlon_path)

        if os.path.exists(latlon_path):
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
               files: object = None,
               rotate_wind: bool = True):
        """

        :param files:
        """
        filelist = self._files_downloaded if not files else files
        batches = [filelist[b:b + 1000] for b in range(0, len(filelist), 1000)]

        max_workers = min(len(batches), self._max_threads)
        regrid_results = list()

        if max_workers > 0:
            with ThreadPoolExecutor(max_workers=max_workers) \
                    as executor:
                futures = []

                for files in batches:
                    future = executor.submit(self._batch_regrid, files)
                    futures.append(future)

                for future in concurrent.futures.as_completed(futures):
                    try:
                        fut_results = future.result()

                        for res in fut_results:
                            logging.debug("Future result -> regrid_results: {}".
                                          format(res))
                            regrid_results.append(res)
                    except Exception as e:
                        logging.exception("Thread failure: {}".format(e))
        else:
            logging.info("No regrid batches to processing, moving on...")

        if rotate_wind:
            logging.info("Rotating wind data prior to merging")
            self.rotate_wind_data()

        for new_datafile, moved_datafile in regrid_results:
            merge_files(new_datafile, moved_datafile, self._drop_vars)

    def _batch_regrid(self,
                      files: object):
        """

        :param files:
        """
        results = list()

        for datafile in files:
            (datafile_path, datafile_name) = os.path.split(datafile)

            new_filename = re.sub(r'^{}'.format(
                self.pregrid_prefix), '', datafile_name)
            new_datafile = os.path.join(datafile_path, new_filename)

            moved_datafile = None

            if os.path.exists(new_datafile):
                moved_filename = "moved.{}".format(new_filename)
                moved_datafile = os.path.join(datafile_path, moved_filename)
                os.rename(new_datafile, moved_datafile)

                logging.info("{} already existed, moved to {}".
                             format(new_filename, moved_filename))

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

            logging.info("Saving regridded data to {}... ".format(new_datafile))
            iris.save(cube_ease, new_datafile, fill_value=np.nan)
            results.append((new_datafile, moved_datafile))

            if self.delete:
                logging.info("Removing {}".format(datafile))
                os.remove(datafile)

        return results

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
                logging.error("Wind file array is not valid: {}".format(
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
                logging.debug("Overwritten {}".format(name))

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
    def levels(self):
        return self._levels

    @property
    def pregrid_prefix(self):
        return self._pregrid_prefix

    @property
    def var_names(self):
        return self._var_names

