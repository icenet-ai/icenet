import logging
import os

import numpy as np
import pandas as pd
import xarray as xr

from icenet2.data.interfaces.downloader import ClimateDownloader
from icenet2.data.cli import download_args
from icenet2.data.utils import esgf_search
from icenet2.data.interfaces.utils import get_daily_filenames


class CMIP6Downloader(ClimateDownloader):
    """Climate downloader to provide CMIP6 reanalysis data from ESGF APIs

    Args:
        identifier: how to identify this dataset
        source: source ID in ESGF node
        member: member ID in ESGF node
        nodes: list of ESGF nodes to query
        experiments: experiment IDs to download
        frequency: query parameter frequency
        table_map: table map for
        grid_map=GRID_MAP,
        grid_override=None,  # EC-Earth3 wants all 'gr'
        exclude_nodes=[],

    "MRI-ESM2-0", "r1i1p1f1", None
    "MRI-ESM2-0", "r2i1p1f1", None
    "MRI-ESM2-0", "r3i1p1f1", None
    "MRI-ESM2-0", "r4i1p1f1", None
    "MRI-ESM2-0", "r5i1p1f1", None
    "EC-Earth3", "r2i1p1f1", "gr"
    "EC-Earth3", "r7i1p1f1", "gr"
    "EC-Earth3", "r10i1p1f1", "gr"
    "EC-Earth3", "r12i1p1f1", "gr"
    "EC-Earth3", "r14i1p1f1", "gr"

    """
    
    TABLE_MAP = {
        'siconca': 'SIday',
        'tas': 'day',
        'ta': 'day',
        'tos': 'Oday',
        'hus': 'day',
        'psl': 'day',
        'rlds': 'day',
        'rsus': 'day',
        'rsds': 'day',
        'zg': 'day',
        'uas': 'day',
        'vas': 'day',
        'ua': 'day',
    }

    GRID_MAP = {
        'siconca': 'gn',
        'tas': 'gn',
        'ta': 'gn',
        'tos': 'gr',
        'hus': 'gn',
        'psl': 'gn',
        'rlds': 'gn',
        'rsus': 'gn',   # Surface Upwelling Shortwave Radiation
        'rsds': 'gn',   # Surface Downwelling Shortwave Radiation
        'zg': 'gn',
        'uas': 'gn',
        'vas': 'gn',
        'ua': 'gn',
    }

    # Prioritise European first, US last, avoiding unnecessary queries
    # against nodes further afield (all traffic has a cost, and the coverage
    # of local nodes is more than enough)
    ESGF_NODES = ("esgf-data3.ceda.ac.uk",
                  "esg1.umr-cnrm.fr",
                  "vesg.ipsl.upmc.fr",
                  "esgf3.dkrz.de",
                  "esgf.bsc.es",
                  "esgf-data.csc.fi",
                  "noresg.nird.sigma2.no",
                  "esgf-data.ucar.edu",
                  "esgf-data2.diasjp.net",)

    def __init__(self,
                 *args,
                 source: str,
                 member: str,
                 nodes: list = ESGF_NODES,
                 experiments: tuple = ('historical', 'ssp245'),
                 frequency: str = "day",
                 table_map: dict = None,
                 grid_map: dict = None,
                 grid_override: dict = None,
                 exclude_nodes: list = None,
                 **kwargs):
        super().__init__(*args,
                         identifier="cmip6",
                         **kwargs)

        self._source = source
        self._member = member
        self._frequency = frequency
        self._experiments = experiments

        self._nodes = nodes if not exclude_nodes else \
            [n for n in nodes if n not in exclude_nodes]

        self._table_map = table_map if table_map else CMIP6Downloader.TABLE_MAP
        self._grid_map = grid_map if grid_map else CMIP6Downloader.GRID_MAP
        self._grid_map_override = grid_override

    def _get_dates_for_request(self):
        return [None]

    def _single_download(self, var_prefix, pressure, req_date):
        query = {
            'source_id': self._source,
            'member_id': self._member,
            'frequency': self._frequency,
            'variable_id': var_prefix,
            'table_id': self._table_map[var_prefix],
            'grid_label': self._grid_map_override
            if self._grid_map_override
            else self._grid_map[var_prefix],
        }

        var_name = "{}{}".format(var_prefix, "" if not pressure else pressure)
        add_str = ".{}.{}".format(self.dates[0].strftime("%F"),
                                  self.dates[-1].strftime("%F")) \
            if self.dates[0] is not None else ""
        download_name = "download.{}.{}{}.nc".format(
            self._source, self._member, add_str)
        download_path = os.path.join(self.get_data_var_folder(var_name),
                                     download_name)

        # Download the full source data for the experiment
        if not os.path.exists(download_path):
            logging.info("Querying ESGF")
            results = []
            for experiment_id in self._experiments:
                query['experiment_id'] = experiment_id

                for data_node in self._nodes:
                    query['data_node'] = data_node

                    # FIXME: inefficient, we can strip redundant results files
                    #  based on WCRP data management standards for file naming,
                    #  such as based on date. Refactor/rewrite this impl...
                    node_results = esgf_search(**query)

                    if len(node_results):
                        logging.debug("Query: {}".format(query))
                        logging.debug("Found {}: {}".format(experiment_id,
                                                            node_results))
                        results.extend(node_results)
                        break

            logging.info("Found {} {} results from ESGF search".
                         format(len(results), var_prefix))

            try:
                # http://xarray.pydata.org/en/stable/user-guide/io.html?highlight=opendap#opendap
                # Avoid 500MB DAP request limit
                cmip6_da = xr.open_mfdataset(results,
                                             combine='by_coords',
                                             chunks={'time': '499MB'}
                                             )[var_prefix]

                if self.dates[0] is not None:
                    cmip6_da = cmip6_da.sel(time=slice(self.dates[0],
                                                       self.dates[-1]))

                if pressure:
                    cmip6_da = cmip6_da.sel(plev=int(pressure * 100))

                cmip6_da = cmip6_da.sel(lat=slice(self.hemisphere_loc[2],
                                                  self.hemisphere_loc[0]))
                logging.info("Retrieving and saving {}:".format(download_path))
                cmip6_da.compute()
                cmip6_da.to_netcdf(download_path)
            except OSError as e:
                logging.exception("Error encountered: {}".format(e),
                                  exc_info=False)

        # Open the download, reprocess out into individual files
        # TODO: repeated code w.r.t OSISAF (& mars/era?)
        da = xr.open_dataarray(download_path)
        da_daily = da.resample(time='1D').reduce(np.mean)

        for day in da_daily.time.values:
            date_str = pd.to_datetime(day).strftime("%Y_%m_%d")
            logging.debug("Processing var {} for {}".format(var_name, date_str))

            daily_path, regridded_name = get_daily_filenames(
                self.get_data_var_folder(
                    var_name, append=[
                        "{}.{}".format(self._source, self._member),
                        str(pd.to_datetime(day).year)]),
                var_name, date_str)

            if len(da_daily.sel(time=slice(day, day)).time) == 0:
                raise RuntimeError("No information in da_daily: {}".format(
                    da_daily
                ))

            if not os.path.exists(daily_path):
                logging.debug(
                    "Saving new daily file: {}".format(daily_path))
                da_daily.sel(time=slice(day, day)).to_netcdf(daily_path)

            if not os.path.exists(regridded_name):
                self._files_downloaded.append(daily_path)

        # Clean up
        if self.delete:
            logging.info("Deleting download: {}".format(download_path))
            raise NotImplementedError("CMIP downloader doesn't delete yet")

    def additional_regrid_processing(self, datafile, cube_ease):
        (datafile_path, datafile_name) = os.path.split(datafile)
        var_name = datafile_path.split(os.sep)[-1]

        # Preprocessing
        if var_name == 'siconca':
            cube_ease.data[cube_ease.data > 500] = 0.
            cube_ease.data[:, self._masks.get_land_mask()] = 0.

            if self._source == 'MRI-ESM2-0':
                cube_ease.data = cube_ease.data / 100.
        elif var_name == 'tos':
            cube_ease.data[cube_ease.data > 500] = 0.
            cube_ease.data[:, self._masks.get_land_mask()] = 0.

        if cube_ease.data.dtype != np.float32:
            logging.info("Regrid processing, data type not float: {}".
                         format(cube_ease.data.dtype))
            cube_ease.data = cube_ease.data.astype(np.float32)

    def convert_cube(self, cube):
        """Converts Iris cube to be fit for CMIP regrid

        Params:
            cube:   the cube requiring alteration
        Returns:
            cube:   the altered cube
        """

        cs = self.sic_ease_cube.coord_system().ellipsoid

        for coord in ['longitude', 'latitude']:
            cube.coord(coord).coord_system = cs
        return cube


def main():
    args = download_args(
        dates=True,
        dates_optional=True,
        extra_args=[
            (["name"], dict(type=str)),
            (["member"], dict(type=str)),
            (("-xs", "--exclude-server"),
             dict(default=[], nargs="*")),
            (("-o", "--override"), dict(required=None, type=str)),
        ],
        workers=True
    )

    logging.info("CMIP6 Data Downloading")

    dates = [None]

    if args.start_date or args.end_date:
        # FIXME: This is very lazy and inefficient
        dates = [pd.to_datetime(date).date() for date in
                 pd.date_range(
                    args.start_date if args.start_date else "1850-1-1",
                    args.end_date if args.end_date else "2100-12-31",
                    freq="D")]
        logging.info("{} dates specified, downloading subset".
                     format(len(dates)))

    downloader = CMIP6Downloader(
        source=args.name,
        member=args.member,
        var_names=["tas", "ta", "tos", "psl", "zg", "hus", "rlds",
                   "rsds", "uas", "vas", "siconca"],
        pressure_levels=[None, [500], None, None, [250, 500], [1000],
                         None, None, None, None, None],
        dates=dates,
        delete_tempfiles=args.delete,
        grid_override=args.override,
        north=args.hemisphere == "north",
        south=args.hemisphere == "south",
        max_threads=args.workers,
        exclude_nodes=args.exclude_server,
    )
    logging.info("CMIP downloading: {} {}".format(args.name,
                                                     args.member))
    downloader.download()
    logging.info("CMIP regridding: {} {}".format(args.name,
                                                    args.member))
    downloader.regrid()
    logging.info("CMIP rotating: {} {}".format(args.name,
                                                  args.member))
    downloader.rotate_wind_data()

"""
        


        output_name = "latlon_{}.{}.nc".format(self._source, self._member)
        proc_name = re.sub(r'^latlon_', '', output_name)
        # TODO: Yearly output

        proc_path = os.path.join(output_path, proc_name)


                os.path.exists(os.path.join(output_path, proc_name)):









                self._files_downloaded.append(output_path)
        else:
            if not os.path.exists(proc_path):
                logging.info("{} already exists but is not processed".
                             format(output_path))
                if output_path not in self._files_downloaded:
                    self._files_downloaded.append(output_path)
            else:
                logging.info("{} processed file exists".format(proc_path))
"""


