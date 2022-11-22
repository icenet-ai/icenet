import logging
import os
import warnings

import numpy as np
import pandas as pd
import xarray as xr

from icenet.data.interfaces.downloader import ClimateDownloader
from icenet.data.cli import download_args
from icenet.data.utils import esgf_search

"""

"""


class CMIP6Downloader(ClimateDownloader):
    """Climate downloader to provide CMIP6 reanalysis data from ESGF APIs

    Useful CMIP6 guidance: https://pcmdi.llnl.gov/CMIP6/Guide/dataUsers.html

    :param identifier: how to identify this dataset
    :param source: source ID in ESGF node
    :param member: member ID in ESGF node
    :param nodes: list of ESGF nodes to query
    :param experiments: experiment IDs to download
    :param frequency: query parameter frequency
    :param table_map: table map for
    :param grid_map:
    :param grid_override:
    :param exclude_nodes:

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
    ESGF_NODES = ("esgf.ceda.ac.uk",
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
                 nodes: object = ESGF_NODES,
                 experiments: object = ('historical', 'ssp245'),
                 frequency: str = "day",
                 table_map: object = None,
                 grid_map: object = None,
                 grid_override: object = None,
                 exclude_nodes: object = None,
                 **kwargs):
        super().__init__(*args,
                         identifier="cmip6.{}.{}".format(source, member),
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

    def _single_download(self,
                         var_prefix: str,
                         level: object,
                         req_dates: object):
        """Overridden CMIP implementation for downloading from DAP server

        Due to the size of the CMIP set and the fact that we don't want to make
        1850-2100 yearly requests for all downloads, we have a bespoke and
        overridden download implementation for this.

        :param var_prefix:
        :param level:
        :param req_dates:
        """

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

        var = var_prefix if not level else "{}{}".format(var_prefix, level)

        req_dates, merge_files = \
            self.filter_dates_on_data(var_prefix, level, req_dates)

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

            cmip6_da = cmip6_da.sel(time=slice(req_dates[0],
                                               req_dates[-1]))

            # TODO: possibly other attributes, especially with ocean vars
            if level:
                cmip6_da = cmip6_da.sel(plev=int(level) * 100)

            cmip6_da = cmip6_da.sel(lat=slice(self.hemisphere_loc[2],
                                              self.hemisphere_loc[0]))
            self.save_temporal_files(var, cmip6_da)
        except OSError as e:
            logging.exception("Error encountered: {}".format(e),
                              exc_info=False)

    def additional_regrid_processing(self,
                                     datafile: str,
                                     cube_ease: object):
        """

        :param datafile:
        :param cube_ease:
        """
        (datafile_path, datafile_name) = os.path.split(datafile)
        var_name = datafile_path.split(os.sep)[self._var_name_idx]

        # TODO: regrid fixes need better implementations
        if var_name == "siconca":
            cube_ease.data[cube_ease.data.mask] = 0.
            cube_ease.data[:, self._masks.get_land_mask()] = 0.

            if self._source == 'MRI-ESM2-0':
                cube_ease.data = cube_ease.data / 100.
            cube_ease.data = cube_ease.data.data
        elif var_name in ["tos", "hus1000"]:
            cube_ease.data[cube_ease.data.mask] = 0.
            cube_ease.data[:, self._masks.get_land_mask()] = 0.

            cube_ease.data = cube_ease.data.data

        if cube_ease.data.dtype != np.float32:
            logging.info("Regrid processing, data type not float: {}".
                         format(cube_ease.data.dtype))
            cube_ease.data = cube_ease.data.astype(np.float32)

    def convert_cube(self, cube: object) -> object:
        """Converts Iris cube to be fit for CMIP regrid

        :param cube:   the cube requiring alteration
        :return cube:   the altered cube
        """

        cs = self.sic_ease_cube.coord_system().ellipsoid

        for coord in ['longitude', 'latitude']:
            cube.coord(coord).coord_system = cs
        return cube


def main():
    args = download_args(
        dates=True,
        extra_args=[
            (["source"], dict(type=str)),
            (["member"], dict(type=str)),
            (("-xs", "--exclude-server"),
             dict(default=[], nargs="*")),
            (("-o", "--override"), dict(required=None, type=str)),
        ],
        workers=True
    )

    logging.info("CMIP6 Data Downloading")

    downloader = CMIP6Downloader(
        source=args.source,
        member=args.member,
        var_names=args.vars,
        pressure_levels=args.levels,
        dates=[pd.to_datetime(date).date() for date in
               pd.date_range(args.start_date, args.end_date, freq="D")],
        delete_tempfiles=args.delete,
        grid_override=args.override,
        north=args.hemisphere == "north",
        south=args.hemisphere == "south",
        max_threads=args.workers,
        exclude_nodes=args.exclude_server,
    )
    logging.info("CMIP downloading: {} {}".format(args.source, args.member))
    downloader.download()

    logging.info("CMIP regridding: {} {}".format(args.source, args.member))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        downloader.regrid()
