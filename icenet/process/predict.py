import argparse
import datetime as dt
import glob
import logging
import os
import re

import iris
import numpy as np
import pandas as pd
import xarray as xr

from icenet import __version__ as icenet_version
from icenet.data.dataset import IceNetDataSet
from icenet.data.sic.mask import Masks
from icenet.utils import run_command, setup_logging


def get_refsic(north: bool = True, south: bool = False) -> object:
    """

    :param north:
    :param south:
    :return:
    """
    assert north or south, "Select one hemisphere at least..."

    str = "nh" if north else "sh"

    sic_day_fname = 'ice_conc_{}_ease2-250_cdr-v2p0_197901021200.nc'.format(str)
    sic_day_path = os.path.join(".", "_sicfile")

    if not os.path.exists(os.path.join(sic_day_path, sic_day_fname)):
        logging.info("Downloading single daily SIC netCDF file for "
                     "regridding ERA5 data to EASE grid...")

        retrieve_sic_day_cmd = 'wget -m -nH --cut-dirs=6 -P {} ' \
                               'ftp://osisaf.met.no/reprocessed/ice/' \
                               'conc/v2p0/1979/01/{}'. \
            format(sic_day_path, sic_day_fname)

        run_command(retrieve_sic_day_cmd)

    return os.path.join(sic_day_path, sic_day_fname)


def get_refcube(north: bool = True, south: bool = False) -> object:
    """

    :param north:
    :param south:
    :return:
    """
    assert north or south, "Select one hemisphere at least..."

    path = get_refsic(north, south)

    cube = iris.load_cube(path, 'sea_ice_area_fraction')
    return cube


def get_prediction_data(root: object,
                        name: object,
                        date: object) -> object:
    """

    :param root:
    :param name:
    :param date:
    :return:
    """
    logging.info("Post-processing {}".format(date))

    glob_str = os.path.join(root,
                            "results",
                            "predict",
                            name,
                            "*",
                            date.strftime("%Y_%m_%d.npy"))

    np_files = glob.glob(glob_str)
    if not len(np_files):
        logging.warning("No files found")
        return None

    data = [np.load(f) for f in np_files]
    data = np.array(data)

    logging.debug("Data read from disk: {} from: {}".format(data.shape, np_files))

    return np.stack(
        [data.mean(axis=0), data.std(axis=0)],
        axis=-1).squeeze()


def date_arg(string: str) -> object:
    """

    :param string:
    :return:
    """
    date_match = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", string)
    return dt.date(*[int(s) for s in date_match.groups()])


@setup_logging
def get_args():
    """

    :return:
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("dataset")
    ap.add_argument("datefile", type=argparse.FileType("r"))

    ap.add_argument("-m", "--mask", default=False, action="store_true")
    ap.add_argument("-o", "--output-dir", default=".")
    ap.add_argument("-r", "--root", type=str, default=".")

    ap.add_argument("-v", "--verbose", action="store_true", default=False)

    return ap.parse_args()


def create_cf_output():
    """

    """
    args = get_args()

    dataset_config = \
        os.path.join(args.root, "dataset_config.{}.json".format(args.dataset))
    ds = IceNetDataSet(dataset_config)

    ref_sic = xr.open_dataset(get_refsic(ds.north, ds.south))
    ref_cube = get_refcube(ds.north, ds.south)

    dates = [dt.date(*[int(v) for v in s.split("-")])
             for s in args.datefile.read().split()]
    args.datefile.close()

    arr = np.array(
        [get_prediction_data(args.root, args.name, date)
         for date in dates])

    logging.info("Dataset arr shape: {}".format(arr.shape))

    sic_mean = arr[..., 0]
    sic_stddev = arr[..., 1]

    if args.mask:
        mask_gen = Masks(north=ds.north, south=ds.south)

        logging.info("Land masking the forecast output")
        land_mask = mask_gen.get_land_mask()
        mask = land_mask[np.newaxis, ..., np.newaxis]
        mask = np.repeat(mask, sic_mean.shape[-1], axis=-1)
        mask = np.repeat(mask, sic_mean.shape[0], axis=0)

        sic_mean[mask] = 0
        sic_stddev[mask] = 0

        logging.info("Applying active grid cell masks")

        for idx, forecast_date in enumerate(dates):
            for lead_idx in np.arange(0, arr.shape[3], 1):
                lead_dt = forecast_date + dt.timedelta(days=int(lead_idx) + 1)
                logging.debug("Active grid cell mask start {} forecast date {}".
                              format(forecast_date, lead_dt))

                grid_cell_mask = mask_gen.get_active_cell_mask(lead_dt.month)
                sic_mean[idx, ~grid_cell_mask, lead_idx] = 0
                sic_stddev[idx, ~grid_cell_mask, lead_idx] = 0

    xarr = xr.Dataset(
        data_vars=dict(
            Lambert_Azimuthal_Grid=ref_sic.Lambert_Azimuthal_Grid,
            sic_mean=(["time", "yc", "xc", "leadtime"], sic_mean),
            sic_stddev=(["time", "yc", "xc", "leadtime"], sic_stddev),
        ),
        coords=dict(
            time=[pd.Timestamp(d) for d in dates],
            leadtime=np.arange(1, arr.shape[3] + 1, 1),
            xc=ref_cube.coord("projection_x_coordinate").points,
            yc=ref_cube.coord("projection_y_coordinate").points,
            lat=(("yc", "xc"), ref_cube.coord("latitude").points),
            lon=(("yc", "xc"), ref_cube.coord("longitude").points),
        ),
        # https://wiki.esipfed.org/Attribute_Convention_for_Data_Discovery_1-3
        attrs=dict(
            Conventions="CF-1.6 ACDD-1.3",
            comments="",
            creator_email="jambyr@bas.ac.uk",
            creator_institution="British Antarctic Survey",
            creator_name="James Byrne",
            creator_url="www.bas.ac.uk",
            date_created=dt.datetime.now().strftime("%Y-%m-%d"),
            # Issue#18: Overcoming OSI-SAF EPSG ref issue
            geospatial_bounds_crs="EPSG:6931" if ds.north else "EPSG:6932",
            geospatial_lat_min=ref_cube.attributes["geospatial_lat_min"],
            geospatial_lat_max=ref_cube.attributes["geospatial_lat_max"],
            geospatial_lon_min=ref_cube.attributes["geospatial_lon_min"],
            geospatial_lon_max=ref_cube.attributes["geospatial_lon_max"],
            geospatial_vertical_min=0.0,
            geospatial_vertical_max=0.0,
            history="{} - creation".format(dt.datetime.now()),
            id="IceNet.TBC".format(icenet_version),
            institution="British Antarctic Survey",
            keywords="""'Earth Science > Cryosphere > Sea Ice > Sea Ice Concentration
            Earth Science > Oceans > Sea Ice > Sea Ice Concentration
            Earth Science > Climate Indicators > Cryospheric Indicators > Sea Ice
            Geographic Region > {} Hemisphere""".format(
                "Northern" if ds.north else "Southern"
            ),
            # TODO: check we're valid
            keywords_vocabulary="GCMD Science Keywords",
            # TODO: Double check this is good with PDC
            license="Open Government Licece (OGL) V3",
            naming_authority="uk.ac.bas",
            platform="BAS HPC",
            #program="",
            #processing_level="",
            product_version=icenet_version,
            project="IceNet",
            publisher_email="",
            publisher_institution="British Antarctic Survey",
            #publisher_name="Polar Data Center",
            publisher_url="",
            source="""
            IceNet model generation at v{}
            """.format(icenet_version),
            spatial_resolution=ref_cube.attributes["spatial_resolution"],
            # Values for any standard_name attribute must come from the CF
            # Standard Names vocabulary for the data file or product to
            #  comply with CF
            standard_name_vocabulary="CF Standard Name Table v27",
            summary="""
            This is an output of sea ice concentration predictions from the 
            IceNet UNet run in an ensemble, with postprocessing to determine 
            the mean and standard deviation across the runs.
            """,
            # Use ISO 8601:2004 duration format, preferably the extended format
            # as recommended in the Attribute Content Guidance section.
            time_coverage_start="",
            time_coverage_end="",
            time_coverage_duration="P1D",
            time_coverage_resolution="P1D",
            title="Sea Ice Concentration Prediction",
        )
    )

    xarr.time.attrs = dict(
        long_name=ref_cube.coord("time").long_name,
        standard_name=ref_cube.coord("time").standard_name,
        axis="T",
        # TODO: https://github.com/SciTools/cf-units for units methods
        # units=Unit('seconds since 1978-01-01 00:00:00', calendar='gregorian')
        # bounds=array([[31622400., 31708800.]])
    )
    xarr.yc.attrs = dict(
        long_name=ref_cube.coord("projection_y_coordinate").long_name,
        standard_name=ref_cube.coord("projection_y_coordinate").standard_name,
        units=ref_cube.coord("projection_y_coordinate").units.name,
        axis="Y",
        # TODO: iris.coord_systems.LambertAzimuthalEqualArea
    )
    xarr.xc.attrs = dict(
        long_name=ref_cube.coord("projection_x_coordinate").long_name,
        standard_name=ref_cube.coord("projection_x_coordinate").standard_name,
        units=ref_cube.coord("projection_x_coordinate").units.name,
        axis="X",
    )
    xarr.leadtime.attrs = dict(
        long_name="leadtime of forecast in relation to reference time",
        short_name="leadtime",
        #units="1",
    )

    xarr.lat.attrs = dict(
        long_name=ref_cube.coord("latitude").long_name,
        standard_name=ref_cube.coord("latitude").standard_name,
        units=ref_cube.coord("latitude").units.name,
    )
    xarr.lon.attrs = dict(
        long_name=ref_cube.coord("longitude").long_name,
        standard_name=ref_cube.coord("longitude").standard_name,
        units=ref_cube.coord("longitude").units.name,
    )

    xarr.sic_mean.attrs = dict(
        long_name="mean sea ice area fraction across ensemble runs of icenet "
                  "model",
        standard_name="sea_ice_area_fraction",
        short_name="sic",
        valid_min=0,
        valid_max=1,
        ancillary_variables="sic_stddev",
        grid_mapping="Lambert_Azimuthal_Grid",
        units="1",
    )

    xarr.sic_stddev.attrs = dict(
        long_name="total uncertainty (one standard deviation) of concentration of sea ice",
        standard_name="sea_ice_area_fraction standard_error",
        valid_min=0,
        valid_max=1,
        grid_mapping="Lambert_Azimuthal_Grid",
        units="1",
    )

    # TODO: split into daily files
    output_path = os.path.join(args.output_dir, "{}.nc".format(args.name))
    logging.info("Saving to {}".format(output_path))
    xarr.to_netcdf(output_path)
