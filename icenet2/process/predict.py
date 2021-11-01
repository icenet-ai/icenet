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

from icenet2.utils import run_command


def date_arg(string):
    date_match = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", string)
    return dt.date(*[int(s) for s in date_match.groups()])


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("datefile", type=argparse.FileType("r"))

    ap.add_argument("-o", "--output-dir", default=".")
    ap.add_argument("-s", "--hemisphere",
                    choices=("north", "south"), default="north")
    ap.add_argument("-r", "--root", type=str, default="../..")
    ap.add_argument("-v", "--verbose", action="store_true", default=False)

    return ap.parse_args()


# TODO: I don't like doing this, but for the moment low barrier to entry
def get_ease_grid(north=True, south=False):
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

    # Load a single SIC map to obtain the EASE grid for
    # regridding ERA data
    cube = iris.load_cube(os.path.join(sic_day_path, sic_day_fname),
                          'sea_ice_area_fraction')
    return cube


def get_prediction_data(root, name, date):
    logging.info("Post-processing {}".format(date))

    glob_str = os.path.join(root,
                            "results",
                            "predict",
                            name,
                            "*",
                            date.strftime("%Y_%m_%d.npy"))

    data = [np.load(file) for file in glob.glob(glob_str)]
    data = np.array(data)
    return np.stack(
        [data.mean(axis=0), data.std(axis=0)],
        axis=-1).squeeze()


if __name__ == "__main__":
    args = get_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    dates = [dt.date(*[int(v) for v in s.split("-")])
             for s in args.datefile.read().split()]
    args.datefile.close()

    arr = np.array(
        [get_prediction_data(args.root, args.name, date)
         for date in dates])

    # TODO: Highly Recommended Variable Attributes
    xarr = xr.Dataset(
        data_vars=dict(
            sic_mean=(["time", "xc", "yc", "leadtime"], arr[..., 0]),
            sic_stddev=(["time", "xc", "yc", "leadtime"], arr[..., 1]),
        ),
        coords=dict(
            time=[pd.Timestamp(d) for d in dates],
            leadtime=np.arange(1, arr.shape[3] + 1, 1),
            xc=np.linspace(-5387.5, 5387.5, 432),
            yc=np.linspace(-5387.5, 5387.5, 432),
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
            geospatial_bounds="",
            geospatial_bounds_crs="",
            geospatial_bounds_vertical_crs="",
            geospatial_lat_min="",
            geospatial_lat_max="",
            geospatial_lat_units="",
            geospatial_lat_resolution="25km",
            geospatial_lon_min="",
            geospatial_lon_max="",
            geospatial_lon_units="",
            geospatial_lon_resolution="25km",
            geospatial_vertical_min=0.0,
            geospatial_vertical_max=0.0,
            history="{} - creation".format(dt.datetime.now()),
            id="IceNet2.TBC",
            institution="British Antarctic Survey",
            keywords="""'Earth Science > Cryosphere > Sea Ice > Sea Ice Concentration
            Earth Science > Oceans > Sea Ice > Sea Ice Concentration
            Earth Science > Climate Indicators > Cryospheric Indicators > Sea Ice""",
            license="Open Government Licece (OGL) V3",
            naming_authority="uk.ac.bas",
            platform="BAS HPC",
            #program="",
            #processing_level="",
            product_version="",
            project="IceNet",
            publisher_email="",
            publisher_institution="British Antarctic Survey",
            publisher_name="Polar Data Center",
            publisher_url="",
            source="""
            IceNet2 model generation at vTBC
            """,
            # Values for any standard_name attribute must come from the CF
            # Standard Names vocabulary for the data file or product to
            #  comply with CF
            standard_name_vocabulary="CF Standard Name Table v27",
            summary="""
            TBC
            """,
            # Use ISO 8601:2004 duration format, preferably the extended format
            # as recommended in the Attribute Content Guidance section.
            time_coverage_start="",
            time_coverage_end="",
            time_coverage_duration="",
            time_coverage_resolution="",
            title="Ensemble output of mean and standard deviation for sea ice "
                  "concentration probability",
        )
    )

    # FIXME: serializer doesn't like empty fields
    #xarr.time.attrs = dict(
    #    long_name="",
    #    short_name="",
    #    units="",
    #)
    #xarr.yc.attrs = dict(
    #    long_name="",
    #    short_name="",
    #    units="",
    #)
    #xarr.xc.attrs = dict(
    #    long_name="",
    #    short_name="",
    #    units="",
    #)
    #xarr.leadtime.attrs = dict(
    #    long_name="",
    #    short_name="",
    #    units="",
    #)

    #xarr.mean.attrs = dict(
    #    long_name="",
    #    short_name="",
    #    units="",
    #)
    #xarr.stddev.attrs = dict(
    #    long_name="",
    #    short_name="",
    #    units="",
    #)

    # TODO: split into daily files
    output_path = os.path.join(args.output_dir, "{}.nc".format(args.name))
    logging.info("Saving to {}".format(output_path))
    xarr.to_netcdf(output_path)
