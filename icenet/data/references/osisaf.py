import logging
import os

import iris
import xarray as xr

from preprocess_toolbox.cli import BaseArgParser


class GenerateArgParser(BaseArgParser):
    def __init__(self,
                 dataset_name):
        super().__init__(description="Program to generate a consistent {} source file for regridding".
                         format(dataset_name))
        self.add_argument("{}_filename".format(dataset_name.lower()), type=str)

        self.add_argument("-o", "--output-path", default=".", type=str)


def generate():
    args = GenerateArgParser("OSISAF").parse_args()
    logging.info("Opening dataset to provide")
    cube = iris.load_cube(args.osisaf_filename, "sea_ice_area_fraction")
    cube.coord('projection_x_coordinate').convert_units('meters')
    cube.coord('projection_y_coordinate').convert_units('meters')
    iris.save(cube,
              os.path.join(args.output_path,
                           "ref.osisaf.{}.nc".format("north" if "_nh_" in args.osisaf_filename else "south")))
