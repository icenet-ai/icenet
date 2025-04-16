import logging

import iris
import xarray as xr

from preprocess_toolbox.processor import NormalisingChannelProcessor


class SICPreProcessor(NormalisingChannelProcessor):
    def pre_normalisation(self, var_name: str, da: object):
        """

        :param var_name:
        :param da:
        :return:
        """
        if var_name != "siconca":
            raise RuntimeError("OSISAF SIC implementation should be dealing "
                               "with siconca only")
        else:
            # If the OSISAF data has been regridded, we might end up with renamed coords
            da = da.rename(dict(x="xc", y="yc"))

            if "units" in da.xc.attrs and da.xc.attrs["units"] == "km":
                da.xc.attrs["units"] = "metres"
                da.yc.attrs["units"] = "metres"
                da['xc'] = da.xc * 1000
                da['yc'] = da.yc * 1000

        #     masks = Masks(north=self.north, south=self.south)
        #     return sic_interpolate(da, masks)
        return da


def amsr_coordinate_regrid(ref_cube, cube, *args):
    (sic_ref, ) = args
    logging.info(f"Using custom regrid, loading {sic_ref}")
    osi_sic = xr.open_dataset(sic_ref).ice_conc

    # NOTE: this is normally required for raw data, but not if the coordinates have been stripped
    # cube.remove_coord('projection_y_coordinate')
    # cube.remove_coord('projection_x_coordinate')

    # Create coordinate objects
    y_coord = iris.coords.DimCoord(osi_sic.yc * 1000, standard_name='projection_y_coordinate', units='m')
    x_coord = iris.coords.DimCoord(osi_sic.xc * 1000, standard_name='projection_x_coordinate', units='m')
    cube.add_dim_coord(y_coord, len(cube.dim_coords))
    cube.add_dim_coord(x_coord, len(cube.dim_coords))

    # Ref: Version 3.1 April 2023 of the OSI-450 product user manual
    cs = iris.coord_systems.PolarStereographic(central_lat=90.0 if osi_sic.lat.min() > 0 else -90,
                                               central_lon=0,
                                               false_easting=0.0,
                                               false_northing=0.0,
                                               # true_scale_lat=70.0,
                                               ellipsoid=iris.coord_systems.GeogCS(semi_major_axis=6378137.0,
                                                                                   semi_minor_axis=6356752.314245))
    cube.coord('projection_y_coordinate').coord_system = cs
    cube.coord('projection_x_coordinate').coord_system = cs

    return cube
