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

    da = xr.open_dataset(sic_ref).ice_conc
    y_coord = iris.coords.DimCoord(da.yc * 1000, standard_name='projection_y_coordinate', units='m')
    x_coord = iris.coords.DimCoord(da.xc * 1000, standard_name='projection_x_coordinate', units='m')
    cube.add_dim_coord(y_coord, 1)
    cube.add_dim_coord(x_coord, 2)

    cube.coord('projection_y_coordinate').coord_system = ref_cube.coord('projection_y_coordinate')[0].coord_system
    cube.coord('projection_x_coordinate').coord_system = ref_cube.coord('projection_x_coordinate')[0].coord_system
    return cube
