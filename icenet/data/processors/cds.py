import logging

from preprocess_toolbox.processor import NormalisingChannelProcessor


class ERA5PreProcessor(NormalisingChannelProcessor):
    def pre_normalisation(self, var_name: str, da: object):
        if 'expver' in da.coords:
            logging.warning("expvers {} in coordinates, will process out but "
                            "this needs further work: expver needs storing for "
                            "later overwriting".format(da.expver))
            # Ref: https://confluence.ecmwf.int/pages/viewpage.action?pageId=173385064
            da = da.sel(expver=1).combine_first(da.sel(expver=5))

        if var_name == 'tos':
            logging.debug("ERA5 regrid postprocessing replacing zeroes: {}".format(var_name))
            da = da.fillna(0)
        elif var_name in ['zg500', 'zg250']:
            # Convert from geopotential to geopotential height
            logging.debug("ERA5 additional regrid: {}".format(var_name))
            da /= 9.80665

        return da

    def post_normalisation(self, var_name: str, da: object):
        logging.info("Renaming ERA5 spatial coordinates to match SIC")
        if "x" in da.coords and "y" in da.coords:
            da = da.rename(dict(x="xc", y="yc"))
        return da
