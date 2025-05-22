import logging

import numpy as np

from preprocess_toolbox.processor import NormalisingChannelProcessor


class CMIP6PreProcessor(NormalisingChannelProcessor):
    def pre_normalisation(self, var_name: str, da: object):
        """

        :param var_name:
        :param da:
        :return:
        """
        if var_name == "siconca":
            # TODO: if self._source == 'MRI-ESM2-0':
            da /= 100.

        if da.dtype != np.floating:
            logging.info("Regrid processing, data type not float: {}".format(da.dtype))
            da = da.astype(self._dtype)

        return da

    def post_normalisation(self, var_name: str, da: object):
        logging.info("Renaming CMIP spatial coordinates to match sample output requirements")
        if "x" in da.coords and "y" in da.coords:
            da = da.rename(dict(x="xc", y="yc"))
        return da

