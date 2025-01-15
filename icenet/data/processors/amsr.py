import logging

from preprocess_toolbox.processor import NormalisingChannelProcessor


class AMSR2PreProcessor(NormalisingChannelProcessor):
    def pre_normalisation(self, var_name: str, da: object):
        logging.info("Renaming AMSR2 spatial coordinates to match IceNet based on OSISAF")
        da = da.rename(dict(x="xc", y="yc"))
        return da
