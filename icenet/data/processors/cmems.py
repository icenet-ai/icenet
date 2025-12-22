"""
CMEMS (Copernicus Marine Environment Monitoring Service) data processors.

This includes ORAS5 ocean reanalysis data.
"""

import logging
from preprocess_toolbox.processor import NormalisingChannelProcessor


class ORAS5PreProcessor(NormalisingChannelProcessor):
    """
    Processor for ORAS5 ocean reanalysis data from CMEMS.
    
    Handles ORAS5-specific preprocessing including coordinate renaming
    to match IceNet output requirements.
    
    Note: Time coordinate shifting from mid-month to end-of-month is done
    during the regridding stage in orca_grid.py to ensure all regridded
    files have consistent end-of-month timestamps.
    """
    
    def pre_normalisation(self, var_name: str, da: object):
        """Pre-normalisation processing for ORAS5 data."""
        return da
    
    def post_normalisation(self, var_name: str, da: object):
        """Post-normalisation processing - rename coordinates to match output requirements."""
        logging.info("Renaming ORAS5 spatial coordinates to match sample output requirements")
        if "x" in da.coords and "y" in da.coords:
            da = da.rename(dict(x="xc", y="yc"))
        return da


__all__ = ['ORAS5PreProcessor']

