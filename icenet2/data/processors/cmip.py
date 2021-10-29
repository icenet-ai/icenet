import pandas as pd

from icenet2.data.process import IceNetPreProcessor
from icenet2.data.sic.mask import Masks
from icenet2.data.processors.utils import SICInterpolation


# TODO: pick up identifiers from the interfaces
class IceNetCMIPPreProcessor(IceNetPreProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, identifier="cmip", **kwargs)

    def pre_normalisation(self, var_name, da):
        """
        Convert the cmip6 xarray time dimension to use day=1, hour=0 convention
        used in the rest of the project.
        """

        standardised_dates = []
        for datetime64 in da.time.values:
            date = pd.Timestamp(datetime64, unit='s')
            date = date.replace(day=1, hour=0)
            standardised_dates.append(date)
        da = da.assign_coords({'time': standardised_dates})

        if var_name == "siconca":
            masks = Masks(north=self.north, south=self.south)
            return SICInterpolation.interpolate(da, masks)

        return da


def main():
    raise NotImplementedError("CMIP processing not currently implemented")
