import datetime as dt
import logging
import os

import pandas as pd

from icenet2.data.process import IceNetPreProcessor
from icenet2.data.sic.mask import Masks
from icenet2.data.processors.utils import SICInterpolation

# TODO: Split datasets into modules correlating to interfaces


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


class IceNetERA5PreProcessor(IceNetPreProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args,
                         file_filters=["latlon_"],
                         identifier="era5",
                         **kwargs)


class IceNetHRESPreProcessor(IceNetPreProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args,
                         file_filters=["latlon_"],
                         identifier="mars.hres",
                         **kwargs)


class IceNetOSIPreProcessor(IceNetPreProcessor):
    def __init__(self, *args,
                 missing_dates=None,
                 **kwargs):
        super().__init__(*args, identifier="osisaf", **kwargs)

        missing_dates_path = os.path.join(
            self._source_data,
            *self.hemisphere_str,
            "siconca",
            "missing_days.csv")

        missing_dates = [] if missing_dates is None else missing_dates
        assert type(missing_dates) is list

        with open(missing_dates_path, "r") as fh:
            missing_dates += [dt.date(*[int(s)
                                        for s in line.strip().split(",")])
                              for line in fh.readlines()]
        self.missing_dates = missing_dates

    def pre_normalisation(self, var_name, da):
        if var_name != "siconca":
            raise RuntimeError("OSISAF SIC implementation should be dealing "
                               "with siconca, ")
        else:
            masks = Masks(north=self.north, south=self.south)
            return SICInterpolation.interpolate(da, masks)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)

    osi = IceNetOSIPreProcessor(
        ["siconca"],
        [],
        "test1",
        list([
            pd.to_datetime(date).date() for date in
            pd.date_range(dt.date(2010, 1, 1), dt.date(2010, 1, 6), freq='D')
        ]),
        [],
        [],
        include_circday=False,
        include_land=False,
        linear_trends=["siconca"],
        linear_trend_days=3,
    )
    osi.init_source_data()
    osi.process()

