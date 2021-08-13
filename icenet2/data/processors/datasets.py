import datetime
import logging

import pandas as pd

from icenet2.data.process import IceNetPreProcessor


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

        return da


class IceNetERA5PreProcessor(IceNetPreProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, identifier="era5", **kwargs)


class IceNetOSIPreProcessor(IceNetPreProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, identifier="osisaf", **kwargs)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    pp = IceNetERA5PreProcessor(
        ["uas", "vas"],
        ["tas", "ta500", "tos", "psl", "zg500", "zg250", "rsds", "rlds",
         "hus1000"],
        "test1",
        [datetime.date(2021, 1, 1)],
        [],
        [],
    )
    pp.init_source_data()
    pp.process()

    osi = IceNetOSIPreProcessor(
        ["siconca"],
        [],
        "test1",
        [datetime.date(2020, 1, 1)],
        [],
        [],
        include_circday=False,
        include_land=False,
        linear_trends=tuple(),
    )
    osi.init_source_data()
    osi.process()

