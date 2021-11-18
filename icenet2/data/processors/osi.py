import datetime as dt
import os

from icenet2.data.cli import process_args, process_date_args
from icenet2.data.process import IceNetPreProcessor
from icenet2.data.sic.mask import Masks
from icenet2.data.processors.utils import SICInterpolation


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
        self.missing_dates = list(set(missing_dates))

    def pre_normalisation(self, var_name, da):
        if var_name != "siconca":
            raise RuntimeError("OSISAF SIC implementation should be dealing "
                               "with siconca, ")
        else:
            masks = Masks(north=self.north, south=self.south)
            return SICInterpolation.interpolate(da, masks)


def main():
    args = process_args()
    dates = process_date_args(args)

    osi = IceNetOSIPreProcessor(
        ["siconca"],
        [],
        args.name,
        dates["train"],
        dates["val"],
        dates["test"],
        linear_trends=["siconca"],
        linear_trend_days=args.forecast_days,
        north=args.hemisphere == "north",
        ref_procdir=args.ref,
        south=args.hemisphere == "south"
    )
    osi.init_source_data(
        lag_days=args.lag,
        lead_days=args.forecast_days,
    )
    osi.process()
