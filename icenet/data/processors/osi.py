import datetime as dt
import os

from icenet.data.cli import process_args, process_date_args
from icenet.data.process import IceNetPreProcessor
from icenet.data.sic.mask import Masks
from icenet.data.processors.utils import sic_interpolate

"""

"""


class IceNetOSIPreProcessor(IceNetPreProcessor):
    """

    :param missing_dates:
    """
    def __init__(self, *args,
                 missing_dates: object = None,
                 **kwargs):
        super().__init__(*args, identifier="osisaf", **kwargs)

        missing_dates_path = os.path.join(
            self._source_data,
            "siconca",
            "missing_days.csv")

        missing_dates = [] if missing_dates is None else missing_dates
        assert type(missing_dates) is list

        with open(missing_dates_path, "r") as fh:
            missing_dates += [dt.date(*[int(s)
                                        for s in line.strip().split(",")])
                              for line in fh.readlines()]
        self.missing_dates = list(set(missing_dates))

    def pre_normalisation(self,
                          var_name: str,
                          da: object):
        """

        :param var_name:
        :param da:
        :return:
        """
        if var_name != "siconca":
            raise RuntimeError("OSISAF SIC implementation should be dealing "
                               "with siconca, ")
        else:
            masks = Masks(north=self.north, south=self.south)
            return sic_interpolate(da, masks)


def main():
    args = process_args()
    dates = process_date_args(args)

    osi = IceNetOSIPreProcessor(
        args.abs,
        args.anom,
        args.name,
        dates["train"],
        dates["val"],
        dates["test"],
        linear_trends=args.trends,
        linear_trend_steps=args.trend_lead,
        north=args.hemisphere == "north",
        ref_procdir=args.ref,
        south=args.hemisphere == "south"
    )
    osi.init_source_data(
        lag_days=args.lag,
    )
    osi.process()
