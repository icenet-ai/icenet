import logging

from icenet2.data.cli import process_args, process_date_args
from icenet2.data.process import IceNetPreProcessor
from icenet2.data.sic.mask import Masks
from icenet2.data.processors.utils import SICInterpolation


class IceNetCMIPPreProcessor(IceNetPreProcessor):
    def __init__(self,
                 source, member, *args, **kwargs):
        cmip_source = "{}.{}".format(source, member)
        super().__init__(*args,
                         identifier="cmip6.{}".format(cmip_source),
                         **kwargs)

    def pre_normalisation(self, var_name, da):
        if var_name == "siconca":
            masks = Masks(north=self.north, south=self.south)
            return SICInterpolation.interpolate(da, masks)

        return da


def main():
    args = process_args(
        extra_args=[
            (["source"], dict(type=str)),
            (["member"], dict(type=str)),
        ],
    )
    dates = process_date_args(args)

    cmip = IceNetCMIPPreProcessor(
        args.source,
        args.member,
        ["uas", "vas"],
        ["tas", "ta500", "tos", "psl", "zg500", "zg250", "rsds", "rlds",
         "hus1000"],
        args.name,
        dates["train"],
        dates["val"],
        dates["test"],
        linear_trends=tuple(),
        north=args.hemisphere == "north",
        ref_procdir=args.ref,
        south=args.hemisphere == "south",
    )
    cmip.init_source_data(
        lag_days=args.lag,
    )
    cmip.process()

    logging.info("SIC PROCESSING")

    cmip_sic = IceNetCMIPPreProcessor(
        args.source,
        args.member,
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
        south=args.hemisphere == "south",
        update_key="siconca.{}.{}".format(args.source, args.member),
    )
    cmip_sic.init_source_data(
        lag_days=args.lag,
        lead_days=args.forecast_days,
    )
    cmip_sic.process()
