import logging

from icenet2.data.cli import process_args, process_date_args
from icenet2.data.process import IceNetPreProcessor
from icenet2.data.sic.mask import Masks

"""

"""


class IceNetCMIPPreProcessor(IceNetPreProcessor):
    """

    :param source:
    :param member:
    """
    def __init__(self,
                 source: str,
                 member: str,
                 *args, **kwargs):
        cmip_source = "{}.{}".format(source, member)
        super().__init__(*args,
                         identifier="cmip6.{}".format(cmip_source),
                         **kwargs)

    def pre_normalisation(self,
                          var_name: str,
                          da: object):
        """

        :param var_name:
        :param da:
        :return:
        """
        if var_name == "siconca":
            masks = Masks(north=self.north, south=self.south)
            return interpolate(da, masks)

        return da


def main():
    args = process_args(
        extra_args=[
            (["source"], dict(type=str)),
            (["member"], dict(type=str)),
            (["-ss", "--skip-sic"], dict(default=False, action="store_true")),
            (["-sv", "--skip-var"], dict(default=False, action="store_true")),
        ],
    )
    dates = process_date_args(args)

    if not args.skip_var:
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

    if not args.skip_sic:
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
