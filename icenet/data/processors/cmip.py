import logging

from icenet.data.cli import process_args, process_date_args
from icenet.data.process import IceNetPreProcessor
from icenet.data.sic.mask import Masks
from icenet.data.processors.utils import sic_interpolate

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
            return sic_interpolate(da, masks)

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
        args.abs,
        args.anom,
        args.name,
        dates["train"],
        dates["val"],
        dates["test"],
        linear_trends=args.trends,
        linear_trend_days=args.trend_lead,
        north=args.hemisphere == "north",
        ref_procdir=args.ref,
        south=args.hemisphere == "south",
        update_key=args.update_key,
    )
    cmip.init_source_data(
        lag_days=args.lag,
    )
    cmip.process()
