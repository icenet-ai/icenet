from icenet.data.cli import process_args, process_date_args
from icenet.data.process import IceNetPreProcessor

"""

"""


class IceNetHRESPreProcessor(IceNetPreProcessor):
    """

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args,
                         file_filters=["latlon_"],
                         identifier="mars.hres",
                         **kwargs)


def main():
    args = process_args()
    dates = process_date_args(args)

    hres = IceNetHRESPreProcessor(
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
    hres.init_source_data(
        lag_days=args.lag,
    )
    hres.process()

    hres_osi = IceNetHRESPreProcessor(
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
        # TODO: should reconsider the process for double usage (overrides?)
        #  though this does work as is, which is nice
        update_key="mars.siconca",
    )
    hres_osi.init_source_data(
        lag_days=args.lag,
        lead_days=args.forecast_days,
    )
    hres_osi.process()
