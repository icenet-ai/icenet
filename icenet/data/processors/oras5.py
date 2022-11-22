from icenet.data.cli import process_args, process_date_args
from icenet.data.process import IceNetPreProcessor


class IceNetORAS5PreProcessor(IceNetPreProcessor):
    """

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args,
                         file_filters=["latlon_"],
                         identifier="oras5",
                         **kwargs)


def main():
    args = process_args()
    dates = process_date_args(args)

    oras5 = IceNetORAS5PreProcessor(
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
    oras5.init_source_data(
        lag_days=args.lag,
    )
    oras5.process()
