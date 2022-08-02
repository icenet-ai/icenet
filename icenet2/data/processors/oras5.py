from icenet2.data.cli import process_args, process_date_args
from icenet2.data.process import IceNetPreProcessor


class IceNetORAS5PreProcessor(IceNetPreProcessor):
    """

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args,
                         file_filters=["latlon_"],
                         identifier="oras5",
                         **kwargs)


def main():
    args = process_args(extra_args=[
        (tuple(["--abs"]), dict(
            help="Comma separated list of abs vars",
            type=lambda x: x.split(",") if "," in x else [x],
            default=[],
        )),
        (tuple(["--anom"]), dict(
            help="Comma separated list of anom vars",
            type=lambda x: x.split(",") if "," in x else [x],
            default=[],
        )),
        (tuple(["--trend"]), dict(
            help="Comma separated list of vars to produce trends for",
            type=lambda x: x.split(",") if "," in x else [x],
            default=[],
        ))
    ])
    dates = process_date_args(args)

    pp = IceNetORAS5PreProcessor(
        args.abs,
        args.anom,
        args.name,
        dates["train"],
        dates["val"],
        dates["test"],
        linear_trends=args.trend,
        north=args.hemisphere == "north",
        ref_procdir=args.ref,
        south=args.hemisphere == "south"
    )
    pp.init_source_data(
        lag_days=args.lag,
    )
    pp.process()
