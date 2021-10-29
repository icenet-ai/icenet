from icenet2.data.cli import process_args, process_date_args
from icenet2.data.process import IceNetPreProcessor


class IceNetERA5PreProcessor(IceNetPreProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args,
                         file_filters=["latlon_"],
                         identifier="era5",
                         **kwargs)


def main():
    args = process_args()
    dates = process_date_args(args)

    pp = IceNetERA5PreProcessor(
        ["uas", "vas"],
        ["tas", "ta500", "tos", "psl", "zg500", "zg250", "rsds", "rlds",
         "hus1000"],
        args.name,
        dates["train"],
        dates["val"],
        dates["test"],
        linear_trends=tuple(),
    )
    pp.init_source_data(
        lag_days=args.lag,
    )
    pp.process()
