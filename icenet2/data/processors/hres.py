from icenet2.data.cli import process_args, process_date_args
from icenet2.data.process import IceNetPreProcessor


class IceNetHRESPreProcessor(IceNetPreProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args,
                         file_filters=["latlon_"],
                         identifier="mars.hres",
                         **kwargs)


def main():
    args = process_args()
    dates = process_date_args(args)

    hres_clim = IceNetHRESPreProcessor(
        ["uas", "vas"],
        ["tas", "ta500", "tos", "psl", "zg500", "zg250", "rsds", "rlds",
         "hus1000"],
        args.name,
        dates["train"],
        dates["val"],
        dates["test"],
        linear_trends=tuple(),
    )
    hres_clim.init_source_data(
        lag_days=args.lag,
    )
    hres_clim.process()

    hres_osi = IceNetHRESPreProcessor(
        ["siconca"],
        [],
        args.name,
        dates["train"],
        dates["val"],
        dates["test"],
        linear_trends=["siconca"],
        linear_trend_days=args.forecast_days,
        # TODO: should reconsider the process for double usage (overrides?)
        #  though this does work as is, which is nice
        update_key="mars.siconca"
    )
    hres_osi.init_source_data(
        lag_days=args.lag,
        lead_days=args.forecast_days,
    )
    hres_osi.process()
