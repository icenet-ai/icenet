from icenet2.data.cli import process_args, process_date_args
from icenet2.data.process import IceNetPreProcessor
from icenet2.data.sic.mask import Masks
from icenet2.data.processors.utils import SICInterpolation


class IceNetCMIPPreProcessor(IceNetPreProcessor):
    def __init__(self, source, member, *args, **kwargs):
        cmip_source = "{}.{}".format(source, member)
        super().__init__(*args,
                         identifier="cmip.{}".format(cmip_source),
                         source_suffix=cmip_source,
                         **kwargs)

    def pre_normalisation(self, var_name, da):
        if var_name == "siconca":
            masks = Masks(north=self.north, south=self.south)
            return SICInterpolation.interpolate(da, masks)

        return da


def main():
    args = process_args()
    dates = process_date_args(args)

    pp = IceNetCMIPPreProcessor(
        # source and member args please!
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
        south=args.hemisphere == "south"
    )
    # ./data/cmip6/north/vas/MRI-ESM2-0.r2i1p1f1/2050/latlon_2050_01_22.nc
    pp.init_source_data(
        lag_days=args.lag,
    )
    pp.process()
