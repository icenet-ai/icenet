import argparse
import logging

import matplotlib
import matplotlib.pyplot as plt

from icenet2.data.cli import date_arg
from icenet2.plotting.utils import get_forecast_obs_ds

matplotlib.rcParams.update({
    'figure.facecolor': 'w',
    'figure.dpi': 300
})


def plot_sic_error(fc_da, obs_da, land_mask):
    """

    :param fc_da:
    :param obs_da:
    :param land_mask:
    """
    raise NotImplementedError(fc_da, obs_da, land_mask)


def sic_error_args() -> object:
    """

    :return:
    """

    ap = argparse.ArgumentParser()
    ap.add_argument("hemisphere", choices=("north", "south"))
    ap.add_argument("forecast_file", type=str)
    ap.add_argument("forecast_date", type=date_arg)

    ap.add_argument("-v", "--verbose", action="store_true", default=False)

    args = ap.parse_args()
    return args


def sic_error():
    args = sic_error_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    plot_sic_error(*get_forecast_obs_ds(args.hemisphere,
                                        args.forecast_file,
                                        args.forecast_date))
