import argparse

from icenet2.data.cli import date_arg
from icenet2.utils import setup_logging


@setup_logging
def data_args() -> object:
    """

    :return:
    """

    ap = argparse.ArgumentParser()
    ap.add_argument("hemisphere", choices=("north", "south"))

    ap.add_argument("start_date", type=date_arg, default=None)
    ap.add_argument("end_date", type=date_arg, default=None)

    ap.add_argument("vars",
                    help="Comma separated list of vars",
                    nargs="+",
                    default=[])

    ap.add_argument("-v", "--verbose", action="store_true", default=False)

    args = ap.parse_args()
    return args


def main():
    args = data_args()
