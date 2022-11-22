import argparse
import collections
import datetime as dt
import logging
import re

from pprint import pformat

import pandas as pd

from icenet.utils import setup_logging

"""

"""


def date_arg(string: str) -> object:
    """

    :param string: 
    :return: 
    """
    date_match = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", string)
    return dt.date(*[int(s) for s in date_match.groups()])


def dates_arg(string: str) -> object:
    """

    :param string: 
    :return: 
    """
    if string == "none":
        return []

    date_match = re.findall(r"(\d{4})-(\d{1,2})-(\d{1,2})", string)

    if len(date_match) < 1:
        raise argparse.ArgumentError("No dates found for supplied argument {}".
                                     format(string))
    return [dt.date(*[int(s) for s in date_tuple]) for date_tuple in date_match]


def csv_arg(string: str) -> list:
    """

    :param string:
    :return:
    """
    csv_items = []
    for el in string.split(","):
        if len(el) == 0:
            csv_items.append(None)
        else:
            csv_items.append(el)
    return csv_items


def csv_of_csv_arg(string: str) -> list:
    """

    :param string:
    :return:
    """
    csv_items = []
    for el in string.split(","):
        if len(el) == 0:
            csv_items.append(None)
        else:
            csv_items.append(el.split("|"))
    return csv_items


def int_or_list_arg(string: str) -> object:
    """

    :param string:
    :return:
    """
    try:
        val = int(string)
    except ValueError:
        val = string.split(",")
    return val


@setup_logging
def download_args(choices: object = None,
                  dates: bool = True,
                  dates_optional: bool = False,
                  var_specs: bool = True,
                  workers: bool = False,
                  extra_args: object = ()) -> object:
    """

    :param choices:
    :param dates:
    :param dates_optional:
    :param var_specs:
    :param workers:
    :param extra_args:
    :return:
    """

    ap = argparse.ArgumentParser()
    ap.add_argument("hemisphere", choices=("north", "south"))

    if choices and type(choices) == list:
        ap.add_argument("-c", "--choice", choices=choices, default=choices[0])

    if dates:
        pos_args = [["start_date"], ["end_date"]] if not dates_optional else \
            [["-sd", "--start-date"], ["-ed", "--end-date"]]
        ap.add_argument(*pos_args[0], type=date_arg, default=None)
        ap.add_argument(*pos_args[1], type=date_arg, default=None)

    if workers:
        ap.add_argument("-w", "--workers", default=8, type=int)

    ap.add_argument("-d", "--dont-delete", dest="delete",
                    action="store_false", default=True)
    ap.add_argument("-v", "--verbose", action="store_true", default=False)

    if var_specs:
        ap.add_argument("--vars",
                        help="Comma separated list of vars",
                        type=csv_arg,
                        default=[])
        ap.add_argument("--levels",
                        help="Comma separated list of pressures/depths as needed, "
                             "use zero length string if None (e.g. ',,500,,,') and "
                             "pipes for multiple per var (e.g. ',,250|500,,'",
                        type=csv_of_csv_arg,
                        default=[])

    for arg in extra_args:
        ap.add_argument(*arg[0], **arg[1])
    args = ap.parse_args()
    return args


@setup_logging
def process_args(dates: bool = True,
                 ref_option: bool = True,
                 extra_args: object = ()) -> object:
    """

    :param dates:
    :param ref_option:
    :param extra_args:
    :return:
    """

    ap = argparse.ArgumentParser()
    ap.add_argument("name", type=str)
    ap.add_argument("hemisphere", choices=("north", "south"))

    if dates:
        add_date_args(ap)

        # FIXME#11: not allowing this option currently
        # ap.add_argument("-d", "--date-ratio", type=float, default=1.0)

    ap.add_argument("-l", "--lag", type=int, default=2)
    ap.add_argument("-f", "--forecast", type=int, default=93)

    ap.add_argument("--abs",
                    help="Comma separated list of abs vars",
                    type=csv_arg,
                    default=[])
    ap.add_argument("--anom",
                    help="Comma separated list of abs vars",
                    type=csv_arg,
                    default=[])
    ap.add_argument("--trends",
                    help="Comma separated list of abs vars",
                    type=csv_arg,
                    default=[])
    ap.add_argument("--trend-lead",
                    help="Time steps in the future for linear trends",
                    type=int_or_list_arg,
                    default=93)

    for arg in extra_args:
        ap.add_argument(*arg[0], **arg[1])

    if ref_option:
        ap.add_argument("-r", "--ref",
                        help="Reference loader for normalisations etc",
                        default=None, type=str)
    ap.add_argument("-v", "--verbose", action="store_true", default=False)

    ap.add_argument("-u", "--update-key",
                    default=None,
                    help="Add update key to processor to avoid overwriting default"
                         "entries in the loader configuration",
                    type=str)

    args = ap.parse_args()
    return args


def add_date_args(arg_parser: object):
    """

    :param arg_parser:
    """
    arg_parser.add_argument("-ns", "--train_start",
                            type=dates_arg, required=False, default=[])
    arg_parser.add_argument("-ne", "--train_end",
                            type=dates_arg, required=False, default=[])
    arg_parser.add_argument("-vs", "--val_start",
                            type=dates_arg, required=False, default=[])
    arg_parser.add_argument("-ve", "--val_end",
                            type=dates_arg, required=False, default=[])
    arg_parser.add_argument("-ts", "--test-start",
                            type=dates_arg, required=False, default=[])
    arg_parser.add_argument("-te", "--test-end", dest="test_end",
                            type=dates_arg, required=False, default=[])


def process_date_args(args: object) -> dict:
    """

    :param args:
    :return:
    """
    dates = dict(train=[], val=[], test=[])

    for dataset in ("train", "val", "test"):
        dataset_dates = collections.deque()

        for i, period_start in \
                enumerate(getattr(args, "{}_start".format(dataset))):
            period_end = getattr(args, "{}_end".format(dataset))[i]
            dataset_dates += [pd.to_datetime(date).date() for date in
                              pd.date_range(period_start,
                                            period_end, freq="D")]
        logging.info("Got {} dates for {}".format(len(dataset_dates),
                                                  dataset))

        dates[dataset] = sorted(list(dataset_dates))
    return dates

