import argparse
import collections
import datetime as dt
import logging
import random
import re

import pandas as pd


def date_arg(string):
    date_match = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", string)
    return dt.date(*[int(s) for s in date_match.groups()])


def dates_arg(string):
    if string == "none":
        return []

    date_match = re.findall(r"(\d{4})-(\d{1,2})-(\d{1,2})", string)

    if len(date_match) < 1:
        raise argparse.ArgumentError("No dates found for supplied argument {}".
                                     format(string))
    return [dt.date(*[int(s) for s in date_tuple]) for date_tuple in date_match]


def download_args(choices=None,
                  dates=True,
                  dates_optional=False,
                  skip_download=False,
                  workers=False,
                  extra_args=[]):
    ap = argparse.ArgumentParser()
    ap.add_argument("hemisphere", choices=("north", "south"))

    if choices and type(choices) == list:
        ap.add_argument("-c", "--choice", choices=choices, default=choices[0])

    if dates:
        pos_args = [["start_date"], ["end_date"]] if not dates_optional else \
            [["-sd", "--start-date"], ["-ed", "--end-date"]]
        ap.add_argument(*pos_args[0], type=date_arg, default=None)
        ap.add_argument(*pos_args[1], type=date_arg, default=None)

    if skip_download:
        ap.add_argument("-s", "--skip-download", default=False,
                        action="store_true")

    if workers:
        ap.add_argument("-w", "--workers", default=8, type=int)

    ap.add_argument("-d", "--dont-delete", dest="delete",
                    action="store_false", default=True)
    ap.add_argument("-v", "--verbose", action="store_true", default=False)

    for arg in extra_args:
        ap.add_argument(*arg[0], **arg[1])

    args = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("cdsapi").setLevel(logging.WARNING)
    return args


def process_args(dates=True,
                 lag_lead=True,
                 ref_option=True,
                 extra_args=[]):
    ap = argparse.ArgumentParser()
    ap.add_argument("name", type=str)
    ap.add_argument("hemisphere", choices=("north", "south"))

    if dates:
        add_date_args(ap)

        # FIXME#11: not allowing this option currently
        # ap.add_argument("-d", "--date-ratio", type=float, default=1.0)

    if lag_lead:
        ap.add_argument("-l", "--lag", type=int, default=2)
        ap.add_argument("-f", "--forecast-days", type=int, default=93)

    for arg in extra_args:
        ap.add_argument(*arg[0], **arg[1])

    if ref_option:
        ap.add_argument("-r", "--ref",
                        help="Reference loader for normalisations etc",
                        default=None, type=str)
    ap.add_argument("-v", "--verbose", action="store_true", default=False)

    args = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    return args


def add_date_args(arg_parser):
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


def process_date_args(args):
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

