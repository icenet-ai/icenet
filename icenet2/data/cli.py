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


def download_args(dates=True):
    ap = argparse.ArgumentParser()
    ap.add_argument("hemisphere", choices=("north", "south"))

    if dates:
        ap.add_argument("start_date", type=date_arg)
        ap.add_argument("end_date", type=date_arg)

    ap.add_argument("-v", "--verbose", action="store_true", default=False)
    args = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    return args


def process_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", type=str)
    ap.add_argument("hemisphere", choices=("north", "south"))

    ap.add_argument("train_start", type=date_arg)
    ap.add_argument("train_end", type=date_arg)
    ap.add_argument("val_start", type=date_arg)
    ap.add_argument("val_end", type=date_arg)

    ap.add_argument("-ts", "--test-start", dest="test_start",
                    type=date_arg, required=False, default=[])
    ap.add_argument("-te", "--test-end", dest="test_end",
                    type=date_arg, required=False, default=[])

    ap.add_argument("-d", "--date-ratio", type=float, default=1.0)

    ap.add_argument("-v", "--verbose", action="store_true", default=False)

    ap.add_argument("-l", "--lag", type=int, default=2)

    args = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    return args


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
        logging.info("Generated {} dates for {}".format(len(dataset_dates),
                                                        dataset))

        num_dates = len(dataset_dates) * args.date_ratio
        random.shuffle(dataset_dates)

        while len(dataset_dates) > num_dates:
            f = dataset_dates.pop \
                if len(dataset_dates) % 2 == 0 \
                else dataset_dates.popleft
            f()

        logging.info("After reduction we have {} {} dates".
                     format(len(dataset_dates), dataset))
        dates[dataset] = sorted(list(dataset_dates))
    return dates

