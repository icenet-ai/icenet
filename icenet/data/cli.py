import argparse
import collections
import datetime as dt
import re
import logging
import os

import pandas as pd


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
        raise argparse.ArgumentError(
            "No dates found for supplied argument {}".format(string))
    return [dt.date(*[int(s) for s in date_tuple]) for date_tuple in date_match]


def csv_arg(string: str) -> list:
    """

    :param string:
    :return:
    """
    csv_items = []
    string = re.sub(r'^\'(.*)\'$', r'\1', string)

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
    string = re.sub(r'^\'(.*)\'$', r'\1', string)

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


def add_date_args(arg_parser: object):
    """

    :param arg_parser:
    """
    arg_parser.add_argument("-ns",
                            "--train_start",
                            type=dates_arg,
                            required=False,
                            default=[])
    arg_parser.add_argument("-ne",
                            "--train_end",
                            type=dates_arg,
                            required=False,
                            default=[])
    arg_parser.add_argument("-vs",
                            "--val_start",
                            type=dates_arg,
                            required=False,
                            default=[])
    arg_parser.add_argument("-ve",
                            "--val_end",
                            type=dates_arg,
                            required=False,
                            default=[])
    arg_parser.add_argument("-ts",
                            "--test-start",
                            type=dates_arg,
                            required=False,
                            default=[])
    arg_parser.add_argument("-te",
                            "--test-end",
                            dest="test_end",
                            type=dates_arg,
                            required=False,
                            default=[])


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
            dataset_dates += [
                pd.to_datetime(date).date()
                for date in pd.date_range(period_start, period_end, freq="D")
            ]
        logging.info("Got {} dates for {}".format(len(dataset_dates), dataset))

        dates[dataset] = sorted(list(dataset_dates))
    return dates
