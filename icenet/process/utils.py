import datetime as dt
import re
import os


def date_arg(string: str) -> object:
    """

    :param string:
    :return:
    """
    d_match = re.search(r'^(\d+)-(\d+)-(\d+)$', string).groups()

    if d_match:
        return dt.date(*[int(s) for s in d_match])


def destination_filename(destination: object,
                         filename: str,
                         date: object) -> object:
    """

    :param destination:
    :param filename:
    :param date:
    :return:
    """
    return os.path.join(destination,
                        "{}.{}{}".format(
                            os.path.splitext(
                                os.path.basename(filename))[0],
                            date.strftime("%Y-%m-%d"),
                            os.path.splitext(
                                os.path.basename(filename))[1],
                        ))
