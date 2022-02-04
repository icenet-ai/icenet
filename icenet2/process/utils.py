import datetime as dt
import os
import re


def date_arg(string):
    d_match = re.search(r'^(\d+)-(\d+)-(\d+)$', string).groups()

    if d_match:
        return dt.date(*[int(s) for s in d_match])


def destination_filename(destination, filename, date):
    return os.path.join(destination,
                        "{}.{}{}".format(
                            os.path.splitext(
                                os.path.basename(filename))[0],
                            date.strftime("%d%m%Y"),
                            os.path.splitext(
                                os.path.basename(filename))[1],
                        ))
