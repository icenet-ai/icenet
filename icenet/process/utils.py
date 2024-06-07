import os


def destination_filename(destination: object, filename: str,
                         date: object) -> object:
    """

    :param destination:
    :param filename:
    :param date:
    :return:
    """
    return os.path.join(
        destination, "{}.{}{}".format(
            os.path.splitext(os.path.basename(filename))[0],
            date.strftime("%Y-%m-%d"),
            os.path.splitext(os.path.basename(filename))[1],
        ))
