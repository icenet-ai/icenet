import logging
import os
import subprocess as sp

from .constants import *


def get_folder(name, *args):
    """Grab the path under the data directory ensuring it's made first

    Args:
        name (str): the folder prefix
        *args: the path segments to join
    Returns:
        retcode (int): the path prefixed with the config entry folder
    """
    # TODO: Config
    if name not in FOLDERS:
        raise AttributeError("The requested top level folder doesn't exist: "
                             "{}".format(name))

    path = os.path.join(FOLDERS[name], *args)

    os.makedirs(path, exist_ok=True)
    return path


def run_command(commandstr):
    """Run a shell command

    A wrapper in case we want some additional handling to go in here

    Args:
        commandstr (str): command to run in a shell
    Returns:
        retcode (int): return code
    Raises:
        OSError: from subprocess
    """
    retcode = sp.call(commandstr, shell=True)
    if retcode < 0:
        logging.warning("Child was terminated by signal", -retcode)
    else:
        logging.info("Child returned", retcode)

    return retcode
