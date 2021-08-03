import logging
import os
import subprocess as sp

from enum import Flag, auto


class Hemisphere(Flag):
    NONE = 0
    NORTH = auto()
    SOUTH = auto()
    BOTH = NORTH & SOUTH


class HemisphereMixin:
    _hemisphere = Hemisphere.NONE

    @property
    def hemisphere_str(self):
        return ["nh"] if self.north else \
               ["sh"] if self.south else \
               ["nh", "sh"]

    @property
    def hemisphere_loc(self):
        return [90, -180, 0, 180] if self.north else \
               [-90, -180, 0, 180] if self.south else \
               [-90, -180, 90, 180]

    @property
    def north(self):
        return self._hemisphere & Hemisphere.NORTH

    @property
    def south(self):
        return self._hemisphere & Hemisphere.SOUTH

    @property
    def both(self):
        return self._hemisphere & Hemisphere.BOTH


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
