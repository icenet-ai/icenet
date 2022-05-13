import logging
import os
import subprocess as sp

from enum import Flag, auto


class Hemisphere(Flag):
    NONE = 0
    NORTH = auto()
    SOUTH = auto()
    BOTH = NORTH | SOUTH


class HemisphereMixin:
    _hemisphere = Hemisphere.NONE

    @property
    def hemisphere(self):
        return self._hemisphere

    @property
    def hemisphere_str(self):
        return ["north"] if self.north else \
               ["south"] if self.south else \
               ["north", "south"]

    @property
    def hemisphere_loc(self):
        return [90, -180, 0, 180] if self.north else \
               [0, -180, -90, 180] if self.south else \
               [90, -180, -90, 180]

    @property
    def north(self):
        return (self._hemisphere & Hemisphere.NORTH) == Hemisphere.NORTH

    @property
    def south(self):
        return (self._hemisphere & Hemisphere.SOUTH) == Hemisphere.SOUTH

    @property
    def both(self):
        return self._hemisphere & Hemisphere.BOTH


def run_command(commandstr, dry=False):
    """Run a shell command

    A wrapper in case we want some additional handling to go in here

    Args:
        commandstr (str): command to run in a shell
    Returns:
        retcode (int): return code
    Raises:
        OSError: from subprocess
    """
    if dry:
        logging.info("Skipping dry commaand: {}".format(commandstr))
        return 0

    ret = sp.run(commandstr, shell=True)
    if ret.returncode < 0:
        logging.warning("Child was terminated by signal: {}".
                        format(-ret.returncode))
    else:
        logging.info("Child returned: {}".
                     format(-ret.returncode))

    return ret.returncode
