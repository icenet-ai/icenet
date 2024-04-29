import logging
import subprocess as sp

from enum import Flag, auto
from functools import wraps


class Hemisphere(Flag):
    """Representation of hemispheres & both with bitwise operations.

    An enum.Flag derived class representing the different hemispheres
        (north, south, or both), providing methods to check which hemisphere
        is selected via bitwise operations:
            & (AND), | (OR), ^ (XOR), and ~ (INVERT)
    """

    NONE = 0
    NORTH = auto()
    SOUTH = auto()
    BOTH = NORTH | SOUTH


class HemisphereMixin:
    """A mixin relating to Hemisphere checking.

    Attributes:
        _hemisphere: Represents the bitmask value of the hemisphere.
            Defaults to Hemisphere.NONE (i.e., 0).
    """

    _hemisphere: int = Hemisphere.NONE

    @property
    def hemisphere(self) -> int:
        """The bitmask value representing the hemisphere."""
        return self._hemisphere

    @property
    def hemisphere_str(self) -> list:
        """A list of strings representing the selected hemispheres."""
        return ["north"] if self.north else \
               ["south"] if self.south else \
               ["north", "south"]

    @property
    def hemisphere_loc(self) -> list:
        """Get a list of latitude and longitude extent representing the hemisphere's location."""
        # A list of latitude and longitude extent representing the hemisphere's location.
        # [north lat, west lon, south lat, east lon]
        return [90, -180, 0, 180] if self.north else \
               [0, -180, -90, 180] if self.south else \
               [90, -180, -90, 180]

    @property
    def north(self) -> bool:
        """A flag indicating if `_hemisphere` is north. True if the hemisphere is north, False otherwise."""
        return (self._hemisphere & Hemisphere.NORTH) == Hemisphere.NORTH

    @property
    def south(self) -> bool:
        """A flag indicating if `_hemisphere` is south. True if the hemisphere is south, False otherwise."""
        return (self._hemisphere & Hemisphere.SOUTH) == Hemisphere.SOUTH

    @property
    def both(self) -> int:
        """The bitmask value representing both hemispheres."""
        return self._hemisphere & Hemisphere.BOTH


def run_command(command: str, dry: bool = False) -> object:
    """Run a shell command

    A wrapper in case we want some additional handling to go in here

    Args:
        command: Command to run in shell.
        dry (optional): Whether to do a dry run or to run actual command.
            Default is False.

    Returns:
        subprocess.CompletedProcess return of the executed command.
    """
    if dry:
        logging.info("Skipping dry command: {}".format(command))
        return 0

    ret = sp.run(command, shell=True)
    if ret.returncode < 0:
        logging.warning(
            "Child was terminated by signal: {}".format(-ret.returncode))
    else:
        logging.info("Child returned: {}".format(-ret.returncode))

    return ret

