import logging
import subprocess as sp

from dataclasses import dataclass, field
from enum import Flag, auto
from functools import wraps
from logging import Logger

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


@dataclass
class LogFormat:
    """
    Creates a logging formatter for use across IceNet.
    """

    str_format: str = "[%(asctime)-17s :%(levelname)-8s] - %(message)s"
    date_format: str = "%d-%m-%y %T"
    formatter: logging.Formatter = field(init=False)

    def __post_init__(self):
        self.formatter = logging.Formatter(self.str_format, datefmt=self.date_format)


def setup_module_logging(module_name: str, level: int=None):
    """
    Configure logger based on input name

    Args:
        module_name: Name for the logger
        level: Logging level, can be int, or constants
               from logging module (e.g. logging.INFO)

    Returns:
        A configured Logger instance
    """
    root_logger = logging.getLogger()

    logger = logging.getLogger(module_name)
    # Prevent duplication across custom and root loggers
    logger.propagate = False

    # Format custom logger
    formatter = LogFormat().formatter
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    if level is None:
        logger.setLevel(root_logger.level)
    else:
        logger.setLevel(level)
    return logger


def setup_logging(func,
                  log_format=LogFormat().str_format):

    @wraps(func)
    def wrapper(*args, **kwargs):
        parsed_args = func(*args, **kwargs)
        level = logging.INFO

        if hasattr(parsed_args, "verbose") and parsed_args.verbose:
            level = logging.DEBUG

        logging.basicConfig(
            level=level,
            format=log_format,
            datefmt=LogFormat().date_format,
        )

        # TODO: better way of handling these on a case by case basis
        logging.getLogger("cdsapi").setLevel(logging.WARNING)
        logging.getLogger("matplotlib").setLevel(logging.WARNING)
        logging.getLogger("matplotlib.pyplot").setLevel(logging.WARNING)
        logging.getLogger("requests").setLevel(logging.WARNING)
        logging.getLogger("tensorflow").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        return parsed_args

    return wrapper


def check_pytorch_import(logger: Logger) -> bool:
    """
    Check for availability of pytorch module

    Args:
        logger: A Logger instance

    Returns:
        pytorch_available: Whether `torch` was successfully imported

    Example:
        >>> logger = setup_module_logging(__name__) #doctest: +SKIP
        >>> pytorch_available = check_pytorch_import(logger) #doctest: +SKIP
    """
    pytorch_available = False
    try:
        import torch
        pytorch_available = True
    except ModuleNotFoundError:
        pass
    except ImportError:
        logger.warning("PyTorch import failed - not required if not using PyTorch")
    return pytorch_available
