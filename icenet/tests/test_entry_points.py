"""Tests for icenet commands (entry points)"""

from importlib_metadata import entry_points
import pytest


icenet_entry_points = [
    ep for ep in entry_points(group="console_scripts")
    if ep.module.startswith('icenet')
]


def test_have_entry_points():
    """Check that there is at least one entry point (to stop the other
    tests passing vacuously if these are moved)
    """
    assert len(icenet_entry_points) > 0
    

@pytest.mark.parametrize("entry_point", icenet_entry_points)
def test_entry_point_exists(entry_point):
    """Check that the entry point ep can be loaded correctly
    (parameterized over all icenet entry points)
    """
    # Check that the entry point can be loaded without raising an
    # exception
    entry_point.load()
