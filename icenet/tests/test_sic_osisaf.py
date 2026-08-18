"""Tests for `icenet.data.sic.osisaf`.

Addresses https://github.com/icenet-ai/icenet/issues/354: OSI-SAF's OSI-430-a
product (the near-real-time SIC source `SICDownloader` used for 2021-present
dates) was decommissioned by EUMETSAT/met.no on 2025-10-17 -- confirmed
directly against the live FTP server (`osisaf.met.no`), where every monthly
directory under the old OSI-430-a path (`/reprocessed/ice/conc-cont-reproc/v3p0/`)
is empty from 2025-10 onward. The stated, and now implemented, replacement is
OSI-438, whose FTP path (`/reprocessed/ice/conc-cont-reproc-amsr/v3p0/`) was
also confirmed directly against the live server -- including that it has full
backfilled coverage for the same 2021-present range OSI-430-a used to cover,
so no new date threshold is needed, just a path swap.

These tests cover `SICDownloader._resolve_ftp_path`, the (now-extracted, pure,
static) method that picks which of the three FTP path templates applies to a
given date. It needs no FTP connection and no `SICDownloader` instance (which
would otherwise require real mask data on disk), so it runs anywhere.
"""

import datetime as dt

import pytest

from icenet.data.sic.osisaf import SICDownloader

FTP_OSI450 = "/reprocessed/ice/conc/v2p0/{:04d}/{:02d}/"
FTP_OSI430B = "/reprocessed/ice/conc-cont-reproc/v2p0/{:04d}/{:02d}/"
FTP_OSI438 = "/reprocessed/ice/conc-cont-reproc-amsr/v3p0/{:04d}/{:02d}/"

OSI430B_START = dt.date(2016, 1, 1)
OSI438_START = dt.date(2021, 1, 1)


def _resolve(date):
    return SICDownloader._resolve_ftp_path(
        date, FTP_OSI450, FTP_OSI430B, FTP_OSI438, OSI430B_START, OSI438_START
    )


@pytest.mark.parametrize(
    "date,expected",
    [
        # Pre-1979-agg era: OSI-450.
        (dt.date(1990, 6, 15), "/reprocessed/ice/conc/v2p0/1990/06/"),
        (dt.date(2015, 12, 31), "/reprocessed/ice/conc/v2p0/2015/12/"),
        # The 2016-01-01 boundary itself belongs to the *newer* product
        # (`<` not `<=` in the selection, matching the pre-refactor inline
        # logic this was extracted from).
        (dt.date(2016, 1, 1),
         "/reprocessed/ice/conc-cont-reproc/v2p0/2016/01/"),
        (dt.date(2019, 7, 4),
         "/reprocessed/ice/conc-cont-reproc/v2p0/2019/07/"),
        (dt.date(2020, 12, 31),
         "/reprocessed/ice/conc-cont-reproc/v2p0/2020/12/"),
        # This is the specific fix under test: dates >= 2021-01-01 must now
        # resolve to OSI-438 (previously OSI-430-a, decommissioned and dead
        # since 2025-10-17).
        (dt.date(2021, 1, 1),
         "/reprocessed/ice/conc-cont-reproc-amsr/v3p0/2021/01/"),
        # A date after OSI-430-a's real decommission date, which previously
        # would have resolved to a now-dead path.
        (dt.date(2025, 11, 1),
         "/reprocessed/ice/conc-cont-reproc-amsr/v3p0/2025/11/"),
        (dt.date(2026, 8, 19),
         "/reprocessed/ice/conc-cont-reproc-amsr/v3p0/2026/08/"),
        # Single-digit month must be zero-padded, or this would silently
        # produce a non-existent FTP path ("2026/1/" instead of "2026/01/").
        (dt.date(2026, 1, 5),
         "/reprocessed/ice/conc-cont-reproc-amsr/v3p0/2026/01/"),
    ],
)
def test_resolve_ftp_path(date, expected):
    """Check the correct product/path is selected and correctly formatted
    for dates spanning all three products and their boundaries."""
    assert _resolve(date) == expected
