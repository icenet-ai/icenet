import os

import numpy as np
import pandas as pd

from icenet2.data.cli import process_args
from icenet2.data.process import IceNetPreProcessor
from icenet2.data.sic.mask import Masks

"""

"""


class IceNetMetaPreProcessor(IceNetPreProcessor):
    """

    :param name:
    :param include_circday:
    :param include_land:
    """

    def __init__(self,
                 name: str,
                 include_circday: bool = True,
                 include_land: bool = True,
                 **kwargs):
        super().__init__(abs_vars=[],
                         anom_vars=[],
                         identifier="meta",
                         linear_trends=tuple(),
                         name=name,
                         test_dates=[],
                         train_dates=[],
                         val_dates=[],
                         **kwargs)

        self._include_circday = include_circday
        self._include_land = include_land

    def init_source_data(self,
                         lag_days: object = None,
                         lead_days: object = None):
        """

        :param lag_days:
        :param lead_days:
        """
        raise NotImplementedError("No need to execute implementation for meta")

    def process(self):
        """

        """
        if self._include_circday:
            self._save_circday()

        if self._include_land:
            self._save_land()

        # FIXME: this is a bit messy, clarify meta_vars and interface between
        #  processing and dataset generation
        if self._update_loader:
            self.update_loader_config()

    def _save_land(self):
        """

        """
        land_mask = Masks(north=self.north, south=self.south).get_land_mask()
        land_map = np.ones(self._data_shape, dtype=self._dtype)
        land_map[~land_mask] = -1.

        if "land" not in self._meta_vars:
            self._meta_vars.append("land")

        land_path = self.save_processed_file("land", "land.npy", land_map)

        for date in pd.date_range(start='2012-1-1', end='2012-12-31'):
            link_path = os.path.join(os.path.dirname(land_path),
                                     date.strftime('%j.npy'))

            if not os.path.islink(link_path):
                os.symlink(os.path.basename(land_path), link_path)
            self.processed_files["land"].append(link_path)

    def _save_circday(self):
        """

        """
        for date in pd.date_range(start='2012-1-1', end='2012-12-31'):
            if self.north:
                circday = date.dayofyear
            else:
                circday = date.dayofyear + 365.25 / 2

            cos_day = np.cos(2 * np.pi * circday / 366, dtype=self._dtype)
            sin_day = np.sin(2 * np.pi * circday / 366, dtype=self._dtype)

            self.save_processed_file("cos",
                                     date.strftime('%j.npy'),
                                     cos_day)
            self.save_processed_file("sin",
                                     date.strftime('%j.npy'),
                                     sin_day)

        for var_name in ["sin", "cos"]:
            if var_name not in self._meta_vars:
                self._meta_vars.append(var_name)


def main():
    args = process_args(dates=False, lag_lead=False, ref_option=False)

    IceNetMetaPreProcessor(
        args.name,
        north=args.hemisphere == "north",
        south=args.hemisphere == "south"
    ).process()
