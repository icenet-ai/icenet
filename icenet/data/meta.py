import numpy as np
import pandas as pd
import xarray as xr

from download_toolbox.interface import DatasetConfig
from preprocess_toolbox.base import Processor, ProcessingError

from icenet.data.masks.osisaf import Masks


class PeriodProcessor(Processor):
    def __init__(self,
                 dataset_config: DatasetConfig,
                 *args,
                 method: callable,
                 **kwargs):
        super().__init__(dataset_config, *args, **kwargs)
        self._method = method

    def get_config(self,
                   config_funcs: dict = None,
                   strip_keys: list = None):
        return {
            "implementation": "{}:{}".format(self.__module__, self.__class__.__name__),
            "absolute_vars": self.abs_vars,
            "path": self.path,
            "processed_files": self._processed_files,
        }

    def process(self):
        if len(self.abs_vars) != 1:
            raise ProcessingError("{} should be provided ONE absolute var name only, not {}".
                                  format(self.__class__.__name__, self.abs_vars))

        var_name = self.abs_vars[0]
        # TODO: factor in that date.dayofyear in north would be + 365.25 / 2 for south
        values = [self._method(2 * np.pi * date.dayofyear / 366)
                  for date in pd.date_range(start='2012-1-1', end='2012-12-31')]

        da = xr.DataArray(
            data=values,
            dims=["time"],
            coords=dict(
                time=pd.date_range(start='2012-1-1', end='2012-12-31')),
            attrs=dict(
                description="IceNet {} mask metadata".format(var_name)))
        self.save_processed_file(var_name, "{}.nc".format(var_name), da)


class SinProcessor(PeriodProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, method=np.sin, **kwargs)


class CosProcessor(PeriodProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, method=np.cos, **kwargs)
