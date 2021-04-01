import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), 'icenet2'))  # if using jupyter kernel
import misc
import utils
import numpy as np
from datetime import datetime

dataset_name = 'dataset1'

preproc_vars = {
    'siconca': {'anom': False, 'abs': True},
    'tas': {'anom': True, 'abs': False},
    'ta500': {'anom': True, 'abs': False},
    'tos': {'anom': True, 'abs': False},
    'psl': {'anom': True, 'abs': False},
    'zg500': {'anom': True, 'abs': False},
    'zg250': {'anom': True, 'abs': False},
    'land': {'metadata': True, 'include': True},
    'circday': {'metadata': True, 'include': True}
}

preproc_vars = {
    'siconca': {'anom': False, 'abs': False},
    'tas': {'anom': False, 'abs': False},
    'ta500': {'anom': False, 'abs': False},
    'tos': {'anom': True, 'abs': False},
    'psl': {'anom': False, 'abs': False},
    'zg500': {'anom': False, 'abs': False},
    'zg250': {'anom': False, 'abs': False},
    'land': {'metadata': True, 'include': False},
    'circday': {'metadata': True, 'include': False}
}

preproc_hemispheres = ['sh']

start_date = datetime(1979, 1, 1)
end_date = datetime(2012, 1, 1)
obs_train_dates = misc.filled_daily_dates(start_date, end_date)

minmax = False

verbose_level = 2

raw_data_shape = (432, 432)

dtype = np.float32

utils.IceNet2DataPreProcessor(
    dataset_name=dataset_name,
    preproc_vars=preproc_vars,
    preproc_hemispheres=preproc_hemispheres,
    obs_train_dates=obs_train_dates,
    minmax=minmax,
    verbose_level=verbose_level,
    raw_data_shape=raw_data_shape,
    dtype=dtype
)
