import collections
import itertools
import logging
import os
import time

from datetime import datetime
from dateutil.relativedelta import relativedelta

import numpy as np
import tensorflow as tf

from . import config


class IcenetDataPreprocessor(object):
    def __init__(self,
                 source,
                 name,
                 config,
                 n_forecast_months,
                 destination=".",
                 seed=42,
                 data_shape=(432, 432),
                 dtype=np.float32,
                 loss_weight_months=True,
                 loss_weight_classes=True,):
        self._name = name
        self._conf = config
        self._seed = seed
        self._data_shape = data_shape
        self._dtype = dtype
        self._n_forecast_months = n_forecast_months

        self._source = os.path.join(source, self._name)
        self._dest = os.path.join(destination, self._name)
        self._rng = np.random.default_rng(seed)


class IcenetCMIPPreprocessor(IcenetDataPreprocessor):
    def __init__(self,
                 train_dates,
                 val_dates,
                 *args, **kwargs):
        super(IcenetCMIPPreprocessor, self).__init__(*args, **kwargs)


class IcenetERAPreprocessor(IcenetDataPreprocessor):
    def __init__(self,
                 train_dates,
                 val_dates,
                 test_dates,
                 *args,
                 use_polarhole3=False,
                 **kwargs):
        super(IcenetERAPreprocessor, self).__init__(*args, **kwargs)

        self._inputs = collections.defaultdict(dict)
        self._metadata = collections.defaultdict(dict)
        self._channels = 0

        self._polarhole_masks = list()

        self._set_inputs()
        self._set_metadata()
        self._load_polarholes(use_polarhole3)

    def _set_inputs(self):
        logging.debug('Setting up the variable paths and channels for {}... '.format(self._name))

        for varname, vardict in self._conf.items():
            for data_format in vardict.keys():
                path = os.path.join(self._source, 'obs', varname, data_format, '{:04d}_{:02d}.npy')
                channels = self._n_forecast_months if data_format == 'linear_trend' else len(vardict['lookbacks'])

                self._inputs[varname][data_format] = dict(path=path, channels=channels)

    def _set_metadata(self):
        for varname, vardict in self._conf.items():
            if varname == 'land':
                self._metadata[varname] = dict(
                    path=os.path.join(self._source, 'meta', 'land.npy'),
                    channels=1)
                self._channels += 1
            elif varname == 'circmonth':
                self._metadata[varname] = dict(
                    path=os.path.join(self._source, 'meta', '{}_month_{:02d}.npy'),
                    channels=2)
                self._channels += 2

        logging.debug("Done")

    def _load_polarholes(self, use_polarhole3):
        """
        This method loads the polar holes.
        """

        logging.debug("Loading and augmenting the polar holes... ")

        # Zero is none
        self._polarhole_masks.append(np.full((432, 432), False))

        num = 2 if not use_polarhole3 else 3
        for i in range(1, num+1):
            self._polarhole_masks.append(np.load(os.path.join(config.mask_data_folder, "polarhole{}_mask".format(i))))

        logging.debug("Loaded {} polarhole masks".format(len(self._polarhole_masks)))
