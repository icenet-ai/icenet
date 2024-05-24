import logging
import os
import random

import numpy as np

from abc import abstractmethod


class BaseNetwork:
    def __init__(self,
                 run_name: object,
                 dataset: object,
                 callbacks_additional: list = None,
                 callbacks_default: list = None,
                 network_folder: object = None,
                 seed: int = 42):

        if not network_folder:
            self._network_folder = os.path.join(".", "results", "networks", run_name)

        if not os.path.exists(self._network_folder):
            logging.info("Creating network folder: {}".format(network_folder))
            os.makedirs(self._network_folder, exist_ok=True)

        self._model_path = os.path.join(
            self._network_folder, "{}.model_{}.{}".format(run_name,
                                                          dataset.identifier,
                                                          seed))

        self._callbacks = self.get_default_callbacks() if callbacks_default is None else callbacks_default
        self._callbacks += callbacks_additional if callbacks_additional is not None else []
        self._dataset = dataset
        self._run_name = run_name
        self._seed = seed

        self._attempt_seed_setup()

    def _attempt_seed_setup(self):
        logging.warning(
            "Setting seed for best attempt at determinism, value {}".format(self._seed))
        # determinism is not guaranteed across different versions of TensorFlow.
        # determinism is not guaranteed across different hardware.
        os.environ['PYTHONHASHSEED'] = str(self._seed)
        # numpy.random.default_rng ignores this, WARNING!
        np.random.seed(self._seed)
        random.seed(self._seed)

    def add_callback(self, callback):
        logging.debug("Adding callback {}".format(callback))
        self._callbacks.append(callback)

    def get_default_callbacks(self):
        return list()

    @abstractmethod
    def train(self,
              epochs: int,
              model_creator: callable,
              train_dataset: object,
              model_creator_kwargs: dict = None,
              save: bool = True):
        raise NotImplementedError("Implementation not found")

    @abstractmethod
    def predict(self):
        raise NotImplementedError("Implementation not found")

    @property
    def callbacks(self):
        return self._callbacks

    @property
    def dataset(self):
        return self._dataset

    @property
    def model_path(self):
        return self._model_path

    @property
    def network_folder(self):
        return self._network_folder

    @property
    def run_name(self):
        return self._run_name

    @property
    def seed(self):
        return self._seed

