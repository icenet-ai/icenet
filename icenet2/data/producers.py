from abc import abstractmethod

import logging
import os


class DataProducer:

    @abstractmethod
    def __init__(self, *args,
                 identifier=None,
                 dry=False,
                 overwrite=False,
                 path=os.path.join(".", "data"),
                 **kwargs):
        self.dry = dry
        self.overwrite = overwrite

        self._identifier = identifier
        self._path = path

        if os.path.exists(self._path):
            logging.warning("{} already exists".format(self._path))
        else:
            os.mkdir(self._path)

        assert self._identifier, "No identifier supplied"

    @abstractmethod
    def load_config(self, filename):
        raise NotImplementedError("{} is abstract".format(__name__))

    @property
    def base_path(self):
        return self._path

    @property
    def identifier(self):
        return self._identifier


class Downloader(DataProducer):
    @abstractmethod
    def download(self):
        raise NotImplementedError("{} is abstract".format(__name__))


class Generator(DataProducer):
    @abstractmethod
    def generate(self):
        raise NotImplementedError("{} is abstract".format(__name__))
