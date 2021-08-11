from abc import abstractmethod

import logging
import os


from icenet2.utils import Hemisphere, HemisphereMixin


class DataProducer(HemisphereMixin):

    @abstractmethod
    def __init__(self, *args,
                 identifier=None,
                 dry=False,
                 overwrite=False,
                 north=True,
                 south=False,
                 path=os.path.join(".", "data"),
                 **kwargs):
        self.dry = dry
        self.overwrite = overwrite

        self._identifier = identifier
        self._path = os.path.join(path, identifier)
        self._hemisphere = (Hemisphere.NORTH if north else Hemisphere.NONE) | \
                           (Hemisphere.SOUTH if south else Hemisphere.NONE)

        if os.path.exists(self._path):
            logging.warning("{} already exists".format(self._path))
        else:
            os.mkdir(self._path)

        assert self._identifier, "No identifier supplied"
        assert self._hemisphere != Hemisphere.NONE, "No hemispheres selected"
        # NOTE: specific limitation for the DataProducers, they'll only do one
        # hemisphere per instance
        assert self._hemisphere != Hemisphere.BOTH, "Both hemispheres selected"

    @property
    def base_path(self):
        return self._path

    @base_path.setter
    def base_path(self, path):
        self._path = path

    @property
    def identifier(self):
        return self._identifier

    def get_data_var_folder(self, var, hemisphere=None):
        if not hemisphere:
            # We can make the assumption because this implementation is limited
            # to a single hemisphere
            hemisphere = self.hemisphere_str[0]

        hemi_path = os.path.join(self.base_path, hemisphere)
        if not os.path.exists(hemi_path):
            logging.info("Creating hemisphere path: {}".format(hemi_path))
            os.mkdir(hemi_path)

        var_path = os.path.join(self.base_path, hemisphere, var)
        if not os.path.exists(var_path):
            logging.info("Creating var path: {}".format(var_path))
            os.mkdir(var_path)
        return var_path


class Downloader(DataProducer):
    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)

    @abstractmethod
    def download(self):
        raise NotImplementedError("{}.download is abstract".
                                  format(__class__.__name__))


class Generator(DataProducer):
    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)

    @abstractmethod
    def generate(self):
        raise NotImplementedError("{}.generate is abstract".
                                  format(__class__.__name__))
