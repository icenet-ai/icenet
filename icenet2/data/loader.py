import collections
import json
import logging
import os
import sys

# https://stackoverflow.com/questions/55852831/
# tf-data-vs-keras-utils-sequence-performance

from icenet2.data.producers import Processor


class IceNetDataLoader:
    """
    Custom data loader class for generating batches of input-output tensors for
    training IceNet. Inherits from  keras.utils.Sequence, which ensures each the
    network trains once on each  sample per epoch. Must implement a __len__
    method that returns the  number of batches and a __getitem__ method that
    returns a batch of data. The  on_epoch_end method is called after each
    epoch.
    See: https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence
    """

    def __init__(self,
                 configuration_path,
                 *args,
                 seed=None,
                 **kwargs):
        self._config = {}
        self._seed = seed

        self._load_configuration(configuration_path)

        # TODO: Create processor objects

    def generate(self):
        raise NotImplementedError("Generate not implemented")

    def _load_configuration(self, path):
        if os.path.exists(path):
            logging.info("Loading configuration {}".format(path))

            with open(path, "r") as fh:
                obj = json.load(fh)

                self._config.update(obj)
            logging.debug("LOADER CONFIG:\n{}".format(self._config))
        else:
            raise OSError("{} not found".format(path))


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    dl = IceNetDataLoader("loader.test1.json")
    dl.generate()

