import glob
import json
import logging
import os

import numpy as np
import tensorflow as tf

from icenet2.data.loader import IceNetDataLoader
from icenet2.data.producers import DataCollection

"""


https://stackoverflow.com/questions/55852831/
tf-data-vs-keras-utils-sequence-performance

"""


def get_decoder(shape: object,
                channels: object,
                forecasts: object,
                num_vars: int = 1,
                dtype: str = "float32") -> object:
    """

    :param shape:
    :param channels:
    :param forecasts:
    :param num_vars:
    :param dtype:
    :return:
    """
    xf = tf.io.FixedLenFeature(
        [*shape, channels], getattr(tf, dtype))
    yf = tf.io.FixedLenFeature(
        [*shape, forecasts, num_vars], getattr(tf, dtype))
    sf = tf.io.FixedLenFeature(
        [*shape, forecasts, num_vars], getattr(tf, dtype))

    @tf.function
    def decode_item(proto):
        features = {
            "x": xf,
            "y": yf,
            "sample_weights": sf,
        }

        item = tf.io.parse_example(proto, features)
        return item['x'], item['y'], item['sample_weights']

    return decode_item


# TODO: define a decent interface and sort the inheritance architecture out, as
#  this will facilitate the new datasets in #35
class SplittingMixin:
    """

    """

    _batch_size: int
    _dtype: object
    _num_channels: int
    _n_forecast_days: int
    _shape: int

    train_fns = []
    test_fns = []
    val_fns = []

    def add_records(self, base_path: str, hemi: str):
        """

        :param base_path:
        :param hemi:
        """
        train_path = os.path.join(base_path, hemi, "train")
        val_path = os.path.join(base_path, hemi, "val")
        test_path = os.path.join(base_path, hemi, "test")

        logging.info("Training dataset path: {}".format(train_path))
        self.train_fns += glob.glob("{}/*.tfrecord".format(train_path))
        logging.info("Validation dataset path: {}".format(val_path))
        self.val_fns += glob.glob("{}/*.tfrecord".format(val_path))
        logging.info("Test dataset path: {}".format(test_path))
        self.test_fns += glob.glob("{}/*.tfrecord".format(test_path))

    def get_split_datasets(self, ratio: object = None):
        """

        :param ratio:
        :return:
        """
        if not (len(self.train_fns) + len(self.val_fns) + len(self.test_fns)):
            raise RuntimeError("No files have been found, abandoning...")

        logging.info("Datasets: {} train, {} val and {} test filenames".format(
            len(self.train_fns), len(self.val_fns), len(self.test_fns)))

        if ratio:
            if ratio > 1.0:
                raise RuntimeError("Ratio cannot be more than 1")

            logging.info("Reducing datasets to {} of total files".format(ratio))
            train_idx, val_idx, test_idx = \
                int(len(self.train_fns) * ratio), \
                int(len(self.val_fns) * ratio), \
                int(len(self.test_fns) * ratio)

            if train_idx > 0:
                self.train_fns = self.train_fns[:train_idx]
            if val_idx > 0:
                self.val_fns = self.val_fns[:val_idx]
            if test_idx > 0:
                self.test_fns = self.test_fns[:test_idx]

            logging.info("Reduced: {} train, {} val and {} test filenames".format(
                len(self.train_fns), len(self.val_fns), len(self.test_fns)))

        train_ds, val_ds, test_ds = \
            tf.data.TFRecordDataset(self.train_fns,
                                    num_parallel_reads=self.batch_size), \
            tf.data.TFRecordDataset(self.val_fns,
                                    num_parallel_reads=self.batch_size), \
            tf.data.TFRecordDataset(self.test_fns,
                                    num_parallel_reads=self.batch_size),

        # TODO: Comparison/profiling runs
        # TODO: parallel for batch size while that's small
        # TODO: obj.decode_item might not work here - figure out runtime
        #  implementation based on wrapped function call that can be serialised
        decoder = get_decoder(self.shape,
                              self.num_channels,
                              self.n_forecast_days,
                              dtype=self.dtype.__name__)

        train_ds = train_ds.\
            shuffle(int(min(len(self.train_fns) / 4, 100)),
                    reshuffle_each_iteration=True).\
            map(decoder, num_parallel_calls=self.batch_size).\
            batch(self.batch_size)

        val_ds = val_ds.\
            map(decoder, num_parallel_calls=self.batch_size).\
            batch(self.batch_size)

        test_ds = test_ds.\
            map(decoder, num_parallel_calls=self.batch_size).\
            batch(self.batch_size)

        return train_ds.prefetch(tf.data.AUTOTUNE), \
            val_ds.prefetch(tf.data.AUTOTUNE), \
            test_ds.prefetch(tf.data.AUTOTUNE)

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def dtype(self):
        return self._dtype

    @property
    def n_forecast_days(self):
        return self._n_forecast_days

    @property
    def num_channels(self):
        return self._num_channels

    @property
    def shape(self):
        return self._shape


class IceNetDataSet(SplittingMixin, DataCollection):
    """

    :param configuration_path:
    :param batch_size:
    :param path:
    """

    def __init__(self,
                 configuration_path: str,
                 *args,
                 batch_size: int = 4,
                 path: str = os.path.join(".", "network_datasets"),
                 **kwargs):
        self._config = dict()
        self._configuration_path = configuration_path
        self._load_configuration(configuration_path)

        super().__init__(*args,
                         identifier=self._config["identifier"],
                         north=bool(self._config["north"]),
                         path=path,
                         south=bool(self._config["south"]),
                         **kwargs)

        self._batch_size = batch_size
        self._counts = self._config["counts"]
        self._dtype = getattr(np, self._config["dtype"])
        self._loader_config = self._config["loader_config"]
        self._n_forecast_days = self._config["n_forecast_days"]
        self._num_channels = self._config["num_channels"]
        self._shape = tuple(self._config["shape"])

        if self._config["loader_path"] and \
                os.path.exists(self._config["loader_path"]):
            hemi = self.hemisphere_str[0]
            self.add_records(self.base_path, hemi)
        else:
            logging.warning("Running in configuration only mode, tfrecords "
                            "were not generated for this dataset")

    def _load_configuration(self, path: str):
        """

        :param path:
        """
        if os.path.exists(path):
            logging.info("Loading configuration {}".format(path))

            with open(path, "r") as fh:
                obj = json.load(fh)

                self._config.update(obj)
        else:
            raise OSError("{} not found".format(path))

    def get_data_loader(self):
        """

        :return:
        """
        # TODO: this invocation in config only mode will lead to the
        #  generation of a network_dataset directory unnecessarily. This
        #  loader_path logic needs sorting out a bit better, as it's gotten
        #  messy
        loader = IceNetDataLoader(self.loader_config,
                                  self.identifier,
                                  self._config["var_lag"],
                                  dataset_config_path=os.path.dirname(
                                      self._configuration_path),
                                  loss_weight_days=self._config[
                                      "loss_weight_days"],
                                  north=self.north,
                                  output_batch_size=self._config[
                                      "output_batch_size"],
                                  south=self.south,
                                  var_lag_override=self._config[
                                      "var_lag_override"])
        return loader

    @property
    def loader_config(self):
        return self._loader_config

    @property
    def channels(self):
        return self._config["channels"]

    @property
    def counts(self):
        return self._config["counts"]


class MergedIceNetDataSet(SplittingMixin, DataCollection):
    """

    :param identifier:
    :param configuration_path:
    :param batch_size:
    :param path:
    """

    def __init__(self,
                 identifier: str,
                 configuration_paths: object,
                 *args,
                 batch_size: int = 4,
                 path: str = os.path.join(".", "network_datasets"),
                 **kwargs):
        self._config = dict()
        self._configuration_paths = [configuration_paths] \
            if type(configuration_paths) != list else configuration_paths
        self._load_configurations(configuration_paths)

        super().__init__(*args,
                         identifier=identifier,
                         north=bool(self._config["north"]),
                         path=path,
                         south=bool(self._config["south"]),
                         **kwargs)

        self._batch_size = batch_size
        self._dtype = getattr(np, self._config["dtype"])
        self._num_channels = self._config["num_channels"]
        self._n_forecast_days = self._config["n_forecast_days"]
        self._shape = self._config["shape"]

        self._init_records()

    def _init_records(self):
        """

        """
        for idx, loader_path in enumerate(self._config["loader_paths"]):
            hemi = self._config["loaders"][idx].hemisphere_str[0]
            self.add_records(self.base_path, hemi)

    def _load_configurations(self, paths: object):
        """

        :param paths:
        """
        self._config = dict(
            loader_paths=[],
            loaders=[],
            north=False,
            south=False
        )
        
        for path in paths:
            if os.path.exists(path):
                logging.info("Loading configuration {}".format(path))

                with open(path, "r") as fh:
                    obj = json.load(fh)
                    self._merge_configurations(path, obj)
            else:
                raise OSError("{} not found".format(path))

    def _merge_configurations(self, path: str, other: object):
        """

        :param path:
        :param other:
        """
        loader = IceNetDataLoader(other["loader_config"],
                                  other["identifier"],
                                  other["var_lag"],
                                  dataset_config_path=os.path.dirname(path),
                                  loss_weight_days=other["loss_weight_days"],
                                  north=other["north"],
                                  output_batch_size=other["output_batch_size"],
                                  south=other["south"],
                                  var_lag_override=other["var_lag_override"])

        self._config["loaders"].append(loader)
        self._config["loader_paths"].append(other["loader_path"])

        if "counts" not in self._config:
            self._config["counts"] = other["counts"].copy()
        else:
            for dataset, count in other["counts"].items():
                logging.info("Merging {} samples from {}".format(count, dataset))
                self._config["counts"][dataset] += count

        general_attrs = ["channels", "dtype", "n_forecast_days",
                         "num_channels", "output_batch_size", "shape"]

        for attr in general_attrs:
            if attr not in self._config:
                self._config[attr] = other[attr]
            else:
                assert self._config[attr] == other[attr], \
                    "{} is not the same across configurations".format(attr)

        self._config["north"] = True if loader.north else self._config["north"]
        self._config["south"] = True if loader.south else self._config["south"]

    def get_data_loader(self):
        """

        :return:
        """
        assert len(self._configuration_paths) == 1, "Configuration mode is " \
                                                    "only for single loader" \
                                                    "datasets: {}".format(
            self._configuration_paths
        )
        return self._config["loader"][0]

    @property
    def channels(self):
        return self._config['channels']

    @property
    def counts(self):
        return self._config["counts"]
