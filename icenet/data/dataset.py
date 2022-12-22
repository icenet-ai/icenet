import argparse
import json
import logging
import os

import numpy as np

from icenet.data.datasets.utils import SplittingMixin
from icenet.data.loader import IceNetDataLoaderFactory
from icenet.data.producers import DataCollection
from icenet.utils import setup_logging

"""


https://stackoverflow.com/questions/55852831/
tf-data-vs-keras-utils-sequence-performance

"""


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
                 shuffling: bool = False,
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
        self._shuffling = shuffling

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
        loader = IceNetDataLoaderFactory().create_data_loader(
            "dask",
            self.loader_config,
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
    :param configuration_paths: List of configurations to load
    :param batch_size:
    :param path:
    """

    def __init__(self,
                 configuration_paths: object,
                 *args,
                 batch_size: int = 4,
                 path: str = os.path.join(".", "network_datasets"),
                 shuffling: bool = False,
                 **kwargs):
        self._config = dict()
        self._configuration_paths = [configuration_paths] \
            if type(configuration_paths) != list else configuration_paths
        self._load_configurations(configuration_paths)

        identifier = ".".join([loader.identifier
                               for loader in self._config["loaders"]])

        super().__init__(*args,
                         identifier=identifier,
                         north=bool(self._config["north"]),
                         path=path,
                         south=bool(self._config["south"]),
                         **kwargs)

        self._base_path = path
        self._batch_size = batch_size
        self._dtype = getattr(np, self._config["dtype"])
        self._num_channels = self._config["num_channels"]
        self._n_forecast_days = self._config["n_forecast_days"]
        self._shape = self._config["shape"]
        self._shuffling = shuffling

        self._init_records()

    def _init_records(self):
        """

        """
        for idx, loader_path in enumerate(self._config["loader_paths"]):
            hemi = self._config["loaders"][idx].hemisphere_str[0]
            base_path = os.path.join(self._base_path,
                                     self._config["loaders"][idx].identifier)
            self.add_records(base_path, hemi)

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
        loader = IceNetDataLoaderFactory().create_data_loader(
            "dask",
            other["loader_config"],
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

    def check_dataset(self,
                      split: str = "train"):
        """

        :param split:
        """
        raise NotImplementedError("Checking not implemented for merged sets, "
                                  "consider doing them individually")

    @property
    def channels(self):
        return self._config['channels']

    @property
    def counts(self):
        return self._config["counts"]


@setup_logging
def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("-s", "--split",
                    choices=["train", "val", "test"], default="train")
    ap.add_argument("-v", "--verbose", action="store_true", default=False)
    args = ap.parse_args()
    return args


def check_dataset():
    args = get_args()
    ds = IceNetDataSet(args.dataset)
    ds.check_dataset(args.split)
