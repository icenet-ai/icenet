import argparse
import logging
import os

import dask
import numpy as np
import orjson
import pandas as pd

from icenet.data.datasets.splitting import SplittingMixin
from icenet.data.loader import IceNetDataLoaderFactory
from download_toolbox.base import DataCollection
from icenet.utils import setup_logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

pytorch_available = False
try:
    from torch.utils.data import Dataset
except ModuleNotFoundError:
    print("PyTorch not found - not required if not using PyTorch")
except ImportError:
    print("PyTorch import failed - not required if not using PyTorch")

"""


https://stackoverflow.com/questions/55852831/
tf-data-vs-keras-utils-sequence-performance

"""


class IceNetDataSet(SplittingMixin, DataCollection):
    """Initialises and configures a dataset.

    It loads a JSON configuration file, updates the `_config` attribute with the
    result, creates a data loader, and methods to access the dataset.

    Attributes:
        _config: A dict used to store configuration loaded from JSON file.
        _configuration_path: The path to the JSON configuration file.
        _batch_size: The batch size for the data loader.
        _counts: A dict with number of elements in train, val, test.
        _dtype: The type of the dataset.
        _loader_config: The path to the data loader configuration file.
        _generate_workers: An integer representing number of workers for parallel processing with Dask.
        _lead_time: An integer representing number of days to predict for.
        _num_channels: An integer representing number of channels (input variables) in the dataset.
        _shape: The shape of the dataset.
        _shuffling: A flag indicating whether to shuffle the data or not.
    """

    def __init__(self,
                 configuration_path: str,
                 *args,
                 batch_size: int = 4,
                 path: str = os.path.join(".", "network_datasets"),
                 shuffling: bool = False,
                 **kwargs) -> None:
        """Initialises an instance of the IceNetDataSet class.

        Args:
            configuration_path: The path to the JSON configuration file.
            *args: Additional positional arguments.
            batch_size (optional): How many samples to load per batch. Defaults to 4.
            path (optional): The path to the directory where the processed tfrecord
                protocol buffer files will be stored. Defaults to './network_datasets'.
            shuffling (optional): Flag indicating whether to shuffle the data.
                Defaults to False.
            *args: Additional keyword arguments.
        """

        self._config = dict()
        self._configuration_path = configuration_path
        self._load_configuration(configuration_path)

        super().__init__(*args,
                         identifier=self._config["identifier"],
                         #north=bool(self._config["north"]),
                         base_path=path, #path=
                         #south=bool(self._config["south"]),
                         **kwargs)

        # TODO: ugh!
        self._config = dict()
        self._load_configuration(configuration_path)
        self._batch_size = batch_size
        self._counts = self._config["counts"]
        self._dtype = getattr(np, self._config["dtype"])
        self._loader_config = self._config["loader_config"]
        self._generate_workers = self._config["generate_workers"]
        self._lead_time = self._config["lead_time"]
        self._num_channels = self._config["num_channels"]
        self._shape = tuple(self._config["shape"])
        self._shuffling = shuffling

        if "loader_path" in self._config:
            logging.warning("Configuration uses old \"loader_path\" attribute, "
                            "this should change to \"dataset_path\"")
            path_attr = "loader_path"
        else:
            path_attr = "dataset_path"

        # Check JSON config has attribute for path to tfrecord datasets, and
        #   that the path exists.
        if self._config[path_attr] and \
                os.path.exists(self._config[path_attr]):
            self.add_records(self.path)
        else:
            logging.warning("Running in configuration only mode, tfrecords "
                            "were not generated for this dataset")

    def _load_configuration(self, path: str) -> None:
        """Load the JSON configuration file and update the `_config` attribute of `IceNetDataSet` class.

        Args:
            path: The path to the JSON configuration file.

        Raises:
            OSError: If the specified configuration file is not found.
        """
        if os.path.exists(path):
            logging.info("Loading configuration {}".format(path))

            with open(path, "r") as fh:
                obj = orjson.loads(fh.read())

                self._config.update(obj)
        else:
            raise OSError("{} not found".format(path))

    def get_data_loader(self,
                        lead_time: object = None,
                        generate_workers: object = None) -> object:
        """Create an instance of the IceNetDataLoader class.

        Args:
            lead_time (optional): The number of forecast steps to be used by the data loader.
                If not provided, defaults to the value specified in the configuration file.
            generate_workers (optional): An integer representing number of workers to use for
                parallel processing with Dask. If not provided, defaults to the value specified in
                the configuration file.

        Returns:
            An instance of the DaskMultiWorkerLoader class configured with the specified parameters.
        """
        if lead_time is None:
            lead_time = self._config["lead_time"]
        if generate_workers is None:
            generate_workers = self._config["generate_workers"]
        loader = IceNetDataLoaderFactory().create_data_loader(
            "dask",  # This will load the `DaskMultiWorkerLoader` class.
            self.loader_config,
            self.identifier,
            lag_time=self._config["lag_time"],
            lead_time=lead_time,
            generate_workers=generate_workers,
            dataset_config_path=os.path.dirname(self._configuration_path),
            loss_weight_days=self._config["loss_weight_days"],
            output_batch_size=self._config["output_batch_size"],
            var_lag_override=self._config["var_lag_override"],
        )
        return loader

    @property
    def loader_config(self) -> str:
        """The path to the JSON loader configuration file stored in the dataset config file."""
        # E.g. `/path/to/loader.{identifier}.json`
        return self._loader_config

    @property
    def channels(self) -> list:
        """The list of channels (variable names) specified in the dataset config file."""
        return self._config["channels"]

    @property
    def counts(self) -> dict:
        """A dict with number of elements in train, val, test in the config file."""
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

        identifier = ".".join(
            [loader.identifier for loader in self._config["loaders"]])

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
        self._lead_time = self._config["lead_time"]
        self._shape = self._config["shape"]
        self._shuffling = shuffling

        self._init_records()

    def _init_records(self):
        """

        """
        for idx, loader_path in enumerate(self._config["loader_paths"]):
            base_path = os.path.join(self._base_path,
                                     self._config["loaders"][idx].identifier)
            self.add_records(base_path)

    def _load_configurations(self, paths: object):
        """

        :param paths:
        """
        self._config = dict(loader_paths=[],
                            loaders=[],
                            north=False,
                            south=False)

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

        if "loader_path" in other:
            logging.warning("Configuration uses old \"loader_path\" attribute, "
                            "this should change to \"dataset_path\"")
            self._config["loader_paths"].append(other["loader_path"])
        else:
            self._config["loader_paths"].append(other["dataset_path"])

        if "counts" not in self._config:
            self._config["counts"] = other["counts"].copy()
        else:
            for dataset, count in other["counts"].items():
                logging.info("Merging {} samples from {}".format(
                    count, dataset))
                self._config["counts"][dataset] += count

        general_attrs = [
            "channels", "dtype", "lead_time", "num_channels",
            "output_batch_size", "shape"
        ]

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

    def check_dataset(self, split: str = "train"):
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


if pytorch_available:
    class IceNetDataSetPyTorch(IceNetDataSet, Dataset):
        """Initialises and configures a PyTorch dataset.
        """
        def __init__(
            self,
            configuration_path: str,
            mode: str,
            batch_size: int = 1,
            shuffling: bool = False,
        ):
            """Initialises an instance of the IceNetDataSetPyTorch class.

            Args:
                configuration_path: The path to the JSON configuration file.
                mode: The dataset type, i.e. `train`, `val` or `test`.
                batch_size (optional): How many samples to load per batch. Defaults to 1.
                shuffling (optional): Flag indicating whether to shuffle the data.
                    Defaults to False.
            """
            super().__init__(configuration_path=configuration_path,
                             batch_size=batch_size,
                             shuffling=shuffling)
            self._dl = self.get_data_loader()

            # check mode option
            if mode not in ["train", "val", "test"]:
                raise ValueError("mode must be either 'train', 'val', 'test'")
            self._mode = mode

            self._dates = self._dl._config["sources"]["osisaf"]["dates"][self._mode]

        def __len__(self):
            return self._counts[self._mode]

        def __getitem__(self, idx):
            """Return a sample from the dataloader for given index.
            """
            with dask.config.set(scheduler="synchronous"):
                sample = self._dl.generate_sample(
                    date=pd.Timestamp(self._dates[idx].replace('_', '-')),
                    parallel=False,
                )
            return sample

        @property
        def dates(self):
            return self._dates

@setup_logging
def get_args() -> object:
    """Parse command line arguments using the argparse module.

    Returns:
        An object containing the parsed command line arguments.

    Example:
        Assuming CLI arguments provided.

        args = get_args()
        print(args.dataset)
        print(args.split)
        print(args.verbose)
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("-s",
                    "--split",
                    choices=["train", "val", "test"],
                    default="train")
    ap.add_argument("-v", "--verbose", action="store_true", default=False)
    args = ap.parse_args()
    return args


def check_dataset() -> None:
    """Check the dataset for a specific split."""
    args = get_args()
    ds = IceNetDataSet(args.dataset)
    ds.check_dataset(args.split)
