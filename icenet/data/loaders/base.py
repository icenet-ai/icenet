import datetime as dt
import logging
import os
from abc import abstractmethod

from pprint import pformat

import orjson
import numpy as np

from download_toolbox.interface import DataCollection, get_dataset_config_implementation
from preprocess_toolbox.interface import get_processor_from_source

"""

"""

DATE_FORMAT = "%Y-%m-%d"

# Defaults for configurations written before the target became configurable.
DEFAULT_TARGET_VARIABLE = "siconca_abs"
DEFAULT_TARGET_MASK = "land"
DEFAULT_TARGET_WEIGHT_MASK = "active_grid_cell"


def _resolve_target_config(configuration: dict,
                           override: object = None) -> dict:
    """Determine the forecast target for a loader configuration.

    Resolved independently of the predictor channels, so that a target need not
    also be an input and can be selected from its own dataset by date.

    :param configuration: the loaded loader configuration
    :param override: an explicit target, taking precedence over the
        configuration's own target block
    :return: a target with source, variable, mask and weight_mask keys, the
        latter two possibly None
    """
    sources = configuration.get("sources", {})
    masks = configuration.get("masks", {})
    raw_target = override if override is not None \
        else configuration.get("target")

    if raw_target is None:
        # Fall back to the historically hardcoded variable, but only where one
        # source provides it: guessing would silently change the ground truth.
        candidates = [
            source_name for source_name, source_cfg in sources.items()
            if DEFAULT_TARGET_VARIABLE in source_cfg.get("processed_files", {})
        ]

        if len(candidates) != 1:
            raise RuntimeError(
                "Could not infer the legacy target. Expected exactly one "
                "source containing {!r}, found: {}".format(
                    DEFAULT_TARGET_VARIABLE, candidates))

        # Sea ice masking is only implied for the inferred sea ice target. An
        # explicit target states its own masks, as land and active grid cells
        # mean nothing over, say, a temperature field.
        raw_target = {
            "source": candidates[0],
            "variable": DEFAULT_TARGET_VARIABLE,
            "mask": DEFAULT_TARGET_MASK,
            "weight_mask": DEFAULT_TARGET_WEIGHT_MASK,
        }

        for field_name in ("mask", "weight_mask"):
            if raw_target[field_name] not in masks:
                raw_target[field_name] = None

    if not isinstance(raw_target, dict):
        raise TypeError("target must be a JSON object")

    missing_fields = {"source", "variable"} - set(raw_target.keys())

    if missing_fields:
        raise ValueError("target is missing required fields: {}".format(
            ", ".join(sorted(missing_fields))))

    target = dict(raw_target)
    source_name = target["source"]
    variable_name = target["variable"]

    if not isinstance(source_name, str) or not source_name:
        raise ValueError("target.source must be a non-empty string")

    if not isinstance(variable_name, str) or not variable_name:
        raise ValueError("target.variable must be a non-empty string")

    if source_name not in sources:
        raise ValueError(
            "Unknown target source {!r}; available sources: {}".format(
                source_name, ", ".join(sorted(sources.keys()))))

    processed_files = sources[source_name].get("processed_files", {})
    target_files = processed_files.get(variable_name)

    if not isinstance(target_files, list) or not target_files:
        raise ValueError("Target variable {!r} is not available in "
                         "sources[{!r}].processed_files".format(
                             variable_name, source_name))

    target.setdefault("mask", None)
    target.setdefault("weight_mask", None)

    for field_name in ("mask", "weight_mask"):
        mask_name = target[field_name]

        if mask_name is not None and mask_name not in masks:
            raise ValueError("target.{} references unknown mask {!r}".format(
                field_name, mask_name))

    return target


class IceNetBaseDataLoader(DataCollection):
    """

    :param loader_configuration,
    :param identifier,
    :param var_lag,
    :param dataset_config_path:
    :param generate_workers:
    :param loss_weight_days:
    :param n_forecast_days:
    :param output_batch_size:
    :param path:
    :param target:
    :param var_lag_override:
    """

    def __init__(self,
                 loader_configuration: str,
                 identifier: str,
                 *args,
                 dataset_config_path: str = ".",
                 dates_override: object = None,
                 dry: bool = False,
                 generate_workers: int = 8,
                 lag_time: int = None,
                 lead_time: int = None,
                 loss_weight_days: bool = True,
                 output_batch_size: int = 32,
                 path: str = os.path.join(".", "network_datasets"),
                 pickup: bool = False,
                 target: object = None,
                 var_lag_override: object = None,
                 **kwargs):
        super().__init__(*args, identifier=identifier, base_path=path, **kwargs)

        self._channels = dict()
        self._channel_files = dict()

        self._configuration_path = loader_configuration
        self._dataset_config_path = dataset_config_path
        self._dates_override = dates_override
        self._config = dict()
        self._dry = dry
        self._loss_weight_days = loss_weight_days
        self._meta_channels = []
        self._missing_dates = []
        self._output_batch_size = output_batch_size
        self._pickup = pickup
        self._trend_steps = dict()
        self._workers = generate_workers

        self._load_configuration(loader_configuration)

        # The target defines the output grid and the leadtime frequency, so
        # derive them from it rather than from whichever source is ordered
        # first.
        self._target = _resolve_target_config(self._config, override=target)

        target_source = self._target["source"]
        target_variable = self._target["variable"]
        target_source_cfg = self._config["sources"][target_source]

        self._target_files = tuple(
            target_source_cfg["processed_files"][target_variable])

        processor = get_processor_from_source(target_source, target_source_cfg)
        ds_config = get_dataset_config_implementation(processor.dataset_config)
        ref_ds = processor.get_dataset([target_variable])

        if target_variable not in ref_ds.data_vars:
            raise RuntimeError(
                "Target variable {!r} was not present in the opened target "
                "dataset; available variables: {}".format(
                    target_variable, ", ".join(sorted(ref_ds.data_vars))))

        target_da = ref_ds[target_variable]

        required_dims = {"time", "yc", "xc"}
        missing_dims = required_dims - set(target_da.dims)

        if missing_dims:
            raise RuntimeError(
                "Target variable {!r} is missing required dimensions: "
                "{}".format(target_variable, ", ".join(sorted(missing_dims))))

        ref_da = target_da.transpose("yc", "xc", "time").isel(time=0)

        # Things that come from preprocessing by default
        self._dtype = ref_da.dtype
        # TODO: we shouldn't ideally need this but we do need a concept of location for masks
        self._ds_config = processor.dataset_config
        self._frequency_attr = ds_config.frequency.attribute
        self._lag_time = lag_time if lag_time is not None else processor.lag_time
        self._lead_time = lead_time if lead_time is not None else processor.lead_time
        self._north = ds_config.location.north
        self._shape = ref_da.shape
        self._south = ds_config.location.south
        self._var_lag_override = dict() \
            if not var_lag_override else var_lag_override

        self._construct_channels()

        self._missing_dates = []
        #    # TODO: format needs to be picked up from dataset frequencies
        #    dt.datetime.strptime(s, DATE_FORMAT)
        #    for s in self._config["missing_dates"]
        #]

    def get_data_var_folder(self,
                            var_name: str,
                            append: object = None,
                            missing_error: bool = False) -> os.PathLike:
        """Returns the path for a specific data variable.

        Appends additional folders to the path if specified in the `append` parameter.

        :param var_name: The data variable.
        :param append: Additional folders to append to the path.
        :param missing_error: Flag to specify if missing directories should be treated as an error.

        :return str: The path for the specific data variable.
        """
        if not append:
            append = []

        data_var_path = os.path.join(self.path, *[var_name, *append])

        if not os.path.exists(data_var_path):
            if not missing_error:
                os.makedirs(data_var_path, exist_ok=True)
            else:
                raise OSError("Directory {} is missing and this is "
                              "flagged as an error!".format(data_var_path))

        return data_var_path

    def write_dataset_config_only(self):
        """

        """
        splits = self._config["sources"][list(self._config["sources"].keys())[0]]["splits"]
        counts = {el: 0 for el in splits}

        logging.info("Writing dataset configuration without data generation")

        # FIXME: cloned mechanism from generate() - do we need to treat these as
        #  sets that might have missing data for fringe cases?
        for dataset in splits:
            forecast_dates = sorted(
                list(
                    set([
                        dt.datetime.strptime(
                            s, DATE_FORMAT).date()
                        for identity in self._config["sources"].keys()
                        for s in self._config["sources"][identity]["splits"][dataset]
                    ])))

            logging.info("{} {} dates in total, NOT generating cache "
                         "data.".format(len(forecast_dates), dataset))
            counts[dataset] += len(forecast_dates)

        self._write_dataset_config(counts, network_dataset=False)

    @abstractmethod
    def generate_sample(self, date: object, prediction: bool = False):
        """

        :param date:
        :param prediction:
        :return:
        """
        pass

    def get_sample_files(self) -> object:
        """

        :param date:
        :return:
        """
        # FIXME: is this not just the same as _channel_files now?
        # FIXME: still experimental code, move to multiple implementations
        # FIXME: CLEAN THIS ALL UP ONCE VERIFIED FOR local/shared STORAGE!
        var_files = dict()

        for var_name, num_channels in self._channels.items():
            var_file = self._get_var_file(var_name)

            if not var_file:
                raise RuntimeError("No file returned for {}".format(var_name))

            if var_name not in var_files:
                var_files[var_name] = var_file
            elif var_file != var_files[var_name]:
                raise RuntimeError("Differing files? {} {} vs {}".format(
                    var_name, var_file, var_files[var_name]))

        return var_files

    def _add_channel_files(self, var_name: str, filelist: object):
        """

        :param var_name:
        :param filelist:
        """
        if var_name in self._channel_files:
            logging.warning("{} already has files, but more found, "
                            "this could be an unintentional merge of "
                            "sources".format(var_name))
        else:
            self._channel_files[var_name] = []

        logging.debug("Adding {} to {} channel".format(len(filelist), var_name))
        self._channel_files[var_name] += filelist

    def _construct_channels(self):
        """

        """
        # As of Python 3.7 dict guarantees the order of keys based on
        # original insertion order, which is great for this method
        attr_map = dict(
            abs="absolute_vars",
            anom="anomoly_vars",
            linear_trend="linear_trends"
        )
        lag_vars = [
            (identity, var, data_format)
            for data_format in ("abs", "anom")
            for identity in sorted(self._config["sources"].keys())
            for var in sorted(self._config["sources"][identity][attr_map[data_format]])
        ]

        for identity, var_name, data_format in lag_vars:
            var_prefix = "{}_{}".format(var_name, data_format)
            var_lag = (self._lag_time
                       if var_name not in self._var_lag_override
                       else self._var_lag_override[var_name])

            self._channels[var_prefix] = int(var_lag) + 1
            self._add_channel_files(var_prefix, self._config["sources"][identity]["processed_files"][var_prefix])

        trend_names = [(identity, var,
                        self._config["sources"][identity]["linear_trend_steps"])
                       for identity in sorted(self._config["sources"].keys())
                       for var in sorted(self._config["sources"][identity]
                                         ["linear_trends"])]

        for identity, var_name, trend_steps in trend_names:
            var_prefix = "{}_linear_trend".format(var_name)

            self._channels[var_prefix] = len(trend_steps)
            self._trend_steps[var_prefix] = trend_steps
            self._add_channel_files(var_prefix,
                                    self._config["sources"][identity]["processed_files"][var_prefix])

        # Meta channels
        for var_name, meta_channel in self._config["channels"].items():
            self._meta_channels.append(var_name)
            self._channels[var_name] = 1
            self._add_channel_files(
                var_name,
                meta_channel["processed_files"][var_name])

        logging.debug(
            "Channel quantities deduced:\n{}\n\nTotal channels: {}".format(
                pformat(self._channels), self.num_channels))

    def _get_var_file(self, var_name: str):
        """

        :param var_name:
        :return:
        """

        filename = "{}.nc".format(var_name)
        files = self._channel_files[var_name]

        if len(self._channel_files[var_name]) > 1:
            logging.warning(
                "Multiple files found for {}, only returning {}".format(
                    filename, files[0]))
        elif not len(files):
            logging.warning("No files in channel list for {}".format(filename))
            return None
        return files[0]

    def _load_configuration(self, path: str):
        """

        :param path:
        """
        if os.path.exists(path):
            logging.info("Loading configuration {}".format(path))

            with open(path, "r") as fh:
                obj = orjson.loads(fh.read())

            self._config.update(obj)
        else:
            raise OSError("{} not found".format(path))

    def _write_dataset_config(self,
                              counts: object,
                              network_dataset: bool = True):
        """

        :param counts:
        :param network_dataset:
        :return:
        """

        # TODO: move to utils for this and process
        def _serialize(x):
            if x is dt.date:
                return x.strftime(DATE_FORMAT)
            return str(x)

        configuration = {
            "identifier": self.identifier,
            "implementation": self.__class__.__name__,
            # This is only for convenience ;)
            "channels": [
                "{}_{}".format(channel, i)
                for channel, s in self._channels.items()
                for i in range(1, s + 1)
            ],
            "counts": counts,
            "dtype": str(self._dtype),
            "loader_config": os.path.abspath(self._configuration_path),
            "missing_dates": self._missing_dates,
            "lag_time": self._lag_time,
            "lead_time": self._lead_time,
            "north": self.north,
            "num_channels": self.num_channels,
            # FIXME: this naming is inconsistent, sort it out!!! ;)
            "shape": list(self._shape),
            "south": self.south,
            "target": self.target,

            # For recreating this dataloader
            # "dataset_config_path = ".",
            "dataset_path": self._path if network_dataset else False,
            "generate_workers": self.workers,
            "loss_weight_days": self._loss_weight_days,
            "output_batch_size": self._output_batch_size,
            "var_lag_override": self._var_lag_override,
        }

        output_path = os.path.join(
            self._dataset_config_path,
            "dataset_config.{}.json".format(self.identifier))

        logging.info("Writing configuration to {}".format(output_path))

        with open(output_path, "w") as fh:
            fh.write(orjson.dumps(configuration, option=orjson.OPT_INDENT_2).decode())

    @property
    def channel_names(self):
        return [
            "{}_{}".format(nom, idx) if idx_qty > 1 else nom
            for nom, idx_qty in self._channels.items()
            for idx in range(1, idx_qty + 1)
        ]

    @property
    def config(self):
        return self._config

    @property
    def dates_override(self):
        return self._dates_override

    @property
    def north(self):
        return self._north

    @property
    def num_channels(self):
        return sum(self._channels.values())

    @property
    def pickup(self):
        return self._pickup

    @property
    def south(self):
        return self._south

    @property
    def target(self) -> dict:
        """The resolved forecast target for this loader."""
        return dict(self._target)

    @property
    def target_files(self) -> list:
        """The processed files backing the forecast target."""
        return list(self._target_files)

    @property
    def workers(self):
        return self._workers
