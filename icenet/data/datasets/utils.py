import glob
import logging
import os

import numpy as np
import tensorflow as tf


def get_decoder(shape: object,
                channels: object,
                forecasts: object,
                num_vars: int = 1,
                dtype: str = "float32") -> object:
    """Returns a decoder function used for parsing and decoding data from tfrecord protocol buffer.

    Args:
        shape: The shape of the input data.
        channels: The number of channels in the input data.
        forecasts: The number of days to forecast in prediction
        num_vars (optional): The number of variables in the input data. Defaults to 1.
        dtype (optional): The data type of the input data. Defaults to "float32".

    Returns:
        A function that can be used to parse and decode data. It takes in a protocol buffer
            (tfrecord) as input and returns the parsed and decoded data.
    """
    xf = tf.io.FixedLenFeature([*shape, channels], getattr(tf, dtype))
    yf = tf.io.FixedLenFeature([*shape, forecasts, num_vars],
                               getattr(tf, dtype))
    sf = tf.io.FixedLenFeature([*shape, forecasts, num_vars],
                               getattr(tf, dtype))

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
    """Read train, val, test datasets from tfrecord protocol buffer files.

    Split and shuffle data if specified as well.

    Example:
        This mixin is not to be used directly, but to give an idea of its use:

        # Initialise SplittingMixin
        >>> split_dataset = SplittingMixin()

        # Add file paths to the train, validation, and test datasets
        >>> split_dataset.add_records(base_path="./network_datasets/notebook_data/", hemi="south")
    """
    _batch_size: int
    _dtype: object
    _num_channels: int
    _n_forecast_days: int
    _shape: int
    _shuffling: bool

    train_fns = []
    test_fns = []
    val_fns = []

    def add_records(self, base_path: str, hemi: str) -> None:
        """Add list of paths to train, val, test *.tfrecord(s) to relevant instance attributes.

        Add sorted list of file paths to train, validation, and test datasets in SplittingMixin.

        Args:
            base_path (str): The base path where the datasets are located.
            hemi (str): The hemisphere the datasets correspond to.

        Returns:
            None. Updates `self.train_fns`, `self.val_fns`, `self.test_fns` with list
                of *.tfrecord files.
        """
        train_path = os.path.join(base_path, hemi, "train")
        val_path = os.path.join(base_path, hemi, "val")
        test_path = os.path.join(base_path, hemi, "test")

        logging.info("Training dataset path: {}".format(train_path))
        self.train_fns += sorted(glob.glob("{}/*.tfrecord".format(train_path)))
        logging.info("Validation dataset path: {}".format(val_path))
        self.val_fns += sorted(glob.glob("{}/*.tfrecord".format(val_path)))
        logging.info("Test dataset path: {}".format(test_path))
        self.test_fns += sorted(glob.glob("{}/*.tfrecord".format(test_path)))

    def get_split_datasets(self, ratio: object = None):
        """Retrieves train, val, and test datasets from corresponding attributes of SplittingMixin.

        Retrieves the train, validation, and test datasets from the file paths stored in the
            `train_fns`, `val_fns`, and `test_fns` attributes of SplittingMixin.

        Args:
            ratio (optional): A float representing the truncated list of datasets to be used.
                If not specified, all datasets will be used.
                Defaults to None.

        Returns:
            tuple: A tuple containing the train, validation, and test datasets.

        Raises:
            RuntimeError: If no files have been found in the train, validation, and test datasets.
            RuntimeError: If the ratio is greater than 1.
        """
        if not (len(self.train_fns) + len(self.val_fns) + len(self.test_fns)):
            raise RuntimeError("No files have been found, abandoning. This is "
                               "likely because you're trying to use a config "
                               "only mode dataset in a situation that demands "
                               "tfrecords to be generated (like training...)")

        logging.info("Datasets: {} train, {} val and {} test filenames".format(
            len(self.train_fns), len(self.val_fns), len(self.test_fns)))

        # If ratio is specified, truncate file paths for train, val, test using the ratio.
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

            logging.info(
                "Reduced: {} train, {} val and {} test filenames".format(
                    len(self.train_fns), len(self.val_fns), len(self.test_fns)))

        # Loads from files as bytes exactly as written. Must parse and decode it.
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

        if self.shuffling:
            logging.info("Training dataset(s) marked to be shuffled")
            # FIXME: this is not a good calculation, but we don't have access
            #  in the mixin to the configuration that generated the dataset #57
            train_ds = train_ds.shuffle(
                min(int(len(self.train_fns) * self.batch_size), 366))

        # Since TFRecordDataset does not parse or decode the dataset from bytes,
        # use custom decoder function with map to do so.
        train_ds = train_ds.\
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

    def check_dataset(self, split: str = "train") -> None:
        """Check the dataset for NaN, log debugging info regarding dataset shape and bounds.

        Also logs a warning if any NaN are found.

        Args:
            split: The split of the dataset to check. Default is "train".
        """
        logging.debug("Checking dataset {}".format(split))

        decoder = get_decoder(self.shape,
                              self.num_channels,
                              self.n_forecast_days,
                              dtype=self.dtype.__name__)

        for df in getattr(self, "{}_fns".format(split)):
            logging.debug("Getting records from {}".format(df))
            try:
                raw_dataset = tf.data.TFRecordDataset([df])
                raw_dataset = raw_dataset.map(decoder)

                for i, (x, y, sw) in enumerate(raw_dataset):
                    x = x.numpy()
                    y = y.numpy()
                    sw = sw.numpy()

                    logging.debug(
                        "Got record {}:{} with x {} y {} sw {}".format(
                            df, i, x.shape, y.shape, sw.shape))

                    input_nans = np.isnan(x).sum()
                    output_nans = np.isnan(y[sw > 0.]).sum()
                    input_min = np.min(x)
                    input_max = np.max(x)
                    output_min = np.min(x)
                    output_max = np.max(x)
                    sw_min = np.min(x)
                    sw_max = np.max(x)

                    logging.debug(
                        "Bounds: Input {}:{} Output {}:{} SW {}:{}".format(
                            input_min, input_max, output_min, output_max,
                            sw_min, sw_max))

                    if input_nans > 0:
                        logging.warning("Input NaNs detected in {}:{}".format(
                            df, i))

                    if output_nans > 0:
                        logging.warning(
                            "Output NaNs detected in {}:{}, not "
                            "accounted for by sample weighting".format(df, i))
            except tf.errors.DataLossError as e:
                logging.warning("{}: data loss error {}".format(df, e.message))
            except tf.errors.OpError as e:
                logging.warning("{}: tensorflow error {}".format(df, e.message))
            # We don't except any non-tensorflow errors to prevent progression

    @property
    def batch_size(self) -> int:
        """The dataset's batch size."""
        return self._batch_size

    @property
    def dtype(self) -> str:
        """The dataset's data type."""
        return self._dtype

    @property
    def n_forecast_days(self) -> int:
        """The number of days to forecast in prediction."""
        return self._n_forecast_days

    @property
    def num_channels(self) -> int:
        """The number of channels in dataset."""
        return self._num_channels

    @property
    def shape(self) -> object:
        """The shape of dataset."""
        return self._shape

    @property
    def shuffling(self) -> bool:
        """A flag for whether training dataset(s) are marked to be shuffled."""
        return self._shuffling
