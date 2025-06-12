import datetime as dt
import logging
import os
import time

from dateutil.relativedelta import relativedelta
from pprint import pformat

import dask
import dask.array as da

from dask.distributed import Client, LocalCluster
from dask.delayed import delayed

import numpy as np
import pandas as pd
import tensorflow as tf
import xarray as xr

from icenet.data.loaders.base import IceNetBaseDataLoader, DATE_FORMAT
from icenet.data.loaders.utils import IceNetDataWarning, write_tfrecord

"""
Dask implementations for icenet data loading

Still WIP to re-introduce alternate implementations that might work better in
certain deployments

"""


class DaskBaseDataLoader(IceNetBaseDataLoader):
    """A subclass of IceNetBaseDataLoader that provides functionality for loading data using Dask.

    Attributes:
        _dashboard_port: The port number for the Dask dashboard.
        _timeout: The timeout value for Dask communication.
        _tmp_dir: The temporary directory for Dask.
    """

    def __init__(self,
                 *args,
                 dask_port: int = 8888,
                 dask_timeouts: int = 60,
                 dask_tmp_dir: object = "/tmp",
                 **kwargs) -> None:
        """Initialises the DaskBaseDataLoader object with the specified port, timeouts, and temp directory.

        Args:
            dask_port: The port number for the Dask dashboard. Defaults to 8888.
            dask_timeouts: The timeout value for Dask communication. Defaults to 60.
            dask_tmp_dir: The temporary directory for Dask. Defaults to `/tmp`.
        """
        super().__init__(*args, **kwargs)

        self._dashboard_port = dask_port
        self._timeout = dask_timeouts
        self._tmp_dir = dask_tmp_dir

    def generate(self) -> None:
        """
        Generates data using Dask client by setting up a Dask cluster and client,
        and calling client_generate method.
        """
        dashboard = "localhost:{}".format(self._dashboard_port)

        with dask.config.set({
                "temporary_directory": self._tmp_dir,
                "distributed.comm.timeouts.connect": self._timeout,
                "distributed.comm.timeouts.tcp": self._timeout,
        }):
            with LocalCluster(
                dashboard_address=dashboard,
                n_workers=self.workers,
                threads_per_worker=1,
                scheduler_port=0,
            ) as cluster, Client(cluster) as client:
                logging.info("Dashboard at {}".format(dashboard))

                logging.info("Using dask client {}".format(client))
                self.client_generate(client,
                                     dates_override=self.dates_override,
                                     pickup=self.pickup)

    def client_generate(self,
                        client: object,
                        dates_override: object = None,
                        pickup: bool = False) -> None:
        """Generates data using the Dask client. This method needs to be implemented in subclasses.

        Args:
            client: The Dask client.
            dates_override (optional): A dict with keys `train`, `val`, `test`, each with a list of
                continuous dates for that purpose. Defaults to None.
            pickup (optional): TODO. Defaults to False.

        Raises:
            NotImplementedError: If generate is called without being implemented as a subclass of DaskBaseDataLoader.
        """
        raise NotImplementedError("generate called on non-implementation")


class DaskMultiSharingWorkerLoader(DaskBaseDataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # TODO: https://github.com/icenet-ai/icenet/blob/83fdbf4b23ccf6ac221e77809b47d407b70b707f/icenet2/data/loader.py
        raise NotImplementedError("Not yet adapted from old implementation")

    def client_generate(self,
                        client: object,
                        dates_override: object = None,
                        pickup: bool = False):
        """

        :param client:
        :param dates_override:
        :param pickup:
        """
        pass

    def generate_sample(self, date: object, prediction: bool = False):
        """

        :param date:
        :param prediction:
        """
        pass


class DaskMultiWorkerLoader(DaskBaseDataLoader):

    def __init__(self,
                 *args,
                 futures_per_worker: int = 2,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self._masks = {var_name: mask_cfg["processed_files"][var_name][0]
                       for var_name, mask_cfg in self._config["masks"].items()}

        self._futures = futures_per_worker

    def client_generate(self,
                        client: object,
                        dates_override: dict = None,
                        pickup: bool = False):
        """

        :param client:
        :param dates_override:
        :param pickup:
        """
        # TODO: for each set, validate every variable has an appropriate file
        #  in the configuration arrays, otherwise drop the forecast date

        splits = set([s
                      for identity in self._config["sources"].keys()
                      for s in self._config["sources"][identity]["splits"].keys()])

        if dates_override and type(dates_override) is dict:
            for split in splits:
                assert split in dates_override.keys() \
                       and type(dates_override[split]) is list, \
                       "{} needs to be list in dates_override".format(split)
        elif dates_override:
            raise RuntimeError("dates_override needs to be a dict if supplied")

        counts = {el: 0 for el in splits}
        exec_times = []

        def batch(batch_dates, num):
            i = 0
            while i < len(batch_dates):
                yield batch_dates[i:i + num]
                i += num

        masks = client.scatter(self._masks, broadcast=True)

        for dataset in splits:
            batch_number = 0
            futures = []

            # Bit grim, but we need to take the intersection of sources for a complete date retrieval
            # as data might be different across sources
            sources = list(self._config["sources"].keys())
            convert_dates = lambda source: set([
                dt.datetime.strptime(s, DATE_FORMAT).date()
                for s in self._config["sources"][source]["splits"][dataset]])
            forecast_dates = convert_dates(sources[0])
            for add_source in sources[1:]:
                forecast_dates = forecast_dates.intersection(convert_dates(add_source))

            if dates_override:
                logging.info("{} available {} dates".format(
                    len(forecast_dates), dataset))
                forecast_dates = forecast_dates.intersection(dates_override[dataset])
            forecast_dates = sorted(list(forecast_dates))

            output_dir = self.get_data_var_folder(dataset)
            tf_path = os.path.join(output_dir, "{:08}.tfrecord")

            logging.info("{} {} dates to process, generating cache "
                         "data.".format(len(forecast_dates), dataset))

            for dates in batch(forecast_dates, self._output_batch_size):
                if not pickup or \
                    (pickup and
                     not os.path.exists(tf_path.format(batch_number))):
                    args = [
                        self._channels, self._dtype, self._loss_weight_days,
                        self._meta_channels, self._missing_dates,
                        self._lead_time, self.num_channels, self._shape,
                        self._trend_steps, self._frequency_attr, self._masks, False
                    ]

                    fut = client.submit(generate_and_write,
                                        tf_path.format(batch_number),
                                        self.get_sample_files(),
                                        dates,
                                        args,
                                        dry=self._dry,
                                        pure=False)
                    futures.append(fut)

                    # Use this to limit the future list, to avoid crashing the
                    # distributed scheduler / workers (task list gets too big!)
                    if len(futures) >= self._workers * self._futures:
                        for tf_data, samples, gen_times \
                                in client.gather(futures):
                            logging.info("Finished output {}".format(tf_data))
                            counts[dataset] += samples
                            exec_times += gen_times
                        futures = []

                    # tf_data, samples, times = generate_and_write(
                    #    tf_path.format(batch_number), args, dry=self._dry)
                else:
                    counts[dataset] += len(dates)
                    logging.warning("Skipping {} on pickup run".format(
                        tf_path.format(batch_number)))

                batch_number += 1

            # Hoover up remaining futures
            for tf_data, samples, gen_times \
                    in client.gather(futures):
                logging.info("Finished output {}".format(tf_data))
                counts[dataset] += samples
                exec_times += gen_times

        if len(exec_times) > 0:
            logging.info("Average sample generation time: {}".format(
                np.average(exec_times)))
        self._write_dataset_config(counts)

    def generate_sample(self,
                        date: object,
                        prediction: bool = False,
                        parallel=True):
        """

        :param date:
        :param prediction:
        :param parallel:
        :return:
        """

        ds_kwargs = dict(
            chunks=dict(time=1,
                        yc=self._shape[0], xc=self._shape[1]),
            drop_variables=["month", "plev", "level", "realization"],
            parallel=parallel,
            engine="h5netcdf",
        )
        var_files = self.get_sample_files()

        var_ds = xr.open_mfdataset([
            v for k, v in var_files.items()
            if k not in self._meta_channels and not k.endswith("linear_trend")
        ], **ds_kwargs)

        logging.debug("VAR: {}".format(pformat(var_ds)))
        x_name = "xc"
        y_name = "yc"
        var_ds = var_ds.transpose(y_name, x_name, "time")

        trend_files = \
            [v for k, v in var_files.items()
             if k.endswith("linear_trend")]
        trend_ds = None

        if len(trend_files) > 0:
            trend_ds = xr.open_mfdataset(trend_files, **ds_kwargs)
            logging.debug("TREND: {}".format(pformat(trend_ds)))
            trend_ds = trend_ds.transpose(y_name, x_name, "time")

        args = [
            self._channels, self._dtype, self._loss_weight_days,
            self._meta_channels, self._missing_dates, self._lead_time,
            self.num_channels, self._shape, self._trend_steps, self._frequency_attr,
            self._masks, prediction
        ]

        x, y, sw = generate_sample(date, var_ds, var_files, trend_ds, *args)
        return x.compute(), y.compute(), sw.compute()


def generate_and_write(path: str,
                       var_files: object,
                       dates: object,
                       args: tuple,
                       dry: bool = False):
    """

    :param path:
    :param var_files:
    :param dates:
    :param args:
    :param dry:
    :return:
    """
    count = 0
    times = []

    (channels, dtype, loss_weight_days, meta_channels, missing_dates,
     n_forecast_days, num_channels, shape, trend_steps, frequency_attr, masks,
     prediction) = args

    ds_kwargs = dict(
        chunks=dict(time=1, yc=shape[0], xc=shape[1]),
        drop_variables=["month", "plev", "realization"],
        parallel=True,
        engine="h5netcdf"
    )

    # TODO: use to_dask_dataframe to stop the relentless moaning? Need to investigate
    #   the submission issue for run_specs in dask: https://github.com/dask/dask/issues/9888
    #   Suspect the datasets should be distributed from outside the compute graph
    #   Whichever way, it doesn't like the size of data moving about or lots of run_spec conflicts
    #   but none of these actually seem to be producing errors
    var_ds = xr.open_mfdataset([
        v for k, v in var_files.items()
        if k not in meta_channels and not k.endswith("linear_trend")
    ], **ds_kwargs)
    x_name = "xc"
    y_name = "yc"
    var_ds = var_ds.transpose(y_name, x_name, "time")

    trend_files = [
        v for k, v in var_files.items() if k.endswith("linear_trend")
    ]
    trend_ds = None

    if len(trend_files):
        trend_ds = xr.open_mfdataset(trend_files, **ds_kwargs)
        trend_ds = trend_ds.transpose(y_name, x_name, "time")

    with tf.io.TFRecordWriter(path) as writer:
        for date in dates:
            start = time.time()

            try:
                x, y, sample_weights = generate_sample(date, var_ds, var_files,
                                                       trend_ds, *args)
                if not dry:
                    x, y, sample_weights = dask.compute(x,
                                                        y,
                                                        sample_weights,
                                                        optimize_graph=True)
                    write_tfrecord(writer, x, y, sample_weights)
                count += 1
            except IceNetDataWarning as e:
                logging.error("Data cannot be included in the outputs: {} - {}".format(date, e))
                continue

            end = time.time()
            times.append(end - start)
            logging.debug("Time taken to produce {}: {}".format(
                date, times[-1]))
    return path, count, times


def generate_sample(forecast_date: object,
                    var_ds: object,
                    var_files: object,
                    trend_ds: object,
                    channels: object,
                    dtype: object,
                    loss_weight_days: bool,
                    meta_channels: object,
                    missing_dates: object,
                    n_forecast_steps: int,
                    num_channels: int,
                    shape: object,
                    trend_steps: object,
                    frequency_attr: str,
                    masks: object,
                    prediction: bool = False):
    """


    :param forecast_date:
    :param var_ds:
    :param var_files:
    :param trend_ds:
    :param channels:
    :param dtype:
    :param loss_weight_days:
    :param meta_channels:
    :param missing_dates:
    :param n_forecast_steps:
    :param num_channels:
    :param shape:
    :param trend_steps:
    :param frequency_attr:
    :param masks:
    :param prediction:
    :return:
    """
    relative_attr = "{}s".format(frequency_attr)

    # Prepare data sample
    # To become array of shape (*raw_data_shape, n_forecast_steps)
    forecast_base_idx = list(var_ds.time.values).index(pd.Timestamp(forecast_date))
    forecast_idxs = [forecast_base_idx + n for n in range(0, n_forecast_steps)]

    y = da.zeros((*shape, n_forecast_steps, 1), dtype=dtype)
    sample_weights = da.zeros((*shape, n_forecast_steps, 1), dtype=dtype)

    land_mask = xr.open_dataarray(masks["land"])

    if not prediction:
        try:
            sample_output = var_ds.siconca_abs.isel(time=forecast_idxs)
        except (KeyError, IndexError):
            raise IceNetDataWarning(
                "Issue with y-data for non-prediction sample {}, "
                "please review siconca ground-truth: dates {}".format(forecast_date, forecast_idxs))
        y[:, :, :, 0] = sample_output
        y_mask = da.stack([land_mask.data for _ in range(0, n_forecast_steps)], axis=-1)
        y_mask = da.stack([y_mask], axis=-1)
        y = da.ma.where(y_mask, 0., y)

    # Masked recomposition of output
    for leadtime_idx in range(n_forecast_steps):
        forecast_step = forecast_date + relativedelta(**{relative_attr: leadtime_idx})

        if any([forecast_step == missing_date for missing_date in missing_dates]):
            sample_weight = da.zeros(shape, dtype)
        else:
            # TODO: this is hacky - across the entire sample generation process we need to render all masks down
            # Zero loss outside of 'active grid cells'
            if "active_grid_cell" in masks:
                sample_weight = xr.open_dataarray(masks["active_grid_cell"]).sel(month=forecast_step.month).data
                sample_weight[land_mask.astype("bool")] = 0.
            else:
                # sample_weight = da.ones(shape, dtype)
                sample_weight = da.where(land_mask.isnull(), 0., 1.)
                sample_weight = da.where(land_mask == 1, 0., 1.)

            # TODO: dynamic inclusion of polarhole?
            sample_weight = sample_weight.astype(dtype)

            # We can pick up nans, which messes up training
            sample_weight[da.isnan(y[..., leadtime_idx, 0])] = 0

            # Scale the loss for each month s.t. March is
            #   scaled by 1 and Sept is scaled by 1.77
            # TODO: this isn't generally applicable (e.g. daily / amsr) so have removed it temporarily
            # if loss_weight_days:
            #     sample_weight *= 33928. / sample_weight.sum()

        sample_weights[:, :, leadtime_idx, 0] = sample_weight

    # INPUT FEATURES
    x = da.zeros((*shape, num_channels), dtype=dtype)
    v1, v2 = 0, 0

    for var_name, num_channels in channels.items():
        if var_name in meta_channels:
            continue

        v2 += num_channels

        if var_name.endswith("linear_trend"):
            channel_ds = trend_ds
            # The linear trend indexing is not the same as the normal data channels, so rediscover the base idx
            lt_base_idx = list(channel_ds.time.values).index(pd.Timestamp(forecast_date))
            if type(trend_steps) is list:
                channel_idxs = [lt_base_idx + n for n in trend_steps]
            else:
                channel_idxs = [lt_base_idx + n for n in range(0, num_channels)]
        # If we're not a trend, we're a lag channel looking back historically from the initialisation date
        else:
            channel_ds = var_ds
            channel_idxs = [forecast_base_idx - n for n in range(1, num_channels + 1)]

        channel_data = []
        for idx in channel_idxs:
            try:
                data = getattr(channel_ds, var_name).isel(time=idx)

                # TODO: validate, but this should not be required if the weights are good
                #  except the situation where there're nans in trends, perhaps?
                # if var_name.startswith("siconca"):
                #     data = da.ma.where(masks["land"], 0., data)

                # TODO: this is probably going to slow things up, but will make datasets more resilient
                if da.nansum(data) == 0:
                    raise IceNetDataWarning("We have {} with blank data at time {} for forecast date {}".
                                            format(var_name, channel_ds.time.values[idx], forecast_date))
                channel_data.append(data)

                # logging.info("NANs: {} = {} in {}-{}".format(forecast_date, int(da.isnan(data).sum()), var_name, idx))
            except (KeyError, IndexError) as e:
                raise IceNetDataWarning("Key or Index error detected on channel construction for {} - {}: {}".
                                        format(forecast_date, var_name, e))
                # channel_data.append(da.zeros(shape))

        x[:, :, v1:v2] = da.from_array(channel_data).transpose([1, 2, 0])
        v1 += num_channels

    for var_name in meta_channels:
        if channels[var_name] > 1:
            raise RuntimeError("{} meta variable cannot have more than "
                               "one channel".format(var_name))

        meta_ds = xr.open_dataarray(var_files[var_name])

        if var_name in ["sin", "cos"]:
            ref_date = "2012-{}-{}".format(forecast_date.month,
                                           forecast_date.day)
            trig_val = meta_ds.sel(time=ref_date).to_numpy()
            x[:, :, v1] = da.broadcast_to([trig_val], shape)
        else:
            x[:, :, v1] = da.array(meta_ds.to_numpy())
        v1 += channels[var_name]

    # TODO: we have unwarranted nans which need fixing, probably from broken spatial infilling
    #nan_mask_x = da.isnan(x)# , nan_mask_y, nan_mask_sw = da.isnan(x), da.isnan(y), da.isnan(sample_weights)
    #if nan_mask_x.sum():# + nan_mask_y.sum() + nan_mask_sw.sum() > 0:
    #    logging.warning("NANs {}: zeroing {} in input".format(#, {} in output, {} in weights".format(
    #        forecast_date.strftime("%F"), int(nan_mask_x.sum())# , int(nan_mask_y.sum()), int(nan_mask_sw.sum())
    #    ))
    #    x[nan_mask_x] = 0
    #
    #

    nan_mask_x, nan_mask_y, nan_mask_sw = da.isnan(x), da.isnan(y), da.isnan(sample_weights)
    if nan_mask_x.sum() + nan_mask_y.sum() + nan_mask_sw.sum() > 0:
        logging.warning("NANs {}: zeroing {} in input, {} in output, {} in weights".format(
            forecast_date.strftime("%F"), int(nan_mask_x.sum()), int(nan_mask_y.sum()), int(nan_mask_sw.sum())
        ))
        x[nan_mask_x] = 0
        sample_weights[nan_mask_sw] = 0
        y[nan_mask_y] = 0

    return x, y, sample_weights
