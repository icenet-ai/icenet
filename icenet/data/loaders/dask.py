import datetime as dt
import logging
import os
import time

from dateutil.relativedelta import relativedelta

import dask
import dask.array as da

from dask.distributed import Client, LocalCluster

import numpy as np
import pandas as pd
import tensorflow as tf
import xarray as xr

from icenet.data.loaders.base import IceNetBaseDataLoader, DATE_FORMAT
from icenet.data.loaders.utils import IceNetDataWarning, write_tfrecord
from icenet.data.masks.osisaf import Masks

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

        # FIXME
        # self._masks = da.array(
        #    [np.load(self._config["masks"]["active_grid_cell"][month-1]) for month in range(1, 13)])
        self._masks = {var_name: xr.open_dataarray(mask_cfg["processed_files"][var_name][0])
                       for var_name, mask_cfg in self._config["masks"].items()}

        self._futures = futures_per_worker

    def client_generate(self,
                        client: object,
                        dates_override: object = None,
                        pickup: bool = False):
        """

        :param client:
        :param dates_override:
        :param pickup:
        """
        # TODO: for each set, validate every variable has an appropriate file
        #  in the configuration arrays, otherwise drop the forecast date
        splits = ("train", "val", "test")

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

            forecast_dates = set([
                dt.datetime.strptime(s, DATE_FORMAT).date()
                for identity in self._config["sources"].keys()
                for s in self._config["sources"][identity]["splits"][dataset]
            ])

            if dates_override:
                logging.info("{} available {} dates".format(
                    len(forecast_dates), dataset))
                forecast_dates = forecast_dates.intersection(
                    dates_override[dataset])
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
                        self._trend_steps, self._frequency_attr, masks, False
                    ]

                    fut = client.submit(generate_and_write,
                                        tf_path.format(batch_number),
                                        self.get_sample_files(),
                                        dates,
                                        args,
                                        dry=self._dry)
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
            chunks=dict(time=1, yc=self._shape[0], xc=self._shape[1]),
            drop_variables=["month", "plev", "level", "realization"],
            parallel=parallel,
        )
        var_files = self.get_sample_files()

        var_ds = xr.open_mfdataset([
            v for k, v in var_files.items()
            if k not in self._meta_channels and not k.endswith("linear_trend")
        ], **ds_kwargs)

        var_ds = var_ds.transpose("yc", "xc", "time")

        trend_files = \
            [v for k, v in var_files.items()
             if k.endswith("linear_trend")]
        trend_ds = None

        if len(trend_files) > 0:
            trend_ds = xr.open_mfdataset(trend_files, **ds_kwargs)

            trend_ds = trend_ds.transpose("yc", "xc", "time")

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

    # TODO: refactor, this is very smelly - with new data throughput args
    #  will always be the same
    (channels, dtype, loss_weight_days, meta_channels, missing_dates,
     n_forecast_days, num_channels, shape, trend_steps, frequency_attr, masks,
     prediction) = args

    ds_kwargs = dict(
        chunks=dict(time=1, yc=shape[0], xc=shape[1]),
        drop_variables=["month", "plev", "realization"],
        parallel=True,
    )

    #print([
    #    v for k, v in var_files.items()
    #    if k not in meta_channels and not k.endswith("linear_trend")
    #])
    #print(ds_kwargs)
    #import sys
    #sys.exit(0)

    var_ds = xr.open_mfdataset([
        v for k, v in var_files.items()
        if k not in meta_channels and not k.endswith("linear_trend")
    ], **ds_kwargs)
    var_ds = var_ds.transpose("yc", "xc", "time")

    trend_files = [
        v for k, v in var_files.items() if k.endswith("linear_trend")
    ]
    trend_ds = None

    if len(trend_files):
        trend_ds = xr.open_mfdataset(trend_files, **ds_kwargs)
        trend_ds = trend_ds.transpose("yc", "xc", "time")

    with tf.io.TFRecordWriter(path) as writer:
        for date in dates:
            start = time.time()

            try:
                x, y, sample_weights = generate_sample(date, var_ds, var_files,
                                                       trend_ds, *args)
                if not dry:
                    x[da.isnan(x)] = 0.

                    x, y, sample_weights = dask.compute(x,
                                                        y,
                                                        sample_weights,
                                                        optimize_graph=True)
                    write_tfrecord(writer, x, y, sample_weights)
                count += 1
            except IceNetDataWarning:
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
    forecast_idxs = [
        forecast_base_idx + n for n in range(0, n_forecast_steps)
    ]

    y = da.zeros((*shape, n_forecast_steps, 1), dtype=dtype)
    sample_weights = da.zeros((*shape, n_forecast_steps, 1), dtype=dtype)

    if not prediction:
        try:
            sample_output = var_ds.siconca_abs.isel(time=forecast_idxs)
        except KeyError as sic_ex:
            logging.exception(
                "Issue selecting data for non-prediction sample, "
                "please review siconca ground-truth: dates {}".format(forecast_idxs))
            raise RuntimeError(sic_ex)
        y[:, :, :, 0] = sample_output
        y_mask = da.stack([masks["land"].data for _ in range(0, n_forecast_steps)], axis=-1)
        y_mask = da.stack([y_mask], axis=-1)
        y = da.ma.where(y_mask, 0., y)

    # Masked recomposition of output
    agcm_masks = []

    for leadtime_idx in range(n_forecast_steps):
        forecast_step = forecast_date + relativedelta(**{relative_attr: leadtime_idx})

        if any([forecast_step == missing_date for missing_date in missing_dates]):
            sample_weight = da.zeros(shape, dtype)
        else:
            # Zero loss outside of 'active grid cells'
            #sample_weight = masks["active_grid_cell"][forecast_day.month - 1]
            sample_weight = masks["active_grid_cell"].sel(month=forecast_step.month).data
            # TODO: dynamic inclusion of polarhole
            sample_weight = sample_weight.astype(dtype)
            sample_weight[masks["land"]] = 0.

            # We can pick up nans, which messes up training
            sample_weight[da.isnan(y[..., leadtime_idx, 0])] = 0

            # Scale the loss for each month s.t. March is
            #   scaled by 1 and Sept is scaled by 1.77
            if loss_weight_days:
                sample_weight *= 33928. / sample_weight.sum()

            agcm_masks.append(masks["active_grid_cell"].sel(month=forecast_step.month).data)

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
            if type(trend_steps) is list:
                channel_idxs = [forecast_base_idx + n for n in trend_steps]
            else:
                channel_idxs = [forecast_base_idx + n for n in range(0, num_channels)]
        else:
            channel_ds = var_ds
            channel_idxs = [forecast_base_idx + n for n in range(0, num_channels)]

        channel_data = []
        for idx in channel_idxs:
            try:
                data = getattr(channel_ds, var_name).isel(time=idx)
                if var_name.startswith("siconca"):
                    data = da.ma.where(masks["land"], 0., data)
                channel_data.append(data)
            except KeyError:
                channel_data.append(da.zeros(shape))

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

    # This is an attempt to avoid the following hack, but it doesn't work
    # agcm_masks = da.stack([agcm for agcm in agcm_masks], axis=-1)
    # agcm_masks = da.stack([agcm_masks], axis=-1)
    # y = da.ma.where(agcm_masks, y, 0.)

    # TODO: this is a hack, we have unwarranted nans and sample_weights aren't working with metrics
    x[da.isnan(x)] = 0
    sample_weights[da.isnan(sample_weights)] = 0
    y[da.isnan(y)] = 0

    return x, y, sample_weights
