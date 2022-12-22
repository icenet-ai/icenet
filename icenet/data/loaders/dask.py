import datetime as dt
import logging
import os
import time

import dask
import dask.array as da

from dask.distributed import Client, LocalCluster

import numpy as np
import pandas as pd
import tensorflow as tf
import xarray as xr

from icenet.data.process import IceNetPreProcessor
from icenet.data.loaders.base import IceNetBaseDataLoader
from icenet.data.loaders.utils import IceNetDataWarning, write_tfrecord
from icenet.data.sic.mask import Masks


"""
Dask implementations for icenet data loading

Still WIP to re-introduce alternate implementations that might work better in 
certain deployments

"""


class DaskBaseDataLoader(IceNetBaseDataLoader):
    def __init__(self,
                 *args,
                 dask_port: int = 8888,
                 dask_timeouts: int = 60,
                 dask_tmp_dir: object = "/tmp",
                 **kwargs):
        super().__init__(*args, **kwargs)

        self._dashboard_port = dask_port
        self._timeout = dask_timeouts
        self._tmp_dir = dask_tmp_dir

    def generate(self):
        """

        """
        dashboard = "localhost:{}".format(self._dashboard_port)

        with dask.config.set({
            "temporary_directory": self._tmp_dir,
            "distributed.comm.timeouts.connect": self._timeout,
            "distributed.comm.timeouts.tcp": self._timeout,
        }):
            cluster = LocalCluster(
                dashboard_address=dashboard,
                n_workers=self.workers,
                threads_per_worker=1,
                scheduler_port=0,
            )
            logging.info("Dashboard at {}".format(dashboard))

            with Client(cluster) as client:
                logging.info("Using dask client {}".format(client))
                self.client_generate(client,
                                     dates_override=self.dates_override,
                                     pickup=self.pickup)

    def client_generate(self,
                        client: object,
                        dates_override: object = None,
                        pickup: bool = False):
        """

        :param client:
        :param dates_override:
        :param pickup:
        """
        raise NotImplementedError("generate called on non-implementation")


class DaskMultiSharingWorkerLoader(DaskBaseDataLoader):
    def __init__(self,
                 *args,
                 **kwargs):
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

    def generate_sample(self,
                        date: object,
                        prediction: bool = False):
        """

        :param date:
        :param prediction:
        """
        pass


class DaskMultiWorkerLoader(DaskBaseDataLoader):
    def __init__(self,
                 *args,
                 futures_per_worker: int = 2,
                 **kwargs):
        super().__init__(*args, **kwargs)

        masks = Masks(north=self.north, south=self.south)
        self._masks = da.array([
            masks.get_active_cell_mask(month) for month in range(1, 13)])

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

            forecast_dates = set([dt.datetime.strptime(s,
                                  IceNetPreProcessor.DATE_FORMAT).date()
                                  for identity in
                                  self._config["sources"].keys()
                                  for s in
                                  self._config["sources"][identity]
                                  ["dates"][dataset]])

            if dates_override:
                logging.info("{} available {} dates".
                             format(len(forecast_dates), dataset))
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
                        self._channels,
                        self._dtype,
                        self._loss_weight_days,
                        self._meta_channels,
                        self._missing_dates,
                        self._n_forecast_days,
                        self.num_channels,
                        self._shape,
                        self._trend_steps,
                        masks,
                        False
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
                    logging.warning("Skipping {} on pickup run".
                                    format(tf_path.format(batch_number)))

                batch_number += 1

            # Hoover up remaining futures
            for tf_data, samples, gen_times \
                    in client.gather(futures):
                logging.info("Finished output {}".format(tf_data))
                counts[dataset] += samples
                exec_times += gen_times

        if len(exec_times) > 0:
            logging.info("Average sample generation time: {}".
                         format(np.average(exec_times)))
        self._write_dataset_config(counts)

    def generate_sample(self,
                        date: object,
                        prediction: bool = False):
        """

        :param date:
        :param prediction:
        :return:
        """

        ds_kwargs = dict(
            chunks=dict(time=1, yc=self._shape[0], xc=self._shape[1]),
            drop_variables=["month", "plev", "realization"],
            parallel=True,
        )
        var_files = self.get_sample_files()

        var_ds = xr.open_mfdataset(
            [v for k, v in var_files.items()
             if k not in self._meta_channels
             and not k.endswith("linear_trend")],
            **ds_kwargs)
        var_ds = var_ds.transpose("yc", "xc", "time")

        trend_files = \
            [v for k, v in var_files.items()
             if k.endswith("linear_trend")]
        trend_ds = None
        
        if len(trend_files) > 0:
            trend_ds = xr.open_mfdataset(
                trend_files,
                **ds_kwargs)

            trend_ds = trend_ds.transpose("yc", "xc", "time")

        args = [
            self._channels,
            self._dtype,
            self._loss_weight_days,
            self._meta_channels,
            self._missing_dates,
            self._n_forecast_days,
            self.num_channels,
            self._shape,
            self._trend_steps,
            self._masks,
            prediction
        ]

        x, y, sw = generate_sample(date,
                                   var_ds,
                                   var_files,
                                   trend_ds,
                                   *args)
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
    (channels,
     dtype,
     loss_weight_days,
     meta_channels,
     missing_dates,
     n_forecast_days,
     num_channels,
     shape,
     trend_steps,
     masks,
     prediction) = args

    ds_kwargs = dict(
        chunks=dict(time=1, yc=shape[0], xc=shape[1]),
        drop_variables=["month", "plev", "realization"],
        parallel=True,
    )

    var_ds = xr.open_mfdataset(
        [v for k, v in var_files.items()
         if k not in meta_channels and not k.endswith("linear_trend")],
        **ds_kwargs)
    var_ds = var_ds.transpose("yc", "xc", "time")

    trend_files = [v for k, v in var_files.items()
                   if k.endswith("linear_trend")]
    trend_ds = None

    if len(trend_files):
        trend_ds = xr.open_mfdataset(
            trend_files,
            **ds_kwargs)
        trend_ds = trend_ds.transpose("yc", "xc", "time")

    with tf.io.TFRecordWriter(path) as writer:
        for date in dates:
            start = time.time()

            try:
                x, y, sample_weights = generate_sample(date,
                                                       var_ds,
                                                       var_files,
                                                       trend_ds,
                                                       *args)
                if not dry:
                    x[da.isnan(x)] = 0.

                    x, y, sample_weights = dask.compute(x, y, sample_weights,
                                                        optimize_graph=True)
                    write_tfrecord(writer,
                                   x, y, sample_weights)
                count += 1
            except IceNetDataWarning:
                continue

            end = time.time()
            times.append(end - start)
            logging.debug("Time taken to produce {}: {}".
                          format(date, times[-1]))
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
                    n_forecast_days: int,
                    num_channels: int,
                    shape: object,
                    trend_steps: object,
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
    :param n_forecast_days:
    :param num_channels:
    :param shape:
    :param trend_steps:
    :param masks:
    :param prediction:
    :return:
    """

    # Prepare data sample
    # To become array of shape (*raw_data_shape, n_forecast_days)
    forecast_dts = [forecast_date + dt.timedelta(days=n)
                    for n in range(n_forecast_days)]

    y = da.zeros((*shape, n_forecast_days, 1), dtype=dtype)
    sample_weights = da.zeros((*shape, n_forecast_days, 1), dtype=dtype)

    if not prediction:
        try:
            sample_output = var_ds.siconca_abs.sel(time=forecast_dts)
        except KeyError as sic_ex:
            logging.exception("Issue selecting data for non-prediction sample, "
                              "please review siconca ground-truth: dates {}".
                              format(forecast_dts))
            raise RuntimeError(sic_ex)
        y[:, :, :, 0] = sample_output

    # Masked recomposition of output
    for leadtime_idx in range(n_forecast_days):
        forecast_day = forecast_date + dt.timedelta(days=leadtime_idx)

        if any([forecast_day == missing_date
                for missing_date in missing_dates]):
            sample_weight = da.zeros(shape, dtype)
        else:
            # Zero loss outside of 'active grid cells'
            sample_weight = masks[forecast_day.month - 1]
            sample_weight = sample_weight.astype(dtype)

            # We can pick up nans, which messes up training
            sample_weight[da.isnan(y[..., leadtime_idx, 0])] = 0

            # Scale the loss for each month s.t. March is
            #   scaled by 1 and Sept is scaled by 1.77
            if loss_weight_days:
                sample_weight *= 33928. / sample_weight.sum()

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
            if type(trend_steps) == list:
                channel_dates = [pd.Timestamp(forecast_date +
                                              dt.timedelta(days=int(n)))
                                 for n in trend_steps]
            else:
                channel_dates = [pd.Timestamp(forecast_date +
                                              dt.timedelta(days=n))
                                 for n in range(num_channels)]
        else:
            channel_ds = var_ds
            channel_dates = [pd.Timestamp(forecast_date - dt.timedelta(days=n))
                             for n in range(num_channels)]

        channel_data = []
        for cdate in channel_dates:
            try:
                channel_data.append(getattr(channel_ds, var_name).
                                    sel(time=cdate))
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

    #    x.visualize(filename='x.svg', optimize_graph=True)
    #    y.visualize(filename='y.svg', optimize_graph=True)
    #    sample_weights.visualize(filename='sample_weights.svg', optimize_graph=True)
    #    import sys
    #    sys.exit(0)

    return x, y, sample_weights



