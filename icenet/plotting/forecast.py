import argparse
import logging
import os
import sys

from datetime import timedelta
from dateutil.relativedelta import relativedelta

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_pdf import PdfPages

import seaborn as sns

import numpy as np
import pandas as pd
import dask.array as da
import xarray as xr

from download_toolbox.dataset import DatasetConfig
from download_toolbox.interface import get_dataset_config_implementation, get_implementation
from icenet.process.metrics import compute_binary_accuracy, compute_sea_ice_extent_error, compute_metrics, \
    compute_metrics_leadtime_avg

from icenet import __version__ as icenet_version
from icenet.plotting.utils import (get_forecast_obs_data,
                                   get_seas_forecast_da, get_forecast_data,
                                   get_seas_forecast_init_dates, show_img,
                                   get_plot_axes, process_probes,
                                   process_regions)
from icenet.plotting.video import xarray_to_video
from icenet.plotting.cli import ForecastPlotArgParser, parse_metrics_arg

cm = mpl.colormaps


def plot_binary_accuracy(masks: object,
                         fc_da: object,
                         cmp_da: object,
                         obs_da: object,
                         output_path: object,
                         threshold: float = 0.15) -> object:
    """
    Compute and plot the binary class accuracy of a forecast,
    where we consider a binary class prediction of ice with SIC > 15%.
    In particular, we compute the mean percentage of correct
    classifications over the active grid cell area.

    :param masks: an icenet Masks object
    :param fc_da: the forecasts given as an xarray.DataArray object
                  with leadtime, xc, yc coordinates
    :param cmp_da: a comparison forecast / sea ice data given as an
                   xarray.DataArray object with leadtime, xc, yc coordinates.
                   If None, will ignore plotting a comparison forecast
    :param obs_da: the "ground truth" given as an xarray.DataArray object
                   with leadtime, xc, yc coordinates
    :param output_path: string specifying the path to store the plot
    :param threshold: the SIC threshold of interest (in percentage as a fraction),
                      i.e. threshold is between 0 and 1

    :return: tuple of (binary accuracy for forecast (fc_da),
                       binary accuracy for comparison (cmp_da))
    """
    binacc_fc = compute_binary_accuracy(masks=masks,
                                        fc_da=fc_da,
                                        obs_da=obs_da,
                                        threshold=threshold)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title(
        f"Binary accuracy comparison (threshold SIC = {threshold*100}%)")

    ax.plot(fc_da.forecast_date.values, binacc_fc.values, label="IceNet Leadtime")

    if cmp_da is not None:
        binacc_cmp = compute_binary_accuracy(masks=masks,
                                             fc_da=cmp_da,
                                             obs_da=obs_da,
                                             threshold=threshold)
        ax.plot(fc_da.forecast_date.values, binacc_cmp.values, label="SEAS")
    else:
        binacc_cmp = None

    ax.set_ylabel("Binary accuracy (%)")
    ax.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    # TODO: review, likely not going to want these with longer forecasts
    ax.xaxis.set_minor_locator(mdates.DayLocator())
    ax.set_xlabel("Date")
    ax.legend(loc='lower right')

    output_path = os.path.join("plots", "binacc.png") \
        if not output_path else output_path
    logging.info(f"Saving to {output_path}")
    plt.savefig(output_path)

    return binacc_fc, binacc_cmp


def plot_sea_ice_extent_error(masks: object,
                              fc_da: object,
                              cmp_da: object,
                              obs_da: object,
                              output_path: object,
                              threshold: float = 0.15) -> object:
    """
    Compute and plot sea ice extent (SIE) error of a forecast, where SIE error is
    defined as the total area covered by grid cells with SIC > (threshold*100)%.

    :param masks: an icenet Masks object
    :param fc_da: the forecasts given as an xarray.DataArray object
                  with time, xc, yc coordinates
    :param cmp_da: a comparison forecast / sea ice data given as an
                   xarray.DataArray object with time, xc, yc coordinates.
                   If None, will ignore plotting a comparison forecast
    :param obs_da: the "ground truth" given as an xarray.DataArray object
                   with time, xc, yc coordinates
    :param output_path: string specifying the path to store the plot
    :param threshold: the SIC threshold of interest (in percentage as a fraction),
                      i.e. threshold is between 0 and 1

    :return: tuple of (SIE error for forecast (fc_da), SIE error for comparison (cmp_da))
    """
    forecast_sie_error = compute_sea_ice_extent_error(
        masks=masks,
        fc_da=fc_da,
        obs_da=obs_da,
        threshold=threshold)

    fig, ax = plt.subplots(figsize=(12, 6))
    grid_area_size = float(abs(fc_da.xc[1] - fc_da.xc[0]))
    ax.set_title(f"SIE error comparison ({grid_area_size} km grid resolution) "
                 f"(threshold SIC = {threshold*100}%)")
    ax.plot(forecast_sie_error.forecast_date.values, forecast_sie_error.values, label="IceNet")

    if cmp_da is not None:
        cmp_sie_error = compute_sea_ice_extent_error(
            masks=masks,
            fc_da=cmp_da,
            obs_da=obs_da,
            threshold=threshold)
        ax.plot(forecast_sie_error.forecast_date.values, cmp_sie_error.values, label="SEAS")
    else:
        cmp_sie_error = None

    ax.set_ylabel("Sea ice extent error (km)")
    ax.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    # TODO: review, likely not going to want these with longer forecasts
    ax.xaxis.set_minor_locator(mdates.DayLocator())
    ax.set_xlabel("Date")
    ax.legend(loc='lower right')

    output_path = os.path.join("plots", "sie_error.png") \
        if not output_path else output_path
    logging.info(f"Saving to {output_path}")
    plt.savefig(output_path)

    return forecast_sie_error, cmp_sie_error


def plot_metrics(metrics: object,
                 masks: object,
                 fc_da: object,
                 cmp_da: object,
                 obs_da: object,
                 output_path: object,
                 separate: bool = False) -> object:
    """
    Computes metrics which are passed in as a list of strings,
    and plots them for each forecast.
    Returns a dictionary where the keys are the metrics,
    and the values are the computed metrics.

    :param metrics: a list of strings
    :param masks: an icenet Masks object
    :param fc_da: an xarray.DataArray object with time, xc, yc coordinates
    :param cmp_da: a comparison forecast / sea ice data given as an
                   xarray.DataArray object with time, xc, yc coordinates.
                   If None, will ignore plotting a comparison forecast
    :param obs_da: an xarray.DataArray object with time, xc, yc coordinates
    :param output_path: string specifying the path to store the plot(s).
                        If separate=True, this should be a directory
    :param separate: bool specifying whether there is a plot created for
                     each metric (True) or not (False), default is False

    :return: dictionary with keys as metric names and values as
             xarray.DataArray's storing the computed metrics for each forecast
    """
    # compute metrics
    fc_metric_dict = compute_metrics(metrics=metrics,
                                     masks=masks,
                                     fc_da=fc_da,
                                     obs_da=obs_da)
    if cmp_da is not None:
        cmp_metric_dict = compute_metrics(metrics=metrics,
                                          masks=masks,
                                          fc_da=cmp_da,
                                          obs_da=obs_da)
    else:
        cmp_metric_dict = None

    if separate:
        # produce separate plots for each metric
        for metric in metrics:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.set_title(f"{metric.upper()} comparison")
            ax.plot(fc_metric_dict[metric].forecast_date.values,
                    fc_metric_dict[metric].values,
                    label="IceNet")
            if cmp_metric_dict is not None:
                ax.plot(cmp_metric_dict[metric].forecast_date.values,
                        cmp_metric_dict[metric].values,
                        label="SEAS")

            ax.xaxis.set_major_formatter(
                mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.xaxis.set_minor_locator(mdates.DayLocator())
            ax.legend(loc='lower right')

            outpath = os.path.join("plots", f"{metric}.png") \
                if not output_path else os.path.join(output_path, f"{metric}.png")
            logging.info(f"Saving to {outpath}")
            plt.savefig(outpath)
    else:
        # produce one plot for all metrics
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_title("Metric comparison")
        for metric in metrics:
            ax.plot(fc_metric_dict[metric].leadtime,
                    fc_metric_dict[metric].values,
                    label=f"IceNet {metric.upper()}")
            if cmp_metric_dict is not None:
                ax.plot(cmp_metric_dict[metric].leadtime,
                        cmp_metric_dict[metric].values,
                        label=f"SEAS {metric.upper()}",
                        linestyle="dotted")

        ax.set_ylabel("SIC (%)")
        ax.xaxis.set_major_formatter(
            mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_minor_locator(mdates.DayLocator())
        ax.set_xlabel("Date")
        ax.legend(loc='lower right')

        output_path = os.path.join("plots", "metrics.png") \
            if not output_path else output_path
        logging.info(f"Saving to {output_path}")
        plt.savefig(output_path)

    return fc_metric_dict, cmp_metric_dict


def standard_deviation_heatmap(metric: str,
                               model_name: str,
                               metrics_df: pd.DataFrame,
                               average_over: str,
                               target_date_avg: bool = False,
                               output_path: str = None,
                               fc_std_metric: pd.DataFrame = None,
                               vmax: float = None,
                               date_format: str = None,
                               **kwargs) -> object:
    """
    Produces a heatmap plot of the leadtime standard deviation of a metric.

    :param metric: string specifying which metric to compute
    :param model_name: string specifying the name of the model or forecast
    :param metrics_df: pandas dataframe with columns 'date', 'leadtime' and the metric name
    :param average_over: string to specify how to average the metrics.
                         If average_over="all", averages over all possible
                         forecasts and produces line plot.
                         If average_over="month" or "day", averages
                         over the month or day respectively and produces
                         heat map plot.
    :param target_date_avg: bool to indiciate whether or not to average over
                            the target date of forecast (as opposed to averaging
                            over the initialisation date), by default False
    :param output_path: string specifying the path to store the plot, default None
    :param fc_std_metric: pandas dataframe of the standard deviation of the metric over leadtime.
                          If None, this will be computed here
    :param vmax: float specifying maximum to anchor the colourmap. If None,
                 this is inferred from the data
    :param date_format: str the plot output format for dates

    :return: dataframe of the standard deviation of the metric over leadtime
    """
    logging.info(f"Creating standard deviation over leadtime plot for "
                 f"{metric} metric for {model_name} forecasts")

    if average_over == "day":
        groupby_col = "dayofyear"
    else:
        groupby_col = "month"
    if target_date_avg:
        groupby_col = "target_" + groupby_col

    if fc_std_metric is None:
        # compute standard deviation of metric
        fc_std_metric = metrics_df.groupby([groupby_col, "leadtime"]).std(numeric_only=True).\
            reset_index().pivot(index=groupby_col, columns="leadtime", values=metric).\
            sort_values(groupby_col, ascending=True)
    n_forecast_steps = fc_std_metric.shape[1]

    # set ylabel (if average_over == "all"), or legend label (otherwise)
    if metric in ["mae", "mse", "rmse"]:
        ylabel = f"SIC {metric.upper()} (%)"
    elif metric == "binacc":
        ylabel = "Binary accuracy (%)"
    elif metric == "sie":
        ylabel = "SIE error (km)"

    # plot heatmap of standard deviation for IceNet
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(data=fc_std_metric,
                ax=ax,
                vmin=0,
                vmax=vmax,
                cmap="mako_r",
                cbar_kws=dict(label=f"Standard deviation of {ylabel}"))

    # y-axis
    labels = _heatmap_ylabels(metrics_df=metrics_df,
                              average_over=average_over,
                              groupby_col=groupby_col)
    ax.set_yticks(np.arange(len(metrics_df[groupby_col].unique())) + 0.5)
    ax.set_yticklabels(labels)
    plt.yticks(rotation=0)
    if target_date_avg:
        ax.set_ylabel("Target date of forecast")
    else:
        ax.set_ylabel("Initialisation date of forecast")

    # x-axis
    ax.set_xticks(np.arange(30, n_forecast_steps, 30))
    ax.set_xticklabels(np.arange(30, n_forecast_steps, 30))
    plt.xticks(rotation=0)
    ax.set_xlabel("Lead time (days)")

    # add plot title
    (start_date, end_date) = (metrics_df["date"].min().strftime(date_format),
                              metrics_df["date"].max().strftime(date_format))
    time_coverage = "\nStandard deviation over a minimum of " + \
        f"{round((metrics_df[groupby_col].value_counts()/n_forecast_steps).min())} " + \
        f"forecasts between {start_date} - {end_date}"
    if metric in ["mae", "mse", "rmse"]:
        title = f"{metric.upper()} comparison"
    elif metric == "binacc":
        title = f"Binary accuracy comparison (threshold SIC = {kwargs['threshold'] * 100}%)"
    elif metric == "sie":
        title = f"SIE error comparison ({kwargs['grid_area_size']} km grid resolution, " +\
            f"threshold SIC = {kwargs['threshold'] * 100}%)"
    ax.set_title(title + f" ({model_name})" + time_coverage)

    # save plot
    targ = "target" if target_date_avg and average_over != "all" else "init"
    filename = f"leadtime_averaged_{targ}_{average_over}_{metric}_{model_name}_std.png"
    output_path = os.path.join("plots", filename) \
        if not output_path else output_path
    logging.info(f"Saving to {output_path}")
    plt.savefig(output_path)

    return fc_std_metric


def plot_metrics_leadtime_avg(metric: str,
                              forecast_file: str,
                              ds_config_path: os.PathLike,
                              output_path: str,
                              average_over: str,
                              compare_against: DatasetConfig | None = None,
                              plot_std: bool = False,
                              std_mult: float = 1.0,
                              data_path: str = None,
                              target_date_avg: bool = False,
                              bias_correct: bool = False,
                              region: tuple = None,
                              **kwargs) -> object:
    """
    Plots leadtime averaged metrics either using all the forecasts
    in the forecast file, or averaging them over by month or day.

    :param metric: string specifying which metric to compute
    :param forecast_file: a path to a .nc file
    :param ds_config: dataset config appropriate for the forecast file
    :param compare_against:
    :param output_path: string specifying the path to store the plot
    :param average_over: string to specify how to average the metrics.
                         If average_over="all", averages over all possible
                         forecasts and produces line plot.
                         If average_over="month" or "day", averages
                         over the month or day respectively and produces
                         heat map plot.
    :param target_date_avg: bool to indiciate whether or not to average over
                            the target date of forecast (as opposed to averaging
                            over the initialisation date), by default False
    :param data_path: string specifying a CSV file where metrics dataframe
                      could be loaded from. If loading in the dataframe is
                      not possible, it will compute the metrics dataframe
                      and try to save the dataframe
    :param bias_correct: bool to indicate whether or not to
                         perform a bias correction on comparison forecast,
                         by default False.
    :param region: region to zoom in to
    :param kwargs: any keyword arguments that are required for the computation
                   of the metric, e.g. 'threshold' for SIE error and binary accuracy
                   metrics, or 'grid_area_size' for SIE error metric

    :return: pandas dataframe with columns 'date', 'leadtime' and the metric name
    """
    implemented_metrics = ["binacc", "sie", "mae", "mse", "rmse"]

    if metric not in implemented_metrics:
        raise NotImplementedError(
            f"{metric} metric has not been implemented. "
            f"Please only choose out of {implemented_metrics}.")
    if metric == "binacc":
        # add default kwargs
        if "threshold" not in kwargs.keys():
            kwargs["threshold"] = 0.15
    elif metric == "sie":
        # add default kwargs
        if "threshold" not in kwargs.keys():
            kwargs["threshold"] = 0.15
    elif metric in ["mae", "mse", "rmse"]:
        # remove threshold kwargs if passed
        if "threshold" in kwargs.keys():
            del kwargs["threshold"]

    do_compute_metrics = True
    if data_path is not None:
        # loading in precomputed dataframes for the metrics
        logging.info(
            f"Attempting to read in metrics dataframe from {data_path}")
        try:
            metric_df = pd.read_csv(data_path)
            metric_df["date"] = pd.to_datetime(metric_df["date"])
            do_compute_metrics = False
        except OSError:
            logging.info(
                f"Couldn't load in dataframe from {data_path}, "
                f"will compute metric dataframe and try save to {data_path}")

    if do_compute_metrics:
        # computing the dataframes for the metrics
        # will save dataframe in data_path if data_path is not None
        metric_df = compute_metrics_leadtime_avg(metric=metric,
                                                 forecast_file=forecast_file,
                                                 ds_config_path=ds_config_path,
                                                 compare_against=compare_against,
                                                 data_path=data_path,
                                                 bias_correct=bias_correct,
                                                 region=region,
                                                 **kwargs)

    fc_metric_df = metric_df[metric_df["forecast_name"] == "IceNet"]
    seas_metric_df = metric_df[metric_df["forecast_name"] == "SEAS"]
    seas_metric_df = seas_metric_df \
        if (len(seas_metric_df) != 0) and compare_against else None

    ds_config = get_dataset_config_implementation(ds_config_path)
    logging.info(f"Creating leadtime averaged plot for {metric} metric")
    fig, ax = plt.subplots(figsize=(12, 6))
    (start_date, end_date) = (fc_metric_df["date"].min().strftime(ds_config.frequency.plot_format),
                              fc_metric_df["date"].max().strftime(ds_config.frequency.plot_format))

    # set ylabel (if average_over == "all"), or legend label (otherwise)
    if metric in ["mae", "mse", "rmse"]:
        ylabel = f"SIC {metric.upper()} (%)"
    elif metric == "binacc":
        ylabel = "Binary accuracy (%)"
    elif metric == "sie":
        ylabel = "SIE error (km)"

    if average_over == "all":
        # averaging metric over leadtime for all forecasts
        fc_avg_metric = fc_metric_df.groupby("leadtime").mean(metric).\
            sort_values("leadtime", ascending=True)[metric]
        n_forecast_steps = fc_avg_metric.index.max()

        # plot leadtime averaged metrics
        ax.plot(fc_avg_metric.index,
                fc_avg_metric,
                label="IceNet",
                color="blue")
        if plot_std:
            # obtaining the standard deviation of the metric
            fc_std_metric = fc_metric_df.groupby("leadtime")[metric].std().\
                sort_index(ascending=True)
            ci = std_mult * fc_std_metric
            ax.fill_between(x=fc_avg_metric.index,
                            y1=fc_avg_metric - ci,
                            y2=fc_avg_metric + ci,
                            color="blue",
                            alpha=0.1)
        if seas_metric_df is not None:
            seas_avg_metric = seas_metric_df.groupby("leadtime").mean(metric).\
                sort_values("leadtime", ascending=True)[metric]
            ax.plot(seas_avg_metric.index,
                    seas_avg_metric,
                    label="SEAS",
                    color="darkorange")
            if plot_std:
                # obtaining the standard deviation of the metric
                seas_std_metric = seas_metric_df.groupby("leadtime")[metric].std().\
                    sort_index(ascending=True)
                ci = std_mult * seas_std_metric
                ax.fill_between(x=seas_avg_metric.index,
                                y1=seas_avg_metric - ci,
                                y2=seas_avg_metric + ci,
                                color="darkorange",
                                alpha=0.1)

        # string to add in plot title
        time_coverage = f"\nAveraged over {len(fc_metric_df['date'].unique())} " + \
            f"forecasts between {start_date} - {end_date}"
        if plot_std:
            time_coverage += f"\n(with +/-{std_mult} standard deviations)"
        ax.set_ylabel(ylabel)
        ax.legend(loc='lower right')
    elif average_over in ["day", "month"]:
        if average_over == "day":
            groupby_col = "dayofyear"
        else:
            groupby_col = "month"
        if target_date_avg:
            groupby_col = "target_" + groupby_col

        # compute metric by first grouping the dataframe by groupby_col and leadtime
        fc_avg_metric = fc_metric_df.groupby([groupby_col, "leadtime"]).mean(metric).\
            reset_index().pivot(index=groupby_col, columns="leadtime", values=metric).\
            sort_values(groupby_col, ascending=True)
        n_forecast_steps = fc_avg_metric.shape[1]

        if seas_metric_df is not None:
            # compute the difference in leadtime average to SEAS forecast
            seas_avg_metric = seas_metric_df.groupby([groupby_col, "leadtime"]).mean(metric).\
                reset_index().pivot(index=groupby_col, columns="leadtime", values=metric).\
                sort_values(groupby_col, ascending=True)
            heatmap_df_diff = fc_avg_metric - seas_avg_metric
            max = np.nanmax(np.abs(heatmap_df_diff.values))

            # plot heatmap of the difference between IceNet and SEAS
            sns.heatmap(
                data=heatmap_df_diff,
                ax=ax,
                vmax=max,
                vmin=-max,
                cmap="seismic_r" if metric in ["binacc", "sie"] else "seismic",
                cbar_kws=dict(
                    label=f"{ylabel} difference between IceNet and SEAS"))

        else:
            # plot heatmap of the leadtime averaged metric when grouped by groupby_col
            sns.heatmap(
                data=fc_avg_metric,
                ax=ax,
                cmap="inferno" if metric in ["binacc", "sie"] else "inferno_r",
                cbar_kws=dict(label=ylabel))

        # string to add in plot title
        time_coverage = "\nAveraged over a minimum of " + \
            f"{round((fc_metric_df[groupby_col].value_counts()/n_forecast_steps).min())} " + \
            f"forecasts between {start_date} - {end_date}"

        # y-axis
        labels = _heatmap_ylabels(metrics_df=fc_metric_df,
                                  average_over=average_over,
                                  groupby_col=groupby_col)
        ax.set_yticks(np.arange(len(fc_metric_df[groupby_col].unique())) + 0.5)
        ax.set_yticklabels(labels)

        plt.yticks(rotation=0)
        if target_date_avg:
            ax.set_ylabel("Target date of forecast")
        else:
            ax.set_ylabel("Initialisation date of forecast")
    else:
        raise NotImplementedError(
            f"averaging over {average_over} not a valid option.")

    # add plot title
    if metric in ["mae", "mse", "rmse"]:
        title = f"{metric.upper()} comparison"
    elif metric == "binacc":
        title = f"Binary accuracy comparison (threshold SIC = {kwargs['threshold'] * 100}%)"
    elif metric == "sie":
        # TODO: would be nice to have grid size here
        title = f"SIE error comparison, threshold SIC = {kwargs['threshold'] * 100}%)"
    ax.set_title(title + time_coverage)

    # x-axis
    ax.set_xticks(np.arange(30, n_forecast_steps, 30))
    ax.set_xticklabels(np.arange(30, n_forecast_steps, 30))
    plt.xticks(rotation=0)
    ax.set_xlabel(f"Lead time ({ds_config.frequency.name.lower()}s)")

    # save plot
    targ = "target" if target_date_avg and average_over != "all" else "init"
    filename = f"leadtime_averaged_{targ}_{average_over}_{metric}" + \
        ("_comp" if seas_metric_df is not None else "") + ".png"
    output_path = os.path.join("plots", filename) \
        if not output_path else output_path
    logging.info(f"Saving to {output_path}")
    plt.savefig(output_path)

    if plot_std and average_over in ["day", "month"]:
        # create heapmap for the standard deviation
        if seas_metric_df is not None:
            # compute the standard deviation of the metric for both the forecast and SEAS5
            fc_std_metric = fc_metric_df.groupby([groupby_col, "leadtime"]).std(numeric_only=True).\
                reset_index().pivot(index=groupby_col, columns="leadtime", values=metric).\
                sort_values(groupby_col, ascending=True)
            seas_std_metric = seas_metric_df.groupby([groupby_col, "leadtime"]).std(numeric_only=True).\
                reset_index().pivot(index=groupby_col, columns="leadtime", values=metric).\
                sort_values(groupby_col, ascending=True)
            # compute the maximum standard deviation to obtain a common scale
            vmax = np.nanmax([
                np.nanmax(fc_std_metric.values),
                np.nanmax(seas_std_metric.values)
            ])
            # create heapmap for the standard deviation
            standard_deviation_heatmap(metric=metric,
                                       model_name="SEAS",
                                       metrics_df=seas_metric_df,
                                       average_over=average_over,
                                       output_path=output_path.replace(
                                           ".png", "_SEAS_std.png"),
                                       target_date_avg=target_date_avg,
                                       fc_std_metric=seas_std_metric,
                                       vmax=vmax,
                                       date_format=ds_config.frequency.plot_format,
                                       **kwargs)
        else:
            # no need to pre-compute the scale for the heatmap, hence
            # pre-computing the standard deviation of the metric is not required
            fc_std_metric = None
            vmax = None
        standard_deviation_heatmap(metric=metric,
                                   model_name="IceNet",
                                   metrics_df=fc_metric_df,
                                   average_over=average_over,
                                   output_path=output_path.replace(
                                       ".png", "_IceNet_std.png"),
                                   target_date_avg=target_date_avg,
                                   fc_std_metric=fc_std_metric,
                                   vmax=vmax,
                                   date_format=ds_config.frequency.plot_format,
                                   **kwargs)

    return fc_metric_df, seas_metric_df


def sic_error_video(fc_da: object,
                    obs_da: object,
                    obs_ds_config: object,
                    land_mask: object,
                    output_path: object) -> object:
    """

    :param fc_da:
    :param obs_da:
    :param obs_ds_config:
    :param land_mask:
    :param output_path:

    :return: matplotlib animation
    """

    diff = fc_da - obs_da
    fig, maps = plt.subplots(nrows=1, ncols=3, figsize=(16, 6), layout="tight")
    fig.set_dpi(150)

    leadtime = 1
    fc_plot = fc_da.isel(leadtime=leadtime).to_numpy()
    obs_plot = obs_da.isel(leadtime=leadtime).to_numpy()
    diff_plot = diff.isel(leadtime=leadtime).to_numpy()

    upper_bound = np.max(
        [np.abs(np.nanmin(diff_plot)),
         np.abs(np.nanmax(diff_plot))])
    diff_vmin = -upper_bound
    diff_vmax = upper_bound
    logging.debug("Bounds of differences: {} - {}".format(
        diff_vmin, diff_vmax))

    sic_cmap = cm.get_cmap("Blues_r")
    contour_kwargs = dict(vmin=0, vmax=1, cmap=sic_cmap)

    diff_cmap = cm.get_cmap("RdBu_r")
    im1 = maps[0].imshow(fc_plot, **contour_kwargs)
    im2 = maps[1].imshow(obs_plot, **contour_kwargs)
    im3 = maps[2].imshow(diff_plot,
                         vmin=diff_vmin,
                         vmax=diff_vmax,
                         cmap=diff_cmap)

    tic = maps[0].set_title(
        "IceNet "
        f"{pd.to_datetime(fc_da.isel(leadtime=leadtime).time.values).strftime(obs_ds_config.frequency.plot_format)}"
    )
    tio = maps[1].set_title(
        "OSISAF Obs "
        f"{pd.to_datetime(obs_da.isel(leadtime=leadtime).time.values).strftime(obs_ds_config.frequency.plot_format)}"
    )
    maps[2].set_title("Diff")

    p0 = maps[0].get_position().get_points().flatten()
    p1 = maps[1].get_position().get_points().flatten()
    p2 = maps[2].get_position().get_points().flatten()

    ax_cbar = fig.add_axes([p0[0] - 0.05, 0.04, p1[2] - p0[0], 0.02])
    plt.colorbar(im1, orientation='horizontal', cax=ax_cbar)

    ax_cbar1 = fig.add_axes([p2[0] + 0.05, 0.04, p2[2] - p2[0], 0.02])
    plt.colorbar(im3, orientation='horizontal', cax=ax_cbar1)

    for m_ax in maps[0:3]:
        m_ax.tick_params(
            labelbottom=False,
            labelleft=False,
        )
        m_ax.contourf(land_mask,
                      levels=[.5, 1],
                      cmap=cm.get_cmap("gray"),
                      zorder=3)

    # TODO: tied to daily forecasting date representations
    def update(lt_idx):
        date = "{} + {} {}s".format(
            pd.to_datetime(fc_da.time.values).strftime(obs_ds_config.frequency.plot_format),
            lt_idx,
            obs_ds_config.frequency.name.lower())
        logging.debug(f"Plotting {date}")

        fc_plot = fc_da.isel(leadtime=lt_idx).to_numpy()
        obs_plot = obs_da.isel(leadtime=lt_idx).to_numpy()
        diff_plot = diff.isel(leadtime=lt_idx).to_numpy()

        tic.set_text("IceNet {}".format(date))
        tio.set_text("Observation {}".format(date))

        im1.set_data(fc_plot)
        im2.set_data(obs_plot)
        im3.set_data(diff_plot)

        return tic, tio, im1, im2, im3

    animation = FuncAnimation(fig,
                              update,
                              range(0, len(fc_da.leadtime)),
                              interval=100)

    plt.close()

    output_path = os.path.join("plots", "sic_error.mp4") \
        if not output_path else output_path
    logging.info(f"Saving to {output_path}")
    animation.save(output_path, fps=2) # TODO: needs to be optional, extra_args=['-vcodec', 'libx264'])
    return animation


def sic_error_local_header_data(da: xr.DataArray):
    n_probe = len(da.probe)
    return {
        "probe array index": {
            i_probe: (f"{da.xi.values[i_probe]},"
                      f"{da.yi.values[i_probe]}")
            for i_probe in range(n_probe)
        },
        "probe location (EASE)": {
            i_probe: (f"{da.xc.values[i_probe]},"
                      f"{da.yc.values[i_probe]}")
            for i_probe in range(n_probe)
        },
        "probe location (lat, lon)": {
            i_probe: (f"{da.lat.values[i_probe]},"
                      f"{da.lon.values[i_probe]}")
            for i_probe in range(n_probe)
        },
        "obs_kind": {
            0: "forecast",
            1: "observation",
            2: "forecast error ('0' - '1')",
            # optionally, include SEAS comparison
        },
    }


def sic_error_local_write_fig(combined_da: xr.DataArray, output_prefix: str):
    """A helper function for `sic_error_local_plots`: plot error and
    forecast/observation data.

    :param combined_da: A DataArray with dims ('time', 'probe', 'obs_kind')

    :param output_prefix: A string from which to produce the output
    path (the probe index and file extension will be appended).
    """

    OBS_KIND_FC = 0
    OBS_KIND_OBS = 1
    OBS_KIND_ERR = 2

    plot_series = combined_da.to_dataframe(name="SIC")["SIC"]

    outfile = output_prefix + ".fc.pdf"

    all_figs = []
    with PdfPages(outfile) as output_pdf:
        n_probe = len(combined_da.probe)
        for i_probe in range(n_probe):

            #### Forecast/observed plots ####

            lat = combined_da.probe.lat.values[i_probe]
            lon = combined_da.probe.lon.values[i_probe]

            lat_h = "N" if lat >= 0.0 else "S"
            lat = abs(lat)

            lon_h = "E" if lon >= 0.0 else "W"
            lon = abs(lon)

            fig, ax = plt.subplots()
            all_figs.append(fig)

            ax.set_title(f"Sea ice concentration at location {i_probe + 1}\n"
                         f"{lat:.3f}° {lat_h}, {lon:.3f}° {lon_h}")

            ax.set_xlabel("Date")
            ax.set_ylabel("SIC (%)")

            # dims: (obs_kind, time, probe)
            ax.plot(plot_series.loc[OBS_KIND_FC, :, i_probe], label="IceNet")
            ax.plot(plot_series.loc[OBS_KIND_OBS, :, i_probe],
                    label="Observed")
            ax.legend()

            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            output_pdf.savefig(fig, bbox_inches='tight')

            #### Error plot ####
            fig2, ax2 = plt.subplots()
            all_figs.append(fig2)

            ax2.set_title(
                f"Sea ice concentration error at location {i_probe + 1}\n"
                f"{lat:.3f}° {lat_h}, {lon:.3f}° {lon_h}")

            ax2.set_xlabel("Date")
            ax2.set_ylabel("SIC error (%)")

            ax2.axhline(color='k', lw=0.5, ls='--')
            ax2.plot(plot_series.loc[OBS_KIND_ERR, :, i_probe], color='C2')

            plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
            output_pdf.savefig(fig2, bbox_inches='tight')

    return all_figs


def sic_error_local_plots(fc_da: object,
                          obs_da: object,
                          output_path: object,
                          as_command: bool = False):
    """
    :param fc_da: a DataArray with dims ('time', 'probe')
    :param obs_da: a DataArray with dims ('time', 'probe')
    """

    # convert SIC to percentages (ranging from 0% to 100%) rather than a fraction
    fc_da = fc_da * 100
    obs_da = obs_da * 100
    err_da = (fc_da - obs_da)
    combined_da = xr.concat([fc_da, obs_da, err_da],
                            dim="obs_kind",
                            coords="minimal")

    # convert to a dataframe for csv output
    df = (
        combined_da.to_dataframe(name="SIC")
        # drop unneeded coords (lat, lon, xc, yc)
        .loc[:, "SIC"]
        # Convert mult-indices
        .unstack(2).unstack(0))

    if output_path is None:
        output_path = "sic_error_local.csv"

    header_info = sic_error_local_header_data(combined_da)

    header = "# icenet_plot_sic_error_local\n"
    header += f"# Part of Icenet, version {icenet_version}\n"
    header += "#\n"

    if as_command:
        header += "# Command output from \n"
        cmd = ' '.join([a.__repr__() for a in sys.argv])
        header += f"#   {cmd}\n"
        header += "#\n"

    header += "# Key\n"
    for header_kind, header_data in header_info.items():
        header += f"# {header_kind}\n"
        for k, v in header_data.items():
            header += f"#   {k}: {v}\n"

    with open(output_path, "w") as outfile:
        outfile.write(header)
        df.to_csv(outfile)

    figs = sic_error_local_write_fig(combined_da, "sic_error_local")

    return figs


##
# icenet.plotting.forecast CLI endpoints
#
# TODO: there is a lot of embedded logic in the CLI entrypoints (sometimes full implementations
#  hence why they're not in icenet.plotting.cli)
#
def binary_accuracy_cli():
    """
    Produces plot of the binary classification accuracy of forecasts.
    """
    ap = (ForecastPlotArgParser().allow_comparison().allow_threshold())
    args = ap.parse_args()

    fc, obs, masks = get_forecast_obs_data(args.forecast_file,
                                           args.obs_dataset_config,
                                           args.forecast_date)

    if args.cmp_dataset_config:
        seas = get_seas_forecast_da(
            seas_config_path=args.cmp_dataset_config,
            date=args.forecast_date,
            bias_correct=args.bias_correct) if args.cmp_dataset_config else None
    else:
        seas = None

    if args.region:
        seas, fc, obs, masks = process_regions(args.region,
                                               [seas, fc, obs, masks])

    plot_binary_accuracy(masks=masks,
                         fc_da=fc,
                         cmp_da=seas,
                         obs_da=obs,
                         output_path=args.output_path,
                         threshold=args.threshold)


def sie_error_cli():
    """
    Produces plot of the sea ice extent (SIE) error of forecasts.
    """
    ap = (ForecastPlotArgParser().allow_comparison().allow_threshold())
    args = ap.parse_args()

    fc, obs, masks = get_forecast_obs_data(args.forecast_file,
                                           args.obs_dataset_config,
                                           args.forecast_date)

    if args.cmp_dataset_config:
        # TODO: we need to detect and provide implementation specifics in these calls - not just SEAS
        seas = get_seas_forecast_da(
            seas_config_path=args.cmp_dataset_config,
            date=args.forecast_date,
            bias_correct=args.bias_correct) if args.cmp_dataset_config else None
    else:
        seas = None

    if args.region:
        seas, fc, obs, masks = process_regions(args.region,
                                               [seas, fc, obs, masks])

    plot_sea_ice_extent_error(masks=masks,
                              fc_da=fc,
                              cmp_da=seas,
                              obs_da=obs,
                              output_path=args.output_path,
                              threshold=args.threshold)


def forecast_cli():
    """CLI entry point for icenet_plot_forecast

    :return:
    """
    ap = ForecastPlotArgParser()
    ap.add_argument("-l",
                    "--leadtimes",
                    help="Leadtimes to output, multiple as CSV, range as n..n",
                    type=lambda s: [
                        int(i) for i in list(
                            s.split(",")
                            if "," in s else range(int(s.split("..")[0]),
                                                   int(s.split("..")[1]) + 1)
                            if ".." in s else [s])
                    ])
    ap.add_argument("-c",
                    "--no-coastlines",
                    help="Turn off cartopy integration",
                    action="store_true",
                    default=False)
    ap.add_argument("-f",
                    "--format",
                    help="Format to output in",
                    choices=("mp4", "png", "svg", "tiff"),
                    default="png")
    ap.add_argument("-n",
                    "--cmap-name",
                    help="Color map name if not wanting to use default",
                    default=None)
    ap.add_argument("-s",
                    "--stddev",
                    help="Plot the standard deviation from the ensemble",
                    action="store_true",
                    default=False)
    args = ap.parse_args()

    fc = get_forecast_data(args.forecast_file, args.forecast_date, stddev=args.stddev)
    ds_config = get_dataset_config_implementation(args.obs_dataset_config, dummy=True)
    fc = fc.transpose(..., "yc", "xc")

    output_path = "." if args.output_path is None else args.output_path

    if not os.path.isdir(output_path):
        logging.warning("No directory at: {}".format(output_path))
        os.makedirs(output_path)
    elif os.path.isfile(output_path):
        raise RuntimeError(
            "{} should be a directory and not existent...".format(output_path))

    forecast_name = "{}.{}".format(
        os.path.splitext(os.path.basename(args.forecast_file))[0],
        args.forecast_date)

    cmap_name = "BuPu_r" if args.stddev else "Blues_r"
    if args.cmap_name is not None:
        cmap_name = args.cmap_name

    logging.info("Using cmap {}".format(cmap_name))
    cmap = cm.get_cmap(cmap_name)
    cmap.set_bad("dimgrey")

    if args.region is not None:
        fc = process_regions(args.region, [fc])[0]

    vmax = 1.

    if args.stddev:
        vmax = float(fc.max())
        logging.info("Calculated vmax to be: {}".format(vmax))

    leadtimes = args.leadtimes \
        if args.leadtimes is not None \
        else list(range(1, int(max(fc.leadtime.values)) + 1))
    leadtime_attr = "{}s".format(ds_config.frequency.attribute)

    if args.format == "mp4":
        pred_da = fc.sel(leadtime=leadtimes)
        anim_args = dict(figsize=5)
        if not args.no_coastlines:
            logging.warning("Coastlines will not work with the current "
                            "implementation of xarray_to_video")

        output_filename = os.path.join(
            output_path,
            "{}.{}.{}{}".format(forecast_name,
                                args.forecast_date.strftime(ds_config.frequency.date_format),
                                "" if not args.stddev else "stddev.",
                                args.format))

        xarray_to_video(pred_da.swap_dims({"leadtime": "forecast_date"}),
                        fps=1,
                        cmap=cmap,
                        imshow_kwargs=dict(vmin=0., vmax=vmax)
                        if not args.stddev else None,
                        video_path=output_filename,
                        date_format=ds_config.frequency.plot_format,
                        time_attr="forecast_date",
                        **anim_args)
    else:
        for leadtime in leadtimes:
            pred_da = fc.sel(leadtime=leadtime)
            bound_args = dict(north=ds_config.location.north,
                              south=ds_config.location.south)

            if args.region is not None:
                bound_args.update(x1=args.region[0],
                                  x2=args.region[2],
                                  y1=args.region[1],
                                  y2=args.region[3])

            ax = get_plot_axes(pred_da,
                               **bound_args,
                               do_coastlines=not args.no_coastlines)

            bound_args.update(cmap=cmap)

            im = show_img(ax,
                          pred_da,
                          **bound_args,
                          vmax=vmax,
                          do_coastlines=not args.no_coastlines)

            plt.colorbar(im, ax=ax)
            plot_date = args.forecast_date + relativedelta(**{leadtime_attr: leadtime})
            ax.set_title(plot_date.strftime(ds_config.frequency.plot_format))
            output_filename = os.path.join(
                output_path, "{}.{}.{}{}".format(
                    forecast_name,
                    (args.forecast_date +
                     relativedelta(**{leadtime_attr: leadtime})).strftime(ds_config.frequency.date_format),
                    "" if not args.stddev else "stddev.", args.format))

            logging.info("Saving to {}".format(output_filename))
            plt.savefig(output_filename)
            plt.clf()


def metric_cli():
    """
    Produces plot of requested metrics for forecasts.
    """
    ap = (ForecastPlotArgParser().allow_comparison().allow_metrics())
    args = ap.parse_args()

    fc, obs, masks = get_forecast_obs_data(args.forecast_file,
                                           args.obs_dataset_config,
                                           args.forecast_date)

    metrics = parse_metrics_arg(args.metrics)

    if args.cmp_dataset_config:
        seas = get_seas_forecast_da(
            seas_config_path=args.cmp_dataset_config,
            date=args.forecast_date,
            bias_correct=args.bias_correct) \
            if args.cmp_dataset_config else None
    else:
        seas = None

    if args.region:
        seas, fc, obs, masks = process_regions(args.region,
                                               [seas, fc, obs, masks])

    plot_metrics(metrics=metrics,
                 masks=masks,
                 fc_da=fc,
                 cmp_da=seas,
                 obs_da=obs,
                 output_path=args.output_path,
                 separate=args.separate)


def leadtime_avg_cli():
    """
    Produces plot of leadtime averaged metrics for forecasts.
    """
    ap = (ForecastPlotArgParser(
        forecast_date=False).allow_comparison().allow_threshold())
    ap.add_argument("-m",
                    "--metric",
                    help="Which metric to compute and plot",
                    type=str)
    ap.add_argument("-dp",
                    "--data_path",
                    help="Where to find (or store) metrics dataframe",
                    type=str,
                    default=None)
    ap.add_argument("-ao",
                    "--average_over",
                    help="How to average the forecast metrics",
                    type=str,
                    choices=["all", "month", "day"])
    ap.add_argument("-s",
                    "--std",
                    help="If flagged, standard deviation is plotted",
                    action="store_true",
                    default=False)
    ap.add_argument("-sm",
                    "--std_mult",
                    help="What multiple of the standard deviation to plot",
                    type=float,
                    default=1.0)
    ap.add_argument(
        "-td",
        "--target_date_average",
        help="Averages metric over target date instead of init date",
        action="store_true",
        default=False)
    args = ap.parse_args()

    plot_metrics_leadtime_avg(metric=args.metric,
                              forecast_file=args.forecast_file,
                              ds_config_path=args.obs_dataset_config,
                              compare_against=args.cmp_dataset_config,
                              output_path=args.output_path,
                              average_over=args.average_over,
                              plot_std=args.std,
                              std_mult=args.std_mult,
                              data_path=args.data_path,
                              target_date_avg=args.target_date_average,
                              bias_correct=args.bias_correct,
                              region=args.region,
                              threshold=args.threshold)


def sic_error_cli():
    """
    Produces video visualisation of SIC of forecast and ground truth.
    """
    ap = ForecastPlotArgParser()
    args = ap.parse_args()

    fc, obs, masks = get_forecast_obs_data(args.forecast_file,
                                           args.obs_dataset_config,
                                           args.forecast_date)
    ds_config = get_dataset_config_implementation(args.obs_dataset_config)

    if args.region:
        fc, obs, masks = process_regions(args.region, [fc, obs, masks])

    sic_error_video(fc_da=fc,
                    obs_da=obs,
                    obs_ds_config=ds_config,
                    land_mask=masks.land(),
                    output_path=args.output_path)


def sic_error_local_cli():
    """
    Entry point for the icenet_plot_sic_error_local command
    """

    ap = (ForecastPlotArgParser().allow_probes())
    args = ap.parse_args()

    fc, obs, _ = get_forecast_obs_data(args.forecast_file,
                                       args.obs_dataset_config,
                                       args.forecast_date)

    fc, obs = process_probes(args.probes, [fc, obs])

    sic_error_local_plots(fc, obs, args.output_path, as_command=True)


##
#   Utility functions
#
def _parse_day_of_year(dayofyear: int,
                       leapyear: bool = False) -> int:
    """
    Private function which takes in a day of year (integer or float) and returns
    the integer day of year. Useful for ensuring consistency over leap years,
    as dates after March could have different day of years due to leap years.
    For example, 01/03/00 is 60th day in 2000 but 01/03/01 is the 59th day in 2001.

    :param dayofyear: integer as int or float type
    :param leapyear: bool to indicate if we want to convert a leapyear dayofyear to
                     non-leapyear

    :return: int dayofyear
    """
    if leapyear:
        return (pd.Timestamp("2000-01-01") +
                timedelta(days=int(dayofyear) - 1)).strftime("%m-%d")
    else:
        return (pd.Timestamp("2001-01-01") +
                timedelta(days=int(dayofyear) - 1)).strftime("%m-%d")


def _heatmap_ylabels(metrics_df: pd.DataFrame,
                     average_over: str,
                     groupby_col: str) -> object:
    """
    Private function to return the labels for the y-axis in heatmap plots.

    :param metrics_df: pandas dataframe with columns 'date', 'leadtime' and the metric name
    :param average_over: string to specify how to average the metrics.
                         If average_over="all", averages over all possible
                         forecasts and produces line plot.
                         If average_over="month" or "day", averages
                         over the month or day respectively and produces
                         heat map plot.
    :param groupby_col: string to specify how we are grouping the data.
                        If average_over="all", this is typically "dayofyear"
                        or "target_dayofyear".
                        If average_over="month" or "day", this is typically
                        "month" or "target_month".

    :return: list of labels for the y-axis
    """
    if average_over == "day":
        # only add labels to the start, end dates
        # and any days that represent the start of months
        days_of_interest = np.array([
            metrics_df[groupby_col].min(), 1, 32, 60, 91, 121, 152, 182, 213,
            244, 274, 305, 335, metrics_df[groupby_col].max()
        ])
        labels = [
            _parse_day_of_year(day) if day in days_of_interest else ""
            for day in sorted(metrics_df[groupby_col].unique())
        ]
    else:
        # find out what months have been plotted and add their names
        month_names = np.array([
            "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sept",
            "Oct", "Nov", "Dec"
        ])
        labels = [
            month_names[month - 1]
            for month in sorted(metrics_df[groupby_col].unique())
        ]

    return labels
