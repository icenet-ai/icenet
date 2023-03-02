import argparse
import datetime as dt
import logging
import os

from datetime import timedelta

import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.animation import FuncAnimation
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import dask.array as da

from icenet.data.cli import date_arg
from icenet.data.sic.mask import Masks
from icenet.plotting.utils import \
    filter_ds_by_obs, get_forecast_ds, get_obs_da, get_seas_forecast_da, \
    show_img, get_plot_axes
from icenet.plotting.video import xarray_to_video

# FIXME: This shouldn't be here, 300 dpi is print quality (not always required)
matplotlib.rcParams.update({
    'figure.facecolor': 'w',
    'figure.dpi': 300
})


def region_arg(argument: str):
    """type handler for region arguments with argparse

    :param argument:
    :return:
    """
    try:
        x1, y1, x2, y2 = tuple([int(s) for s in argument.split(",")])

        assert x2 > x1 and y2 > y1, "Region is not valid"
        return x1, y1, x2, y2
    except TypeError:
        raise argparse.ArgumentTypeError(
            "Region argument must be list of four integers")


def process_regions(region: tuple,
                    data: tuple) -> tuple:
    """

    :param region:
    :param data:
    :return:
    """

    assert len(region) == 4, "Region needs to be a list of four integers"
    x1, y1, x2, y2 = region
    assert x2 > x1 and y2 > y1, "Region is not valid"

    for idx, arr in enumerate(data):
        if arr is not None:
            data[idx] = arr[..., y1:y2, x1:x2]
    return data


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
                  with time, xc, yc coordinates
    :param cmp_da: a comparison forecast / sea ice data given as an 
                   xarray.DataArray object with time, xc, yc coordinates.
                   If None, will ignore plotting a comparison forecast
    :param obs_da: the "ground truth" given as an xarray.DataArray object
                   with time, xc, yc coordinates
    :param output_path: string specifying the path to store the plot
    :param threshold: the SIC threshold of interest (in percentage as a fraction),
                      i.e. threshold is between 0 and 1
    
    :return: tuple of (binary accuracy for forecast (fc_da), 
                       binary accuracy for comparison (cmp_da))
    """
    if (threshold < 0) or (threshold > 1):
        raise ValueError("threshold must be a float between 0 and 1")
    
    agcm = masks.get_active_cell_da(obs_da)
    binary_obs_da = obs_da > threshold

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title(f"Binary accuracy comparison (threshold SIC = {threshold*100}%)")

    binary_fc_da = fc_da > threshold
    binary_fc_da = (binary_fc_da == binary_obs_da).\
        astype(np.float16).weighted(agcm)
    binacc_fc = (binary_fc_da.mean(dim=['yc', 'xc']) * 100)
    ax.plot(binacc_fc.time, binacc_fc.values, label="IceNet")

    if cmp_da is not None:
        binary_cmp_da = cmp_da > threshold
        binary_cmp_da = (binary_cmp_da == binary_obs_da).\
            astype(np.float16).weighted(agcm)
        binacc_cmp = (binary_cmp_da.mean(dim=['yc', 'xc']) * 100)
        ax.plot(binacc_cmp.time, binacc_cmp.values, label="SEAS")
    else:
        binacc_cmp = None

    ax.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_minor_locator(mdates.DayLocator())
    ax.legend(loc='lower right')

    output_path = os.path.join("plot", "binacc.png") \
        if not output_path else output_path
    logging.info(f"Saving to {output_path}")
    plt.savefig(output_path)

    return binacc_fc, binacc_cmp


def plot_sea_ice_extent_error(masks: object,
                              fc_da: object,
                              cmp_da: object,
                              obs_da: object,
                              output_path: object,
                              grid_area_size: int = 25,
                              threshold: float = 0.15) -> object:
    """
    Compute sea ice extent (SIE) error of a forecast, where SIE is
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
    :param grid_area_size: the length of the sides of the grid (in km),
                           by default set to 25 (so area of grid is 25*25)
    :param threshold: the SIC threshold of interest (in percentage as a fraction),
                      i.e. threshold is between 0 and 1
    
    :return: tuple of (SIE for forecast (fc_da), SIE for comparison (cmp_da))
    """
    if (threshold < 0) or (threshold > 1):
        raise ValueError("threshold must be a float between 0 and 1")
    
    # obtain mask
    agcm = masks.get_active_cell_da(obs_da)
    
    # binary for observed (i.e. truth)
    binary_obs_da = obs_da > threshold
    binary_obs_weighted_da = binary_obs_da.astype(int).weighted(agcm)

    # binary for forecast
    binary_fc_da = fc_da > threshold
    binary_fc_weighted_da = binary_fc_da.astype(int).weighted(agcm)
    
    # sie error
    forecast_sie_error = (
        binary_fc_weighted_da.sum(['xc', 'yc']) -
        binary_obs_weighted_da.sum(['xc', 'yc'])
    ) * (grid_area_size**2)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title(f"SIE comparison ({grid_area_size} km grid resolution) "
                 f"(threshold SIC = {threshold*100}%)")
    ax.plot(forecast_sie_error.time, forecast_sie_error.values, label="IceNet")

    if cmp_da is not None:
        binary_cmp_da = cmp_da > threshold
        binary_cmp_weighted_da = binary_cmp_da.astype(int).weighted(agcm)
        cmp_sie_error = (
            binary_cmp_weighted_da.sum(['xc', 'yc']) -
            binary_obs_weighted_da.sum(['xc', 'yc'])
        ) * (grid_area_size**2)
        ax.plot(cmp_sie_error.time, cmp_sie_error.values, label="SEAS")
    else:
        cmp_sie_error = None

    ax.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_minor_locator(mdates.DayLocator())
    ax.legend(loc='lower right')

    output_path = os.path.join("plot", "sie_error.png") \
        if not output_path else output_path
    logging.info(f"Saving to {output_path}")
    plt.savefig(output_path)

    return forecast_sie_error, cmp_sie_error


def compute_metrics(metrics: object,
                    masks: object,
                    fc_da: object,
                    obs_da: object) -> object:
    """
    Computes metrics which are passed in as a list of strings.
    Returns a dictionary where the keys are the metrics,
    and the values are the computed metrics.

    :param metrics: a list of strings
    :param masks: an icenet Masks object
    :param fc_da: an xarray.DataArray object with time, xc, yc coordinates
    :param obs_da: an xarray.DataArray object with time, xc, yc coordinates
    
    :return: dictionary with keys as metric names and values as 
             xarray.DataArray's storing the computed metrics for each forecast
    """
    # check requested metrics have been implemented
    implemented_metrics = ['MAE', 'MSE', 'RMSE']
    for metric in metrics:
        if metric not in implemented_metrics:
            raise NotImplementedError(f"{metric} metric has not been implemented. "
                                      f"Please only choose out of {implemented_metrics}.")
    
    # obtain mask
    mask_da = masks.get_active_cell_da(obs_da)
    
    metric_dict = {}
    # compute raw error
    err_da = (fc_da-obs_da)*100
    if "MAE" in metrics:
        # compute absolute SIC errors
        abs_err_da = da.fabs(err_da)
        abs_weighted_da = abs_err_da.weighted(mask_da)
    if "MSE" in metrics or "RMSE" in metrics:
        # compute squared SIC errors
        square_err_da = err_da**2
        square_weighted_da = square_err_da.weighted(mask_da)
        
    for metric in metrics:
        if metric == "MAE":
            metric_dict[metric] = abs_weighted_da.mean(dim=['yc', 'xc'])
        elif metric == "MSE":
            if "MSE" not in metric_dict.keys():
                # might've already been computed if RMSE came first
                metric_dict["MSE"] = square_weighted_da.mean(dim=['yc', 'xc'])
        elif metric == "RMSE":
            if "MSE" not in metric_dict.keys():
                # check if MSE already been computed
                metric_dict["MSE"] = square_weighted_da.mean(dim=['yc', 'xc'])
            metric_dict[metric] = da.sqrt(metric_dict["MSE"])

    # only return metrics requested (might've computed MSE when computing RMSE)
    return {k: metric_dict[k] for k in metrics}
    
    
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
    :param separate: logical value specifying whether there is a plot created for
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
            ax.set_title(f"{metric} comparison")
            ax.plot(fc_metric_dict[metric].time,
                    fc_metric_dict[metric].values,
                    label="IceNet")
            if cmp_metric_dict is not None:
                ax.plot(cmp_metric_dict[metric].time,
                        cmp_metric_dict[metric].values,
                        label=f"SEAS")
                
            ax.xaxis.set_major_formatter(
                mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.xaxis.set_minor_locator(mdates.DayLocator())
            ax.legend(loc='lower right')
            
            outpath = os.path.join("plot", f"{metric}.png") \
                if not output_path else os.path.join(output_path, f"{metric}.png")
            logging.info(f"Saving to {outpath}")
            plt.savefig(outpath)
    else:
        # produce one plot for all metrics
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_title("Metric comparison")
        for metric in metrics:
            ax.plot(fc_metric_dict[metric].time,
                    fc_metric_dict[metric].values,
                    label=f"IceNet {metric}")
            if cmp_metric_dict is not None:
                ax.plot(cmp_metric_dict[metric].time,
                        cmp_metric_dict[metric].values,
                        label=f"SEAS {metric}",
                        linestyle="dotted")
        
        ax.xaxis.set_major_formatter(
            mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_minor_locator(mdates.DayLocator())
        ax.legend(loc='lower right')
        
        output_path = os.path.join("plot", "metrics.png") \
            if not output_path else output_path
        logging.info(f"Saving to {output_path}")
        plt.savefig(output_path)
    
    return fc_metric_dict, cmp_metric_dict


def sic_error_video(fc_da: object,
                    obs_da: object,
                    land_mask: object,
                    output_path: object) -> object:
    """

    :param fc_da:
    :param obs_da:
    :param land_mask:
    :param output_path:
    
    :returns: matplotlib animation
    """

    diff = fc_da - obs_da

    fig, maps = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))
    fig.set_dpi(150)

    leadtime = 0
    fc_plot = fc_da.isel(time=leadtime).to_numpy()
    obs_plot = obs_da.isel(time=leadtime).to_numpy()
    diff_plot = diff.isel(time=leadtime).to_numpy()

    contour_kwargs = dict(
        vmin=0,
        vmax=1,
        cmap='YlOrRd'
    )

    im1 = maps[0].imshow(fc_plot, **contour_kwargs)
    im2 = maps[1].imshow(obs_plot, **contour_kwargs)
    im3 = maps[2].imshow(diff_plot, 
                         vmin=-1, vmax=1, cmap="RdBu_r")

    tic = maps[0].set_title("IceNet "
                            f"{pd.to_datetime(fc_da.isel(time=leadtime).time.values).strftime('%d/%m/%Y')}")
    tio = maps[1].set_title("OSISAF Obs "
                            f"{pd.to_datetime(obs_da.isel(time=leadtime).time.values).strftime('%d/%m/%Y')}")
    maps[2].set_title("Diff")

    p0 = maps[0].get_position().get_points().flatten()
    p1 = maps[1].get_position().get_points().flatten()
    p2 = maps[2].get_position().get_points().flatten()

    ax_cbar = fig.add_axes([p0[0], 0, p1[2]-p0[0], 0.05])
    plt.colorbar(im1, cax=ax_cbar, orientation='horizontal')

    ax_cbar1 = fig.add_axes([p2[0], 0, p2[2]-p2[0], 0.05])
    plt.colorbar(im3, cax=ax_cbar1, orientation='horizontal')

    for m_ax in maps[0:3]:
        m_ax.contourf(land_mask,
                      levels=[.5, 1],
                      colors=[matplotlib.cm.gray(180)],
                      zorder=3)

    fig.subplots_adjust(hspace=0.2, wspace=0.2)

    def update(date):
        logging.debug(f"Plotting {date}")

        fc_plot = fc_da.isel(time=date).to_numpy()
        obs_plot = obs_da.isel(time=date).to_numpy()
        diff_plot = diff.isel(time=date).to_numpy()
        
        tic.set_text("IceNet "
                     f"{pd.to_datetime(fc_da.isel(time=date).time.values).strftime('%d/%m/%Y')}")
        tio.set_text("OSISAF Obs "
                     f"{pd.to_datetime(obs_da.isel(time=date).time.values).strftime('%d/%m/%Y')}")

        im1.set_data(fc_plot)
        im2.set_data(obs_plot)
        im3.set_data(diff_plot)

        return tic, tio, im1, im2, im3

    animation = FuncAnimation(fig,
                              update,
                              range(0, len(fc_da.time)),
                              interval=100)

    plt.close()

    output_path = os.path.join("plot", "sic_error.mp4") \
        if not output_path else output_path
    logging.info(f"Saving to {output_path}")
    animation.save(output_path,
                   fps=10,
                   extra_args=['-vcodec', 'libx264'])
    return animation


def forecast_plot_args(ecmwf: bool = True,
                       threshold: bool = False,
                       sie: bool = False,
                       metrics: bool = False,
                       extra_args: object = None) -> object:
    """
    Process command line arguments.
    
    :param ecmwf:
    :param threshold:
    :param metrics:
    :param sie:
    :param extra_args:
    :return:
    """

    ap = argparse.ArgumentParser()
    ap.add_argument("hemisphere", choices=("north", "south"))
    ap.add_argument("forecast_file", type=str)
    ap.add_argument("forecast_date", type=date_arg)

    ap.add_argument("-r", "--region", default=None, type=region_arg,
                    help="Region specified x1, y1, x2, y2")

    ap.add_argument("-o", "--output-path", type=str, default=None)
    ap.add_argument("-v", "--verbose", action="store_true", default=False)

    if ecmwf:
        ap.add_argument("-b", "--bias-correct",
                        help="Bias correct SEAS forecast array",
                        action="store_true",
                        default=False)
        ap.add_argument("-e", "--ecmwf", action="store_true", default=False)

    if threshold:
        ap.add_argument("-t",
                        "--threshold",
                        help="The SIC threshold of interest",
                        type=float,
                        default=0.15)

    if sie:
        ap.add_argument("-ga",
                        "--grid-area",
                        help="The length of the sides of the grid used (in km)",
                        type=int,
                        default=25)

    if metrics:
        ap.add_argument("-m", 
                        "--metrics",
                        help="Which metrics to compute and plot",
                        type=str,
                        default="MAE,MSE,RMSE")
        ap.add_argument("-s",
                        "--separate",
                        help="Whether or not to produce separate plots for each metric",
                        action="store_true",
                        default=False)

    if type(extra_args) == list:
        for arg in extra_args:
            ap.add_argument(*arg[0], **arg[1])
    elif extra_args is not None:
        logging.warning("Implementation error: extra_args is invalid")

    args = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    return args

##
# CLI endpoints
#


def binary_accuracy():
    """
    Produces plot of the binary classification accuracy of forecasts.
    """
    args = forecast_plot_args(ecmwf=True, threshold=True)

    masks = Masks(north=args.hemisphere == "north",
                  south=args.hemisphere == "south")

    fc = get_forecast_ds(args.forecast_file,
                         args.forecast_date)
    obs = get_obs_da(args.hemisphere,
                     pd.to_datetime(args.forecast_date) +
                     timedelta(days=1),
                     pd.to_datetime(args.forecast_date) +
                     timedelta(days=int(fc.leadtime.max())))
    fc = filter_ds_by_obs(fc, obs, args.forecast_date)

    if args.ecmwf:
        seas = get_seas_forecast_da(
            args.hemisphere,
            args.forecast_date,
            bias_correct=args.bias_correct) \
            if args.ecmwf else None

        seas = seas.assign_coords(dict(xc=seas.xc / 1e3, yc=seas.yc / 1e3))
        seas = seas.isel(time=slice(1, None))
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


def sie_error():
    """
    Produces plot of the sea-ice extent (SIE) error of forecasts.
    """
    args = forecast_plot_args(ecmwf=True, threshold=True, sie=True)

    masks = Masks(north=args.hemisphere == "north",
                  south=args.hemisphere == "south")

    fc = get_forecast_ds(args.forecast_file,
                         args.forecast_date)
    obs = get_obs_da(args.hemisphere,
                     pd.to_datetime(args.forecast_date) +
                     timedelta(days=1),
                     pd.to_datetime(args.forecast_date) +
                     timedelta(days=int(fc.leadtime.max())))
    fc = filter_ds_by_obs(fc, obs, args.forecast_date)

    if args.ecmwf:
        seas = get_seas_forecast_da(
            args.hemisphere,
            args.forecast_date,
            bias_correct=args.bias_correct) \
            if args.ecmwf else None

        seas = seas.assign_coords(dict(xc=seas.xc / 1e3, yc=seas.yc / 1e3))
        seas = seas.isel(time=slice(1, None))
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
                              grid_area_size=args.grid_area,
                              threshold=args.threshold)


def plot_forecast():
    """CLI entry point for icenet_plot_forecast

    :return:
    """
    args = forecast_plot_args(ecmwf=False,
                              extra_args=[
                                  (("-l", "--leadtimes"), dict(
                                      help="Leadtimes to output, multiple as CSV, range as n..n",
                                      type=lambda s: [int(i) for i in
                                                      list(s.split(",") if "," in s else
                                                           range(int(s.split("..")[0]),
                                                                 int(s.split("..")[1])) if ".." in s else
                                                           [s])])),
                                  (("-c", "--no-coastlines"), dict(
                                      help="Turn off cartopy integration",
                                      action="store_true", default=False,
                                  )),
                                  (("-f", "--format"), dict(
                                      help="Format to output in",
                                      choices=("mp4", "png", "svg", "tiff"),
                                      default="png"
                                  )),
                                  (("-s", "--stddev"), dict(
                                      help="Plot the standard deviation from the ensemble",
                                      action="store_true",
                                      default=False
                                  ))
                              ])
    fc = get_forecast_ds(args.forecast_file,
                         args.forecast_date,
                         stddev=args.stddev)
    fc = fc.transpose(..., "yc", "xc")

    if not os.path.isdir(args.output_path):
        logging.warning("No directory at: {}".format(args.output_path))
        os.makedirs(args.output_path)
    elif os.path.isfile(args.output_path):
        raise RuntimeError("{} should be a directory and not existent...".
                           format(args.output_path))

    forecast_name = "{}.{}".format(
        os.path.splitext(os.path.basename(args.forecast_file))[0],
        args.forecast_date)

    cmap = None
    if args.region is not None:
        cmap = cm.get_cmap("tab20")
        cmap.set_bad("dimgrey")

        fc = process_regions(args.region, [fc])[0]

    leadtimes = args.leadtimes \
        if args.leadtimes is not None \
        else list(range(1, int(max(fc.leadtime.values)) + 1))

    if args.format == "mp4":
        pred_da = fc.isel(time=0).sel(leadtime=leadtimes)

        if "forecast_date" not in pred_da:
            forecast_dates = [
                pd.Timestamp(args.forecast_date) + dt.timedelta(lt)
                for lt in args.leadtimes]
            pred_da = pred_da.assign_coords(
                forecast_date=("leadtime", forecast_dates))

        pred_da = pred_da.drop("time").drop("leadtime").\
            rename(leadtime="time", forecast_date="time").set_index(time="time")

        anim_args = dict(
            figsize=5
        )
        if not args.no_coastlines:
            logging.warning("Coastlines will not work with the current "
                            "implementation of xarray_to_video")

        output_filename = os.path.join(args.output_path, "{}.{}.{}{}".format(
            forecast_name,
            args.forecast_date.strftime("%Y%m%d"),
            "" if not args.stddev else "stddev.",
            args.format
        ))
        xarray_to_video(pred_da, fps=1, cmap=cmap,
                        imshow_kwargs=dict(vmin=0., vmax=1.),
                        video_path=output_filename,
                        **anim_args)
    else:
        for leadtime in leadtimes:
            pred_da = fc.sel(leadtime=leadtime).isel(time=0)    #.sic_mean. \
                      # .where(~lm)

            bound_args = dict()

            if args.region is not None:
                bound_args.update(x1=args.region[0],
                                  x2=args.region[2],
                                  y1=args.region[1],
                                  y2=args.region[3])

            ax = get_plot_axes(**bound_args,
                               do_coastlines=not args.no_coastlines)

            if cmap:
                bound_args.update(cmap=cmap)

            im = show_img(ax, pred_da, **bound_args,
                          do_coastlines=not args.no_coastlines)

            plt.colorbar(im, ax=ax)
            plot_date = args.forecast_date + dt.timedelta(leadtime)
            ax.set_title("{:04d}/{:02d}/{:02d}".format(plot_date.year,
                                                       plot_date.month,
                                                       plot_date.day))
            output_filename = os.path.join(args.output_path, "{}.{}.{}{}".format(
                forecast_name,
                (args.forecast_date + dt.timedelta(
                    days=leadtime)).strftime("%Y%m%d"),
                "" if not args.stddev else "stddev.",
                args.format
            ))

            logging.info("Saving to {}".format(output_filename))
            plt.savefig(output_filename)
            plt.clf()


def parse_metrics_arg(argument: str) -> object:
    """
    Splits a string into a list by separating on commas.
    Will remove any whitespace and removes duplicates.
    Used to parsing metrics argument in metric_plots.
    
    :param argument: string
    
    :return: list of metrics to compute
    """
    return list(set([s.replace(" ", "") for s in argument.split(",")]))


def metric_plots():
    """
    Produces plot of requested metrics for forecasts.
    """
    args = forecast_plot_args(ecmwf=True, metrics=True)

    masks = Masks(north=args.hemisphere == "north",
                  south=args.hemisphere == "south")

    fc = get_forecast_ds(args.forecast_file,
                         args.forecast_date)
    obs = get_obs_da(args.hemisphere,
                     pd.to_datetime(args.forecast_date) +
                     timedelta(days=1),
                     pd.to_datetime(args.forecast_date) +
                     timedelta(days=int(fc.leadtime.max())))
    fc = filter_ds_by_obs(fc, obs, args.forecast_date)

    metrics = parse_metrics_arg(args.metrics)

    if args.ecmwf:
        seas = get_seas_forecast_da(
            args.hemisphere,
            args.forecast_date,
            bias_correct=args.bias_correct) \
            if args.ecmwf else None

        seas = seas.assign_coords(dict(xc=seas.xc / 1e3, yc=seas.yc / 1e3))
        seas = seas.isel(time=slice(1, None))
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


def sic_error():
    """
    Produces video visualisation of SIC of forecast and ground truth.
    """
    args = forecast_plot_args(ecmwf=False)

    masks = Masks(north=args.hemisphere == "north",
                  south=args.hemisphere == "south")

    fc = get_forecast_ds(args.forecast_file,
                         args.forecast_date)
    obs = get_obs_da(args.hemisphere,
                     pd.to_datetime(args.forecast_date) +
                     timedelta(days=1),
                     pd.to_datetime(args.forecast_date) +
                     timedelta(days=int(fc.leadtime.max())))
    fc = filter_ds_by_obs(fc, obs, args.forecast_date)

    if args.region:
        fc, obs, masks = process_regions(args.region, [fc, obs, masks])

    sic_error_video(fc_da=fc,
                    obs_da=obs,
                    land_mask=masks.get_land_mask(),
                    output_path=args.output_path)
