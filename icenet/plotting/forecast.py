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

from icenet.data.cli import date_arg
from icenet.data.sic.mask import Masks
from icenet.plotting.utils import \
    filter_ds_by_obs, get_forecast_ds, get_obs_da, get_seas_forecast_da, \
    show_img, get_plot_axes

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
                         output_path: object) -> object:
    """

    :param masks:
    :param fc_da:
    :param cmp_da:
    :param obs_da:
    :param output_path:
    :return:
    """
    agcm = masks.get_active_cell_da(obs_da)
    binary_obs_da = obs_da > 0.15

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title("Binary accuracy comparison")

    binary_fc_da = fc_da > 0.15
    binary_fc_da = (binary_fc_da == binary_obs_da).\
        astype(np.float16).weighted(agcm)
    binacc_fc = (binary_fc_da.mean(dim=['yc', 'xc']) * 100)
    ax.plot(binacc_fc.time, binacc_fc.values, label="IceNet")

    if cmp_da is not None:
        binary_cmp_da = cmp_da > 0.15
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
    logging.info("Saving to {}".format(output_path))
    plt.savefig(output_path)

    return binacc_fc, binacc_cmp


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

    tic = maps[0].set_title("IceNet {}".format(
        pd.to_datetime(fc_da.isel(time=leadtime).time.values).strftime("%d/%m/%Y")))
    tio = maps[1].set_title("OSISAF Obs {}".format(
        pd.to_datetime(obs_da.isel(time=leadtime).time.values).strftime("%d/%m/%Y")))
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
        logging.debug("Plotting {}".format(date))

        fc_plot = fc_da.isel(time=date).to_numpy()
        obs_plot = obs_da.isel(time=date).to_numpy()
        diff_plot = diff.isel(time=date).to_numpy()
        
        tic.set_text("IceNet {}".format(
            pd.to_datetime(fc_da.isel(time=date).time.values).strftime("%d/%m/%Y")))
        tio.set_text("OSISAF Obs {}".format(
            pd.to_datetime(obs_da.isel(time=date).time.values).strftime("%d/%m/%Y")))

        im1.set_data(fc_plot)
        im2.set_data(obs_plot)
        im3.set_data(diff_plot)

        return tic, tio, im1, im2, im3

    animation = FuncAnimation(fig,
                              update,
                              range(0, len(fc_da.time)),
                              interval=100)

    plt.close()

    output_path  = os.path.join("plot", "sic_error.mp4") \
        if not output_path else output_path
    logging.info("Saving plot to {}".format(output_path))
    animation.save(output_path,
                   fps=10,
                   extra_args=['-vcodec', 'libx264'])
    return animation


def forecast_plot_args(ecmwf: bool = True,
                       extra_args: object = None) -> object:
    """

    :param ecmwf:
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
                        action="store_true", default=False)
        ap.add_argument("-e", "--ecmwf", action="store_true", default=False)

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

    """
    args = forecast_plot_args()

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

    plot_binary_accuracy(masks,
                         fc,
                         seas,
                         obs,
                         args.output_path)


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
                                      action="store_false", default=True,
                                  )),
                                  (("-f", "--format"), dict(
                                      help="Format to output in",
                                      choices=("png", "svg", "tiff"),
                                      default="png"
                                  ))
                              ])
    fc = get_forecast_ds(args.forecast_file, args.forecast_date)

    if not os.path.isdir(args.output_path):
        logging.warning("No directory at: {}".format(args.output_path))
        os.makedirs(args.output_dir)
    elif os.path.isfile(args.output_path):
        raise RuntimeError("{} should be a directory and not existent...".
                           format(args.output_path))

    forecast_name = "{}.{}".format(
        os.path.splitext(os.path.basename(args.forecast_file))[0],
        args.forecast_date)

    for leadtime in args.leadtimes:
        pred_da = fc.sel(leadtime=leadtime).isel(time=0)    #.sic_mean. \
                  # .where(~lm)

        if args.region:
            pred_da = process_regions(args.region, [pred_da])[0]

        if args.format == "geotiff":
            raise RuntimeError("GeoTIFF will be supported in a future commit")
        else:
            if args.region is None:
                bound_args = dict()
            else:
                cmap = cm.get_cmap("tab20")
                cmap.set_bad("dimgrey")
                bound_args = dict(x1=args.region[0],
                                  x2=args.region[2],
                                  y1=args.region[1],
                                  y2=args.region[3])

            ax = get_plot_axes(**bound_args,
                               do_coastlines=args.no_coastlines)

            if cmap is not None:
                bound_args.update(cmap=cmap)

            im = show_img(ax, pred_da, **bound_args,
                          do_coastlines=args.no_coastlines)

            plt.colorbar(im, ax=ax)
            plot_date = args.forecast_date + dt.timedelta(leadtime)
            ax.set_title("{:04d}/{:02d}/{:02d}".format(plot_date.year,
                                                       plot_date.month,
                                                       plot_date.day))
            output_filename = os.path.join(args.output_path, "{}.{}.{}".format(
                forecast_name,
                (args.forecast_date + dt.timedelta(
                    days=leadtime)).strftime("%Y%m%d"),
                args.format
            ))

            logging.info("Saving to {}".format(output_filename))
            plt.savefig(output_filename)
            plt.clf()


def sic_error():
    """

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

    sic_error_video(fc,
                    obs,
                    masks.get_land_mask(),
                    args.output_path)
