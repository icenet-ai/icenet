import argparse
import datetime as dt
import logging
import os
import re

from concurrent.futures import as_completed, ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable

from icenet.process.predict import get_refcube
from icenet.utils import setup_logging


# TODO: This can be a plotting or analysis util function elsewhere
def get_dataarray_from_files(files: object,
                             numpy: bool = False) -> object:
    """

    :param files:
    :param numpy:
    :return:
    """
    if not numpy:
        ds = xr.open_mfdataset(files)
        # TODO: We're relying on single variable files from downloaders
        #  so maybe allow a specifier for this for multi var files?
        da = ds.to_array(dim=list(ds.data_vars)[0])[0]
    else:
        first_file = np.load(files[0])
        arr = np.zeros((len(files), *first_file.shape))
        dates = []

        assert len(first_file.shape) == 2, \
            "Wrong number of dims for use in videos".\
                format(len(first_file.shape))

        for np_idx in range(0, len(files)):
            arr[np_idx] = np.load(files[np_idx])
            nom = os.path.basename(files[np_idx])

            # TODO: error handling
            date_match = re.search(r"(\d{4})_(\d{1,2})_(\d{1,2})", nom)
            dates.append(pd.to_datetime(
                dt.date(*[int(s) for s in date_match.groups()])))

        # FIXME: naive implementations abound
        path_comps = os.path.dirname(files[0]).split(os.sep)
        ref_cube = get_refcube("north" in path_comps, "south" in path_comps)
        var_name = path_comps[-2]

        da = xr.DataArray(
            data=arr,
            dims=("time", "yc", "xc"),
            coords=dict(
                time=[pd.Timestamp(d) for d in dates],
                xc=ref_cube.coord("projection_x_coordinate").points,
                yc=ref_cube.coord("projection_y_coordinate").points,
            ),
            name=var_name,
        )

    return da


def xarray_to_video(da: object,
                    fps: int,
                    video_path: object = None,
                    mask: object = None,
                    mask_type: str = 'contour',
                    clim: object = None,
                    crop: object = None,
                    data_type: str = 'abs',
                    video_dates: object = None,
                    cmap: object = 'viridis',
                    figsize: int = 12,
                    dpi: int = 150) -> object:

    """
    Generate video of an xarray.DataArray. Optionally input a list of
    `video_dates` to show, otherwise the full set of time coordiantes
    of the dataset is used.

    :param da: Dataset to create video of.
    :param video_path: Path to save the video to.
    :param fps: Frames per second of the video.
    :param mask: Boolean mask with True over masked elements to overlay
    as a contour or filled contour. Defaults to None (no mask plotting).
    :param mask_type: 'contour' or 'contourf' dictating whether the mask is
    overlaid as a contour line or a filled contour.
    :param data_type: 'abs' or 'anom' describing whether the data is in absolute
    or anomaly format. If anomaly, the colorbar is centred on 0.
    :param video_dates: List of Pandas Timestamps or datetime.datetime objects
    to plot video from the dataset.
    :param crop: [(a, b), (c, d)] to crop the video from a:b and c:d
    :param clim: Colormap limits. Default is None, in which case the min and
    max values of the array are used.
    :param cmap: Matplotlib colormap.
    :param figsize: Figure size in inches.
    :param dpi: Figure DPI.

    """

    def update(date):
        logging.debug("Plotting {}".format(date.strftime("%D")))
        image.set_data(da.sel(time=date))

        image_title.set_text("{:04d}/{:02d}/{:02d}".format(
            date.year, date.month, date.day))

        return image, image_title

    logging.info("Inspecting data")

    if clim is not None:
        n_min = clim[0]
        n_max = clim[1]
    else:
        n_max = da.max().values
        n_min = da.min().values

        if data_type == 'anom':
            if np.abs(n_max) > np.abs(n_min):
                n_min = -n_max
            elif np.abs(n_min) > np.abs(n_max):
                n_max = -n_min

    if video_dates is None:
        video_dates = [pd.Timestamp(date).to_pydatetime()
                       for date in da.time.values]

    if crop is not None:
        a = crop[0][0]
        b = crop[0][1]
        c = crop[1][0]
        d = crop[1][1]
        da = da.isel(xc=np.arange(a, b), yc=np.arange(c, d))
        if mask is not None:
            mask = mask[a:b, c:d]

    logging.info("Initialising plot")

    fig, ax = plt.subplots(figsize=(figsize, figsize))
    fig.set_dpi(dpi)

    if mask is not None:
        if mask_type == 'contour':
            ax.contour(mask, levels=[.5, 1], colors='k', zorder=3)
        elif mask_type == 'contourf':
            ax.contourf(mask, levels=[.5, 1], colors='k', zorder=3)

    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)

    divider = make_axes_locatable(ax)
    date = pd.Timestamp(da.time.values[0]).to_pydatetime()
    image = ax.imshow(da.sel(time=date),
                      cmap=cmap,
                      clim=(n_min, n_max),
                      animated=True,
                      zorder=1)

    image_title = ax.set_title("{:04d}/{:02d}/{:02d}".
                               format(date.year, date.month, date.day),
                               fontsize="medium", zorder=2)
    cax = divider.append_axes('right', size='5%', pad=0.05, zorder=2)
    plt.colorbar(image, cax)

    logging.info("Animating")
    # TODO: Investigate blit, but it causes a few problems with masks/titles.
    animation = FuncAnimation(fig,
                              update,
                              video_dates,
                              interval=1000/fps)

    plt.close()

    if not video_path:
        logging.info("Not saving plot, will return animation")
    else:
        logging.info("Saving plot to {}".format(video_path))
        animation.save(video_path,
                       fps=fps,
                       extra_args=['-vcodec', 'libx264'])
    return animation


def recurse_data_folders(base_path: object,
                         lookups: object,
                         children: object,
                         filetype: str = "nc") -> object:
    """

    :param base_path:
    :param lookups:
    :param children:
    :param filetype:
    :return:
    """
    logging.info("Looking at {}".format(base_path))
    files = []

    if children is None and lookups is None:
        # TODO: should ideally use scandir for performance
        # TODO: naive hardcoded filtering of files
        logging.debug("CHILDREN: {} or LOOKUPS: {}".format(children, lookups))
        files = sorted(
            [os.path.join(base_path, f) for f in os.listdir(base_path)
             if os.path.splitext(f)[1] == ".{}".format(filetype)
             and (re.match(r'^\d{4}\.nc$', f)
                  or
                  re.search(r'(abs|anom|linear_trend)\.nc$', f))])
        
        if not len(files):
            return None
    else:
        for subdir in os.listdir(base_path):
            logging.debug("SUBDIR: {}".format(subdir))
            new_path = os.path.join(base_path, subdir)

            if not os.path.isdir(new_path):
                continue

            if not len(lookups) or \
                    (len(lookups) and subdir in [str(s) for s in lookups]):
                subdir_files = recurse_data_folders(
                    new_path,
                    children[0]
                    if children is not None and len(children) > 0 else None,
                    children[1:]
                    if children is not None and len(children) > 1 else None,
                    filetype
                )
                if subdir_files:
                    files.append(subdir_files)

    return files


def video_process(files: object,
                  numpy: object,
                  output_dir: object,
                  fps: int) -> object:
    """

    :param files:
    :param numpy:
    :param output_dir:
    :param fps:
    :return:
    """
    path_comps = os.path.dirname(files[0]).split(os.sep)
    os.makedirs(output_dir, exist_ok=True)
    output_name = os.path.join(output_dir,
                               "{}.mp4".format("_".join(path_comps)))

    if not os.path.exists(output_name):
        logging.debug("Supplied: {} files for processing".format(len(files)))
        da = get_dataarray_from_files(files, numpy)
        logging.info("Saving to {}".format(output_name))
        xarray_to_video(da, fps, video_path=output_name)
    else:
        logging.warning("Not overwriting existing: {}".format(output_name))
        return None

    return output_name


@setup_logging
def cli_args():
    """

    :return:
    """
    args = argparse.ArgumentParser()

    args.add_argument("-f", "--fps", default=15, type=int)
    args.add_argument("-n", "--numpy", action="store_true", default=False)
    args.add_argument("-o", "--output-dir", dest="output_dir", type=str,
                      default="plot")
    args.add_argument("-p", "--path", default="data", type=str)
    args.add_argument("-w", "--workers", default=8, type=int)

    args.add_argument("-v", "--verbose", action="store_true", default=False)

    args.add_argument("data", type=lambda s: s.split(","))
    args.add_argument("hemisphere", default=[],
                      choices=["north", "south"], nargs="?")

    args.add_argument("--vars", default=[], type=lambda s: s.split(","))
    args.add_argument("--years", default=[], type=lambda s: s.split(","))

    return args.parse_args()


def data_cli():
    """

    """
    args = cli_args()

    hemis = [args.hemisphere] if len(args.hemisphere) else ["north", "south"]
    logging.info("Looking into {}".format(args.path))

    path_children = [hemis, args.vars]
    video_batches = recurse_data_folders(args.path,
                                         args.data,
                                         path_children,
                                         filetype="nc"
                                         if not args.numpy else "npy")
    video_batches = [
        v_el for h_list in video_batches
        for v_list in h_list
        for v_el in v_list
    ]

    if len(args.years) > 0:
        new_batches = []
        for batch in video_batches:
            batch = [el for el in batch
                     if os.path.basename(el)[0:4] in args.years]
            if len(batch):
                new_batches.append(batch)
            video_batches = new_batches

    logging.debug("Batches {}".format(video_batches))

    with ProcessPoolExecutor(
            max_workers=min(len(video_batches), args.workers)) as executor:
        futures = []

        for batch in video_batches:
            futures.append(executor.submit(video_process,
                                           batch,
                                           args.numpy,
                                           args.output_dir,
                                           args.fps))

        for future in as_completed(futures):
            try:
                res = future.result()

                if res:
                    logging.info("Produced {}".format(res))
            except Exception as e:
                logging.error(e)


