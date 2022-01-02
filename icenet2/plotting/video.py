import argparse
import datetime as dt
import logging
import os
import re

from pprint import pformat, pprint
from concurrent.futures import as_completed, ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
import xarray as xr

from matplotlib.animation import ArtistAnimation, FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable

from icenet2.process.predict import get_refcube


# TODO: This can be a plotting or analysis util function elsewhere
def get_dataarray_from_files(files,
                             numpy=False):
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
        ref_cube = get_refcube("nh" in path_comps, "sh" in path_comps)
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


def xarray_to_video(da, fps, video_path=None, mask=None, mask_type='contour',
                    clim=None, crop=None, data_type='abs', video_dates=None,
                    cmap='viridis', figsize=12, dpi=150):

    """
    Generate video of an xarray.DataArray. Optionally input a list of
    `video_dates` to show, otherwise the full set of time coordiantes
    of the dataset is used.

    Parameters:
    da (xr.DataArray): Dataset to create video of.

    video_path (str): Path to save the video to.

    fps (int): Frames per second of the video.

    mask (np.ndarray): Boolean mask with True over masked elements to overlay
    as a contour or filled contour. Defaults to None (no mask plotting).

    mask_type (str): 'contour' or 'contourf' dictating whether the mask is
    overlaid as a contour line or a filled contour.

    data_type (str): 'abs' or 'anom' describing whether the data is in absolute
    or anomaly format. If anomaly, the colorbar is centred on 0.

    video_dates (list): List of Pandas Timestamps or datetime.datetime objects
    to plot video from the dataset.

    crop (list): [(a, b), (c, d)] to crop the video from a:b and c:d

    clim (list): Colormap limits. Default is None, in which case the min and
    max values of the array are used.

    cmap (str): Matplotlib colormap.

    figsize (int or float): Figure size in inches.

    dpi (int): Figure DPI.
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
            ax.contour(mask, levels=[.5, 1], colors='k')
        elif mask_type == 'contourf':
            ax.contourf(mask, levels=[.5, 1], colors='k')

    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)

    divider = make_axes_locatable(ax)
    date = pd.Timestamp(da.time.values[0]).to_pydatetime()
    image = ax.imshow(da.sel(time=date),
                      cmap=cmap,
                      clim=(n_min, n_max),
                      animated=True)

    image_title = ax.set_title("{:04d}/{:02d}/{:02d}".
                               format(date.year, date.month, date.day),
                               fontsize="medium")
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(image, cax)

    logging.info("Animating")
    animation = FuncAnimation(fig,
                              update,
                              video_dates,
                              interval=1000/fps)

    if not video_path:
        logging.info("Showing plot")
        plt.show()
    else:
        logging.info("Saving plot to {}".format(video_path))
        animation.save(video_path,
                       fps=fps,
                       extra_args=['-vcodec', 'libx264'])


def cli_args():
    args = argparse.ArgumentParser()

    args.add_argument("-f", "--fps", default=15, type=int)
    args.add_argument("-n", "--numpy", action="store_true", default=False)
    args.add_argument("-o", "--output-dir", dest="output_dir", type=str,
                      default="plot")
    args.add_argument("-p", "--path", default="data", type=str)
    args.add_argument("-sy", "--skip-years",
                      help="Don't include years in paths",
                      default=False, action="store_true")
    args.add_argument("-w", "--workers", default=8, type=int)

    args.add_argument("-v", "--verbose", action="store_true", default=False)

    args.add_argument("datasets", type=lambda s: s.split(","))
    args.add_argument("hemisphere", default=[],
                      choices=["nh", "sh"], nargs="?")
    args.add_argument("vars", default=[],
                      nargs="?", type=lambda s: s.split(","))
    args.add_argument("years", default=[],
                      nargs="?", type=lambda s: s.split(","))

    return args.parse_args()


def recurse_data_folders(base_path, lookups, children,
                         filetype="nc"):
    logging.info("Looking at {}".format(base_path))
    files = []

    if children is None and lookups is None:
        # TODO: should ideally use scandir for performance
        # TODO: naive lexicographical sorting
        files = sorted(
            [os.path.join(base_path, f) for f in os.listdir(base_path)
             if os.path.splitext(f)[1] == ".{}".format(filetype)])

        if not len(files):
            return None
    else:
        for subdir in os.listdir(base_path):
            new_path = os.path.join(base_path, subdir)

            if not os.path.isdir(new_path):
                continue

            if not len(lookups) or \
                    (len(lookups) and subdir in lookups):
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


def video_process(files, numpy, output_dir, fps):
    logging.debug("Supplied: {} files for processing".format(len(files)))
    da = get_dataarray_from_files(files, numpy)
    path_comps = os.path.dirname(files[0]).split(os.sep)
    os.makedirs(output_dir, exist_ok=True)
    output_name = os.path.join(output_dir,
                               "{}.mp4".format("_".join(path_comps)))
    logging.info("Saving to {}".format(output_name))
    xarray_to_video(da, fps, video_path=output_name)


def data_cli():
    args = cli_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    logging.getLogger("matplotlib").setLevel(level=logging.WARNING)

    hemis = [args.hemisphere] if len(args.hemisphere) else ["nh", "sh"]
    years = [int(year) for year in args.years] if args.years else args.years

    logging.info("Looking into {}".format(args.path))

    path_children = [hemis, args.vars]
    if not args.skip_years:
        # TODO: GH#3
        path_children += [years]

    video_batches = recurse_data_folders(args.path,
                                         args.datasets,
                                         path_children,
                                         filetype="nc"
                                         if not args.numpy else "npy")

    video_batches = np.array(video_batches).squeeze()
    if len(video_batches.shape) == 1:
        video_batches = video_batches[np.newaxis, :]

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
                future.result()
            except Exception as e:
                logging.error(e)


