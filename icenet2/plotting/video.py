
import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm

from mpl_toolkits.axes_grid1 import make_axes_locatable


def xarray_to_video(da, video_path, fps, mask=None, mask_type='contour',
                    clim=None, crop=None, data_type='abs', video_dates=None,
                    cmap='viridis', figsize=15, dpi=300):

    '''
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
    '''

    if clim is not None:
        min = clim[0]
        max = clim[1]
    elif clim is None:
        max = da.max().values
        min = da.min().values

        if data_type == 'anom':
            if np.abs(max) > np.abs(min):
                min = -max
            elif np.abs(min) > np.abs(max):
                max = -min

    def make_frame(date):
        fig, ax = plt.subplots(figsize=(figsize, figsize))
        fig.set_dpi(dpi)
        im = ax.imshow(da.sel(time=date), cmap=cmap, clim=(min, max))
        if mask is not None:
            if mask_type == 'contour':
                ax.contour(mask, levels=[.5, 1], colors='k')
            elif mask_type == 'contourf':
                ax.contourf(mask, levels=[.5, 1], colors='k')
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)

        ax.set_title('{:04d}/{:02d}/{:02d}'.
                     format(date.year, date.month, date.day),
                     fontsize=figsize*4)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax)

        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        plt.close()
        return image

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

    imageio.mimsave(video_path,
                    [make_frame(date) for date in tqdm(video_dates)],
                    fps=fps)
