import numpy as np
# from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
import imageio
import matplotlib.pyplot as plt
from tqdm import tqdm

###############################################################################
############### LOSS FUNCTIONS
###############################################################################

# TODO

###############################################################################
############### METRICS
###############################################################################

# TODO

###############################################################################
############### ARCHITECTURES
###############################################################################

# TODO

###############################################################################
############### MISC
###############################################################################


def filled_daily_dates(start_date, end_date):
    """
    Return a numpy array of datetimes, incrementing daily, starting at start_date and
    going up to (but not including) end_date.
    """

    monthly_list = []
    date = start_date

    while date < end_date:
        monthly_list.append(date)
        date += relativedelta(days=1)

    return np.array(monthly_list)


def gen_frame_func(da, mask=None, mask_type='contour', cmap='viridis', figsize=15):

    '''
    Create imageio frame function for xarray.DataArray visualisation.

    Parameters:
    da (xr.DataArray): Dataset to create video of.

    mask (np.ndarray): Boolean mask with True over masked elements to overlay
    as a contour or filled contour. Defaults to None (no mask plotting).

    mask_type (str): 'contour' or 'contourf' dictating whether the mask is overlaid
    as a contour line or a filled contour.

    '''

    max = da.max().values
    min = da.min().values

    def make_frame(date):
        fig, ax = plt.subplots(figsize=(figsize, figsize))
        ax.imshow(da.sel(time=date), cmap=cmap, clim=(min, max))
        if mask is not None:
            if mask_type == 'contour':
                ax.contour(mask, levels=[.5, 1], colors='k')
            elif mask_type == 'contourf':
                ax.contourf(mask, levels=[.5, 1], colors='k')
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)

        ax.set_title('{:04d}/{:02d}/{:02d}'.format(date.year, date.month, date.day), fontsize=figsize*4)

        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        plt.close()
        return image

    return make_frame


def xarray_to_video(da, video_path, fps, mask=None, mask_type='contour',
                    video_dates=None, cmap='viridis', figsize=15):

    '''
    Generate video of an xarray.DataArray. Optionally input a list of
    `video_dates` to show, otherwise the full set of time coordiantes
    of the dataset is used.
    '''

    if video_dates is None:
        video_dates = [pd.Timestamp(date).to_pydatetime() for date in da.time.values]

    make_frame = gen_frame_func(da=da, mask=mask, mask_type=mask_type,
                                cmap=cmap, figsize=figsize)

    imageio.mimsave(video_path,
                    [make_frame(date) for date in tqdm(video_dates)],
                    fps=fps)
