import os
import sys
import numpy as np
sys.path.insert(0, os.path.join(os.getcwd(), 'icenet2'))  # if using jupyter kernel
from dateutil.relativedelta import relativedelta
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import xarray as xr
import imageio
import matplotlib.pyplot as plt
from tqdm import tqdm


###############################################################################
############### MISC
###############################################################################


def filled_daily_dates(start_date, end_date, include_end=False):
    """
    Return a numpy array of datetimes, incrementing daily, starting at start_date and
    going up to (possibly including) end_date.
    """

    daily_list = []
    date = start_date

    if include_end:
        end_date += relativedelta(days=1)

    while date < end_date:
        daily_list.append(date)
        date += relativedelta(days=1)

    return np.array(daily_list)


def filled_monthly_dates(start_date, end_date, include_end=False):
    """
    Return a numpy array of datetimes, incrementing monthly, starting at start_date and
    going up to (possibly including) end_date.
    """

    monthly_list = []
    date = start_date

    if include_end:
        end_date += relativedelta(months=1)

    while date < end_date:
        monthly_list.append(date)
        date += relativedelta(months=1)

    return np.array(monthly_list)


def xarray_to_video(da, video_path, fps, mask=None, mask_type='contour', clim=None,
                    crop=None, data_type='abs', video_dates=None, cmap='viridis',
                    figsize=15, dpi=300):

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

    mask_type (str): 'contour' or 'contourf' dictating whether the mask is overlaid
    as a contour line or a filled contour.

    data_type (str): 'abs' or 'anom' describing whether the data is in absolute
    or anomaly format. If anomaly, the colorbar is centred on 0.

    video_dates (list): List of Pandas Timestamps or datetime.datetime objects
    to plot video from the dataset.

    crop (list): [(a, b), (c, d)] to crop the video from a:b and c:d

    clim (list): Colormap limits. Default is None, in which case the min and max values
    of the array are used.

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

        ax.set_title('{:04d}/{:02d}/{:02d}'.format(date.year, date.month, date.day), fontsize=figsize*4)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax)

        # TEMP crop to image
        # fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        plt.close()
        return image

    if video_dates is None:
        video_dates = [pd.Timestamp(date).to_pydatetime() for date in da.time.values]

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


class StretchOutNormalize(plt.Normalize):
    '''
    Taken from https://stackoverflow.com/questions/59270751/how-to-make-a-diverging-color-bar-with-white-space-between-two-values-matplotlib

    Expand the white area of a diverging colormap.

    '''
    def __init__(self, vmin=None, vmax=None, low=None, up=None, clip=False):
        self.low = low
        self.up = up
        plt.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.low, self.up, self.vmax], [0, 0.5-1e-9, 0.5+1e-9, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def gen_forecast_video(video_fpath, forecast_folder_path, forecast_date, n_forecast_days,
                       fps, video_format='fixed_init', third_axis='sic_error', day_error_map=None,
                       init_pause=10, mask=None, figsize=(15, 5), dpi=300):

    '''
    Generate a video of a forecast and compare with ground truth.
    This assumes the forecast dataset has already been saved in the
    data/forecasts/ folder (see icenet2/predict_validation.py). The IceNet2 forecast
    paths take the form:

        data/forecasts/icenet2/{dataloader_name}/{network_name}/{seed}/{year}.nc

    Parameters:

        video_fpath (str): Path to save the froecast video to.

        forecast_folder_path (str): Path to the folder storing the NetCDF forecasts.

        forecast_date (date-like): If video_format=='fixed_init', this corresponds
            to the date the forecast was initialised (i.e. the first day being
            forecasted). If video_format=='fixed_target', this corresponds to the
            target date of the forecast.

        n_forecast_days (int): Maximum lead time of the forecast.

        fps (int): Frames per second of the video.

        video_format (str): 'fixed_init' (default) or 'fixed_target'. This
            dictates whether the forecast video takes the form of a fixed
            initialisation date with growing lead time, or a
            fixed target date with shrinking lead time.

        third_axis (str): 'sic_error' for a forecast SIC error video, 'day_error' for
            static break up dat plot (not yet implemented).

        day_error_map (np.ndarray): if third_axis=='day_error', supply the array
            of the ice transition day error. Defaults to None.

        init_pause (int): Number of frames to 'pause' on the first forecast day.
            Defaults to 10. Set to 1 for no pause.

        mask (np.ndarray): Boolean mask with True over masked elements to overlay
            as a contour or filled contour. Defaults to None (no mask plotting).

        figsize (tuple of floats): Figure (width, height) in inches.

        dpi (int): Figure dpi (default 300). Reduce to increase video generation speed.

    Example use:

        gen_forecast_video(
            video_fpath='forecast_video_icenet2_more_recent.mp4',
            forecast_folder_path='data/forecasts/icenet2/2021_04_03_1421_icenet2_nh_sh_thinned5_weeklyinput/unet_batchnorm/42/',
            forecast_date=pd.Timestamp('2014-08-01'),
            n_forecast_days=31*3,
            fps=4,
            video_format='fixed_init',
            third_axis='sic_error',
            day_error_map=None,
            init_pause=10,
            mask=np.load('data/nh/masks/land_mask.npy'),
            figsize=(15, 5),
            dpi=300
        )

    '''

    #### Load forecast dataset
    print('Loading forecast dataset... ', end='', flush=True)
    validation_prediction_fpaths = [
        os.path.join(forecast_folder_path, f) for f in os.listdir(forecast_folder_path)
    ]
    all_forecasts_ds = xr.open_mfdataset(
        validation_prediction_fpaths,
    )
    all_forecast_da = next(iter(all_forecasts_ds.data_vars.values()))  # Convert to DataArray
    print('Done.')

    #### Load ground truth
    print('Loading ground truth... ', end='', flush=True)
    true_sic_fpath = os.path.join('data', 'nh', 'siconca', 'siconca_all_interp.nc')
    true_sic_da = xr.open_dataarray(true_sic_fpath)

    # Replace 12:00 hour with 00:00 hour by convention
    dates = [pd.Timestamp(date).to_pydatetime().replace(hour=0) for date in true_sic_da.time.values]
    true_sic_da = true_sic_da.assign_coords(dict(time=dates))
    print('Done.')

    #### Set up forecast array for the video

    print('Slicing forecast (this can take a minute)... ', end='', flush=True)
    if video_format == 'fixed_init':
        target_dates = pd.date_range(
            start=forecast_date,
            end=forecast_date + pd.DateOffset(days=n_forecast_days),
            freq='D'
        )
        leadtimes = np.arange(1, n_forecast_days+1)

        shape = (len(target_dates), 432, 432)
        forecast_da = xr.DataArray(
            data=np.zeros(shape, dtype=np.float32),
            dims=('time', 'yc', 'xc'),
            coords={
                'time': target_dates,
                'yc': true_sic_da.coords['yc'],
                'xc': true_sic_da.coords['xc'],
            }
        )

        for target_date, leadtime in zip(target_dates, leadtimes):
            forecast_da.loc[target_date, :] = \
                all_forecast_da.sel(time=target_date, leadtime=leadtime)

    elif video_format == 'fixed_target':
        # Shrinking lead times
        leadtimes = np.arange(n_forecast_days, 0, -1)
        target_dates = [forecast_date]*n_forecast_days

        shape = (len(leadtimes), 432, 432)
        forecast_da = xr.DataArray(
            data=np.zeros(shape, dtype=np.float32),
            dims=('leadtime', 'yc', 'xc'),
            coords={
                'leadtime': leadtimes,
                'yc': true_sic_da.coords['yc'],
                'xc': true_sic_da.coords['xc'],
            }
        )

        for leadtime in leadtimes:
            # It is faster to use .sel on one element at a time
            forecast_da.loc[leadtime, :] = \
                all_forecast_da.sel(time=forecast_date, leadtime=leadtime)
        forecast_da.load()  # Load into memory
    print('Done.')

    #### Pause
    def pause(input_list, num=init_pause):
        ''' Repeat the first element of a list `num` times. '''
        return [*input_list[0:1]*num, *input_list[1:]]

    target_dates = pause(list(target_dates))
    leadtimes = pause(list(leadtimes))

    # Also pause at the end of the video
    def pause_end(input_list, num=init_pause):
        ''' Repeat the last element of a list `num` times. '''
        return [*input_list[:-1], *input_list[-1:]*num]

    target_dates = pause_end(list(target_dates))
    leadtimes = pause_end(list(leadtimes))

    #### Make video
    def make_frame(date, leadtime):
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=figsize)
        fig.set_dpi(dpi)

        if video_format == 'fixed_init':
            pred = forecast_da.sel(time=date)
        elif video_format == 'fixed_target':
            pred = forecast_da.sel(leadtime=leadtime)
        true = true_sic_da.sel(time=date)

        # Forecast
        ax = axes[0]
        ax.set_title('Forecast')
        im = ax.imshow(pred, cmap='Blues_r', clim=(0, 1))
        if mask is not None:
            ax.contourf(mask, levels=[.5, 1], colors='k')

        # Ground truth
        ax = axes[1]
        ax.set_title('Observed')
        im = ax.imshow(true, cmap='Blues_r', clim=(0, 1))
        if mask is not None:
            ax.contourf(mask, levels=[.5, 1], colors='k')

        # Sea ice transition map
        ax = axes[2]
        if third_axis == 'sic_error':
            ax.set_title('SIC error')
            norm = StretchOutNormalize(vmin=-1, vmax=1, low=-.1, up=0.1)
            im = ax.imshow(pred - true, cmap='seismic', clim=(-1 , 1), norm=norm)
            if mask is not None:
                ax.contourf(mask, levels=[.5, 1], colors='k')

            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            plt.colorbar(im, cax)

        elif third_axis == 'day_error':
            ax.set_title('Sea ice transtion day error map')
            ax.text(x=0.5, y=0.5, s="Ellie's sea ice transition error map here :-)",
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes)

            # TODO: colorbar, logic for which transition is being plotted, etc

        plt.tight_layout()
        plt.subplots_adjust(top=0.85)  # Make room for title

        fig.suptitle(date.strftime('%Y/%m/%d') + ', {} days leadtime'.format(leadtime),
                     fontsize=figsize[0]*1.5)

        for ax in axes:
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)

        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        plt.close()
        return image

    print('Generating video:\n')
    imageio.mimsave(video_fpath,
                    [make_frame(date, leadtime) for date, leadtime
                     in zip(tqdm(target_dates), leadtimes)],
                    fps=fps)
    print('\nVideo saved to {}'.format(video_fpath))

# gen_forecast_video(
#     video_fpath='fixed_target_2012.mp4',
#     forecast_folder_path='data/forecasts/icenet2/2021_04_03_1421_icenet2_nh_sh_thinned5_weeklyinput/unet_batchnorm/42/',
#     forecast_date=pd.Timestamp('2012-09-15'),
#     n_forecast_days=31*3,
#     fps=3,
#     video_format='fixed_target',
#     third_axis='sic_error',
#     day_error_map=None,
#     init_pause=10,
#     mask=np.load('data/nh/masks/land_mask.npy'),
#     figsize=(15, 5),
#     dpi=300
# )
