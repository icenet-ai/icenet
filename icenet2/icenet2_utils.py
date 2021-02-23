import os
import sys
import warnings
import numpy as np
from datetime import datetime
sys.path.insert(0, os.path.join(os.getcwd(), 'icenet2'))  # if using jupyter kernel
import config
from dateutil.relativedelta import relativedelta
from mpl_toolkits.axes_grid1 import make_axes_locatable
import xarray as xr
import pandas as pd
import regex as re
import imageio
import matplotlib.pyplot as plt
import time
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
############### DATA PROCESSING & LOADING
###############################################################################


class IceNet2DataPreProcessor(object):
    """
    Normalises IceNet2 input data and saves the normalised daily averages
    as .npy files.

    Data is stored in the following form:
     - data/network_datasets/dataset_name/obs/tas/2006_04_12.npy
    """

    def __init__(self, dataset_name, preproc_vars, obs_train_dates,
                 minmax, verbose_level, raw_data_shape,
                 dtype=np.float32):
        """
        Parameters:
        dataset_name (str): Name of this network dataset (used for the folder to
        store data)

        preproc_vars (dict): Which variables to preprocess. Example:

                preproc_vars = {
                    'siconca': {'anom': True, 'abs': True},
                    'tas': {'anom': True, 'abs': False},
                    'tos': {'anom': True, 'abs': False},
                    'rsds': {'anom': True, 'abs': False},
                    'rsus': {'anom': True, 'abs': False},
                    'psl': {'anom': False, 'abs': True},
                    'zg500': {'anom': False, 'abs': True},
                    'zg250': {'anom': False, 'abs': True},
                    'ua10': {'anom': False, 'abs': True},
                    'uas': {'anom': False, 'abs': True},
                    'vas': {'anom': False, 'abs': True},
                    'sfcWind': {'anom': False, 'abs': True},
                    'land': {'metadata': True, 'include': True},
                    'circday': {'metadata': True, 'include': True}
                }

        obs_train_dates (tuple): Tuple of months (stored as datetimes)
        to be used for the training set by the data loader.

        minmax (bool): Whether to use min-max normalisation to (-1, 1)

        verbose_level (int): Controls how much to print. 0: Print nothing.
        1: Print key set-up stages. 2: Print debugging info.

        raw_data_shape (tuple): Shape of input satellite data as (rows, cols).

        dtype (type): Data type for the data (default np.float32)

        """
        self.dataset_name = dataset_name
        self.preproc_vars = preproc_vars
        self.obs_train_dates = obs_train_dates
        self.minmax = minmax
        self.verbose_level = verbose_level
        self.raw_data_shape = raw_data_shape
        self.dtype = dtype

        self.set_up_folder_hierarchy()
        self.preproc_and_save_icenet_data()

        if self.verbose_level >= 1:
            print("Setup complete.\n")

    def set_up_folder_hierarchy(self):

        """
        Initialise the folders to store the datasets.
        """

        if self.verbose_level >= 1:
            print('Setting up the folder hierarchy for {}... '.format(self.dataset_name),
                  end='', flush=True)

        # Root folder for this dataset
        self.dataset_path = os.path.join(config.folders['data'], 'network_datasets', self.dataset_name)

        # Dictionary data structure to store folder paths
        self.paths = {}

        # Set up the folder hierarchy
        self.paths['obs'] = {}

        for varname, vardict in self.preproc_vars.items():

            if 'metadata' not in vardict.keys():
                self.paths['obs'][varname] = {}

                for data_format in vardict.keys():

                    if vardict[data_format] is True:
                        path = os.path.join(self.dataset_path, 'obs',
                                            varname, data_format)

                        self.paths['obs'][varname][data_format] = path

                        if not os.path.exists(path):
                            os.makedirs(path)

            elif 'metadata' in vardict.keys():

                if vardict['include'] is True:
                    path = os.path.join(self.dataset_path, 'meta')

                    self.paths['meta'] = path

                    if not os.path.exists(path):
                        os.makedirs(path)

        if self.verbose_level >= 1:
            print('Done.')

    @staticmethod
    def mean_and_std(list, verbose_level=2):

        """
        Return the mean and standard deviation of an array-like object (intended
        use case is for normalising a raw satellite data array based on a list
        of samples used for training).
        """

        mean = np.nanmean(list)
        std = np.nanstd(list)

        if verbose_level >= 2:
            print("Mean: {:.3f}, std: {:.3f}".format(mean.item(), std.item()))

        return mean, std

    def normalise_array_using_all_training_data(self, da, minmax=False,
                                                mean=None, std=None,
                                                min=None, max=None):

        """
        Using the *training* data only, compute the mean and
        standard deviation of the input raw satellite DataArray (`da`)
        and return a normalised version. If minmax=True,
        instead normalise to lie between min and max of the elements of `array`.

        If min, max, mean, or std are given values other than None,
        those values are used rather than being computed from the training months.

        Returns:
        new_da (xarray.DataArray): Normalised array.

        mean, std (float): Mean and standard deviation used or computed for the
        normalisation.

        min, max (float): Min and max used or computed for the normalisation.
        """

        training_samples = da.sel(time=self.obs_train_dates).data
        training_samples = training_samples.ravel()

        if not minmax:
            # Normalise by mean and standard deviation (compute them if not provided)

            if mean is None and std is None:
                # Compute mean and std
                mean, std = IceNet2DataPreProcessor.mean_and_std(training_samples,
                                                                 self.verbose_level)
            elif mean is not None and std is None:
                # Compute std only
                _, std = IceNet2DataPreProcessor.mean_and_std(training_samples,
                                                              self.verbose_level)
            elif mean is None and std is not None:
                # Compute mean only
                mean, _ = IceNet2DataPreProcessor.mean_and_std(training_samples,
                                                               self.verbose_level)

            new_da = (da - mean) / std

        elif minmax:
            # Normalise by min and max (compute them if not provided)

            if min is None:
                min = np.nanmin(training_samples)
            if max is None:
                max = np.nanmax(training_samples)

            new_da = (da - min) / (max - min)

        if minmax:
            return new_da, min, max
        elif not minmax:
            return new_da, mean, std

    def save_xarray_in_daily_averages(self, da, dataset_type, varname, data_format,
                                      member_id=None):

        """
        Saves an xarray DataArray as daily averaged .npy files using the
        self.paths data structure.

        Parameters:
        da (xarray.DataArray): The DataArray to save.

        dataset_type (str): Either 'obs' or 'transfer' (for CMIP6 data) - the type
        of dataset being saved.

        varname (str): Variable name being saved.

        data_format (str): Either 'abs' or 'anom' - the format of the data
        being saved.
        """

        if self.verbose_level >= 2:
            print('Saving {} {} daily averages... '.format(data_format, varname), end='', flush=True)

        for date in da.time.values:
            slice = da.sel(time=date).data
            date_datetime = datetime.utcfromtimestamp(date.tolist() / 1e9)
            fname = date_datetime.strftime('%Y_%m_%d.npy')

            if dataset_type == 'obs':
                np.save(os.path.join(self.paths[dataset_type][varname][data_format], fname),
                        slice)

        if self.verbose_level >= 2:
            print('Done.')

    def save_variable(self, varname, data_format, dates=None):

        """
        Save a normalised 3-dimensional satellite/reanalysis dataset as daily
        averages (either the absolute values or the daily anomalies
        computed with xarray).

        This method assumes there is only one variable stored in the NetCDF files.

        Parameters:
        varname (str): Name of the variable to load & save

        data_format (str): 'abs' for absolute values, or 'anom' to compute the
        anomalies, or 'linear_trend' for SIC linear trend projections.

        dates (list of datetime): Months to use to compute the daily
        climatologies (defaults to the months used for training).
        """

        if data_format == 'anom':
            if dates is None:
                dates = self.obs_train_dates

        ########################################################################
        ################# Observational variable
        ########################################################################

        if self.verbose_level >= 2:
            print("Preprocessing {} data for {}...  ".format(data_format, varname), end='', flush=True)
            tic2 = time.time()

        daily_folder = os.path.join(config.folders['data'], varname)

        # Open all the NetCDF files in the given variable folder
        netcdf_regex = re.compile('^.*\\.nc$'.format(varname))
        filenames = sorted(os.listdir(daily_folder))  # List of files in month folder
        filenames = [filename for filename in filenames if netcdf_regex.match(filename)]
        paths = [os.path.join(daily_folder, filename) for filename in filenames]

        # Extract the first DataArray in the dataset
        with xr.open_mfdataset(paths, concat_dim='time', combine='nested') as ds:
            da = next(iter(ds.data_vars.values()))
            if len(ds.data_vars) > 1:
                warnings.warn('warning: there is more than one variable in the netcdf '
                              'file, but it is assumed that there is only one.')
                print('the loaded variable is: {}'.format(da.name))

        if data_format == 'anom':
            climatology = da.sel(time=dates).groupby('time.dayofyear', restore_coord_dims=True).mean()
            da = da.groupby('time.dayofyear') - climatology

        # Realise the array
        da.data = np.asarray(da.data, dtype=self.dtype)

        # Normalise the array
        if varname == 'siconca':
            # Don't normalsie SIC values - already betw 0 and 1
            mean, std = None, None
            min, max = None, None
        else:
            if self.minmax:
                da, min, max = self.normalise_array_using_all_training_data(da, self.minmax)
            elif not self.minmax:
                da, mean, std = self.normalise_array_using_all_training_data(da, self.minmax)

        da.data[np.isnan(da.data)] = 0.  # Convert any NaNs to zeros

        self.save_xarray_in_daily_averages(da, 'obs', varname, data_format)

        if self.verbose_level >= 2:
            print("Done in {:.3f}s.\n".format(time.time() - tic2))

    def preproc_and_save_icenet_data(self):

        '''
        Docstring TODO
        '''

        if self.verbose_level == 1:
            print("Loading and normalising the raw input maps... ", end='', flush=True)
            tic = time.time()

        for varname, vardict in self.preproc_vars.items():

            if 'metadata' not in vardict.keys():

                for data_format in vardict.keys():

                    if vardict[data_format] is True:

                        self.save_variable(varname, data_format)

            elif 'metadata' in vardict.keys():

                if vardict['include']:
                    if varname == 'land':
                        if self.verbose_level >= 2:
                            print("Setting up the land map: ", end='', flush=True)

                        land_mask = np.load(os.path.join(config.folders['masks'], config.fnames['land_mask']))
                        land_map = np.ones(self.raw_data_shape, self.dtype)
                        land_map[~land_mask] = -1.

                        np.save(os.path.join(self.paths['meta'], 'land.npy'), land_map)

                        print('\n')

                    elif varname == 'circday':
                        if self.verbose_level >= 2:
                            print("Computing circular day values... ", end='', flush=True)
                            tic2 = time.time()

                        # 2012 used arbitrarily as leap year
                        for date in pd.date_range(start='2012-1-1', end='2012-12-31'):

                            cos_month = np.cos(2 * np.pi * date.dayofyear / 366, dtype=self.dtype)
                            sin_month = np.sin(2 * np.pi * date.dayofyear / 366, dtype=self.dtype)

                            np.save(os.path.join(self.paths['meta'], date.strftime('cos_month_%m_%d.npy')), cos_month)
                            np.save(os.path.join(self.paths['meta'], date.strftime('sin_month_%m_%d.npy')), sin_month)

                        if self.verbose_level >= 2:
                            print("Done in {:.3f}s.\n".format(time.time() - tic2))

        if self.verbose_level == 1:
            print("Done in {:.3f}s.\n".format(time.time() - tic))


###############################################################################
############### MISC
###############################################################################


def filled_daily_dates(start_date, end_date):
    """
    Return a numpy array of datetimes, incrementing daily, starting at start_date and
    going up to (but not including) end_date.
    """

    daily_list = []
    date = start_date

    while date < end_date:
        daily_list.append(date)
        date += relativedelta(days=1)

    return np.array(daily_list)


def gen_frame_func(da, mask=None, mask_type='contour', data_type='abs',
                   clim=None, cmap='viridis', figsize=15):

    '''
    Create imageio frame function for xarray.DataArray visualisation.

    Parameters:
    da (xr.DataArray): Dataset to create video of.

    mask (np.ndarray): Boolean mask with True over masked elements to overlay
    as a contour or filled contour. Defaults to None (no mask plotting).

    mask_type (str): 'contour' or 'contourf' dictating whether the mask is overlaid
    as a contour line or a filled contour.

    data_type (str): 'abs' or 'anom' describing whether the data is in absolute
    or anomaly format. If anomaly, the colorbar is centred on 0.

    cmap (str): Matplotlib colormap.

    figsize (int or float): Figure size in inches.

    Returns:
    make_frame (function): Function to return a frame for imageio to
    turn into a video.
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

    return make_frame


def xarray_to_video(da, video_path, fps, mask=None, mask_type='contour', clim=None,
                    data_type='abs', video_dates=None, cmap='viridis', figsize=15):

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

    cmap (str): Matplotlib colormap.

    figsize (int or float): Figure size in inches.
    '''

    if video_dates is None:
        video_dates = [pd.Timestamp(date).to_pydatetime() for date in da.time.values]

    make_frame = gen_frame_func(da=da, mask=mask, mask_type=mask_type, clim=clim,
                                data_type=data_type, cmap=cmap, figsize=figsize)

    imageio.mimsave(video_path,
                    [make_frame(date) for date in tqdm(video_dates)],
                    fps=fps)
