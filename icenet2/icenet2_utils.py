import os
import sys
import warnings
import numpy as np
from datetime import datetime
sys.path.insert(0, os.path.join(os.getcwd(), 'icenet2'))  # if using jupyter kernel
import config
import icenet2_utils
from dateutil.relativedelta import relativedelta
from mpl_toolkits.axes_grid1 import make_axes_locatable
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, \
    Conv2D, BatchNormalization, UpSampling2D, concatenate, MaxPooling2D, \
    Input, TimeDistributed, ConvLSTM2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import xarray as xr
import pandas as pd
import regex as re
import json
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


def unet_batchnorm(input_shape, loss, metrics, learning_rate=1e-4, filter_size=3,
                   n_filters_factor=1, n_forecast_days=1, **kwargs):
    inputs = Input(shape=input_shape)

    conv1 = Conv2D(np.int(64*n_filters_factor), filter_size, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(np.int(64*n_filters_factor), filter_size, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    bn1 = BatchNormalization(axis=-1)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(bn1)

    conv2 = Conv2D(np.int(128*n_filters_factor), filter_size, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(np.int(128*n_filters_factor), filter_size, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    bn2 = BatchNormalization(axis=-1)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(bn2)

    conv3 = Conv2D(np.int(256*n_filters_factor), filter_size, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(np.int(256*n_filters_factor), filter_size, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    bn3 = BatchNormalization(axis=-1)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(bn3)

    conv4 = Conv2D(np.int(256*n_filters_factor), filter_size, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(np.int(256*n_filters_factor), filter_size, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    bn4 = BatchNormalization(axis=-1)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(bn4)

    conv5 = Conv2D(np.int(512*n_filters_factor), filter_size, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(np.int(512*n_filters_factor), filter_size, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    bn5 = BatchNormalization(axis=-1)(conv5)

    up6 = Conv2D(np.int(256*n_filters_factor), 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2,2), interpolation='nearest')(bn5))
    merge6 = concatenate([bn4, up6], axis=3)
    conv6 = Conv2D(np.int(256*n_filters_factor), filter_size, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(np.int(256*n_filters_factor), filter_size, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    bn6 = BatchNormalization(axis=-1)(conv6)

    up7 = Conv2D(np.int(256*n_filters_factor), 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2,2), interpolation='nearest')(bn6))
    merge7 = concatenate([bn3,up7], axis=3)
    conv7 = Conv2D(np.int(256*n_filters_factor), filter_size, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(np.int(256*n_filters_factor), filter_size, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    bn7 = BatchNormalization(axis=-1)(conv7)

    up8 = Conv2D(np.int(128*n_filters_factor), 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2,2), interpolation='nearest')(bn7))
    merge8 = concatenate([bn2,up8], axis=3)
    conv8 = Conv2D(np.int(128*n_filters_factor), filter_size, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(np.int(128*n_filters_factor), filter_size, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    bn8 = BatchNormalization(axis=-1)(conv8)

    up9 = Conv2D(np.int(64*n_filters_factor), 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2,2), interpolation='nearest')(bn8))
    merge9 = concatenate([conv1,up9], axis=3)
    conv9 = Conv2D(np.int(64*n_filters_factor), filter_size, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(np.int(64*n_filters_factor), filter_size, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(np.int(64*n_filters_factor), filter_size, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

    final_layer = [(Conv2D(3, 1, activation='sigmoid')(conv9)) for i in range(n_forecast_days)]

    model = Model(inputs, final_layer)

    model.compile(optimizer=Adam(lr=learning_rate), loss=loss, metrics=metrics)

    return model




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


class IceNet2DataLoader(tf.keras.utils.Sequence):
    """
    Generates batches of input-output tensors for training IceNet2. Inherits from
    keras.utils.Sequence which ensures each the network trains once on each
    sample per epoch. Must implement a __len__ method that returns the
    number of batches and a __getitem__ method that returns a batch of data. The
    on_epoch_end method is called after each epoch.

    See: https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence

    """

    def __init__(self, data_loader_config_path):

        with open(data_loader_config_path, 'r') as readfile:
            self.config = json.load(readfile)

        forecast_start_date_ends = [
            pd.Timestamp(date).to_pydatetime() for date in self.config['obs_train_dates']
        ]

        self.all_forecast_start_dates = icenet2_utils.filled_daily_dates(
            forecast_start_date_ends[0],
            forecast_start_date_ends[1])

        self.load_missing_dates()
        self.remove_missing_dates()
        self.set_variable_path_formats()
        self.set_seed(self.config['default_seed'])
        self.set_number_of_input_channels_for_each_input_variable()
        self.load_polarholes()
        self.determine_tot_num_channels()
        self.on_epoch_end()

        if self.config['verbose_level'] >= 1:
            print("Setup complete.\n")

    def set_variable_path_formats(self):

        """
        Initialise the paths to the .npy files of each variable based on
        `self.config['input_data']`. If `self.do_transfer_learning` is True (as set by
        IceNet2DataLoader.turn_on_transfer_learning), then the paths to the
        CMIP6 .npy files will be used instead.
        """

        if self.config['verbose_level'] >= 1:
            print('Setting up the variable paths for {}... '.format(self.config['dataset_name']),
                  end='', flush=True)

        # Parent folder for this dataset
        self.dataset_path = os.path.join(config.folders['data'], 'network_datasets', self.config['dataset_name'])

        # Dictionary data structure to store image variable_paths
        self.variable_paths = {}

        for varname, vardict in self.config['input_data'].items():

            if 'metadata' not in vardict.keys():
                self.variable_paths[varname] = {}

                for data_format in vardict.keys():

                    if vardict[data_format]['include'] is True:

                        path = os.path.join(self.dataset_path, 'obs',
                                            varname, data_format, '{:04d}_{:02d}_{:02d}.npy')

                        self.variable_paths[varname][data_format] = path

            elif 'metadata' in vardict.keys():

                if vardict['include'] is True:

                    if varname == 'land':
                        path = os.path.join(self.dataset_path, 'meta', 'land.npy')
                        self.variable_paths['land'] = path

                    elif varname == 'circday':
                        path = os.path.join(self.dataset_path, 'meta',
                                            '{}_month_{:02d}_{:02d}.npy')
                        self.variable_paths['circday'] = path

        if self.config['verbose_level'] >= 1:
            print('Done.')

    def reset_data_loader_with_new_input_data(self):
        """
        If the data loader object's `input_data` field is updated, this method
        must be called to update the other object parameters.
        """
        self.set_variable_path_formats()
        self.set_number_of_input_channels_for_each_input_variable()
        self.determine_tot_num_channels()

    def set_seed(self, seed):
        """
        Set the seed used by the random generator (used to randomly shuffle
        the ordering of training samples after each epoch).
        """
        if self.config['verbose_level'] >= 1:
            print("Setting the data generator's random seed to {}".format(seed))
        self.rng = np.random.default_rng(seed)

    def determine_variable_names(self):
        """
        Set up a list of strings for the names of each input variable (in the
        correct order) by looping over the `input_data` dictionary.
        """
        variable_names = []

        for varname, vardict in self.config['input_data'].items():
            # Input variables that span time
            if 'metadata' not in vardict.keys():
                for data_format in vardict.keys():
                    if vardict[data_format]['include']:
                        if data_format != 'linear_trend':
                            for lag in np.arange(1, vardict[data_format]['max_lag']+1):
                                variable_names.append(varname+'_{}_{}'.format(data_format, lag))
                        elif data_format == 'linear_trend':
                            for leadtime in np.arange(1, self.config['n_forecast_days']+1):
                                variable_names.append(varname+'_{}_{}'.format(data_format, leadtime))

            # Metadata input variables that don't span time
            elif 'metadata' in vardict.keys() and vardict['include']:
                if varname == 'land':
                    variable_names.append(varname)

                elif varname == 'circday':
                    variable_names.append('cos(day)')
                    variable_names.append('sin(day)')

        return variable_names

    def set_number_of_input_channels_for_each_input_variable(self):
        """
        Build up the dict `self.num_input_channels_dict` to store the number of input
        channels spanned by each input variable.
        """

        if self.config['verbose_level'] >= 1:
            print("Setting the number of input months for each input variable.")

        self.num_input_channels_dict = {}

        for varname, vardict in self.config['input_data'].items():
            if 'metadata' not in vardict.keys():
                # Variables that span time
                for data_format in vardict.keys():
                    if vardict[data_format]['include']:
                        varname_format = varname+'_{}'.format(data_format)
                        if data_format != 'linear_trend':
                            self.num_input_channels_dict[varname_format] = vardict[data_format]['max_lag']
                        elif data_format == 'linear_trend':
                            self.num_input_channels_dict[varname_format] = self.config['n_forecast_days']

            # Metadata input variables that don't span time
            elif 'metadata' in vardict.keys() and vardict['include']:
                if varname == 'land':
                    self.num_input_channels_dict[varname] = 1

                if varname == 'circday':
                    self.num_input_channels_dict[varname] = 2

    def determine_tot_num_channels(self):
        """
        Determine the number of channels for the input 3D volumes.
        """

        self.tot_num_channels = 0
        for varname, num_channels in self.num_input_channels_dict.items():
            self.tot_num_channels += num_channels

    def all_sic_input_dates_from_forecast_start_date(self, forecast_start_date):
        """
        Return a list of all the SIC dates used as input for a particular forecast
        date, based on the "max_lag" options of self.config['input_data'].
        """

        # Find all SIC lags
        max_lags = []
        if self.config['input_data']['siconca']['abs']['include']:
            max_lags.append(self.config['input_data']['siconca']['abs']['max_lag'])
        if self.config['input_data']['siconca']['anom']['include']:
            max_lags.append(self.config['input_data']['siconca']['anom']['max_lag'])
        max_lag = np.max(max_lags)

        input_dates = [
            forecast_start_date - relativedelta(days=int(lag)) for lag in np.arange(1, max_lag+1)
        ]

        return input_dates

    def check_for_missing_date(self, forecast_start_date):
        """
        Return a bool used to block out forecast_start_dates
        if any of the input SIC maps are missing.

        Note: If one of the _forecast_ dates are missing but not _input_ dates,
        the sample weight matrix for that date will be all zeroes so that the
        samples for that date do not appear in the loss function.
        """
        contains_missing_date = False

        # Check SIC input dates
        input_dates = self.all_sic_input_dates_from_forecast_start_date(forecast_start_date)

        for input_date in input_dates:
            if any([input_date == missing_date for missing_date in self.missing_dates]):
                contains_missing_date = True

        return contains_missing_date

    def load_missing_dates(self):

        '''
        Load missing SIC day spreadsheet and use it to build up a list of all
        missing days
        '''

        missing_date_df = pd.read_csv(os.path.join(config.folders['data'], config.fnames['missing_sic_days']))
        self.missing_dates = []
        for idx, row in missing_date_df.iterrows():
            # Ensure hour = 0 convention for daily dates
            start = pd.Timestamp(row['start']).to_pydatetime().replace(hour=0)
            end = pd.Timestamp(row['end']).to_pydatetime().replace(hour=0)
            self.missing_dates.extend(
                icenet2_utils.filled_daily_dates(start, end, include_end=True)
            )

    def remove_missing_dates(self):

        '''
        Remove dates from self.all_forecast_start_dates that depend on a missing
        observation of SIC.
        '''

        if self.config['verbose_level'] >= 2:
            print('Checking forecast start dates for missing SIC dates... ', end='', flush=True)

        new_all_forecast_start_dates = []
        for idx, forecast_start_date in enumerate(self.all_forecast_start_dates):
            if self.check_for_missing_date(forecast_start_date):
                if self.config['verbose_level'] >= 3:
                    print('Removing {}, '.format(forecast_start_date.strftime('%Y_%m_%d')), end='', flush=True)

            else:
                new_all_forecast_start_dates.append(forecast_start_date)

        self.all_forecast_start_dates = np.array(new_all_forecast_start_dates)

    def load_polarholes(self):
        """
        This method loads the polar holes.
        """

        if self.config['verbose_level'] >= 1:
            tic = time.time()
            print("Loading and augmenting the polar holes... ", end='', flush=True)

        polarhole_path = os.path.join(config.folders['masks'], config.fnames['polarhole1'])
        self.polarhole1_mask = np.load(polarhole_path)

        polarhole_path = os.path.join(config.folders['masks'], config.fnames['polarhole2'])
        self.polarhole2_mask = np.load(polarhole_path)

        self.nopolarhole_mask = np.full((432, 432), False)

        if self.config['verbose_level'] >= 1:
            print("Done in {:.3f}s.\n".format(time.time() - tic))

    def determine_polar_hole_mask(self, forecast_date):
        """
        Determine which polar hole mask to use (if any) by finding the oldest SIC
        input month based on the current output month. The polar hole active for
        the oldest input month is used (because the polar hole size decreases
        monotonically over time, and we wish to use the largest polar hole for
        the input data).

        Parameters:
        forecast_date (datetime): Month timepoint for the output date being
        predicted in (year, month, 1) format.

        Returns:
        polarhole_mask: Mask array with NaNs on polar hole grid cells and 1s
        elsewhere.
        """

        oldest_input_date = min(self.all_sic_input_dates_from_forecast_start_date(forecast_date))

        if oldest_input_date <= config.polarhole1_final_date:
            polarhole_mask = self.polarhole1_mask
            if self.config['verbose_level'] >= 3:
                print("Output date: {}, polar hole: {}".format(
                    forecast_date.strftime("%Y_%m"), 1))

        elif oldest_input_date <= config.polarhole2_final_date:
            polarhole_mask = self.polarhole2_mask
            if self.config['verbose_level'] >= 3:
                print("Output date: {}, polar hole: {}".format(
                    forecast_date.strftime("%Y_%m"), 2))

        else:
            polarhole_mask = self.nopolarhole_mask
            if self.config['verbose_level'] >= 3:
                print("Output date: {}, polar hole: {}".format(
                    forecast_date.strftime("%Y_%m"), "none"))

        return polarhole_mask

    def determine_active_grid_cell_mask(self, forecast_date):
        """
        Determine which active grid cell mask to use (a boolean array with
        True on active cells and False on inactive cells). The cells with 'True'
        are where predictions are to be made with IceNet. The active grid cell
        mask for a particular month is determined by the sum of the land cells,
        the ocean cells (for that month), and the missng polar hole.

        The mask is used for removing 'inactive' cells (such as land or polar
        hole cells) from the loss function in self.data_generation.
        """

        output_month_str = '{:02d}'.format(forecast_date.month)
        output_active_grid_cell_mask_fname = config.formats['active_grid_cell_mask']. \
            format(output_month_str)
        output_active_grid_cell_mask_path = os.path.join(config.folders['masks'],
                                                         output_active_grid_cell_mask_fname)
        output_active_grid_cell_mask = np.load(output_active_grid_cell_mask_path)

        # Only use the polar hole mask if predicting observational data
        polarhole_mask = self.determine_polar_hole_mask(forecast_date)

        # Add the polar hole mask to that land/ocean mask for the current month
        output_active_grid_cell_mask[polarhole_mask] = False

        return output_active_grid_cell_mask

    def convert_to_validation_data_loader(self):

        """
        This method resets the `all_forecast_start_dates` array to correspond to the
        validation months defined by `self.obs_val_dates`.
        """

        self.all_forecast_start_dates = self.obs_val_dates
        self.remove_missing_dates()

    def convert_to_test_data_loader(self):

        """
        As above but for the testing months defined by `self.obs_test_dates`
        """

        self.all_forecast_start_dates = self.obs_test_dates
        self.remove_missing_dates()

    def data_generation(self, forecast_start_dates):
        """
        Generate input-output data for IceNet at defined indexes into the SIC
        satellite array.

        Parameters:
        forecast_start_dates (ndarray): If self.do_transfer_learning is False, this is
        an (N_samps,) array of datetime objects corresponding to the output (forecast) months.
        If self.do_transfer_learning is True, this is an (N_samps, 3) object array of tuples
        of the form (cmip6_model_name, member_id, forecast_start_date).

        Returns:
        X (ndarray): Set of input 3D volumes.

        y (ndarray): Set of categorical segmented output maps with pixel
        weighting as first channel.

        """
        current_batch_size = forecast_start_dates.shape[0]

        ########################################################################
        # OUTPUT LABELS
        ########################################################################

        # Build up the set of N_samps output SIC time-series
        #   (each n_forecast_days long in the time dimension)

        # To become array of shape (N_samps, *self.config['raw_data_shape'], self.config['n_forecast_days'])
        batch_sic_list = []

        for sample_idx, forecast_date in enumerate(forecast_start_dates):

            # To become array of shape (*config['raw_data_shape'], config['n_forecast_days'])
            sample_sic_list = []

            for forecast_leadtime_idx in range(self.config['n_forecast_days']):

                forecast_date = forecast_start_dates[sample_idx] + relativedelta(days=forecast_leadtime_idx)

                if not os.path.exists(
                    self.variable_paths['siconca']['abs'].format(
                        forecast_date.year,
                        forecast_date.month,
                        forecast_date.day)):
                    # Output file does not exist - fill it with NaNs
                    sample_sic_list.append(np.full(self.config['raw_data_shape'], np.nan))

                else:
                    # Output file exists
                    sample_sic_list.append(
                        np.load(self.variable_paths['siconca']['abs'].format(
                            forecast_date.year, forecast_date.month, forecast_date.day))
                    )

            batch_sic_list.append(np.stack(sample_sic_list, axis=0))

        batch_sic = np.stack(batch_sic_list, axis=0)

        # Move day index from axis 1 to axis 3
        batch_sic = np.moveaxis(batch_sic, source=1, destination=3)

        # 'Hacky' solution for pixelwise loss function weighting: also output
        #   the pixelwise sample weights as the last channel of y
        y = np.zeros((current_batch_size,
                      *self.config['raw_data_shape'],
                      self.config['n_forecast_days'],
                      2),
                     dtype=np.float32)

        y[:, :, :, :, 0] = batch_sic

        for sample_idx, forecast_date in enumerate(forecast_start_dates):

            for forecast_leadtime_idx in range(self.config['n_forecast_days']):

                forecast_date = forecast_start_dates[sample_idx] + relativedelta(days=forecast_leadtime_idx)

                if any([forecast_date == missing_date for missing_date in self.missing_dates]):
                    sample_weight = np.zeros(self.config['raw_data_shape'], np.float32)

                else:
                    # Zero loss outside of 'active grid cells'
                    sample_weight = self.determine_active_grid_cell_mask(forecast_date)
                    sample_weight = sample_weight.astype(np.float32)

                    # Scale the loss for each month s.t. March is
                    #   scaled by 1 and Sept is scaled by 1.77
                    if self.config['loss_weight_months']:
                        sample_weight *= 33928. / np.sum(sample_weight)

                y[sample_idx, :, :, forecast_leadtime_idx, 1] = sample_weight

        ########################################################################
        # INPUT FEATURES
        ########################################################################

        # Batch tensor
        X = np.zeros((current_batch_size, *self.config['raw_data_shape'], self.tot_num_channels),
                     dtype=np.float32)

        # Build up the batch of inputs
        for sample_idx, forecast_start_date in enumerate(forecast_start_dates):

            present_date = forecast_start_date - relativedelta(days=1)

            # Initialise variable indexes used to fill the input tensor `X`
            variable_idx1 = 0
            variable_idx2 = 0

            for varname, vardict in self.config['input_data'].items():

                if 'metadata' not in vardict.keys():

                    for data_format in vardict.keys():

                        if vardict[data_format]['include']:

                            varname_format = '{}_{}'.format(varname, data_format)

                            if data_format != 'linear_trend':
                                max_lag = vardict[data_format]['max_lag']
                                input_months = [present_date - relativedelta(days=int(lag))
                                                for lag in np.arange(1, max_lag+1)]
                            elif data_format == 'linear_trend':
                                input_months = [present_date + relativedelta(days=int(lead))
                                                for lead in np.arange(1, self.config['n_forecast_days']+1)]

                            variable_idx2 += self.num_input_channels_dict[varname_format]

                            X[sample_idx, :, :, variable_idx1:variable_idx2] = \
                                np.stack([np.load(self.variable_paths[varname][data_format].format(
                                          date.year, date.month, date.day))
                                          for date in input_months], axis=-1)

                            variable_idx1 += self.num_input_channels_dict[varname_format]

                elif 'metadata' in vardict.keys() and vardict['include']:

                    variable_idx2 += self.num_input_channels_dict[varname]

                    if varname == 'land':
                        X[sample_idx, :, :, variable_idx1] = np.load(self.variable_paths['land'])

                    elif varname == 'circday':
                        # Broadcast along row and col dimensions
                        X[sample_idx, :, :, variable_idx1] = \
                            np.load(self.variable_paths['circday'].format(
                                'cos',
                                forecast_start_date.month,
                                forecast_start_date.day))
                        X[sample_idx, :, :, variable_idx1 + 1] = \
                            np.load(self.variable_paths['circday'].format(
                                'sin',
                                forecast_start_date.month,
                                forecast_start_date.day))

                    variable_idx1 += self.num_input_channels_dict[varname]

        return X, y

    def __getitem__(self, index):
        '''
        Generate one batch of data of size `batch_size` at index `index`
        into the set of batches in the epoch.
        '''
        batch_forecast_start_dates = \
            self.all_forecast_start_dates[index*self.config['batch_size']:(index+1)*self.config['batch_size']]

        return self.data_generation(batch_forecast_start_dates)

    def __len__(self):
        ''' Returns the number of batches per training epoch. '''
        return int(np.ceil(self.all_forecast_start_dates.shape[0] / self.config['batch_size']))

    def on_epoch_end(self):
        """ Randomly shuffles training samples after each epoch. """

        if self.config['verbose_level'] >= 2:
            print("on_epoch_end called")

        # Randomly shuffle the output months in-place
        self.rng.shuffle(self.all_forecast_start_dates)

    def time_batch_generation(self, num_batches):
        """ Print the time taken to generate `num_batches` batches """
        tot_dur = 0
        tic_batch_gen = time.time()
        for batch_idx in range(num_batches):
            X, y = self.__getitem__(batch_idx)
        tot_dur = time.time() - tic_batch_gen

        dur_per_batch = 1000 * tot_dur / num_batches  # in ms
        dur_per_epoch = dur_per_batch * len(self) / 1000  # in seconds
        dur_per_epoch_min = np.floor(dur_per_epoch / 60)
        dur_per_epoch_sec = dur_per_epoch % 60

        print("Duration: {:.2f}s for {} batches, {:.2f}ms per batch, {:.0f}m:{:.0f}s per epoch".
              format(tot_dur, num_batches, dur_per_batch, dur_per_epoch_min, dur_per_epoch_sec))


###############################################################################
############### MISC
###############################################################################


def filled_daily_dates(start_date, end_date, include_end=False):
    """
    Return a numpy array of datetimes, incrementing daily, starting at start_date and
    going up to (but not including) end_date.
    """

    daily_list = []
    date = start_date

    if include_end:
        end_date += relativedelta(days=1)

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
