import os
import itertools
import time

from datetime import datetime
from dateutil.relativedelta import relativedelta

import numpy as np
import tensorflow as tf

from . import config


class IceUNetDataLoader(tf.keras.utils.Sequence):
    """
    Generates batches of input-output tensors for training IceNet. Inherits from
    keras.utils.Sequence which ensures each the network trains once on each
    sample per epoch. Must implement a __len__ method that returns the
    number of batches and a __getitem__ method that returns a batch of data. The
    on_epoch_end method is called after each epoch.

    See: https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence

    """

    def __init__(self, input_data, dataset_name, batch_size, shuffle, n_forecast_months,
                 obs_train_dates, obs_val_dates, obs_test_dates,
                 verbose_level, raw_data_shape, default_seed,
                 unit_test_data_loader=False, dtype=np.float32,
                 loss_weight_months=True, loss_weight_classes=True,
                 cmip6_transfer_train_dict=None, cmip6_transfer_val_dict=None,
                 convlstm=False, n_convlstm_input_months=12):
        """
        Parameters:
        input_data (dict): Dictionary of dictionaries dictating which
        variables to include for IceNet's input 3D volumes and, if appropriate,
        a list of past month lags to grab the data from. The nested dictionaries
        have keys of "include" (a bool for whether to input that variable), and
        "lookbacks" (a list of ints for which past months to input, indexing
        from 0 relative to the most recent month).

            Example:

                input_data = {
                    "siconca":
                        {"abs": {"include": True, 'lookbacks': np.arange(0,12)},
                                 "anom": {"include": True, 'lookbacks': np.arange(0,3)},
                                 "linear_trend": {"include": True}},
                    "tas":
                        {"abs": {"include": False, 'lookbacks': np.arange(0,3)},
                         "anom": {"include": True, 'lookbacks': np.arange(0,3)}},
                    "rsds":
                        {"abs": {"include": True, 'lookbacks': np.arange(0,3)},
                         "anom": {"include": False, 'lookbacks': np.arange(0,3)}},
                    "rsus":
                        {"abs": {"include": True, 'lookbacks': np.arange(0,3)},
                         "anom": {"include": False, 'lookbacks': np.arange(0,3)}},
                    "tos":
                        {"abs": {"include": False, 'lookbacks': np.arange(0,3)},
                         "anom": {"include": True, 'lookbacks': np.arange(0,3)}},
                    "psl":
                        {"abs": {"include": True, 'lookbacks': np.arange(0,3)},
                         "anom": {"include": False, 'lookbacks': np.arange(0,3)}},
                    "zg500":
                        {"abs": {"include": True, 'lookbacks': np.arange(0,3)},
                         "anom": {"include": False, 'lookbacks': np.arange(0,3)}},
                    "zg250":
                        {"abs": {"include": True, 'lookbacks': np.arange(0,3)},
                         "anom": {"include": False, 'lookbacks': np.arange(0,3)}},
                    "ua10":
                        {"abs": {"include": True, 'lookbacks': np.arange(0,3)},
                         "anom": {"include": False, 'lookbacks': np.arange(0,3)}},
                    "uas":
                        {"abs": {"include": True, 'lookbacks': np.arange(0,3)},
                         "anom": {"include": False, 'lookbacks': np.arange(0,3)}},
                    "vas":
                        {"abs": {"include": True, 'lookbacks': np.arange(0,3)},
                         "anom": {"include": False, 'lookbacks': np.arange(0,3)}},
                    "sfcWind":
                        {"abs": {"include": True, 'lookbacks': np.arange(0,3)},
                         "anom": {"include": False, 'lookbacks': np.arange(0,3)}},
                    "land":
                        {"metadata": True,
                         "include": True},
                    "circmonth":
                        {"metadata": True,
                         "include": True},
                }

        dataset_name (str): Folder name of the dataset stored in
        data/network_datasets/.

        batch_size (int): Number of samples per training batch.

        shuffle (bool): Whether to shuffle the order of the individual batches
        between epochs.

        n_forecast_months (int): Total number of months ahead to predict.

        obs_train_dates (tuple): Tuple of output months (stored as datetimes)
        to use to build up the training set.

        obs_val_dates (tuple): As above but for the validation set.

        obs_test_dates (tuple): As above but for the test set.

        verbose_level (int): Controls how much to print. 0: Print nothing.
        1: Print key set-up stages. 2: Print debugging info. 3: Print when an
        output month is skipped due to missing data.

        raw_data_shape (tuple): Shape of input satellite data as (rows, cols).

        default_seed (int): Default random seed to use for shuffling the order
        of training samples a) before training, and b) after each training epoch.

        unit_test_data_loader (bool): Whether this IceUNetDataLoader instance
        is to be used for unit testing of its methods, in which case the raw
        data loading & normalising will not occur.

        dtype (type): Data type for the input-output data (default np.float32)

        cmip6_transfer_train_dict (dict): Data structure storing the output dates
        to use from each individual CMIP6 climate model and member_id.
        This is used to train on all cmip6 models and ensemble members at once.

            Example:

            cmip6_transfer_train_dict = {
                cmip6_model_name: {
                    'r1i1p1f1': transfer_train_dates_r1,
                    'r2i1p1f1': transfer_train_dates_r2_and_up,
                    'r3i1p1f1': transfer_train_dates_r2_and_up,
                    'r4i1p1f1': transfer_train_dates_r2_and_up,
                    'r5i1p1f1': transfer_train_dates_r2_and_up,
                }
            }

        cmip6_transfer_val_dict (dict): As above but for the CMIP6 validation
        set (used for early stopping during pre-training).

            Example:

            cmip6_transfer_val_dict = {
                cmip6_model_name: {
                    'r1i1p1f1': transfer_val_dates_r1,
                    'r2i1p1f1': transfer_val_dates_r2_and_up,
                    'r3i1p1f1': transfer_val_dates_r2_and_up,
                    'r4i1p1f1': transfer_val_dates_r2_and_up,
                    'r5i1p1f1': transfer_val_dates_r2_and_up,
                }
            }

        convlstm (bool): If true, generate batches for a ConvLSTM architecture
        rather than a U-Net architecture.

        n_convlstm_input_months (int): If generating ConvLSTM batches, this
        sets the number of input months to use for the input sequence

        """
#         self.input_data = input_data
#         self.dataset_name = dataset_name
#         self.batch_size = batch_size
#         self.shuffle = shuffle
#         self.n_forecast_months = n_forecast_months
#         self.obs_train_dates = np.array(obs_train_dates)
#         self.obs_val_dates = np.array(obs_val_dates)
#         self.obs_test_dates = np.array(obs_test_dates)
#         self.verbose_level = verbose_level
#         self.raw_data_shape = raw_data_shape
#         self.default_seed = default_seed
#         self.unit_test_data_loader = unit_test_data_loader
#         self.dtype = dtype
#         self.loss_weight_months = loss_weight_months
#         self.loss_weight_classes = loss_weight_classes
#         self.cmip6_transfer_train_dict = cmip6_transfer_train_dict
#         self.cmip6_transfer_val_dict = cmip6_transfer_val_dict
#         self.convlstm = convlstm
#         self.n_convlstm_input_months = n_convlstm_input_months

#        self.all_forecast_start_dates = self.obs_train_dates

#        self.do_transfer_learning = False

#        if self.unit_test_data_loader:
#            return

#        if cmip6_transfer_train_dict is not None and cmip6_transfer_val_dict is not None:
#            self.set_transfer_train_and_val_ids()

#        self.remove_missing_months()
#        self.set_variable_path_formats()
#        self.set_seed(self.default_seed)
#        self.set_number_of_input_channels_for_each_input_variable()
#        self.load_polarholes()
        self.determine_tot_num_channels()
        self.on_epoch_end()

        if self.verbose_level >= 1:
            print("Setup complete.\n")

    def set_transfer_train_and_val_ids(self):

        '''
        Use self.cmip6_transfer_train_dict and self.cmip6_transfer_val_dict
        to set up a numpy object array of 3-tuples of the form:

            (cmip6_model_name, member_id, output_start_date)

        These are used as IDs into the transfer data hierarchy
        to train on all cmip6 models and ensemble members at once.
        '''

        # Transfer training set IDs
        self.transfer_train_dates = []
        for cmip6_model_name, member_id_dict in self.cmip6_transfer_train_dict.items():
            for member_id, member_id_dates in member_id_dict.items():
                self.transfer_train_dates.extend(
                    itertools.product([cmip6_model_name], [member_id], member_id_dates)
                )
        self.transfer_train_dates = np.array(self.transfer_train_dates)

        # Transfer validation set IDs
        self.transfer_val_dates = []
        for cmip6_model_name, member_id_dict in self.cmip6_transfer_val_dict.items():
            for member_id, member_id_dates in member_id_dict.items():
                self.transfer_val_dates.extend(
                    itertools.product([cmip6_model_name], [member_id], member_id_dates)
                )
        self.transfer_val_dates = np.array(self.transfer_val_dates)

#     def set_variable_path_formats(self):
#
#         """
#         Initialise the paths to the .npy files of each variable based on
#         `self.input_data`. If `self.do_transfer_learning` is True (as set by
#         IceUNetDataLoader.turn_on_transfer_learning), then the paths to the
#         CMIP6 .npy files will be used instead.
#         """
#
#         if self.verbose_level >= 1:
#             print('Setting up the variable paths for {}... '.format(self.dataset_name))
#
#         # Parent folder for this dataset
#         self.dataset_path = os.path.join(config.data_folder, 'network_datasets', self.dataset_name)
#
#         # Dictionary data structure to store image variable_paths
#         self.variable_paths = {}
#
#         for varname, vardict in self.input_data.items():
#
#             if 'metadata' not in vardict.keys():
#                 self.variable_paths[varname] = {}
#
#                 for data_format in vardict.keys():
#
#                     if vardict[data_format]['include'] is True:
#
#                         if not self.do_transfer_learning:
#                             path = os.path.join(self.dataset_path, 'obs',
#                                                 varname, data_format, '{:04d}_{:02d}.npy')
#                         elif self.do_transfer_learning:
#                             path = os.path.join(self.dataset_path, 'transfer',
#                                                 '{}', '{}',
#                                                 varname, data_format, '{:04d}_{:02d}.npy')
#
#                         self.variable_paths[varname][data_format] = path
#
#             elif 'metadata' in vardict.keys():
#
#                 if vardict['include'] is True:
#
#                     if varname == 'land':
#                         path = os.path.join(self.dataset_path, 'meta', 'land.npy')
#                         self.variable_paths['land'] = path
#
#                     elif varname == 'circmonth':
#                         path = os.path.join(self.dataset_path, 'meta',
#                                             '{}_month_{:02d}.npy')
#                         self.variable_paths['circmonth'] = path
#
#         if self.verbose_level >= 1:
#             print('Done.')

#     def reset_data_loader_with_new_input_data(self):
#         """
#         If the data loader object's `input_data` field is updated, this method
#         must be called to update the other object parameters.
#         """
#         self.set_variable_path_formats()
#         self.set_number_of_input_channels_for_each_input_variable()
#         self.determine_tot_num_channels()
#
#     def set_seed(self, seed):
#         """
#         Set the seed used by the random generator (used to randomly shuffle
#         the ordering of training samples after each epoch).
#         """
#         if self.verbose_level >= 1:
#             print("Setting the data generator's random seed to {}".format(seed))
#         self.rng = np.random.default_rng(seed)

    def determine_variable_names(self):
        """
        Set up a list of strings for the names of each input variable (in the
        correct order) by looping over the `input_data` dictionary.
        """
        variable_names = []

        for varname, vardict in self.input_data.items():
            # Input variables that span time
            if 'metadata' not in vardict.keys():
                for data_format in vardict.keys():
                    if vardict[data_format]['include']:
                        if data_format != 'linear_trend':
                            if not self.convlstm:
                                for lb in vardict[data_format]['lookbacks']:
                                    variable_names.append(varname+'_{}_{}'.format(data_format, lb))
                            elif self.convlstm:
                                variable_names.append(varname+'_{}'.format(data_format))
                        elif data_format == 'linear_trend':
                            for leadtime in np.arange(1, self.n_forecast_months+1):
                                variable_names.append(varname+'_{}_{}'.format(data_format, leadtime))

            # Metadata input variables that don't span time
            elif 'metadata' in vardict.keys() and vardict['include']:
                if varname == 'land':
                    variable_names.append(varname)

                elif varname == 'circmonth':
                    variable_names.append('cos(month)')
                    variable_names.append('sin(month)')

        return variable_names

#     def set_number_of_input_channels_for_each_input_variable(self):
#         """
#         Build up the dict `self.num_input_channels_dict` to store the number of input
#         channels spanned by each input variable.
#         """
#
#         if self.verbose_level >= 1:
#             print("Setting the number of input months for each input variable.")
#
#         self.num_input_channels_dict = {}
#
#         for varname, vardict in self.input_data.items():
#             if 'metadata' not in vardict.keys():
#                 # Variables that span time
#                 for data_format in vardict.keys():
#                     if vardict[data_format]['include']:
#                         varname_format = varname+'_{}'.format(data_format)
#                         if data_format != 'linear_trend':
#                             if not self.convlstm:
#                                 lbs = vardict[data_format]['lookbacks']
#                                 self.num_input_channels_dict[varname_format] = len(lbs)
#                             else:
#                                 self.num_input_channels_dict[varname_format] = 1
#                         elif data_format == 'linear_trend':
#                             self.num_input_channels_dict[varname_format] = self.n_forecast_months

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
        date, based on the "lookbacks" options of self.input_data.
        """

        # Check input SIC months for missing data
        sic_lookback_list = []
        if self.input_data['siconca']['abs']['include']:
            lbs = self.input_data['siconca']['abs']['lookbacks']
            sic_lookback_list.extend(lbs)
        if self.input_data['siconca']['anom']['include']:
            lbs = self.input_data['siconca']['anom']['lookbacks']
            sic_lookback_list.extend(lbs)

        # Remove repeat lookbacks
        sic_lookback_set = list(set(sic_lookback_list))

        # Get deltas from the forecast month
        sic_lookback_list = np.array(sic_lookback_set) + 1

        input_dates = []
        for lookback in sic_lookback_list:
            input_dates.append(forecast_start_date - relativedelta(months=lookback))

        return input_dates

#     def check_for_missing_month(self, forecast_start_date):
#         """
#         Return a bool used to block out forecast_start_dates
#         if any of the input SIC maps are missing.
#
#         Note: If one of the forecast months are missing, the sample weight matrix
#         for that month will be all zeroes so that the samples for that month
#         do not appear in the loss function.
#         """
#         contains_missing_months = False
#
#         # # Check forecast dates
#         # for month_lead_i in range(self.n_forecast_months):
#         #     forecast_date = forecast_start_date + relativedelta(months=month_lead_i)
#         #     if any([forecast_date == missing_date for missing_date in config.missing_dates]):
#         #         contains_missing_months = True
#
#         # Check SIC input dates
#         input_dates = self.all_sic_input_dates_from_forecast_start_date(forecast_start_date)
#
#         for input_date in input_dates:
#             if any([input_date == missing_date for missing_date in config.missing_dates]):
#                 contains_missing_months = True
#
#         if contains_missing_months and self.verbose_level >= 3:
#             date_str = "{:04d}_{:02d}".format(forecast_start_date.year, forecast_start_date.month)
#             print("Skipping due to missing data for forecast month: {}.".format(date_str))
#
#         return contains_missing_months
#
#     def remove_missing_months(self):
#
#         '''
#         Remove dates from self.all_forecast_start_dates that depend on a missing
#         observation of monthly SIC.
#         '''
#
#         if self.do_transfer_learning:
#             pass
#         else:
#             if self.verbose_level >= 2:
#                 print('Checking output dates for missing SIC months... ')
#
#             for idx, output_start_date in enumerate(self.all_forecast_start_dates):
#                 if self.check_for_missing_month(output_start_date):
#                     if self.verbose_level >= 2:
#                         print('Removing {}, '.format(output_start_date.strftime('%Y_%m')))
#
#                     np.delete(self.all_forecast_start_dates, idx)
#
#     def load_polarholes(self):
#         """
#         This method loads the polar holes.
#         """
# 
#         if self.verbose_level >= 1:
#             tic = time.time()
#             print("Loading and augmenting the polar holes... ")
# 
#         polarhole_path = os.path.join(config.mask_data_folder, config.polarhole1_fname)
#         self.polarhole1_mask = np.load(polarhole_path)
# 
#         polarhole_path = os.path.join(config.mask_data_folder, config.polarhole2_fname)
#         self.polarhole2_mask = np.load(polarhole_path)
# 
#         if config.use_polarhole3:
#             polarhole_path = os.path.join(config.mask_data_folder, config.polarhole3_fname)
#             self.polarhole3_mask = np.load(polarhole_path)
# 
#         self.nopolarhole_mask = np.full((432, 432), False)
# 
#         if self.verbose_level >= 1:
#             print("Done in {:.3f}s.\n".format(time.time() - tic))

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
            if self.verbose_level >= 3:
                print("Output date: {}, polar hole: {}".format(
                    forecast_date.strftime("%Y_%m"), 1))

        elif oldest_input_date <= config.polarhole2_final_date:
            polarhole_mask = self.polarhole2_mask
            if self.verbose_level >= 3:
                print("Output date: {}, polar hole: {}".format(
                    forecast_date.strftime("%Y_%m"), 2))

        elif oldest_input_date <= config.polarhole3_final_date and config.use_polarhole3:
            polarhole_mask = self.polarhole3_mask
            if self.verbose_level >= 3:
                print("Output date: {}, polar hole: {}".format(
                    forecast_date.strftime("%Y_%m"), 3))

        else:
            polarhole_mask = self.nopolarhole_mask
            if self.verbose_level >= 3:
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
        output_active_grid_cell_mask_fname = config.active_grid_cell_file_format. \
            format(output_month_str)
        output_active_grid_cell_mask_path = os.path.join(config.mask_data_folder,
                                                         output_active_grid_cell_mask_fname)
        output_active_grid_cell_mask = np.load(output_active_grid_cell_mask_path)

        # Only use the polar hole mask if predicting observational data
        if not self.do_transfer_learning:
            polarhole_mask = self.determine_polar_hole_mask(forecast_date)

            # Add the polar hole mask to that land/ocean mask for the current month
            output_active_grid_cell_mask[polarhole_mask] = False

        return output_active_grid_cell_mask

    def turn_on_transfer_learning(self):

        '''
        Converts the data loader to use CMIP6 pre-training data
        for transfer learning.
        '''

        self.do_transfer_learning = True
        self.all_forecast_start_dates = self.transfer_train_dates

        self.on_epoch_end()  # Shuffle transfer training indexes

        self.set_variable_path_formats()

    def turn_off_transfer_learning(self):

        '''
        Converts the data loader back to using ERA5/NSIDC observational
        training data.
        '''

        self.do_transfer_learning = False

        self.all_forecast_start_dates = self.obs_train_dates
        self.remove_missing_months()
        self.on_epoch_end()  # Shuffle obs training indexes

        self.set_variable_path_formats()

    def convert_to_validation_data_loader(self):

        """
        This method resets the `all_forecast_start_dates` array to correspond to the
        validation months defined by `self.obs_val_dates`.
        """

        if self.do_transfer_learning:
            self.all_forecast_start_dates = self.transfer_val_dates
        elif not self.do_transfer_learning:
            self.all_forecast_start_dates = self.obs_val_dates
            self.remove_missing_months()

    def convert_to_test_data_loader(self):

        """
        As above but for the testing months defined by `self.obs_test_dates`
        """

        if self.do_transfer_learning:
            raise ValueError('Test set not intended for transfer learning dataset')
        else:
            self.all_forecast_start_dates = self.obs_test_dates
            self.remove_missing_months()

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

        if self.do_transfer_learning:
            cmip6_model_names = forecast_start_dates[:, 0]
            cmip6_member_ids = forecast_start_dates[:, 1]
            forecast_start_dates = forecast_start_dates[:, 2]

        ########################################################################
        # OUTPUT LABELS
        ########################################################################

        # Build up the set of N_samps output SIC time-series
        #   (each n_forecast_months long in the time dimension)

        # To become array of shape (N_samps, *raw_data_shape, n_forecast_months)
        batch_sic_list = []

        for sample_idx, forecast_date in enumerate(forecast_start_dates):

            # To become array of shape (*raw_data_shape, n_forecast_months)
            sample_sic_list = []

            for forecast_leadtime_idx in range(self.n_forecast_months):

                forecast_date = forecast_start_dates[sample_idx] + relativedelta(months=forecast_leadtime_idx)

                if self.do_transfer_learning:
                    sample_sic_list.append(
                        np.load(self.variable_paths['siconca']['abs'].format(
                            cmip6_model_names[sample_idx], cmip6_member_ids[sample_idx],
                            forecast_date.year, forecast_date.month))
                    )

                elif not self.do_transfer_learning:
                    if not os.path.exists(self.variable_paths['siconca']['abs'].format(forecast_date.year, forecast_date.month)):
                        # Output file does not exist - fill it with NaNs
                        sample_sic_list.append(np.full(self.raw_data_shape, np.nan))

                    else:
                        sample_sic_list.append(
                            np.load(self.variable_paths['siconca']['abs'].format(
                                forecast_date.year, forecast_date.month))
                        )

            batch_sic_list.append(np.stack(sample_sic_list, axis=0))

        batch_sic = np.stack(batch_sic_list, axis=0)

        # Move month index from axis 1 to axis 3
        batch_sic = np.moveaxis(batch_sic, source=1, destination=3)

        no_ice_gridcells = batch_sic <= 0.15
        ice_gridcells = batch_sic >= 0.80
        marginal_ice_gridcells = ~((no_ice_gridcells) | (ice_gridcells))

        # Categorical representation with channel dimension for class probs
        #   (length 4 with mask for zeroth channel)
        y = np.zeros((current_batch_size, *self.raw_data_shape, self.n_forecast_months, 4), dtype=self.dtype)

        y[no_ice_gridcells, 1] = 1
        y[marginal_ice_gridcells, 2] = 1
        y[ice_gridcells, 3] = 1

        y = np.moveaxis(y, source=3, destination=4)

        # 'Hacky' solution for pixelwise loss function weighting: also output
        #   the pixelwise sample weights as the first channel of y
        for sample_idx, forecast_date in enumerate(forecast_start_dates):

            for forecast_leadtime_idx in range(self.n_forecast_months):

                forecast_date = forecast_start_dates[sample_idx] + relativedelta(months=forecast_leadtime_idx)

                if any([forecast_date == missing_date for missing_date in config.missing_dates]):
                    sample_weight = np.zeros(self.raw_data_shape, self.dtype)

                else:
                    # Zero loss outside of 'active grid cells'
                    sample_weight = self.determine_active_grid_cell_mask(forecast_date)
                    sample_weight = sample_weight.astype(self.dtype)

                    # Scale the loss for each month s.t. March is
                    #   scaled by 1 and Sept is scaled by 1.77
                    if self.loss_weight_months:
                        sample_weight *= 33928. / np.sum(sample_weight)

                    # Weight each class for the loss by its frequency rel. to 'no ice'
                    if self.loss_weight_classes:
                        true_class = np.argmax(y[sample_idx, :, :, 1:, forecast_leadtime_idx], axis=-1)
                        sample_weight[true_class == 1] *= 6.83
                        sample_weight[true_class == 2] *= 0.49

                y[sample_idx, :, :, 0, forecast_leadtime_idx] = sample_weight

        ########################################################################
        # INPUT FEATURES
        ########################################################################


        # Batch tensor
        if not self.convlstm:
            X = np.zeros((current_batch_size, *self.raw_data_shape, self.tot_num_channels),
                         dtype=self.dtype)
        elif self.convlstm:
            X = np.zeros((current_batch_size, self.n_convlstm_input_months,
                          *self.raw_data_shape, self.tot_num_channels),
                         dtype=self.dtype)

        # Build up the batch of inputs
        for sample_idx, forecast_start_date in enumerate(forecast_start_dates):

            present_date = forecast_start_date - relativedelta(months=1)

            # Initialise variable indexes used to fill the input tensor `X`
            variable_idx1 = 0
            variable_idx2 = 0

            for varname, vardict in self.input_data.items():

                if 'metadata' not in vardict.keys():

                    for data_format in vardict.keys():

                        if vardict[data_format]['include']:

                            varname_format = '{}_{}'.format(varname, data_format)

                            if data_format != 'linear_trend':
                                if not self.convlstm:
                                    lbs = vardict[data_format]['lookbacks']
                                    input_months = [present_date - relativedelta(months=lb) for lb in lbs]
                                elif self.convlstm:
                                    lbs = np.arange(0, self.n_convlstm_input_months)
                                    input_months = [present_date - relativedelta(months=lb) for lb in lbs]
                                    input_months.reverse()  # Time from past to present
                            else:
                                input_months = [present_date + relativedelta(months=forecast_leadtime)
                                                for forecast_leadtime in np.arange(1, self.n_forecast_months+1)]

                            variable_idx2 += self.num_input_channels_dict[varname_format]

                            if not self.do_transfer_learning:
                                if not self.convlstm:
                                    X[sample_idx, :, :, variable_idx1:variable_idx2] = \
                                        np.stack([np.load(self.variable_paths[varname][data_format].format(
                                                  date.year, date.month))
                                                  for date in input_months], axis=-1)
                                elif self.convlstm:
                                    X[sample_idx, :, :, :, variable_idx1:variable_idx2] = \
                                        np.stack([np.load(self.variable_paths[varname][data_format].format(
                                                  date.year, date.month))
                                                  for date in input_months], axis=0)[:, :, :, np.newaxis]
                            elif self.do_transfer_learning:
                                cmip6_model_name = cmip6_model_names[sample_idx]
                                cmip6_member_id = cmip6_member_ids[sample_idx]

                                if not self.convlstm:
                                    X[sample_idx, :, :, variable_idx1:variable_idx2] = \
                                        np.stack([np.load(self.variable_paths[varname][data_format].format(
                                                  cmip6_model_name, cmip6_member_id, date.year, date.month))
                                                  for date in input_months], axis=-1)
                                elif self.convlstm:
                                    X[sample_idx, :, :, :, variable_idx1:variable_idx2] = \
                                        np.stack([np.load(self.variable_paths[varname][data_format].format(
                                                  cmip6_model_name, cmip6_member_id, date.year, date.month))
                                                  for date in input_months], axis=0)[:, :, :, np.newaxis]

                            variable_idx1 += self.num_input_channels_dict[varname_format]

                elif 'metadata' in vardict.keys() and vardict['include']:

                    variable_idx2 += self.num_input_channels_dict[varname]

                    if varname == 'land':
                        if not self.convlstm:
                            X[sample_idx, :, :, variable_idx1] = np.load(self.variable_paths['land'])
                        elif self.convlstm:
                            # Broadcast along time dimension
                            X[sample_idx, :, :, :, variable_idx1] = np.load(self.variable_paths['land'])[np.newaxis, :, :]

                    elif varname == 'circmonth':
                        if not self.convlstm:
                            # Broadcast along row and col dimensions
                            X[sample_idx, :, :, variable_idx1] = \
                                np.load(self.variable_paths['circmonth'].format('cos', forecast_start_date.month))
                            X[sample_idx, :, :, variable_idx1 + 1] = \
                                np.load(self.variable_paths['circmonth'].format('sin', forecast_start_date.month))
                        elif self.convlstm:
                            # Broadcast along time, row and col dimensions
                            X[sample_idx, :, :, :, variable_idx1] = \
                                np.load(self.variable_paths['circmonth'].format('cos', forecast_start_date.month))
                            X[sample_idx, :, :, :, variable_idx1 + 1] = \
                                np.load(self.variable_paths['circmonth'].format('sin', forecast_start_date.month))

                    variable_idx1 += self.num_input_channels_dict[varname]

        return X, y

    def __getitem__(self, index):
        '''
        Generate one batch of data of size `batch_size` at index `index`
        into the set of batches in the epoch.
        '''
        batch_forecast_start_dates = self.all_forecast_start_dates[index*self.batch_size:(index+1)*self.batch_size]

        return self.data_generation(batch_forecast_start_dates)

    def __len__(self):
        """ Returns the number of batches per training epoch. """
        return np.int(np.ceil(self.all_forecast_start_dates.shape[0] / self.batch_size))

    def get_month_of_input_output_data(self, forecast_start_date):
        """
        Generate the input-output data for IceNet for a particular forecast
        month, as well as the mask needed to convert the output predicted vector
        `y` into a map/2D array.

        Parameters:
        forecast_start_date (datetime.datetime): Datetime object corresponding to the
        first forecast month to generate the input-output data for.

        forecast_start_date (ndarray): If self.do_transfer_learning is False, this is
        a datetime object corresponding to the forecast start month.
        If self.do_transfer_learning is True, this is a 3-tuple
        of the form (cmip6_model_name, member_id, forecast_start_date).

        Returns:
        X (ndarray): Set of input 3D volumes.

        y (ndarray): Set of segmented output maps.

        output_masks (Boolean index array): Boolean index array with True on
        active grid cells where network predictions are made and False elsewhere.
        Shape: (self.n_forecast_months, *self.raw_data_shape).
        """

        sample_idx = forecast_start_date

        if self.do_transfer_learning:
            forecast_start_date = forecast_start_date[2]

        if self.verbose_level >= 2:
            print("Getting input-output data for the output month {}... ".
                  format(forecast_start_date.strftime("%Y_%m")))
            tic = time.time()

        X, y = self.data_generation(np.array([sample_idx]))

        output_masks = []
        for month_lead_i in range(self.n_forecast_months):
            forecast_date = forecast_start_date + relativedelta(months=month_lead_i)
            output_masks.append(self.determine_active_grid_cell_mask(forecast_date))
        output_masks = np.stack(output_masks, axis=0)

        if self.verbose_level >= 2:
            print("Done in {:.3f}s.\n".format(time.time() - tic))

        return X, y, output_masks

#     def on_epoch_end(self):
#         """ Randomly shuffles training samples after each epoch. """
# 
#         if self.verbose_level >= 2:
#             print("on_epoch_end called")
#         if self.shuffle:
#             # Randomly shuffle the output months in-place
#             self.rng.shuffle(self.all_forecast_start_dates)

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


class CachingDataLoader(IceUNetDataLoader):
    def __init__(self, *args, cache_path=os.path.join(".", "Xy_cache"), **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: These caches don't need to be retained after disk cache generation
        self._grid_cell_mask = {}
        self._sic_filecache = {}

        self._cache_path = cache_path

    def clear_memory_caches(self):
        if self.verbose_level >= 2:
            print("Grid cell cache: {}".format(len(self._grid_cell_mask.keys())))
            print("SIC file cache: {}".format(len(self._sic_filecache.keys())))
        self._grid_cell_mask = {}
        self._sic_filecache = {}

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

        if forecast_date.month not in self._grid_cell_mask:
            output_month_str = '{:02d}'.format(forecast_date.month)
            output_active_grid_cell_mask_fname = config.active_grid_cell_file_format. \
                format(output_month_str)
            output_active_grid_cell_mask_path = os.path.join(config.mask_data_folder,
                                                             output_active_grid_cell_mask_fname)
            self._grid_cell_mask[forecast_date.month] = np.load(output_active_grid_cell_mask_path)

        output_active_grid_cell_mask = np.copy(self._grid_cell_mask[forecast_date.month])

        # Only use the polar hole mask if predicting observational data
        if not self.do_transfer_learning:
            polarhole_mask = self.determine_polar_hole_mask(forecast_date)

            # Add the polar hole mask to that land/ocean mask for the current month
            output_active_grid_cell_mask[polarhole_mask] = False

        return output_active_grid_cell_mask

    def data_generation(self, forecast_start_dates):
        raise RuntimeError("data_generation not usable in it's previous context")

    def _sample_generation(self, forecast_start_dates):
        current_batch_size = forecast_start_dates.shape[0]

        if self.do_transfer_learning:
            cmip6_model_names = forecast_start_dates[:, 0]
            cmip6_member_ids = forecast_start_dates[:, 1]
            forecast_start_dates = forecast_start_dates[:, 2]

        # BEGIN

        ########################################################################
        # OUTPUT LABELS
        ########################################################################

        # Build up the set of N_samps output SIC time-series
        #   (each n_forecast_months long in the time dimension)

        # To become array of shape (N_samps, *raw_data_shape, n_forecast_months)
        batch_sic_list = []

        for sample_idx, forecast_date in enumerate(forecast_start_dates):

            # To become array of shape (*raw_data_shape, n_forecast_months)
            sample_sic_list = []

            for forecast_leadtime_idx in range(self.n_forecast_months):

                forecast_date = forecast_start_dates[sample_idx] + relativedelta(months=forecast_leadtime_idx)

                fname_args = [] if not self.do_transfer_learning else \
                    [cmip6_model_names[sample_idx], cmip6_member_ids[sample_idx]]
                fname_args += [forecast_date.year, forecast_date.month]

                sic_fname = self.variable_paths['siconca']['abs'].format(*fname_args)

                # Output file does not exist - fill it with NaNs
                if not os.path.exists(sic_fname):
                    sample_sic_list.append(np.full(self.raw_data_shape, np.nan))
                else:
                    if sic_fname not in self._sic_filecache:
                        self._sic_filecache[sic_fname] = np.load(sic_fname)

                    sample_sic_list.append(self._sic_filecache[sic_fname])

            batch_sic_list.append(np.stack(sample_sic_list, axis=0))

        batch_sic = np.stack(batch_sic_list, axis=0)

        # Move month index from axis 1 to axis 3
        batch_sic = np.moveaxis(batch_sic, source=1, destination=3)

        no_ice_gridcells = batch_sic <= 0.15
        ice_gridcells = batch_sic >= 0.80
        marginal_ice_gridcells = ~((no_ice_gridcells) | (ice_gridcells))

        # Categorical representation with channel dimension for class probs
        #   (length 4 with mask for zeroth channel)
        y = np.zeros((1, *self.raw_data_shape, self.n_forecast_months, 4), dtype=self.dtype)

        y[no_ice_gridcells, 1] = 1
        y[marginal_ice_gridcells, 2] = 1
        y[ice_gridcells, 3] = 1

        y = np.moveaxis(y, source=3, destination=4)

        # 'Hacky' solution for pixelwise loss function weighting: also output
        #   the pixelwise sample weights as the first channel of y
        for sample_idx, forecast_date in enumerate(forecast_start_dates):

            for forecast_leadtime_idx in range(self.n_forecast_months):

                forecast_date = forecast_start_dates[sample_idx] + relativedelta(months=forecast_leadtime_idx)

                if any([forecast_date == missing_date for missing_date in config.missing_dates]):
                    sample_weight = np.zeros(self.raw_data_shape, self.dtype)

                else:
                    # Zero loss outside of 'active grid cells'
                    sample_weight = self.determine_active_grid_cell_mask(forecast_date)
                    sample_weight = sample_weight.astype(self.dtype)

                    # Scale the loss for each month s.t. March is
                    #   scaled by 1 and Sept is scaled by 1.77
                    if self.loss_weight_months:
                        sample_weight *= 33928. / np.sum(sample_weight)

                    # Weight each class for the loss by its frequency rel. to 'no ice'
                    if self.loss_weight_classes:
                        true_class = np.argmax(y[sample_idx, :, :, 1:, forecast_leadtime_idx], axis=-1)
                        sample_weight[true_class == 1] *= 6.83
                        sample_weight[true_class == 2] *= 0.49

                y[sample_idx, :, :, 0, forecast_leadtime_idx] = sample_weight

        ########################################################################
        # INPUT FEATURES
        ########################################################################

        # Batch tensor
        x_sz = (1, *self.raw_data_shape, self.tot_num_channels)

        if self.convlstm:
            x_sz = (1, self.n_convlstm_input_months, *self.raw_data_shape, self.tot_num_channels)

        X = np.zeros(x_sz, dtype=self.dtype)

        # Build up the batch of inputs
        for sample_idx, forecast_start_date in enumerate(forecast_start_dates):

            present_date = forecast_start_date - relativedelta(months=1)

            # Initialise variable indexes used to fill the input tensor `X`
            vi1 = 0
            vi2 = 0

            for varname, vardict in self.input_data.items():

                if 'metadata' not in vardict.keys():

                    for data_format in vardict.keys():
                        varname_format = '{}_{}'.format(varname, data_format)

                        if vardict[data_format]['include']:

                            if data_format != 'linear_trend':
                                lbs = vardict[data_format]['lookbacks'] if not self.convlstm \
                                    else np.arange(0, self.n_convlstm_input_months)
                                input_months = [present_date - relativedelta(months=lb) for lb in lbs]

                                if self.convlstm:
                                    input_months.reverse()  # Time from past to present
                            else:
                                input_months = [present_date + relativedelta(months=forecast_leadtime)
                                                for forecast_leadtime in np.arange(1, self.n_forecast_months + 1)]

                            vi2 += self.num_input_channels_dict[varname_format]

                            x_idx = tuple([sample_idx, ..., slice(vi1, vi2)])

                            cmip_args = [] if not self.do_transfer_learning else \
                                [cmip6_model_names[sample_idx], cmip6_member_ids[sample_idx]]

                            if self.convlstm:
                                axis = 0
                                stack_slice = tuple([..., np.newaxis])
                            else:
                                axis = -1
                                stack_slice = tuple([...])

                            X[x_idx] = np.stack([
                                np.load(self.variable_paths[varname][data_format].format(
                                    *cmip_args, date.year, date.month)) for date in input_months],
                                axis=axis)[stack_slice]

                            vi1 += self.num_input_channels_dict[varname_format]

                elif 'metadata' in vardict.keys() and vardict['include']:

                    vi2 += self.num_input_channels_dict[varname]

                    if varname == 'land':
                        arr_slice = [np.newaxis, ...] if self.convlstm else [...]
                        meta_fn = self.variable_paths['land']
                        X[sample_idx, ..., vi1] = np.load(meta_fn)[tuple(arr_slice)]

                    elif varname == 'circmonth':
                        # Broadcast along row and col dimensions
                        X[sample_idx, ..., vi1] = \
                            np.load(self.variable_paths['circmonth'].format('cos', forecast_start_date.month))
                        X[sample_idx, ..., vi1 + 1] = \
                            np.load(self.variable_paths['circmonth'].format('sin', forecast_start_date.month))

                    vi1 += self.num_input_channels_dict[varname]

        ## END

        if len(X[:, ...]) != 1 or len(y[:, ...]) != 1:
            raise RuntimeWarning("__sample_generation should only be producing a single sample at a time")

        x = X[0, ...]
        y = y[0, ...]
        return x, y

    def __getitem__(self, index):
        '''
        Generate one batch of data of size `batch_size` at index `index`
        into the set of batches in the epoch.
        '''
        batch_forecast_start_dates = self.all_forecast_start_dates[
                                     index * self.batch_size:(index + 1) * self.batch_size]
        print("Index: {}".format(index))
        return self._retrieve_data(batch_forecast_start_dates)

    def _retrieve_data(self, batch_forecast_start_dates):
        x_arr = []
        y_arr = []

        for dt in batch_forecast_start_dates:
            if type(dt) == datetime:
                cache_fn = "{}.npz"
                cache_path = os.path.join(self._cache_path, cache_fn.format(dt.strftime("%d%m%Y")))
            else:
                cache_fn = ".".join(["{}"] * 3) + ".npz"
                cache_path = os.path.join(self._cache_path, cache_fn.format(dt[0], dt[1], dt[2].strftime("%d%m%Y")))

            # Restore the structure expected by data_generator
            dt = np.array([dt])

            if os.path.exists(cache_path):
                if self.verbose_level >= 2:
                    print("Cached: {}".format(cache_path))
                cache_data = np.load(cache_path)
                (x, y) = (cache_data['x'], cache_data['y'])
            else:
                (x, y) = self.__sample_generation(dt)

                if not os.path.exists(self._cache_path):
                    if self.verbose_level >= 2:
                        print("New cache directory created: {}".format(os.path.join(os.getcwd(), self._cache_path)))
                    os.makedirs(self._cache_path)

                if self.verbose_level >= 2:
                    print("Saving: {}".format(cache_path))
                np.savez_compressed(cache_path, x=x, y=y)

            x_arr.append(x)
            y_arr.append(y)

        X, y = np.stack(x_arr), np.stack(y_arr)

        return X, y

    def __len__(self):
        ''' Returns the number of batches per training epoch. '''
        return np.int(np.ceil(self.all_forecast_start_dates.shape[0] / self.batch_size))

    def get_month_of_input_output_data(self, forecast_start_date):
        """
        Generate the input-output data for IceNet for a particular forecast
        month, as well as the mask needed to convert the output predicted vector
        `y` into a map/2D array.

        Parameters:
        forecast_start_date (datetime.datetime): Datetime object corresponding to the
        first forecast month to generate the input-output data for.

        forecast_start_date (ndarray): If self.do_transfer_learning is False, this is
        a datetime object corresponding to the forecast start month.
        If self.do_transfer_learning is True, this is a 3-tuple
        of the form (cmip6_model_name, member_id, forecast_start_date).

        Returns:
        X (ndarray): Set of input 3D volumes.

        y (ndarray): Set of segmented output maps.

        output_masks (Boolean index array): Boolean index array with True on
        active grid cells where network predictions are made and False elsewhere.
        Shape: (self.n_forecast_months, *self.raw_data_shape).
        """

        sample_idx = forecast_start_date

        if self.do_transfer_learning:
            forecast_start_date = forecast_start_date[2]

        if self.verbose_level >= 2:
            print("Getting input-output data for the output month {}... ".
                  format(forecast_start_date.strftime("%Y_%m")), end='', flush=True)
            tic = time.time()

        X, y = self._retrieve_data(np.array([sample_idx]))

        output_masks = []
        for month_lead_i in range(self.n_forecast_months):
            forecast_date = forecast_start_date + relativedelta(months=month_lead_i)
            output_masks.append(self.determine_active_grid_cell_mask(forecast_date))
        output_masks = np.stack(output_masks, axis=0)

        if self.verbose_level >= 2:
            print("Done in {:.3f}s.\n".format(time.time() - tic))

        return X, y, output_masks

