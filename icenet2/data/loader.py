class IceNetDataLoader(tf.keras.utils.Sequence):
    """
    Custom data loader class for generating batches of input-output tensors for
    training IceNet. Inherits from  keras.utils.Sequence, which ensures each the
    network trains once on each  sample per epoch. Must implement a __len__
    method that returns the  number of batches and a __getitem__ method that
    returns a batch of data. The  on_epoch_end method is called after each
    epoch.
    See: https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence
    """

    def __init__(self, dataloader_config_fpath, seed=None):

        '''
        Params:
        dataloader_config_fpath (str): Path to the data loader configuration
            settings JSON file, defining IceNet's input-output data configuration.
        seed (int): Random seed used for shuffling the training samples before
            each epoch.
        '''

        with open(dataloader_config_fpath, 'r') as readfile:
            self.config = json.load(readfile)

        if seed is None:
            self.set_seed(self.config['default_seed'])
        else:
            self.set_seed(seed)

        self.do_transfer_learning = False

        self.set_obs_forecast_IDs(dataset='train')
        self.set_transfer_forecast_IDs()
        self.all_forecast_IDs = self.obs_forecast_IDs
        self.remove_missing_dates()
        self.set_variable_path_formats()
        self.set_number_of_input_channels_for_each_input_variable()
        self.load_polarholes()
        self.determine_tot_num_channels()
        self.on_epoch_end()

        if self.config['verbose_level'] >= 1:
            print("Setup complete.\n")

    def set_forecast_IDs(self, dataset='train'):
        """
        Build up a list of forecast initialisation dates for the train, val, or
        test sets based on the configuration JSON file start & end points for
        each dataset.
        """

        self.all_forecast_IDs = []

        for hemisphere, sample_ID_dict in self.config['sample_IDs'].items():
            forecast_start_date_ends = sample_ID_dict['obs_{}_dates'.format(dataset)]

            if forecast_start_date_ends is not None:
                # Convert to Pandas Timestamps
                forecast_start_date_ends = [
                    pd.Timestamp(date).to_pydatetime() for date in forecast_start_date_ends
                ]

                forecast_start_dates = misc.filled_daily_dates(
                    forecast_start_date_ends[0],
                    forecast_start_date_ends[1])

                self.all_forecast_IDs.extend([
                    (hemisphere, start_date) for start_date in forecast_start_dates]
                )

        if dataset == 'train' or dataset == 'val':
            if '{}_sample_thin_factor'.format(dataset) in self.config.keys():
                if self.config['{}_sample_thin_factor'.format(dataset)] is not None:
                    reduce = self.config['{}_sample_thin_factor'.format(dataset)]
                    prev_n_samps = len(self.all_forecast_IDs)
                    new_n_samps = int(prev_n_samps / reduce)

                    self.all_forecast_IDs = self.rng.choice(
                        a=self.all_forecast_IDs,
                        size=new_n_samps,
                        replace=False
                    )

                    if self.config['verbose_level'] >= 1:
                        print('Number of {} samples thinned from {} '
                              'to {}.'.format(dataset, prev_n_samps, len(self.all_forecast_IDs)))


    def set_variable_path_formats(self):

        """
        Initialise the paths to the .npy files of each variable based on
        `self.config['input_data']`.
        """

        if self.config['verbose_level'] >= 1:
            print('Setting up the variable paths for {}... '.format(self.config['dataset_name']),
                  end='', flush=True)

        # Parent folder for this dataset
        self.dataset_path = os.path.join(config.network_dataset_folder, self.config['dataset_name'])

        # Dictionary data structure to store image variable paths
        self.variable_paths = {}

        for varname, vardict in self.config['input_data'].items():

            if 'metadata' not in vardict.keys():
                self.variable_paths[varname] = {}

                for data_format in vardict.keys():

                    if vardict[data_format]['include'] is True:

                        if not self.do_transfer_learning:
                            path = os.path.join(
                                self.dataset_path, 'obs',
                                varname, data_format, '{:04d}_{:02d}.npy'
                            )
                        elif self.do_transfer_learning:
                            path = os.path.join(
                                self.dataset_path, 'transfer', '{}', '{}',
                                varname, data_format, '{:04d}_{:02d}.npy'
                            )

                        self.variable_paths[varname][data_format] = path

            elif 'metadata' in vardict.keys():

                if vardict['include'] is True:

                    if varname == 'land':
                        path = os.path.join(self.dataset_path, 'meta', 'land.npy')
                        self.variable_paths['land'] = path

                    elif varname == 'circmonth':
                        path = os.path.join(self.dataset_path, 'meta',
                                            '{}_month_{:02d}.npy')
                        self.variable_paths['circmonth'] = path

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
                            for leadtime in np.arange(1, self.config['n_forecast_months']+1):
                                variable_names.append(varname+'_{}_{}'.format(data_format, leadtime))

            # Metadata input variables that don't span time
            elif 'metadata' in vardict.keys() and vardict['include']:
                if varname == 'land':
                    variable_names.append(varname)

                elif varname == 'circmonth':
                    variable_names.append('cos(month)')
                    variable_names.append('sin(month)')

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
                            self.num_input_channels_dict[varname_format] = self.config['n_forecast_months']

            # Metadata input variables that don't span time
            elif 'metadata' in vardict.keys() and vardict['include']:
                if varname == 'land':
                    self.num_input_channels_dict[varname] = 1

                if varname == 'circmonth':
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
            forecast_start_date - pd.DateOffset(months=int(lag)) for lag in np.arange(1, max_lag+1)
        ]

        return input_dates

    def check_for_missing_date_dependence(self, forecast_start_date):
        """
        Check a forecast ID and return a bool for whether any of the input SIC maps
        are missing. Used to remove forecast IDs that depend on missing SIC data.
        Note: If one of the _forecast_ dates are missing but not _input_ dates,
        the sample weight matrix for that date will be all zeroes so that the
        samples for that date do not appear in the loss function.
        """
        contains_missing_date = False

        # Check SIC input dates
        input_dates = self.all_sic_input_dates_from_forecast_start_date(forecast_start_date)

        for input_date in input_dates:
            if any([input_date == missing_date for missing_date in config.missing_dates]):
                contains_missing_date = True
                break

        return contains_missing_date

    def load_missing_dates(self):

        '''
        Load missing SIC day spreadsheet and use it to build up a list of all
        missing days
        '''

        self.missing_dates = {}

        for hemisphere in ['nh', 'sh']:
            self.missing_dates[hemisphere] = []
            missing_date_df = pd.read_csv(
                os.path.join('data', hemisphere, config.fnames['missing_sic_days']))
            for idx, row in missing_date_df.iterrows():
                # Ensure hour = 0 convention for daily dates
                start = pd.Timestamp(row['start']).to_pydatetime().replace(hour=0)
                end = pd.Timestamp(row['end']).to_pydatetime().replace(hour=0)
                self.missing_dates[hemisphere].extend(
                    misc.filled_daily_dates(start, end, include_end=True)
                )

    def remove_missing_dates(self):

        '''
        Remove dates from self.obs_forecast_IDs that depend on a missing
        observation of SIC.
        '''

        if self.config['verbose_level'] >= 2:
            print('Checking forecast start dates for missing SIC dates... ', end='', flush=True)

        new_obs_forecast_IDs = []
        for forecast_start_date in self.obs_forecast_IDs:
            if self.check_for_missing_date_dependence(forecast_start_date):
                if self.config['verbose_level'] >= 3:
                    print('Removing {}, '.format(
                        forecast_start_date.strftime('%Y_%m_%d')), end='', flush=True)

            else:
                new_obs_forecast_IDs.append(forecast_start_date)

        self.obs_forecast_IDs = new_obs_forecast_IDs

    def load_polarholes(self):
        """
        Loads each of the polar holes.
        """

        if self.config['verbose_level'] >= 1:
            tic = time.time()
            print("Loading and augmenting the polar holes... ", end='', flush=True)

        polarhole_path = os.path.join(config.mask_data_folder, config.polarhole1_fname)
        self.polarhole1_mask = np.load(polarhole_path)

        polarhole_path = os.path.join(config.mask_data_folder, config.polarhole2_fname)
        self.polarhole2_mask = np.load(polarhole_path)

        if config.use_polarhole3:
            polarhole_path = os.path.join(config.mask_data_folder, config.polarhole3_fname)
            self.polarhole3_mask = np.load(polarhole_path)

        self.nopolarhole_mask = np.full((432, 432), False)

        if self.config['verbose_level'] >= 1:
            print("Done in {:.0f}s.\n".format(time.time() - tic))

    def determine_polar_hole_mask(self, forecast_start_date):
        """
        Determine which polar hole mask to use (if any) by finding the oldest SIC
        input month based on the current output month. The polar hole active for
        the oldest input month is used (because the polar hole size decreases
        monotonically over time, and we wish to use the largest polar hole for
        the input data).
        Parameters:
        forecast_start_date (pd.Timestamp): Timepoint for the forecast initialialisation.
        Returns:
        polarhole_mask: Mask array with NaNs on polar hole grid cells and 1s
        elsewhere.
        """

        oldest_input_date = min(self.all_sic_input_dates_from_forecast_start_date(forecast_start_date))

        if oldest_input_date <= config.polarhole1_final_date:
            polarhole_mask = self.polarhole1_mask
            if self.config['verbose_level'] >= 3:
                print("Forecast start date: {}, polar hole: {}".format(
                    forecast_start_date.strftime("%Y_%m"), 1))

        elif oldest_input_date <= config.polarhole2_final_date:
            polarhole_mask = self.polarhole2_mask
            if self.config['verbose_level'] >= 3:
                print("Forecast start date: {}, polar hole: {}".format(
                    forecast_start_date.strftime("%Y_%m"), 2))

        else:
            polarhole_mask = self.nopolarhole_mask
            if self.config['verbose_level'] >= 3:
                print("Forecast start date: {}, polar hole: {}".format(
                    forecast_start_date.strftime("%Y_%m"), "none"))

        return polarhole_mask

    def determine_active_grid_cell_mask(self, forecast_date):
        """
        Determine which active grid cell mask to use (a boolean array with
        True on active cells and False on inactive cells). The cells with 'True'
        are where predictions are to be made with IceNet. The active grid cell
        mask for a particular month is determined by the sum of the land cells,
        the ocean cells (for that month), and the missing polar hole.
        The mask is used for removing 'inactive' cells (such as land or polar
        hole cells) from the loss function in self.data_generation.
        """

        output_month_str = '{:02d}'.format(forecast_date.month)
        output_active_grid_cell_mask_fname = config.active_grid_cell_file_format. \
            format(output_month_str)
        output_active_grid_cell_mask_path = os.path.join(
            config.mask_data_folder, output_active_grid_cell_mask_fname)
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
        self.all_forecast_IDs = self.transfer_forecast_IDs
        self.on_epoch_end()  # Shuffle transfer training indexes
        self.set_variable_path_formats()

    def turn_off_transfer_learning(self):

        '''
        Converts the data loader back to using ERA5/OSI-SAF observational
        training data.
        '''

        self.do_transfer_learning = False
        self.all_forecast_IDs = self.obs_forecast_IDs
        self.on_epoch_end()  # Shuffle transfer training indexes
        self.set_variable_path_formats()

    def convert_to_validation_data_loader(self):

        """
        Resets the `all_forecast_IDs` array to correspond to the validation
        months defined by the data loader configuration file.
        """

        self.set_forecast_IDs(dataset='val')
        self.remove_missing_dates()

    def convert_to_test_data_loader(self):

        """
        As above but for the testing months.
        """

        self.set_forecast_IDs(dataset='test')
        self.remove_missing_dates()

    def data_generation(self, forecast_IDs):
        """
        Generate input-output data for IceNet for a given forecast ID.
        Parameters:
        forecast_IDs (list):
            If self.do_transfer_learning is False, a list of pd.Timestamp objects
            corresponding to the forecast initialisation dates (first month
            being forecast) for the batch of X-y data to load.
            If self.do_transfer_learning is True, a list of tuples
            of the form (cmip6_model_name, member_id, forecast_start_date).
        Returns:
        X (ndarray): Batch of input 3D volumes.
        y (ndarray): Batch of ground truth output SIC class maps
        sample_weight (ndarray): Batch of pixelwise weights for weighting the
            loss function (masking outside the active grid cell region and
            up-weighting summer months).
        """

        # Allow non-list input for single forecasts
        forecast_IDs = pd.Timestamp(forecast_IDs) if isinstance(forecast_IDs, str) else forecast_IDs
        forecast_IDs = [forecast_IDs] if not isinstance(forecast_IDs, list) else forecast_IDs

        current_batch_size = len(forecast_IDs)

        ########################################################################
        # OUTPUT LABELS
        ########################################################################

        # Build up the set of N_samps output SIC time-series
        #   (each n_forecast_months long in the time dimension)

        # To become array of shape (N_samps, *raw_data_shape, n_forecast_months)
        batch_sic_list = []

        # True = forecasts months corresponding to no data
        missing_month_dict = {}

        for sample_idx, forecast_date in enumerate(forecast_start_dates):

            # To become array of shape (*raw_data_shape, n_forecast_months)
            sample_sic_list = []

            # List of forecast indexes with missing data
            missing_month_dict[sample_idx] = []

            for forecast_leadtime_idx in range(self.config['n_forecast_months']):

                forecast_date = forecast_start_dates[sample_idx] + pd.DateOffset(months=forecast_leadtime_idx)

                if not os.path.exists(
                    self.variable_paths[hemisphere]['siconca']['abs'].format(
                        forecast_target_date.year,
                        forecast_target_date.month,
                        forecast_target_date.day)):
                    # Output file does not exist - fill it with NaNs
                    sample_sic_list.append(np.full(self.config['raw_data_shape'], np.nan))

                else:
                    # Output file exists
                    sample_sic_list.append(
                        np.load(self.variable_paths['siconca']['abs'].format(
                            cmip6_model_names[sample_idx], cmip6_member_ids[sample_idx],
                            forecast_date.year, forecast_date.month))
                    )

            batch_sic_list.append(np.stack(sample_sic_list, axis=2))

        batch_sic = np.stack(batch_sic_list, axis=0)

        no_ice_gridcells = batch_sic <= 0.15
        ice_gridcells = batch_sic >= 0.80
        marginal_ice_gridcells = ~((no_ice_gridcells) | (ice_gridcells))

        # Categorical representation with channel dimension for class probs
        y = np.zeros((
            current_batch_size,
            *self.config['raw_data_shape'],
            self.config['n_forecast_months'],
            3
        ), dtype=np.float32)

        y[no_ice_gridcells, 0] = 1
        y[marginal_ice_gridcells, 1] = 1
        y[ice_gridcells, 2] = 1

        # Move lead time to final axis
        y = np.moveaxis(y, source=3, destination=4)

        # Missing months
        for sample_idx, forecast_leadtime_idx_list in missing_month_dict.items():
            if len(forecast_leadtime_idx_list) > 0:
                y[sample_idx, :, :, :, forecast_leadtime_idx_list] = 0

        ########################################################################
        # PIXELWISE LOSS FUNCTION WEIGHTING
        ########################################################################

        sample_weight = np.zeros((
            current_batch_size,
            *self.config['raw_data_shape'],
            1,  # Broadcastable class dimension
            self.config['n_forecast_months']
        ), dtype=np.float32)
        for sample_idx, forecast_date in enumerate(forecast_start_dates):

            for forecast_leadtime_idx in range(self.config['n_forecast_months']):

                forecast_date = forecast_start_dates[sample_idx] + pd.DateOffset(months=forecast_leadtime_idx)

                if any([forecast_date == missing_date for missing_date in config.missing_dates]):
                    # Leave sample weighting as all-zeros
                    pass

                else:
                    # Zero loss outside of 'active grid cells'
                    sample_weight_ij = self.determine_active_grid_cell_mask(forecast_date)
                    sample_weight_ij = sample_weight_ij.astype(np.float32)

                    # Scale the loss for each month s.t. March is
                    #   scaled by 1 and Sept is scaled by 1.77
                    if self.config['loss_weight_months']:
                        sample_weight_ij *= 33928. / np.sum(sample_weight_ij)

                    sample_weight[sample_idx, :, :, 0, forecast_leadtime_idx] = \
                        sample_weight_ij

        ########################################################################
        # INPUT FEATURES
        ########################################################################

        # Batch tensor
        X = np.zeros((
            current_batch_size,
            *self.config['raw_data_shape'],
            self.tot_num_channels
        ), dtype=np.float32)

        # Build up the batch of inputs
        for sample_idx, forecast_start_date in enumerate(forecast_start_dates):

            present_date = forecast_start_date - relativedelta(months=1)

            # Initialise variable indexes used to fill the input tensor `X`
            variable_idx1 = 0
            variable_idx2 = 0

            for varname, vardict in self.config['input_data'].items():

                if 'metadata' not in vardict.keys():

                    for data_format in vardict.keys():

                        if vardict[data_format]['include']:

                            varname_format = '{}_{}'.format(varname, data_format)

                            if data_format != 'linear_trend':
                                lbs = range(vardict[data_format]['max_lag'])
                                input_months = [present_date - relativedelta(months=lb) for lb in lbs]
                            elif data_format == 'linear_trend':
                                input_months = [present_date + relativedelta(months=forecast_leadtime)
                                                for forecast_leadtime in np.arange(1, self.config['n_forecast_months']+1)]

                            variable_idx2 += self.num_input_channels_dict[varname_format]

                            X[sample_idx, :, :, variable_idx1:variable_idx2] = \
                                np.stack([np.load(self.variable_paths[hemisphere][varname][data_format].format(
                                          date.year, date.month, date.day))
                                          for date in input_months], axis=-1)

                            variable_idx1 += self.num_input_channels_dict[varname_format]

                elif 'metadata' in vardict.keys() and vardict['include']:

                    variable_idx2 += self.num_input_channels_dict[varname]

                    if varname == 'land':
                        X[sample_idx, :, :, variable_idx1] = np.load(self.variable_paths[hemisphere]['land'])

                    elif varname == 'circday':
                        # Broadcast along row and col dimensions
                        X[sample_idx, :, :, variable_idx1] = \
                            np.load(self.variable_paths[hemisphere]['circday'].format(
                                'cos',
                                forecast_start_date.month,
                                forecast_start_date.day))
                        X[sample_idx, :, :, variable_idx1 + 1] = \
                            np.load(self.variable_paths[hemisphere]['circday'].format(
                                'sin',
                                forecast_start_date.month,
                                forecast_start_date.day))

                    variable_idx1 += self.num_input_channels_dict[varname]

        return X, y, sample_weight

    def __getitem__(self, batch_idx):
        '''
        Generate one batch of data of size `batch_size` at batch index `batch_idx`
        into the set of batches in the epoch.
        '''

        batch_start = batch_idx * self.config['batch_size']
        batch_end = np.min([(batch_idx + 1) * self.config['batch_size'], len(self.all_forecast_IDs)])

        sample_idxs = np.arange(batch_start, batch_end)
        batch_IDs = [self.all_forecast_IDs[sample_idx] for sample_idx in sample_idxs]

        return self.data_generation(batch_IDs)

    def __len__(self):
        ''' Returns the number of batches per training epoch. '''
        return int(np.ceil(len(self.all_forecast_IDs) / self.config['batch_size']))

    def on_epoch_end(self):
        """ Randomly shuffles training samples after each epoch. """

        if self.config['verbose_level'] >= 2:
            print("on_epoch_end called")

        # Randomly shuffle the forecast IDs in-place
        self.rng.shuffle(self.all_forecast_IDs)

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

