import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), 'icenet2'))
import pandas as pd
import json

"""

Script to generate a .json file storing data loader configuration settings using
a dictionary. The dictionary settings are:

    dataloader_name (str): Name for these data loader configuration settings,
    used.

    dataset_name (str): Name for of the network dataset (generated with
    preproc_icenet2_data.py) to load data from.

    input_data (dict): Dictionary of dictionaries dictating which
    variables to include for IceNet2's input 3D volumes and, if appropriate,
    a maximum lag (in days) to grab the data from. The nested dictionaries
    have keys of "include" (a bool for whether to input that variable), and
    "lookbacks" (a list of ints for which past months to input, indexing
    from 0 relative to the most recent month).

        Example:
            'input_data': {
                "siconca":
                    {"abs": {"include": True, 'max_lag': 31*3},
                     "anom": {"include": False, 'max_lag': 31},
                     "linear_trend": {"include": True}},
                "tas":
                    {"abs": {"include": False, 'max_lag': 31},
                     "anom": {"include": True, 'max_lag': 31}},
                "ta500":
                    {"abs": {"include": False, 'max_lag': 31},
                     "anom": {"include": True, 'max_lag': 31}},
                "tos":
                    {"abs": {"include": False, 'max_lag': 31*3},
                     "anom": {"include": True, 'max_lag': 31*3}},
                "psl":
                    {"abs": {"include": False, 'max_lag': 31},
                     "anom": {"include": True, 'max_lag': 31}},
                "zg500":
                    {"abs": {"include": False, 'max_lag': 31},
                     "anom": {"include": True, 'max_lag': 31}},
                "zg250":
                    {"abs": {"include": False, 'max_lag': 31},
                     "anom": {"include": True, 'max_lag': 31}},
                "land":
                    {"metadata": True,
                     "include": True},
                "circday":
                    {"metadata": True,
                     "include": True},
            },
    batch_size (int): Number of samples per training batch.

    n_forecast_days (int): Total number of days ahead to predict.

    sample_IDs (dict): Dictionary of dictionaries storing the train-val-test
    set splits for the Northern and Southern hemispheres. Splits are defined in
    terms of start & end dates for the forecast initialisation dates
    used to define sampled IDs. If None, that hemisphere does not contribute
    to the given dataset split.

        Example:
            'sample_IDs': {
                'nh': {
                    'obs_train_dates': ('1979-6-1', '2011-6-1'),
                    'obs_val_dates': ('2012-1-1', '2017-6-1'),
                    'obs_test_dates': ('2018-1-1', '2019-6-1'),
                },
                'sh': {
                    'obs_train_dates': ('1979-6-1', '2014-12-31'),
                    'obs_val_dates': None,
                    'obs_test_dates': None,
                },
            },

    training_sample_thin_factor (int): Factor by which to downsample the training
    samples due to high correlation between days. For training efficiency during model
    improvement phase.

    raw_data_shape (tuple): Shape of input satellite data as (rows, cols).

    default_seed (int): Default random seed to use for shuffling the order
    of training samples a) before training, and b) after each training epoch.

    loss_weight_months (bool): Whether to weight the loss function for different
    months based on the size of the active grid cell mask.

    verbose_level (int): Controls how much to print. 0: Print nothing.
    1: Print key set-up stages. 2: Print debugging info. 3: Print when an
    output month is skipped due to missing data.
"""

dataloder_config = {
    'dataloader_name': 'icenet2_nh_thinned7_weeklyinput_wind_3month',
    'dataset_name': 'dataset1',
    'input_data': {
        "siconca":
            {"abs": {"include": True, 'max_lag': 7*1},
             "anom": {"include": False, 'max_lag': 7*1},
             "linear_trend": {"include": False}},
        "tas":
            {"abs": {"include": False, 'max_lag': 7*1},
             "anom": {"include": True, 'max_lag': 7*1}},
        "ta500":
            {"abs": {"include": False, 'max_lag': 7*1},
             "anom": {"include": True, 'max_lag': 7*1}},
        "tos":
            {"abs": {"include": False, 'max_lag': 7*1},
             "anom": {"include": True, 'max_lag': 7*1}},
        "psl":
            {"abs": {"include": False, 'max_lag': 7*1},
             "anom": {"include": True, 'max_lag': 7*1}},
        "zg500":
            {"abs": {"include": False, 'max_lag': 7*1},
             "anom": {"include": True, 'max_lag': 7*1}},
        "zg250":
            {"abs": {"include": False, 'max_lag': 7*1},
             "anom": {"include": True, 'max_lag': 7*1}},
        "rsds":
            {"abs": {"include": False, 'max_lag': 7*1},
             "anom": {"include": True, 'max_lag': 7*1}},
        "rlds":
            {"abs": {"include": False, 'max_lag': 7*1},
             "anom": {"include": True, 'max_lag': 7*1}},
        "hus1000":
            {"abs": {"include": False, 'max_lag': 7*1},
             "anom": {"include": True, 'max_lag': 7*1}},
        "uas":
            {"abs": {"include": True, 'max_lag': 7*1},
             "anom": {"include": False, 'max_lag': 7*1}},
        "vas":
            {"abs": {"include": True, 'max_lag': 7*1},
             "anom": {"include": False, 'max_lag': 7*1}},
        "land":
            {"metadata": True,
             "include": True},
        "circday":
            {"metadata": True,
             "include": True},
    },
    'batch_size': 2,
    'n_forecast_days': 31*3,
    'sample_IDs': {
        'nh': {
            'obs_train_dates': ('1979-6-1', '2011-6-1'),
            'obs_val_dates': ('2012-1-1', '2017-6-1'),
            'obs_test_dates': ('2018-1-1', '2019-6-1'),
        },
        'sh': {
            # 'obs_train_dates': ('1979-6-1', '2014-12-31'),
            'obs_train_dates': None,
            'obs_val_dates': None,
            'obs_test_dates': None,
        },
    },
    'train_sample_thin_factor': 7,
    'val_sample_thin_factor': 7,
    'raw_data_shape': (432, 432),
    'default_seed': 42,
    'loss_weight_months': True,
    'verbose_level': 1,
}

now = pd.Timestamp.now()
fname = now.strftime('%Y_%m_%d_%H%M_{}.json').format(dataloder_config['dataloader_name'])
folder = 'dataloader_configs'
fpath = os.path.join(folder, fname)
if not os.path.exists(folder):
    os.makedirs(folder)

with open(fpath, 'w') as outfile:
    json.dump(dataloder_config, outfile)

print('Data loader config saved to {}\n'.format(fpath))
print('Data loader name: {}'.format(fname[:-5]))
