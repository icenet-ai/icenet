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

    obs_train_dates (tuple): Tuple of start and end initialisation date for
    training (storedas strings, e.g '1979-9-1')

    obs_val_dates (tuple): As above but for the validation set.

    obs_test_dates (tuple): As above but for the test set.

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
    'dataloader_name': 'icenet2_init',
    'dataset_name': 'dataset1',
    'input_data': {
        "siconca":
            {"abs": {"include": True, 'max_lag': 31*3},
             "anom": {"include": False, 'max_lag': 31},
             "linear_trend": {"include": False}},
        "tas":
            {"abs": {"include": False, 'max_lag': 31*1},
             "anom": {"include": True, 'max_lag': 31*1}},
        "ta500":
            {"abs": {"include": False, 'max_lag': 31*1},
             "anom": {"include": True, 'max_lag': 31*1}},
        "tos":
            {"abs": {"include": False, 'max_lag': 31*3},
             "anom": {"include": True, 'max_lag': 31*3}},
        "psl":
            {"abs": {"include": False, 'max_lag': 31*1},
             "anom": {"include": True, 'max_lag': 31*1}},
        "zg500":
            {"abs": {"include": False, 'max_lag': 31*1},
             "anom": {"include": True, 'max_lag': 31*1}},
        "zg250":
            {"abs": {"include": False, 'max_lag': 31*1},
             "anom": {"include": True, 'max_lag': 31*1}},
        "land":
            {"metadata": True,
             "include": True},
        "circday":
            {"metadata": True,
             "include": True},
    },
    'batch_size': 2,
    'n_forecast_days': 31*6,
    'obs_train_dates': ('1979-6-1', '2011-6-1'),
    'obs_val_dates': ('2012-1-1', '2017-6-1'),
    'obs_test_dates': ('2018-1-1', '2019-6-1'),
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

print('Data loader config saved to {}'.format(fpath))
