import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), 'icenet2'))
import config
import utils
import metrics
import losses
import numpy as np
import xarray as xr
import pandas as pd
from tqdm import tqdm
import plotly.express as px
from datetime import datetime
from dateutil.relativedelta import relativedelta
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

####################################################################

models = ['IceNet2', 'Day_persistence']
models = ['Day_persistence']  # TEMP

# TODO: generic predict functions for the different models that take init date
#   as input?

icenet2_name = 'unet_batchnorm'
dataloader_name = '2021_03_03_1928_icenet2_init'

seed = 42

verbose = False

#### Load network and dataloader
####################################################################

if 'IceNet2' in models:

    network_folder = os.path.join(config.folders['results'], dataloader_name, icenet2_name, 'networks')
    if not os.path.exists(network_folder):
        os.makedirs(network_folder)
    network_fpath = os.path.join(network_folder, 'network_{}.h5'.format(seed))

    network = load_model(
        network_fpath,
        custom_objects={
            'weighted_MSE': losses.weighted_MSE,
            'weighted_RMSE': metrics.weighted_RMSE
        }
    )

dataloader_config_fpath = os.path.join('dataloader_configs', dataloader_name+'.json')

dataloader = utils.IceNet2DataLoader(dataloader_config_fpath)

forecast_folders_dict = {}

for model in models:

    if model == 'IceNet2':
        forecast_folders_dict[model] = os.path.join(
            config.folders['data'], 'forecasts', 'icenet2', dataloader_name, icenet2_name)

    else:
        forecast_folders_dict[model] = os.path.join(
            config.folders['data'], 'forecasts', model)

    if not os.path.exists(forecast_folders_dict[model]):
        os.makedirs(forecast_folders_dict[model])

#### Load ground truth SIC for statistical model benchmarks
####################################################################

true_sic_fpath = os.path.join(config.folders['data'], 'siconca', 'siconca_all_interp.nc')
true_sic_da = xr.open_dataarray(true_sic_fpath)

# Replace 12:00 hour with 00:00 hour by convention
dates = [pd.Timestamp(date).to_pydatetime().replace(hour=0) for date in true_sic_da.time.values]
true_sic_da = true_sic_da.assign_coords(dict(time=dates))

#### Set up DataArrays for forecasts
####################################################################

n_forecast_days = dataloader.config['n_forecast_days']

# TODO: get this from dataloader config?
all_forecast_target_dates = utils.filled_daily_dates(
    start_date=datetime(2012, 1, 1), end_date=datetime(2020, 12, 31),
    include_end=True
)

all_forecast_start_dates = utils.filled_daily_dates(
    start_date=all_forecast_target_dates[0] - relativedelta(days=n_forecast_days-1),
    end_date=all_forecast_target_dates[-1],
    include_end=True)

da_with_coords = xr.open_dataarray('data/siconca/raw_yearly_data/siconca_1979.nc')

model_forecast_dict = {}
for model in models:

    yearly_forecast_da_dict = {}

    # TODO: don't hard code years
    for year in np.arange(2012, 2020+1):

        year_forecast_target_dates = utils.filled_daily_dates(
            start_date=datetime(year, 1, 1), end_date=datetime(year, 12, 31),
            include_end=True
        )

        shape = (len(year_forecast_target_dates), *dataloader.config['raw_data_shape'], n_forecast_days)

        yearly_forecast_da_dict[year] = xr.DataArray(
            data=np.zeros(shape, dtype=np.float32),
            dims=('time', 'yc', 'xc', 'leadtime'),
            coords={
                'time': year_forecast_target_dates,  # To be sliced to target dates
                'yc': da_with_coords.coords['yc'],
                'xc': da_with_coords.coords['xc'],
                # 'lon': (['yc, xc'], da_with_coords.coords['lon'].values),
                # 'lat': (['yc, xc'], da_with_coords.coords['lat'].values),
                'leadtime': np.arange(1, n_forecast_days+1)
            }
        )

    model_forecast_dict[model] = yearly_forecast_da_dict

#### Build up forecasts
####################################################################

leadtimes = np.arange(1, n_forecast_days+1)

# TODO: don't hard code start year
year_to_save = 2012

# forecast_start_date = all_forecast_start_dates[0]
print('Building up forecast DataArrays...\n')

for model in models:

    print(model + ':\n')

    for forecast_start_date in tqdm(all_forecast_start_dates):

        if model == 'IceNet2':

            X, y = dataloader.data_generation(np.array([forecast_start_date]))
            mask = y[0, :, :, :, 1] == 0

            pred = network.predict(X)[0]
            pred[mask] = 0.

        if model == 'Day_persistence':

            # Date for most recent SIC observation to persist
            persistence_date = forecast_start_date - relativedelta(days=1)
            pred = true_sic_da.sel(time=persistence_date).data

        forecast_target_dates = utils.filled_daily_dates(
            start_date=forecast_start_date,
            end_date=forecast_start_date + relativedelta(days=n_forecast_days-1),
            include_end=True
        )

        for i, (forecast_target_date, leadtime) in enumerate(zip(forecast_target_dates, leadtimes)):
            if forecast_target_date in all_forecast_target_dates:
                year = forecast_target_date.year

                if model == 'Day_persistence':
                    # Same forecast at each lead time
                    yearly_forecast_da_dict[year].\
                        loc[forecast_target_date, :, :, leadtime] = pred

                else:
                    yearly_forecast_da_dict[year].\
                        loc[forecast_target_date, :, :, leadtime] = pred[:, :, i]

        # End of year reached - save completed yearly NetCDF
        if forecast_start_date == datetime(year_to_save, 12, 31):
            if verbose:
                print('Saving forecast NetCDF for {}... '.format(year_to_save), end='', flush=True)
            yearly_forecast_fpath = os.path.join(forecast_folders_dict[model], '{:04d}.nc'.format(year_to_save))
            if os.path.exists(yearly_forecast_fpath):
                os.remove(yearly_forecast_fpath)
            yearly_forecast_da_dict[year_to_save].to_netcdf(yearly_forecast_fpath)
            if verbose:
                print('Done.')

            del(yearly_forecast_da_dict[year_to_save])  # Does this do with memory management anything?

            year_to_save += 1

print('Done.')
