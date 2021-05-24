import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), 'icenet2'))
import config
import misc
from tqdm import tqdm
from datetime import datetime
import numpy as np
import regex as re
import xarray as xr
import dask
from distributed import Client, progress
from dask.diagnostics import ProgressBar
import dask.array as da
from time import time
import pandas as pd
import matplotlib
matplotlib.rcParams.update({
    'figure.facecolor': 'w',
    'figure.dpi': 300
})

use_bias_corrected_seas5 = True

metric_compute_list = ['MAE', 'RMSE', 'Binary_accuracy']

n_forecast_days = 93

if use_bias_corrected_seas5:
    model = 'SEAS5'
else:
    model = 'SEAS5_noBC'

biascorrection_folder = os.path.join('data', 'forecasts', 'seas5_biascorrection')

### Initialise results dataframe
####################################################################

leadtimes = np.arange(1, n_forecast_days+1)

forecast_target_dates = pd.date_range(
    start='2012-01-01', end='2018-01-01', closed='left'
)

results_df_fnames = sorted([f for f in os.listdir('results') if re.compile('.*.csv').match(f)])
if len(results_df_fnames) >= 1:
    old_results_df_fname = results_df_fnames[-1]
    old_results_df_fpath = os.path.join('results', old_results_df_fname)
    print('\n\nLoading previous results dataset from {}'.format(old_results_df_fpath))

now = pd.Timestamp.now()
new_results_df_fname = now.strftime('%Y_%m_%d_%H%M%S_results.csv')
new_results_df_fpath = os.path.join('results', new_results_df_fname)

print('New results will be saved to {}\n\n'.format(new_results_df_fpath))

results_df = pd.read_csv(old_results_df_fpath)
# Drop spurious index column if present
results_df = results_df.drop('Unnamed: 0', axis=1, errors='ignore')
results_df['Forecast date'] = [pd.Timestamp(date) for date in results_df['Forecast date']]

existing_models = results_df.Model.unique()
results_df = results_df.set_index(['Leadtime', 'Forecast date', 'Model'])
existing_metrics = results_df.columns

if model not in existing_models:
    # Add new model to the dataframe
    new_index = pd.MultiIndex.from_product(
        [leadtimes, forecast_target_dates, [model]],
        names=['Leadtime', 'Forecast date', 'Model'])
    results_df = results_df.append(pd.DataFrame(index=new_index)).sort_index()

### Load forecasts
####################################################################

validation_forecast_folder = os.path.join('data', 'forecasts', 'seas5', 'EASE')

if not os.path.exists(validation_forecast_folder) or len(os.listdir(validation_forecast_folder)) == 0:
    # No forecast data - do not compute
    print('Could not find forecast data for {}.'.format(model))
    sys.exit()

validation_prediction_fpaths = [
    os.path.join(validation_forecast_folder, f) for f
    in sorted(os.listdir(validation_forecast_folder))
]

fname_regex = re.compile('^.*_([0-9]{4}_[0-9]{2}_[0-9]{2}).nc$')

all_forecasts_dict = {}

print('Loading SEAS5 forecasts:\n')
for fpath in tqdm(validation_prediction_fpaths):
    init_date_str = fname_regex.match(fpath)[1].replace('_', '-')
    init_date = pd.Timestamp(init_date_str)

    forecast_dates = pd.date_range(
        start=init_date, periods=n_forecast_days, freq='d',
    )

    if use_bias_corrected_seas5:
        biascorrection_fpath = os.path.join(
            biascorrection_folder, '{:02d}_01.nc'.format(init_date.month))

        biascorrection_field = xr.open_dataarray(biascorrection_fpath)
        biascorrection_field = biascorrection_field.rename({'leadtime': 'time'})
        biascorrection_field = biascorrection_field.assign_coords({'time': forecast_dates})
        biascorrection_field.load()

    if init_date <= forecast_target_dates[-1]:
        forecast = xr.open_dataset(fpath)['siconc']

        # Assume SEAS5 forecast for 24:00 is close to daily average forecast for that day,
        #   and use coordinate convention of 00:00 for daily average
        dates = [pd.Timestamp(date) - pd.DateOffset(1) for date
                 in forecast.time.values]
        forecast = forecast.assign_coords(dict(time=dates))

        # Convert spatial coords to km
        forecast = forecast.assign_coords(dict(xc=forecast.xc/1e3, yc=forecast.yc/1e3))

        # Remove initialisation state
        forecast = forecast.loc[forecast.time >= init_date]

        if use_bias_corrected_seas5:
            forecast = forecast - biascorrection_field

        # Remove forecasts outside of analysis period
        valid_dates = [date for date in forecast.time.values if date in forecast_target_dates]
        forecast = forecast.loc[valid_dates]

        forecast.load()

        all_forecasts_dict[init_date] = forecast
print('Done.\n')

### True SIC
####################################################################

true_sic_fpath = os.path.join('data', 'nh', 'siconca', 'siconca_all_interp.nc')
true_sic_da = xr.open_dataarray(true_sic_fpath, chunks=dict(time=20))

# Replace 12:00 hour with 00:00 hour by convention
dates = [pd.Timestamp(date).to_pydatetime().replace(hour=0) for date in true_sic_da.time.values]
true_sic_da = true_sic_da.assign_coords(dict(time=dates))

true_sic_da = true_sic_da.sel(time=forecast_target_dates)
true_sic_da.load()

if 'Binary_accuracy' in metric_compute_list:
    true_sic_binary_da = true_sic_da > 0.15

### Monthwise masks
####################################################################

mask_fpath_format = os.path.join('data', 'nh', 'masks',
                                 config.formats['active_grid_cell_mask'])

month_mask_da = xr.DataArray(np.array(
    [np.load(mask_fpath_format.format('{:02d}'.format(month))) for
     month in np.arange(1, 12+1)],
))

months = [pd.Timestamp(date).month-1 for date in true_sic_da.time.values]
mask_da = xr.DataArray(
    [month_mask_da[month] for month in months],
    dims=('time', 'yc', 'xc'),
    coords={
        'time': true_sic_da.time.values,
        'yc': true_sic_da.yc.values,
        'xc': true_sic_da.xc.values,
    }
)
mask_da.load()

### Compute performance metrics for each forecast
####################################################################

print('\n\n\n')

# Metrics based on raw SIC error
sic_err_metrics = ['MAE', 'MSE', 'RMSE']

compute_sic_err_metrics = [metric for metric in metric_compute_list if metric in sic_err_metrics]
compute_non_sic_err_metrics = [metric for metric in metric_compute_list if metric not in sic_err_metrics]

compute_ds_list = []

tic = time()
for i, (init_date, forecast) in enumerate(tqdm(all_forecasts_dict.items())):

    if len(compute_sic_err_metrics) >= 1:

        # Absolute SIC errors
        err_da = (forecast - true_sic_da) * 100
        abs_err_da = da.fabs(err_da)
        abs_weighted_da = abs_err_da.weighted(mask_da)

        # Squared errors
        square_err_da = err_da**2
        square_weighted_da = square_err_da.weighted(mask_da)

        compute_ds = xr.Dataset()
        for metric in compute_sic_err_metrics:

            if metric == 'MAE':
                ds_mae = abs_weighted_da.mean(dim=['yc', 'xc'])
                # compute_ds[metric] = next(iter(ds_mae.data_vars.values()))
                compute_ds[metric] = ds_mae

            elif metric == 'MSE':

                ds_mse = square_weighted_da.mean(dim=['yc', 'xc'])
                # compute_ds[metric] = next(iter(ds_mse.data_vars.values()))
                compute_ds[metric] = ds_mse

            elif metric == 'RMSE':

                if 'MSE' not in compute_sic_err_metrics:
                    ds_mse = square_weighted_da.mean(dim=['yc', 'xc'])

                ds_rmse = da.sqrt(ds_mse)
                # compute_ds[metric] = next(iter(ds_rmse.data_vars.values()))
                compute_ds[metric] = ds_rmse

    if len(compute_non_sic_err_metrics) >= 1:

        for metric in compute_non_sic_err_metrics:

            if metric == 'Binary_accuracy':
                forecast_binary_da = forecast > 0.15
                binary_correct_da = (forecast_binary_da == true_sic_binary_da).astype(np.float32)
                binary_correct_weighted_da = binary_correct_da.weighted(mask_da)

                # Mean percentage of correct classifications over the active
                #   grid cell area
                ds_binacc = (binary_correct_weighted_da.mean(dim=['yc', 'xc']) * 100)
                # Compute_ds[metric] = next(iter(ds_binacc.data_vars.values()))
                compute_ds[metric] = ds_binacc

    compute_ds = compute_ds.drop('number')

    compute_ds_list.append(compute_ds)

    # Determine forecast lead times
    leadtimes = [(date - init_date).days+1 for date in forecast.time.values]

    mapping = {'time': 'Forecast date'}

    forecast_compute_df = compute_ds.to_dataframe().reset_index().rename(columns=mapping).\
        assign(Model=model, Leadtime=leadtimes).set_index(['Leadtime', 'Forecast date', 'Model'])

    compute_df = forecast_compute_df if i == 0 else pd.concat((compute_df, forecast_compute_df))

dur = time() - tic
print("Computations finished in {}m:{:.0f}s.\n\n ".format(np.floor(dur / 60), dur % 60))

print('Writing to results dataset (this can take a minute)...')
tic = time()
results_df.loc[compute_df.index.values, compute_df.columns] = \
    compute_df.values
dur = time() - tic
print("Done in {}m:{:.0f}s.\n\n ".format(np.floor(dur / 60), dur % 60))

### Checkpoint results
print('\nCheckpointing results dataset... ', end='', flush=True)
tic = time()
results_df.index = results_df.index.rename(['Leadtime', 'Forecast date', 'Model'])
results_df.to_csv(new_results_df_fpath)
dur = time() - tic
print("Done in {}m:{:.0f}s.\n\n ".format(np.floor(dur / 60), dur % 60))

print('\n\nNEW RESULTS: ')
print(results_df.head(10))
print('\n...\n')
print(results_df.tail(10))

# # TEMP seas5 video
# land_mask = np.load('data/nh/masks/land_mask.npy')
#
# err_da = (all_forecasts_dict[pd.Timestamp('2012-08-01')] - true_sic_da) * 100
# err_da.data[:, land_mask] = 0
#
# from misc import xarray_to_video
# xarray_to_video(err_da, 'temp_summer_bc.mp4', mask=land_mask, cmap='seismic',
#                 clim=(-100, 100), fps=5)
