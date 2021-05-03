import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), 'icenet2'))
import config
import misc
import json
from datetime import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
import regex as re
import xarray as xr
import dask
import dask.array as da
from dask.diagnostics import ProgressBar
from time import time
import pandas as pd
from distributed import Client, progress
import matplotlib
matplotlib.rcParams.update({
    'figure.facecolor': 'w',
    'figure.dpi': 300
})

compute_in_memory = True  # Not recomended unless you have lots of RAM!
use_local_distributed_cluster = False
use_multiprocessing_scheduler = False

if np.sum([compute_in_memory, use_local_distributed_cluster, use_multiprocessing_scheduler]) != 1:
    raise ValueError('You must specify a single compute strategy '
                     'at the beginning of the script.')

# temp_dir = 'tmp_dask'
temp_dir = '/local/tmp'
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)

if __name__ == '__main__':

    n_workers = 8
    # n_workers = 32
    threads_per_worker = 2

    dask.config.set(temporary_directory=os.path.expandvars(temp_dir))

    ####################################################################

    network_name = 'unet_batchnorm'
    # dataloader_name = '2021_04_03_1421_icenet2_nh_sh_thinned5_weeklyinput'
    # dataloader_name = '2021_04_25_1351_icenet2_nh_thinned7_weeklyinput_wind_3month'
    # dataloader_name = '2021_04_08_1205_icenet2_nh_sh_thinned5_weeklyinput_wind_3month'
    dataloader_name = 'icenet2_10ensemble'
    seed = 'ensemble'

    # None if you want to determine from dataloader config
    n_forecast_days = 93

    # Format for storing different IceNet2 results in one dataframe
    icenet2_name = 'IceNet2__{}__{}__{}'.format(dataloader_name, network_name, seed)

    pre_load_results_df = True

    # What to compute
    # model_compute_list = ['Year_persistence', 'Day_persistence', icenet2_name]
    model_compute_list = [icenet2_name]

    # OPTIONS: ['MAE', 'MSE', 'RMSE', 'Binary_accuracy']
    #   Note: ensure RMSE is after MSE
    # metric_compute_list = ['MAE', 'MSE', 'RMSE']
    # metric_compute_list = ['MAE', 'MSE', 'RMSE', 'Binary_accuracy']
    metric_compute_list = ['MAE', 'RMSE', 'Binary_accuracy']

    ### Dataloader
    ####################################################################

    if n_forecast_days is None:
        dataloader_config_fpath = os.path.join('dataloader_configs', dataloader_name+'.json')
        with open(dataloader_config_fpath, 'r') as f:
            dataloader_config = json.load(f)

        n_forecast_days = dataloader_config['n_forecast_days']

    # chunking = dict(time=7, leadtime=n_forecast_days)
    # chunking = dict(time=2, leadtime=n_forecast_days/2)
    # chunking = dict(time=14, leadtime=n_forecast_days)
    chunking = dict(time=20, leadtime=n_forecast_days)

    ### Monthly masks
    ####################################################################

    mask_fpath_format = os.path.join('data', 'nh', 'masks',
                                     config.formats['active_grid_cell_mask'])

    month_mask_da = xr.DataArray(np.array(
        [np.load(mask_fpath_format.format('{:02d}'.format(month))) for
         month in np.arange(1, 12+1)],
    ))

    ### Initialise results dataframe
    ####################################################################

    leadtimes = np.arange(1, n_forecast_days+1)

    forecast_target_dates = misc.filled_daily_dates(
        start_date=datetime(2012, 1, 1), end_date=datetime(2018, 1, 1)
    )

    # # TEMP
    # forecast_target_dates = misc.filled_daily_dates(
    #     start_date=datetime(2012, 1, 1), end_date=datetime(2014, 1, 1)
    # )

    results_df_fnames = sorted([f for f in os.listdir('results') if re.compile('.*.csv').match(f)])
    if len(results_df_fnames) >= 1:
        old_results_df_fname = results_df_fnames[-1]
        old_results_df_fpath = os.path.join('results', old_results_df_fname)
        print('\n\nLoading previous results dataset from {}'.format(old_results_df_fpath))

    now = pd.Timestamp.now()
    new_results_df_fname = now.strftime('%Y_%m_%d_%H%M%S_results.csv')
    new_results_df_fpath = os.path.join('results', new_results_df_fname)

    print('New results will be saved to {}\n\n'.format(new_results_df_fpath))

    if pre_load_results_df:
        results_df = pd.read_csv(old_results_df_fpath)
        # Drop spurious index column if present
        results_df = results_df.drop('Unnamed: 0', axis=1, errors='ignore')
        results_df['Forecast date'] = [pd.Timestamp(date) for date in results_df['Forecast date']]

        existing_models = results_df.Model.unique()
        results_df = results_df.set_index(['Leadtime', 'Forecast date', 'Model'])
        existing_metrics = results_df.columns

        new_models = [model for model in model_compute_list if model not in existing_models]
        new_metrics = [metric for metric in metric_compute_list if metric not in existing_metrics]

        compute_dict = {}
        for new_model in new_models:
            # Compute all metrics for new models
            compute_dict[new_model] = metric_compute_list

        # Add new metrics to the dataframe
        if len(new_metrics) > 0:
            for existing_model in existing_models:
                # Compute new metrics for existing models
                compute_dict[existing_model] = new_metrics

            results_df = pd.concat(
                [results_df, pd.DataFrame(columns=new_metrics)], axis=1)

        # Add new models to the dataframe
        if len(new_models) > 0:
            new_index = pd.MultiIndex.from_product(
                [leadtimes, forecast_target_dates, new_models],
                names=['Leadtime', 'Forecast date', 'Model'])
            results_df = results_df.append(pd.DataFrame(index=new_index)).sort_index()

    else:
        # Instantiate new results dataframe
        multi_index = pd.MultiIndex.from_product(
            [leadtimes, forecast_target_dates, model_compute_list],
            names=['Leadtime', 'Forecast date', 'Model'])
        results_df = pd.DataFrame(index=multi_index, columns=metric_compute_list)

        compute_dict = {
            model: metric_compute_list for model in model_compute_list
        }

    print('COMPUTATIONS:')
    print(compute_dict)
    print('\n\n')

    ### Load forecasts
    ####################################################################

    validation_forecasts_dict = {}

    remove_models = []  # Models for which no forecast data is found

    for model in compute_dict.keys():

        if model == 'Year_persistence':
            # Forecasts computed locally in this script because they are so simple
            continue

        matchobj = re.compile('^IceNet2__(.*)__(.*)__(.*)$').match(model)
        if matchobj:
            dataloader_name = matchobj[1]
            network_name = matchobj[2]
            seed = matchobj[3]
            validation_forecast_folder = os.path.join(
                'data', 'forecasts', 'icenet2', dataloader_name, network_name, seed
            )
        else:
            validation_forecast_folder = os.path.join(
                'data', 'forecasts', model
            )

        if not os.path.exists(validation_forecast_folder) or len(os.listdir(validation_forecast_folder)) == 0:
            # No forecast data - do not compute
            print('Could not find forecast data for {} -- removing from computations.'.format(model))
            remove_models.append(model)
            continue

        validation_prediction_fpaths = [
            os.path.join(validation_forecast_folder, f) for f in os.listdir(validation_forecast_folder)
        ]
        with dask.config.set(**{'array.slicing.split_large_chunks': False}):
            forecast_ds = xr.open_mfdataset(
                validation_prediction_fpaths, chunks=chunking
            )
            forecast_ds = next(iter(forecast_ds.data_vars.values()))  # Convert to DataArray
            validation_forecasts_dict[model] = forecast_ds.sel(time=forecast_target_dates)

    for model in remove_models:
        del(compute_dict[model])

    ### True SIC
    ####################################################################

    true_sic_fpath = os.path.join('data', 'nh', 'siconca', 'siconca_all_interp.nc')
    true_sic_da = xr.open_dataarray(true_sic_fpath,
                                    chunks=dict(time=chunking['time']))

    # Replace 12:00 hour with 00:00 hour by convention
    dates = [pd.Timestamp(date).to_pydatetime().replace(hour=0) for date in true_sic_da.time.values]
    true_sic_da = true_sic_da.assign_coords(dict(time=dates))

    if 'Year_persistence' in compute_dict.keys():
        persistence_forecast_da = true_sic_da.copy()
        dates = [pd.Timestamp(date) + relativedelta(days=365) for date in persistence_forecast_da.time.values]
        persistence_forecast_da = persistence_forecast_da.assign_coords(dict(time=dates))
        persistence_forecast_da = persistence_forecast_da.sel(time=forecast_target_dates)
        validation_forecasts_dict['Year_persistence'] = persistence_forecast_da

    true_sic_da = true_sic_da.sel(time=forecast_target_dates)

    if 'Binary_accuracy' in metric_compute_list:
        true_sic_binary_da = true_sic_da > 0.15

    ### Monthwise masks
    ####################################################################

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

    ### Compute
    ####################################################################

    print('\n\n\n')

    # Metrics based on raw SIC error
    sic_err_metrics = ['MAE', 'MSE', 'RMSE']

    for model, model_metric_compute_list in compute_dict.items():

        if use_local_distributed_cluster:
            client = Client(n_workers=n_workers, threads_per_worker=threads_per_worker)
            print(client)

        compute_sic_err_metrics = [metric for metric in model_metric_compute_list if metric in sic_err_metrics]
        compute_non_sic_err_metrics = [metric for metric in model_metric_compute_list if metric not in sic_err_metrics]

        if len(compute_sic_err_metrics) >= 1:

            # Absolute SIC errors
            err_da = (validation_forecasts_dict[model] - true_sic_da) * 100
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
                    forecast_binary_da = validation_forecasts_dict[model] > 0.15
                    binary_correct_da = (forecast_binary_da == true_sic_binary_da).astype(np.float32)
                    binary_correct_weighted_da = binary_correct_da.weighted(mask_da)

                    # Mean percentage of correct classifications over the active
                    #   grid cell area
                    ds_binacc = (binary_correct_weighted_da.mean(dim=['yc', 'xc']) * 100)
                    # Compute_ds[metric] = next(iter(ds_binacc.data_vars.values()))
                    compute_ds[metric] = ds_binacc

        print('Computing all metrics:')
        print(model_metric_compute_list)
        tic = time()

        # TEMP: loading all into memory
        if compute_in_memory:
            compute_ds = compute_ds.compute()

        if use_local_distributed_cluster:
            compute_ds = compute_ds.persist()
            progress(compute_ds)
            compute_ds = compute_ds.compute()

            # print('\nRestarting client... ', end='', flush=True)
            # client.restart()
            # print('Done.')

            print('\nClosing client... ', end='', flush=True)
            client.close()
            print('Done.')

        if use_multiprocessing_scheduler:
            with ProgressBar():  # Does this not work with local distributed client?
                with dask.config.set(num_workers=8):
                    compute_ds = compute_ds.compute(scheduler='processes')

        dur = time() - tic
        print("Computations finished in {}m:{:.0f}s.\n\n ".format(np.floor(dur / 60), dur % 60))
        # print(compute_ds)

        if model == 'Year_persistence':
            compute_ds = compute_ds.expand_dims({'leadtime': leadtimes})

        mapping = {'leadtime': 'Leadtime', 'time': 'Forecast date'}

        compute_df = compute_ds.to_dataframe().reset_index().rename(columns=mapping).\
            assign(Model=model).set_index(['Leadtime', 'Forecast date', 'Model'])

        print('Writing to results dataset (this can take a minute)...')
        tic = time()
        results_df.loc[pd.IndexSlice[leadtimes, forecast_target_dates, model], compute_df.columns] = \
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
