import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), 'icenet2'))
import config
import utils
import json
from datetime import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
import xarray as xr
import dask
import dask.array as da
from dask.diagnostics import ProgressBar
from time import time
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from distributed import Client, progress
import seaborn as sns
import matplotlib
matplotlib.rcParams.update({
    'figure.facecolor': 'w',
    'figure.dpi': 300
})

use_local_distributed_cluster = True
use_multiprocessing_scheduler = False

if (
    (use_local_distributed_cluster and use_multiprocessing_scheduler) or
    not (use_local_distributed_cluster or use_multiprocessing_scheduler)
    ):
    raise ValueError('You must specify a single Dask parallelisation strategy '
                     'at the beginning of the script.')

# temp_dir = 'tmp_dask'
temp_dir = '/local/tmp'
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)

if __name__ == '__main__':

    # dask.config.set(temporary_directory=os.path.expandvars("/tmp/$USER/dask-tmp"))
    dask.config.set(temporary_directory=os.path.expandvars(temp_dir))

    if use_local_distributed_cluster:
        # client = Client(n_workers=8, threads_per_worker=3)
        client = Client(n_workers=8)
        # client = Client(n_workers=16)
        # client = Client(n_workers=16, threads_per_worker=8)
        # client = Client()
        print(client)

    ####################################################################

    network_name = 'unet_batchnorm'
    dataloader_name = '2021_03_03_1928_icenet2_init'

    # Format for storing different IceNet2 results in one dataframe
    icenet2_name = 'IceNet2__{}__{}'.format(dataloader_name, network_name)

    pre_load_results_df = True

    # What to compute
    model_compute_list = ['Year_persistence', 'Day_persistence', icenet2_name]
    # TODO: what about adding a new model for which none of the metrics have been done?
    # TODO: what about icenet2 results changing but benchmarks not changing?
    # TODO: compute_dict of model: metric pairs determined from loaded results_df?

    # metric_compute_list = ['MAE', 'RMSE', 'MSE']
    # metric_compute_list = ['Binary_accuracy', 'MAE']
    # metric_compute_list = ['Binary_accuracy']
    # metric_compute_list = ['Binary_accuracy', 'foo']
    metric_compute_list = ['RMSE']

    do_plotting = True

    ### Dataloader
    ####################################################################

    dataloader_config_fpath = os.path.join('dataloader_configs', dataloader_name+'.json')
    with open(dataloader_config_fpath, 'r') as f:
        dataloader_config = json.load(f)

    n_forecast_days = dataloader_config['n_forecast_days']

    # TODO: work out optimal chunking
    chunking = dict(time=31, leadtime=n_forecast_days)

    # TEMP
    chunking = dict(time=8, leadtime=n_forecast_days)
    # chunking = dict(time=7, leadtime=int(n_forecast_days/4))

    # TEMP
    # chunking = dict(time=1, leadtime=3)

    ### Monthly masks
    ####################################################################

    mask_fpath_format = os.path.join(config.folders['masks'],
                                     config.formats['active_grid_cell_mask'])

    month_mask_da = xr.DataArray(np.array(
        [np.load(mask_fpath_format.format('{:02d}'.format(month))) for
         month in np.arange(1, 12+1)],
    ))

    ### Initialise results dataframe
    ####################################################################

    leadtimes = np.arange(1, dataloader_config['n_forecast_days']+1)

    forecast_target_dates = utils.filled_daily_dates(
        start_date=datetime(2012, 1, 1), end_date=datetime(2018, 1, 1)
    )

    # results_df_fpath = os.path.join(
    #     config.folders['results'], dataloader_name, network_name, 'results.csv'
    # )

    results_df_fpath = os.path.join(
        config.folders['results'], 'results.csv'
    )

    if pre_load_results_df:
        results_df = pd.read_csv(results_df_fpath)
        # results_df = results_df.drop('Unnamed: 0', axis=1)  # Drop spurious index column
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

    ### Load forecasts
    ####################################################################

    validation_forecasts_dict = {}

    for model in compute_dict.keys():

        if model == 'Year_persistence':
            # Forecasts computed locally in this script as they are so simple
            continue

        if model == icenet2_name:
            validation_forecast_folder = os.path.join(
                config.folders['data'], 'forecasts', 'icenet2', dataloader_name, network_name
            )
        else:
            validation_forecast_folder = os.path.join(
                config.folders['data'], 'forecasts', model
            )
        validation_prediction_fpaths = [
            os.path.join(validation_forecast_folder, f) for f in os.listdir(validation_forecast_folder)
        ]
        with dask.config.set(**{'array.slicing.split_large_chunks': False}):
            forecast_ds = xr.open_mfdataset(
                validation_prediction_fpaths, chunks=chunking
            )
            validation_forecasts_dict[model] = forecast_ds.sel(time=forecast_target_dates)

    ### True SIC
    ####################################################################

    true_sic_fpath = os.path.join(config.folders['data'], 'siconca', 'siconca_all_interp.nc')
    true_sic_da = xr.open_dataarray(true_sic_fpath,
                                    chunks=dict(time=chunking['time']))

    # Replace 12:00 hour with 00:00 hour by convention
    dates = [pd.Timestamp(date).to_pydatetime().replace(hour=0) for date in true_sic_da.time.values]
    true_sic_da = true_sic_da.assign_coords(dict(time=dates))

    if 'Year_persistence' in model_compute_list:
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

    for model, model_metric_compute_list in compute_dict.items():

        print(model)

        for metric in model_metric_compute_list:

            if metric == 'MAE':

                err_da = validation_forecasts_dict[model] - true_sic_da

                abs_err_da = da.fabs(err_da)

                # Compute weighted MAE
                abs_weighted_da = abs_err_da.weighted(mask_da)
                compute_da = (abs_weighted_da.mean(dim=['yc', 'xc']) * 100)

                # print('visualising')
                # g = compute_da.data.__dask_graph__()
                # g.visualize(filename='graph.pdf', rankdir='LR')
                # # compute_da.data.visualize(filename='graph.pdf')
                # print('done')

            if metric == 'RMSE':

                # TODO: save compute if doing MSE and RMSE

                err_da = validation_forecasts_dict[model] - true_sic_da

                square_err_da = err_da**2

                # Computed weighted RMSE
                abs_weighted_da = square_err_da.weighted(mask_da)
                compute_da = da.sqrt(abs_weighted_da.mean(dim=['yc', 'xc'])) * 100

            if metric == 'Binary_accuracy':
                forecast_binary_da = validation_forecasts_dict[model] > 0.15
                binary_correct_da = (forecast_binary_da == true_sic_binary_da).astype(np.float32)
                binary_correct_weighted_da = binary_correct_da.weighted(mask_da)

                # Mean percentage of correct classifications over the active
                #   grid cell area
                compute_da = (binary_correct_weighted_da.mean(dim=['yc', 'xc']) * 100)

            print('Computing {}.'.format(metric))
            tic = time()

            if use_local_distributed_cluster:
                compute_da = compute_da.persist()
                progress(compute_da)
                compute_da.compute()

            if use_multiprocessing_scheduler:
                with ProgressBar():  # Does this not work with local distributed client?
                    with dask.config.set(num_workers=8):
                        compute_da = compute_da.compute(scheduler='processes')

            dur = time() - tic
            print("Done in {}m:{:.0f}s.\n\n ".format(np.floor(dur / 60), dur % 60))

            compute_df = compute_da.to_dataframe()

            idx = pd.IndexSlice
            if model == 'Year_persistence':
                # Not a function of lead time
                for leadtime in leadtimes:
                    results_df.loc[idx[leadtime, :, model], metric] = \
                        compute_df.values
            else:
                results_df.loc[idx[:, :, model], metric] = \
                    compute_df.values

    if use_local_distributed_cluster:
        client.close()

    # Make sure index names are correct
    results_df.index = results_df.index.rename(['Leadtime', 'Forecast date', 'Model'])

    results_df.to_csv(results_df_fpath)
