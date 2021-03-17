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

    icenet2_name = 'unet_batchnorm'
    dataloader_name = '2021_03_03_1928_icenet2_init'

    # TODO model list e.g. ['IceNet2', 'Persistence', 'SEAS5']

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

    config.folders['masks']
    mask_fpath_format = os.path.join(config.folders['masks'],
                                     config.formats['active_grid_cell_mask'])

    mask_da = xr.DataArray(np.array(
        [np.load(mask_fpath_format.format('{:02d}'.format(month))) for month in np.arange(1, 12+1)],
    ))

    ### Initialise results dataframe
    ####################################################################

    leadtimes = np.arange(1, dataloader_config['n_forecast_days']+1)

    forecast_target_dates = utils.filled_daily_dates(
        start_date=datetime(2012, 1, 1), end_date=datetime(2018, 1, 1)
    )

    model_list = ['Year_persistence', 'Day_persistence', 'IceNet2']

    # metric_list = ['MAE', 'RMSE', 'MSE']
    metric_list = ['MAE']

    multi_index = pd.MultiIndex.from_product(
        [leadtimes, forecast_target_dates, model_list],
        names=['Leadtime', 'Forecast date', 'Model'])
    results_df = pd.DataFrame(index=multi_index, columns=metric_list)
    results_df = pd.concat([results_df, pd.DataFrame(columns=metric_list)], axis=1)

    results_df_fpath = os.path.join(
        config.folders['results'], dataloader_name, icenet2_name, 'results.csv'
    )

    ### Load forecasts
    ####################################################################

    # TODO loop through model_list
    # TODO only need to run on statistical benchmarks once technically???

    validation_forecasts_dict = {}

    for model in model_list:

        if model == 'IceNet2':
            validation_forecast_folder = os.path.join(
                config.folders['data'], 'forecasts', 'icenet2', dataloader_name, icenet2_name
            )
        else:
            validation_forecast_folder = os.path.join(
                config.folders['data'], 'forecasts', model
            )
        validation_prediction_fpaths = [
            os.path.join(validation_forecast_folder, f) for f in os.listdir(validation_forecast_folder)
        ]
        forecast_ds = xr.open_mfdataset(
            validation_prediction_fpaths, chunks=chunking
        )
        validation_forecasts_dict[model] = forecast_ds.sel(time=forecast_target_dates)

    ### True SIC
    ####################################################################

    true_sic_fpath = os.path.join(config.folders['data'], 'siconca', 'siconca_all_interp.nc')
    true_sic_da = xr.open_dataarray(true_sic_fpath)

    # Replace 12:00 hour with 00:00 hour by convention
    dates = [pd.Timestamp(date).to_pydatetime().replace(hour=0) for date in true_sic_da.time.values]
    true_sic_da = true_sic_da.assign_coords(dict(time=dates))

    if 'Year_persistence' in model_list:
        persistence_forecast_da = true_sic_da.copy()
        dates = [pd.Timestamp(date) + relativedelta(days=365) for date in persistence_forecast_da.time.values]
        persistence_forecast_da = persistence_forecast_da.assign_coords(dict(time=dates))
        persistence_forecast_da = persistence_forecast_da.sel(time=forecast_target_dates)
        validation_forecasts_dict['Year_persistence'] = persistence_forecast_da

    true_sic_da = true_sic_da.sel(time=forecast_target_dates)

    ### Compute
    ####################################################################

    for model in model_list:

        print(model)

        for metric in metric_list:

            if metric == 'MAE':

                err_da = validation_forecasts_dict[model] - true_sic_da

                abs_err_da = da.fabs(err_da)

                months = [pd.Timestamp(date).month-1 for date in abs_err_da.time.values]
                mask_arr = xr.DataArray(
                    [mask_da[month] for month in months],
                    dims=('time', 'yc', 'xc'),
                    coords={
                        'time': abs_err_da.time.values,
                        'yc': abs_err_da.yc.values,
                        'xc': abs_err_da.xc.values,
                    }
                )

                # Computed weighted MAE
                abs_weighted = abs_err_da.weighted(mask_arr)
                compute_da = (abs_weighted.mean(dim=['yc', 'xc']) * 100)

                # print('visualising')
                # g = compute_da.data.__dask_graph__()
                # g.visualize(filename='graph.pdf', rankdir='LR')
                # # compute_da.data.visualize(filename='graph.pdf')
                # print('done')

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
            # compute_df = compute_da.to_dataframe(name='compute_da')

            if model == 'Year_persistence':
                # Not a function of lead time
                for leadtime in leadtimes:
                    results_df.loc[leadtime, :, model] = \
                        compute_df.values
            else:
                results_df.loc[:, :, model] = \
                    compute_df.values

    if use_local_distributed_cluster:
        client.close()

    results_df.to_csv(results_df_fpath)

    # TEMP plotting
    results_df = pd.read_csv(results_df_fpath)
    results_df['Forecast date'] = [pd.Timestamp(date) for date in results_df['Forecast date']]
    results_df['dayofyear'] = results_df['Forecast date'].dt.dayofyear
    results_df = results_df.set_index(['Model', 'Leadtime', 'Forecast date'])

    # results_df = results_df.rename(columns={'__xarray_dataarray_variable__': 'MAE'})

    heatmap_dfs = {}
    for model in model_list:
        heatmap_dfs[model] = results_df.loc[model].groupby(['dayofyear', 'Leadtime']).mean().reset_index().\
            pivot('dayofyear', 'Leadtime', 'MAE')

        heatmap_dfs[model].index = pd.to_datetime(heatmap_df.index, unit='D', origin='2012-01-01') - pd.Timedelta(days=1)

    heatmap_dfs['Day_persistence'] - heatmap_dfs['IceNet2']

    import matplotlib
    matplotlib.rcParams.update({
        'figure.facecolor': 'w',
        'figure.dpi': 300
    })

    for model in model_list:
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.heatmap(
            data=heatmap_dfs[model],
            ax=ax,
            cbar_kws=dict(label='SIC MAE (%)')
        )
        ax.yaxis.set_major_locator(matplotlib.dates.DayLocator(bymonthday=15))
        ax.tick_params(axis='y', which='major',length=0)
        ax.yaxis.set_major_formatter(matplotlib.dates.DateFormatter('%m'))
        ax.yaxis.set_minor_locator(matplotlib.dates.DayLocator(bymonthday=1))
        ax.set_xticks(np.arange(30, n_forecast_days, 30))
        ax.set_xticklabels(np.arange(30, n_forecast_days, 30))
        ax.set_title('{} MAE'.format(model))
        ax.set_ylabel('Calendar month')
        ax.set_xlabel('Lead time (days)')
        plt.tight_layout()
        plt.savefig('mae_heatmap_{}.png'.format(model.lower()))
        plt.close()

    for model in ['Day_persistence', 'Year_persistence']:

        heatmap_df_diff = heatmap_dfs['IceNet2'] - heatmap_dfs[model]
        max = np.max(np.abs(heatmap_df_diff.values))

        fig, ax = plt.subplots(figsize=(6, 6))
        sns.heatmap(
            data=heatmap_df_diff,
            cmap='seismic',
            ax=ax,
            vmax=max,
            vmin=-max,
            cbar_kws=dict(label='SIC MAE (%)')
        )

        ax.yaxis.set_major_locator(matplotlib.dates.DayLocator(bymonthday=15))
        ax.tick_params(axis='y', which='major',length=0)
        ax.yaxis.set_major_formatter(matplotlib.dates.DateFormatter('%m'))
        ax.yaxis.set_minor_locator(matplotlib.dates.DayLocator(bymonthday=1))
        ax.set_xticks(np.arange(30, n_forecast_days, 30))
        ax.set_xticklabels(np.arange(30, n_forecast_days, 30))
        ax.set_title('IceNet2 MAE improvement over {}'.format(model))
        ax.set_ylabel('Calendar month')
        ax.set_xlabel('Lead time (days)')
        plt.tight_layout()
        plt.savefig('diff_mae_heatmap_{}.png'.format(model.lower()))
        plt.close()

    fig, ax = plt.subplots()
    sns.lineplot(
        x='Leadtime',
        y='MAE',
        ci=None,
        hue='Model',
        data=results_df,
        ax=ax
    )
    ax.set_ylabel('MAE (%)')
    plt.tight_layout()
    plt.savefig('mae_vs_leadtime.png')
    ax.set_xticks(np.arange(30, n_forecast_days, 30))
    ax.set_xticklabels(np.arange(30, n_forecast_days, 30))
    plt.close()

    fig, ax = plt.subplots()
    sns.lineplot(
        x='Leadtime',
        y='MAE',
        hue='Model',
        data=results_df.loc[:, :, datetime(2012,9,15)],
        ax=ax
    )
    ax.set_ylabel('MAE (%)')
    ax.set_xticks(np.arange(30, n_forecast_days, 30))
    ax.set_xticklabels(np.arange(30, n_forecast_days, 30))
    plt.tight_layout()
    plt.savefig('mae_vs_leadtime_2012_09_15.png')
    plt.close()

    fig, ax = plt.subplots()
    sns.lineplot(
        x='Forecast date',
        y='MAE',
        data=results_df[results_df.Leadtime.isin([1, 30, 60, 90])],
        hue='Leadtime',
        ax=ax
    )
    ax.set_ylabel('MAE (%)')
    ax.set_xlabel('Forecast date')
    plt.tight_layout()
    plt.savefig('mae_vs_forecast_date.png')
    plt.close()

    # TODO: assign values to results_df pandas DataFrame
    # results_df.loc[:, :, 'IceNet2'].MAE = df.values  # ???
