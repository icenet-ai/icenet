import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), 'icenet2'))
import config
import utils
import json
from datetime import datetime
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

if __name__ == '__main__':

    dask.config.set(temporary_directory=os.path.expandvars("/tmp/$USER/dask-tmp"))

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

    model_name_list = ['IceNet2']

    # metrics_list = ['MAE', 'RMSE', 'MSE']
    metrics_list = ['MAE']

    multi_index = pd.MultiIndex.from_product(
        [leadtimes, forecast_target_dates, model_name_list],
        names=['Leadtime', 'Forecast date', 'Model'])
    results_df = pd.DataFrame(index=multi_index)
    results_df = pd.concat([results_df, pd.DataFrame(columns=metrics_list)], axis=1)

    ### IceNet2 validation predictions
    ####################################################################

    validation_forecast_folder = os.path.join(
        config.folders['results'], dataloader_name, icenet2_name, 'validation'
    )
    validation_prediction_fpaths = [
        os.path.join(validation_forecast_folder, f) for f in os.listdir(validation_forecast_folder)
    ]
    forecast_ds = xr.open_mfdataset(
        validation_prediction_fpaths, chunks=chunking
    )
    forecast_ds = forecast_ds.sel(time=forecast_target_dates)

    ### True SIC
    ####################################################################

    true_sic_fpath = os.path.join(config.folders['data'], 'siconca', 'siconca_all_interp.nc')
    true_sic_da = xr.open_dataarray(true_sic_fpath)

    # Replace 12:00 hour with 00:00 hour by convention
    dates = [pd.Timestamp(date).to_pydatetime().replace(hour=0) for date in true_sic_da.time.values]
    true_sic_da = true_sic_da.assign_coords(dict(time=dates))
    true_sic_da = true_sic_da.sel(time=forecast_target_dates)

    ### Compute
    ####################################################################

    err_da = forecast_ds - true_sic_da

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
    mae_da = (abs_weighted.mean(dim=['yc', 'xc']) * 100)

    # print('visualising')
    # g = mae_da.data.__dask_graph__()
    # g.visualize(filename='graph.pdf', rankdir='LR')
    # # mae_da.data.visualize(filename='graph.pdf')
    # print('done')

    print('Computing MAE.')
    tic = time()

    if use_local_distributed_cluster:
        mae_da = mae_da.persist()
        progress(mae_da)
        mae_da.compute()

    if use_multiprocessing_scheduler:
        with ProgressBar():  # Does this not work with local distributed client?
            with dask.config.set(num_workers=8):
                mae_da = mae_da.compute(scheduler='processes')

    dur = time() - tic
    print("Done in {}m:{:.0f}s.\n\n ".format(np.floor(dur / 60), dur % 60))

    mae_df = mae_da.to_dataframe()
    # mae_df = mae_da.to_dataframe(name='mae_da')

    if use_local_distributed_cluster:
        client.close()

    # TEMP saving
    mae_df.to_csv('temp.csv')

    mae_df = pd.read_csv('temp.csv')
    mae_df.time = [pd.Timestamp(date) for date in mae_df.time]
    mae_df['dayofyear'] = mae_df.time.dt.dayofyear

    heatmap_df = mae_df.groupby(['dayofyear', 'leadtime']).mean().reset_index().\
        pivot('dayofyear', 'leadtime', 'mae_da')

    heatmap_df.index = pd.to_datetime(heatmap_df.index, unit='D', origin='2012-01-01') - pd.Timedelta(days=1)

    import matplotlib
    matplotlib.rcParams.update({
        'figure.facecolor': 'w',
        'figure.dpi': 300
    })

    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(
        data=heatmap_df,
        ax=ax,
        cbar_kws=dict(label='SIC MAE (%)')
    )
    ax.yaxis.set_major_locator(matplotlib.dates.DayLocator(bymonthday=15))
    ax.tick_params(axis='y', which='major',length=0)
    ax.yaxis.set_major_formatter(matplotlib.dates.DateFormatter('%m'))
    ax.yaxis.set_minor_locator(matplotlib.dates.DayLocator(bymonthday=1))
    ax.set_xticks(np.arange(30, n_forecast_days, 30))
    ax.set_xticklabels(np.arange(30, n_forecast_days, 30))
    ax.set_title('IceNet2 MAE')
    ax.set_ylabel('Calendar month')
    ax.set_xlabel('Lead time (days)')
    plt.tight_layout()
    plt.savefig('mae_heatmap.png')
    plt.close()

    fig, ax = plt.subplots()
    sns.lineplot(
        x='leadtime',
        y='mae_da',
        data=mae_df,
        ax=ax
    )
    ax.set_ylabel('MAE (%)')
    plt.tight_layout()
    plt.savefig('mae_vs_leadtime.png')
    plt.close()

    fig, ax = plt.subplots()
    sns.lineplot(
        x='leadtime',
        y='mae_da',
        data=mae_df[mae_df.time==datetime(2012,9,15)],
        ax=ax
    )
    ax.set_ylabel('MAE (%)')
    plt.tight_layout()
    plt.savefig('mae_vs_leadtime_2012_09_15.png')
    plt.close()

    fig, ax = plt.subplots()
    sns.lineplot(
        x='time',
        y='mae_da',
        data=mae_df[mae_df.leadtime.isin([1, 30, 60, 90])],
        hue='leadtime',
        ax=ax
    )
    ax.set_ylabel('MAE (%)')
    ax.set_xlabel('Forecast date')
    plt.tight_layout()
    plt.savefig('mae_vs_forecast_date.png')
    plt.close()

    # TODO: assign values to results_df pandas DataFrame
    # results_df.loc[:, :, 'IceNet2'].MAE = df.values  # ???
