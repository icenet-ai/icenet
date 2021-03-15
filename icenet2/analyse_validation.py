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
from dask.diagnostics import ProgressBar
from time import time
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from dask.distributed import Client, progress
import seaborn as sns

use_local_distributed_cluster = False
use_multiprocessing_scheduler = True

if (
    (use_local_distributed_cluster and use_multiprocessing_scheduler) or
    not (use_local_distributed_cluster or use_multiprocessing_scheduler)
    ):
    raise ValueError('You must specify a single Dask parallelisation strategy '
                     'at the beginning of the script.')

if __name__ == '__main__':
    if use_local_distributed_cluster:
        client = Client(asynchronous=True, n_workers=16, threads_per_worker=8)
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
    chunking = dict(time=7, leadtime=int(n_forecast_days/4))

    # TEMP
    # chunking = dict(time=1, leadtime=3)

    ### Monthly masks
    ####################################################################

    config.folders['masks']
    mask_fpath_format = os.path.join(config.folders['masks'],
                                     config.formats['active_grid_cell_mask'])

    mask_dict = {}
    for month in np.arange(1, 12+1):
        month_str = '{:02d}'.format(month)
        mask_dict[month] = np.load(mask_fpath_format.format(month_str))

    ### TEMP: map_funcs
    ####################################################################

    def get_masks(da):
        '''
        Returns a boolean mask array of shape (n_dates, n_x, n_y) of month-wise masks
        for computing the accuracy over.
        '''

        months = [pd.Timestamp(date).month for date in da.time.values]

        mask_arr = np.array([mask_dict[month] for month in months])

        return mask_arr

    # TODO: generic chunk processing function to return any metric
    # TODO: logic for preloading results_df and determining if already computed
    def chunk_mae_func(abs_err_chunk):
        '''
        Compute of forecast SIC data MAE over a chunk of a Dask array of
        absolute errors. MAE is computed over an active grid cell area
        that is dependent on the forecast month.

        TODO: this needs to return one MAE value per time and leadtime

        Parameters:
        abs_err_chunk (xr.DataArray): Subset of (time, yc, xc, leadtime) dimensional
        DataArray.

        Returns:
        mae (np.float32): Mean absolute error over the active grid cell area.
        '''

        # TODO: if this is computing bin acc, chunk is error bits
        # TODO: if this is computing some func of error, chunk is error reals

        # Get the month-wise masks to compute accuracy over
        mask_da = xr.DataArray(
            data=get_masks(abs_err_chunk),
            dims=('time', 'yc', 'xc'),
            coords={
                'time': abs_err_chunk.time.values,
                'yc': abs_err_chunk.yc.values,
                'xc': abs_err_chunk.xc.values,
            }
        )

        # Computed weighted MAE
        # TODO: is this a dask array?
        chunk_weighted = abs_err_chunk.weighted(mask_da)
        mae = chunk_weighted.mean(dim=('yc', 'xc'))

        return mae

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
    forecast_da = xr.open_mfdataset(
        validation_prediction_fpaths, chunks=chunking
    )
    forecast_da = forecast_da.to_array()[0].drop('variable')  # Convert to DataArray
    forecast_da = forecast_da.sel(time=forecast_target_dates)

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

    abs_err_da = xr.ufuncs.fabs(forecast_da - true_sic_da)

    abs_err_da = abs_err_da.chunk(chunking)

    # TEMP decreasing dataset size for debugging
    # abs_err_da = abs_err_da.sel(time=slice('2012-1-1', '2012-1-31'))
    # abs_err_da = abs_err_da.sel(time=slice('2012-1-1', '2012-1-1'),
    #                             leadtime=[1])

    template_da = xr.DataArray(
        np.zeros((len(abs_err_da.time.values), len(abs_err_da.leadtime.values))),
        dims=('time', 'leadtime'),
        coords={
            'time': abs_err_da.time.values,
            'leadtime': abs_err_da.leadtime.values,
        },
    ).chunk(chunking)

    mae_da = abs_err_da.map_blocks(chunk_mae_func, template=template_da)
    mae_da *= 100  # Convert to SIC (%)

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
        mae_da = mae_da.compute()

    if use_multiprocessing_scheduler:
        with ProgressBar():  # Does this not work with local distributed client?
            with dask.config.set(num_workers=32):
                mae_da = mae_da.compute(scheduler='processes')

    dur = time() - tic
    print("Done in {}m:{:.0f}s.\n\n ".format(np.floor(dur / 60), dur % 60))

    df = mae_da.to_dataframe(name='mae_da')

    # TEMP saving
    df.to_csv('temp.csv')

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
