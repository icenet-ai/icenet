import argparse
import os
import sys

import json

from datetime import datetime
from time import time

import numpy as np
import xarray as xr

import dask
import dask.array as da
import dask.dataframe as df
from dask import delayed
from distributed.diagnostics.progressbar import progress

import pandas as pd

import icenet2.config as config
import icenet2.utils as utils

from distributed import Client
#from dask_jobqueue import SLURMCluster



####################################################################
def parse_args():
    a = argparse.ArgumentParser()
    a.add_argument("-t", "--time", default=8, type=int)
    return a.parse_args()


# TODO: Context, where should this be run icenet2 or icenet2/icenet2 (import config)
if __name__ == '__main__':
    args = parse_args()

    dask.config.set(temporary_directory=os.path.expandvars("/tmp/$USER/dask-tmp"))

    # TODO: This will run, but don't do it, it needs more testing
#    cluster = SLURMCluster(cores=64,
#                           memory="32 GB",
#                           project="short",
#                           walltime="00:30:00",
#                           queue="short")
#    cluster.scale(jobs=args.time)
    client = Client(n_workers=8)

    ####################################################################
    icenet2_name = 'unet_batchnorm'
    dataloader_name = '2021_03_03_1928_icenet2_init'

    ### Dataloader
    ####################################################################

    dataloader_config_fpath = os.path.join('dataloader_configs', dataloader_name+'.json')
    with open(dataloader_config_fpath, 'r') as f:
        dataloader_config = json.load(f)

    n_forecast_days = dataloader_config['n_forecast_days']
    print("Forecast days: {}".format(n_forecast_days))

    ## TEMP
    chunking = dict(time=args.time, leadtime=n_forecast_days)

    ### Monthly masks
    ####################################################################

    mask_fpath_format = os.path.join(config.folders['masks'],
                                     config.formats['active_grid_cell_mask'])

    mask_da = xr.DataArray(np.array(
        [np.load(mask_fpath_format.format('{:02d}'.format(month))) for month in np.arange(1, 12+1)],
    ))

    ### IceNet2 validation predictions
    ####################################################################

    validation_forecast_folder = os.path.join(
        #config.folders['results'], dataloader_name, icenet2_name, 'validation'
        "/data/hpcdata/users/tomand/code/icenet2/results/", dataloader_name, icenet2_name, 'validation'
    )

    #/data/hpcdata/users/tomand/code/icenet2/results/2021_03_03_1928_icenet2_init/unet_batchnorm/validation/2012.nc (48G)
    # Reduce to selected dates for any prod version?
    validation_prediction_fpaths = [
        os.path.join(validation_forecast_folder, f) for f in os.listdir(validation_forecast_folder)
    ]

    forecast_target_dates = utils.filled_daily_dates(
        #start_date=datetime(2012, 1, 1), end_date=datetime(2018, 1, 1)
        start_date=datetime(2012, 1, 1), end_date=datetime(2018, 1, 1)
    )

    print(validation_prediction_fpaths)

    forecast_ds = xr.open_mfdataset(
        validation_prediction_fpaths, chunks=dict(time=args.time, leadtime=n_forecast_days)
    )
    # Just realised what this might be, don't delete!
    #forecast_da = forecast_da.to_array()[0].drop('variable')
    forecast_ds = forecast_ds.sel(time=forecast_target_dates)

    #    dask.visualize(forecast_da, format='svg', filename="test")

    print("Forecast DS")
    forecast_ds.info()

    ### True SIC
    ####################################################################

    true_sic_fpath = os.path.join(config.folders['data'], 'siconca', 'siconca_all_interp.nc')
    true_sic_da = xr.open_dataarray(true_sic_fpath)

    # Replace 12:00 hour with 00:00 hour by convention
    dates = [pd.Timestamp(date).to_pydatetime().replace(hour=0) for date in true_sic_da.time.values]
    true_sic_da = true_sic_da.assign_coords(dict(time=dates))
    true_sic_da = true_sic_da.sel(time=forecast_target_dates)

    print("truesic DA")
    print(true_sic_da.shape)

    ### Compute
    ####################################################################

    err_da = forecast_ds - true_sic_da
    abs_err_da = da.fabs(err_da)

    print("abs err DA")
    abs_err_da.info()

    print('Setting up MAE.')
    tic = time()

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

    abs_weighted = abs_err_da.weighted(mask_arr)
    mae_da = (abs_weighted.mean(['yc', 'xc']) * 100)


    #print("Visualising")
    #dask.visualize(mae_da, format='svg', filename="test")
    #client.close()
    #sys.exit(0)

    print("Computing")

    mae = mae_da.persist()
    progress(mae)
    mae.compute()

    dur = time() - tic
    print("Done in {}m:{:.0f}s.\n\n ".format(np.floor(dur / 60), dur % 60))

    mae_df = mae.to_dataframe()
    mae_df.info()
    mae_df.to_csv('temp.csv')

    client.close()
    #cluster.close()
