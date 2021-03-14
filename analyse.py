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
from dask.diagnostics import ProgressBar

import pandas as pd

import icenet2.config as config
import icenet2.utils as utils

from distributed import Client

### TEMP: map_funcs
####################################################################


# TODO: Context, where should this be run icenet2 or icenet2/icenet2 (import config)
if __name__ == '__main__':
    dask.config.set(temporary_directory=os.path.expandvars("/tmp/$USER/dask-tmp"))
    client = Client(n_workers=16)

    ####################################################################
    icenet2_name = 'unet_batchnorm'
    dataloader_name = '2021_03_03_1928_icenet2_init'

    ### Dataloader
    ####################################################################

    dataloader_config_fpath = os.path.join('dataloader_configs', dataloader_name+'.json')
    with open(dataloader_config_fpath, 'r') as f:
        dataloader_config = json.load(f)

    n_forecast_days = dataloader_config['n_forecast_days']

    ## TEMP
    chunking = dict(time=1, leadtime=n_forecast_days)

    ### Monthly masks
    ####################################################################

    mask_fpath_format = os.path.join(config.folders['masks'],
                                     config.formats['active_grid_cell_mask'])

    mask_da = xr.DataArray(np.array(
        [np.load(mask_fpath_format.format('{:02d}'.format(month))) for month in np.arange(1, 12+1)],
    ))
    # dask.array<array, shape=(12, 432, 432), dtype=bool, chunksize=(12, 432, 432), chunktype=numpy.ndarray>

    ### IceNet2 validation predictions
    ####################################################################

    validation_forecast_folder = os.path.join(
        #config.folders['results'], dataloader_name, icenet2_name, 'validation'
        "/data/hpcdata/users/tomand/code/icenet2/results/", dataloader_name, icenet2_name, 'validation'
    )
    validation_prediction_fpaths = [
        os.path.join(validation_forecast_folder, f) for f in os.listdir(validation_forecast_folder)
    ]

    forecast_target_dates = utils.filled_daily_dates(
        start_date=datetime(2012, 1, 1), end_date=datetime(2018, 1, 1)
    )

    #/data/hpcdata/users/tomand/code/icenet2/results/2021_03_03_1928_icenet2_init/unet_batchnorm/validation/2012.nc (48G)
    forecast_da = xr.open_mfdataset(
        validation_prediction_fpaths, chunks=dict(time=1, leadtime=1)
    )
    forecast_da = forecast_da.to_array()[0].drop('variable')
    forecast_da = forecast_da.sel(time=forecast_target_dates)

    # <xarray.DataArray 'stack-3c744aedb18a4507d557016ee243bcc2' (time: 2192, yc: 432, xc: 432, leadtime: 186)>
    # dask.array<getitem, shape=(2192, 432, 432, 186), dtype=float32, chunksize=(1, 432, 432, 1), chunktype=numpy.ndarray>
    # Coordinates:
    #   * time      (time) datetime64[ns] 2012-01-01 2012-01-02 ... 2017-12-31
    #   * yc        (yc) float64 5.388e+03 5.362e+03 ... -5.362e+03 -5.388e+03
    #   * xc        (xc) float64 -5.388e+03 -5.362e+03 ... 5.362e+03 5.388e+03
    #   * leadtime  (leadtime) int64 1 2 3 4 5 6 7 8 ... 180 181 182 183 184 185 186

    print("Forecast DA")
    print(forecast_da.shape)

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

    abs_err_da = xr.ufuncs.fabs(forecast_da - true_sic_da)

    print("abs err DA")
    print(abs_err_da.shape)

    print('Computing MAE.')
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
    with ProgressBar():
        mae = mae_da.compute()

    # >>> mae_da
    # <xarray.DataArray (time: 9, leadtime: 186)>
    # dask.array<mul, shape=(9, 186), dtype=float64, chunksize=(1, 1), chunktype=numpy.ndarray>
    # Coordinates:
    #   * time      (time) datetime64[ns] 2012-01-01 2012-01-02 ... 2012-01-09
    #   * leadtime  (leadtime) int64 1 2 3 4 5 6 7 8 ... 180 181 182 183 184 185 186
    # >>> mae_da.compute()
    # <xarray.DataArray (time: 9, leadtime: 186)>
    # array([[4.47474815, 5.0133635 , 4.9317736 , ..., 6.11015493, 5.96323838,
    #         6.07493676],
    #        [4.57382763, 5.0326483 , 4.93432715, ..., 6.35651186, 6.33104949,
    #         6.2287885 ],
    #        [4.84749644, 5.184488  , 5.16307038, ..., 6.5933675 , 6.49747025,
    #         6.52474704],
    #        ...,
    #        [4.62127685, 5.25285173, 5.09580409, ..., 6.06727758, 5.90394006,
    #         5.95896276],
    #        [4.58940307, 4.9673342 , 4.99307693, ..., 5.87748405, 5.85600548,
    #         5.86364249],
    #        [4.85256254, 5.06698623, 4.96427311, ..., 5.99312671, 5.82945896,
    #         5.94444338]])
    # Coordinates:
    #   * time      (time) datetime64[ns] 2012-01-01 2012-01-02 ... 2012-01-09
    #   * leadtime  (leadtime) int64 1 2 3 4 5 6 7 8 ... 180 181 182 183 184 185 186

    dur = time() - tic
    print("Done in {}m:{:.0f}s.\n\n ".format(np.floor(dur / 60), dur % 60))

    mae_df = mae.to_dataframe(name='mae')
    mae_df.to_csv('temp.csv')















