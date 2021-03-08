import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), 'icenet2'))
import config
import utils
import metrics
import losses
import numpy as np
import xarray as xr
from tqdm import tqdm
import plotly.express as px
from datetime import datetime
from dateutil.relativedelta import relativedelta
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

####################################################################

# TODO: user options for which models to run validation predictions for

# TODO: generic predict functions for the different models that take init date
#   as input?

icenet2_name = 'unet_batchnorm'
dataloader_name = '2021_03_03_1928_icenet2_init'

seed = 42

#### Load network and dataloader
####################################################################

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

validation_forecast_folder = os.path.join(config.folders['results'], dataloader_name,
                                          icenet2_name, 'validation')
if not os.path.exists(validation_forecast_folder):
    os.makedirs(validation_forecast_folder)

#### Set up DataArray of forecasts
####################################################################

n_forecast_days = dataloader.config['n_forecast_days']

# TODO: get this from dataloader config?
all_forecast_target_dates = utils.filled_daily_dates(
    start_date=datetime(2012, 1, 1), end_date=datetime(2020, 12, 31),
    include_end=True
)

all_forecast_start_dates = utils.filled_daily_dates(
    start_date=all_forecast_target_dates[0] - relativedelta(days=n_forecast_days),
    end_date=all_forecast_target_dates[-1],
    include_end=True)

da_with_coords = xr.open_dataarray('data/siconca/raw_yearly_data/siconca_1979.nc')

# TODO: make this a dataset with ground turth (time xc yx) as well?
yearly_forecast_da_dict = {}

# TODO: don't hard code
for year in np.arange(2012, 2020+1):

    year_forecast_target_dates = utils.filled_daily_dates(
        start_date=datetime(year, 1, 1), end_date=datetime(year, 12, 31),
        include_end=True
    )

    shape = (len(year_forecast_target_dates), *dataloader.config['raw_data_shape'], n_forecast_days)

    yearly_forecast_da_dict[year] = xr.DataArray(
        data=np.zeros(shape, dtype=np.float32),
        dims=('time', 'xc', 'yc', 'leadtime'),
        coords={
            'time': year_forecast_target_dates,  # To be sliced to target dates
            'xc': da_with_coords.coords['xc'],
            'yc': da_with_coords.coords['yc'],
            # 'lon': (['yc, xc'], da_with_coords.coords['lon'].values),
            # 'lat': (['yc, xc'], da_with_coords.coords['lat'].values),
            'leadtime': np.arange(1, n_forecast_days+1)
        }
    )

#### Build up forecasts
####################################################################

# TODO: make this save in inidiv yearly xarray datasets according to target date

leadtimes = np.arange(1, n_forecast_days+1)

# TODO: don't hard code
year_to_save = 2012

# forecast_start_date = forecast_start_dates[0]
print('Building up forecast DataArrays...\n')
for forecast_start_date in tqdm(all_forecast_start_dates):

    X, y = dataloader.data_generation(np.array([forecast_start_date]))
    mask = y[0, :, :, :, 1] == 0

    pred = network.predict(X)[0]
    pred[mask] = 0.

    forecast_target_dates = utils.filled_daily_dates(
        start_date=forecast_start_date,
        end_date=forecast_start_date + relativedelta(days=n_forecast_days),
        include_end=True
    )

    for i, (forecast_target_date, leadtime) in enumerate(zip(forecast_target_dates, leadtimes)):
        if forecast_target_date in all_forecast_target_dates:
            year = forecast_target_date.year
            yearly_forecast_da_dict[year].\
                loc[forecast_target_date, :, :, leadtime] = pred[:, :, i]

    # End of year reached - save completed yearly NetCDF
    if forecast_start_date == datetime(year_to_save, 12, 31):
        print('Saving forecast NetCDF for {}'.format(year_to_save), end='', flush=True)
        yearly_forecast_da_dict[year].to_netcdf(
            os.path.join(validation_forecast_folder, '{:04d}.nc'.format(year_to_save))
        )
        print('Done.')

        # TODO: check this stops memory growing
        del(yearly_forecast_da_dict[year_to_save])  # Does this do with memory management anything?

        year_to_save += 1
print('Done.')


# from tqdm import tqdm
# land_mask = np.load(os.path.join(config.folders['masks'], config.fnames['land_mask']))
# accs = []
#
# for year in tqdm(np.arange(2012,2021)):
#     print(year)
#     for month in np.arange(1, 13):
#         print(month)
#         X, y = dataloader.data_generation(np.array([datetime(year,month,1)]))
#         y*=100
#         pred = 100*network.predict(X)
#
#         pred_monthly_mean = np.mean(pred[0], axis=-1)
#         y_monthly_mean = np.mean(y[0, :, :, :, 0], axis=-1)
#         mask = y[0, :, :, 0, 1] == 0
#         pred_monthly_mean[mask] = 0
#
#         err = pred_monthly_mean - y_monthly_mean
#
#         mae = np.mean(np.abs(err[~mask]))
#
#         fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 6))
#
#         ax = axes[0]
#         im = ax.imshow(y_monthly_mean, cmap='Blues_r', clim=(0, 100))
#         ax.contour(land_mask, levels=[.5], colors='k')
#         divider = make_axes_locatable(ax)
#         cax = divider.append_axes('right', size='5%', pad=0.05)
#         fig.colorbar(im, cax)
#         ax.set_title('True map', fontsize=20)
#
#         ax = axes[1]
#         im = ax.imshow(pred_monthly_mean, cmap='Blues_r', clim=(0, 100))
#         ax.contour(land_mask, levels=[.5], colors='k')
#         divider = make_axes_locatable(ax)
#         cax = divider.append_axes('right', size='5%', pad=0.05)
#         fig.colorbar(im, cax)
#         ax.set_title('IceNet2 prediction, MAE: {:.2f}%'.format(mae), fontsize=20)
#
#         ax = axes[2]
#         im = ax.imshow(err, cmap='seismic', clim=(-100, 100))
#         ax.contour(land_mask, levels=[.5], colors='k')
#         divider = make_axes_locatable(ax)
#         cax = divider.append_axes('right', size='5%', pad=0.05)
#         fig.colorbar(im, cax)
#         ax.set_title('Prediction minus true', fontsize=20)
#
#         # for ax in axes:
#         #     ax.axes.xaxis.set_visible(False)
#         #     ax.axes.yaxis.set_visible(False)
#         plt.tight_layout()
#
#         plt.savefig('init_validation/{:04d}_{:02d}.png'.format(year, month), facecolor='white', dpi=300)
#         plt.close()
#
#         acc = accuracy_score(y_monthly_mean[~mask]>15, pred_monthly_mean[~mask]>15)
#         accs.append(acc)
#
# accs = np.array(accs)*100
# print(np.mean(accs))
#
# # px.line(y=accs)
