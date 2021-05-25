import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), 'icenet2'))
import config
import re
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
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
mpl.rcParams.update({
    'figure.facecolor': 'w',
    'figure.dpi': 300
})

####################################################################

network_name = 'unet_batchnorm'
# dataloader_name = '2021_03_03_1928_icenet2_init'
dataloader_name = '2021_04_25_1351_icenet2_nh_thinned7_weeklyinput_wind_3month'
# dataloader_name = '2021_04_08_1205_icenet2_nh_sh_thinned5_weeklyinput_wind_3month'
# dataloader_name = 'icenet2_10ensemble'
seed = 'ensemble'

# Format for storing different IceNet2 results in one dataframe
icenet2_name = 'IceNet2__{}__{}__{}'.format(dataloader_name, network_name, seed)

n_forecast_days = 93

####################################################################

results_df_fnames = sorted([f for f in os.listdir('results') if re.compile('.*.csv').match(f)])
if len(results_df_fnames) >= 1:
    results_df_fname = results_df_fnames[-1]
    results_df_fpath = os.path.join('results', results_df_fname)
    print('\n\nLoading previous results dataset from {}'.format(results_df_fpath))

results_df = pd.read_csv(results_df_fpath)
print('Done.')

parent_fig_folder = os.path.join(
    config.folders['results'], dataloader_name, network_name, 'validation_figs'
)

if not os.path.exists(parent_fig_folder):
    os.makedirs(parent_fig_folder)

### Forecasts
####################################################################

validation_forecast_folder = os.path.join(
    'data', 'forecasts', 'icenet2', dataloader_name, network_name, seed
)

validation_prediction_fpaths = [
    os.path.join(validation_forecast_folder, f) for f in os.listdir(validation_forecast_folder)
]
forecast_ds = xr.open_mfdataset(validation_prediction_fpaths)
forecast_da = next(iter(forecast_ds.data_vars.values()))  # Convert to DataArray

################################################################################

model_list = results_df.Model.unique()

# Chop off lead times beyond user-specified max
results_df = results_df[results_df.Leadtime <= n_forecast_days]

results_df['Forecast date'] = [pd.Timestamp(date) for date in results_df['Forecast date']]
results_df = results_df.set_index(['Model', 'Leadtime', 'Forecast date'])
metric_list = results_df.columns
results_df['dayofyear'] = results_df.index.get_level_values(2).dayofyear
results_df = results_df.sort_index()

month_names = np.array(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec'])

set(results_df.index.get_level_values(0))

### Skill heatmaps
################################################################################

metric_dict = {
    'MAE': {
        'label': 'SIC MAE (%)',
        'cmap': 'inferno_r',
        'cmap_diff': 'seismic',
    },
    'RMSE': {
        'label': 'SIC RMSE (%)',
        'cmap': 'inferno_r',
        'cmap_diff': 'seismic',
    },
    'MSE': {
        'label': 'SIC MSE (%)',
        'cmap': 'inferno_r',
        'cmap_diff': 'seismic',
    },
    'Binary_accuracy': {
        'label': 'Binary accuracy (%)',
        'cmap': 'inferno',
        'cmap_diff': 'seismic_r',
    },
}

### Skill heatmaps
################################################################################

fig_folder = os.path.join(parent_fig_folder, 'skill_heatmaps')
if not os.path.exists(fig_folder):
    os.makedirs(fig_folder)

for metric in metric_list:
    print(metric)

    fontsize = 13
    context = {
        'font.size': fontsize,
        'axes.labelsize': fontsize,
        'axes.titlesize': fontsize,
        'ytick.labelsize': fontsize,
        'xtick.labelsize': fontsize
    }
    with plt.rc_context(context):

        heatmap_dfs = {}
        for model in model_list:
            heatmap_dfs[model] = results_df.loc[model].groupby(['dayofyear', 'Leadtime']).\
                mean().reset_index().pivot('dayofyear', 'Leadtime', metric)

            heatmap_dfs[model].index = \
                (pd.to_datetime(heatmap_dfs[model].index, unit='D', origin='2012-01-01') -
                 pd.Timedelta(days=1))

        for model in model_list:
            fig, ax = plt.subplots(figsize=(6, 6))
            sns.heatmap(
                data=heatmap_dfs[model],
                ax=ax,
                cmap=metric_dict[metric]['cmap'],
                # cmap=sns.cm.rocket_r,
                # cmap='inferno_r',
                cbar_kws=dict(label=metric_dict[metric]['label'])
            )
            ax.yaxis.set_major_locator(mpl.dates.DayLocator(bymonthday=15))
            ax.tick_params(axis='y', which='major',length=0)
            ax.yaxis.set_major_formatter(mpl.dates.DateFormatter('%m'))
            ax.yaxis.set_minor_locator(mpl.dates.DayLocator(bymonthday=1))
            ax.set_yticklabels(month_names)

            ax.set_xticks(np.arange(30, n_forecast_days, 30))
            ax.set_xticklabels(np.arange(30, n_forecast_days, 30))
            plt.xticks(rotation=0)

            ax.set_title('{} {}'.format(model, metric))

            ax.set_ylabel('Calendar month')
            ax.set_xlabel('Lead time (days)')
            plt.tight_layout()
            fname = '{}_heatmap_{}.png'.format(metric.lower(), model.lower())
            plt.savefig(os.path.join(fig_folder, fname))
            plt.close()

        for model in ['Day_persistence', 'Year_persistence']:

            heatmap_df_diff = heatmap_dfs[icenet2_name] - heatmap_dfs[model]
            max = np.nanmax(np.abs(heatmap_df_diff.values))

            fig, ax = plt.subplots(figsize=(6, 6))
            sns.heatmap(
                data=heatmap_df_diff,
                cmap=metric_dict[metric]['cmap_diff'],
                ax=ax,
                vmax=max,
                vmin=-max,
                cbar_kws=dict(label=metric_dict[metric]['label'] + ' difference'),
            )

            ax.yaxis.set_major_locator(mpl.dates.DayLocator(bymonthday=15))
            ax.tick_params(axis='y', which='major',length=0)
            ax.yaxis.set_major_formatter(mpl.dates.DateFormatter('%m'))
            ax.yaxis.set_minor_locator(mpl.dates.DayLocator(bymonthday=1))
            ax.set_xticks(np.arange(30, n_forecast_days, 30))
            ax.set_xticklabels(np.arange(30, n_forecast_days, 30))
            ax.set_title('IceNet2 {} improvement over {}'.format(metric, model))
            ax.set_ylabel('Calendar month')
            ax.set_xlabel('Lead time (days)')
            plt.tight_layout()
            fname = 'diff_{}_heatmap_{}.png'.format(metric.lower(), model.lower())
            plt.savefig(os.path.join(fig_folder, fname))
            plt.close()

    print('Done.')


### Metric vs. lead time
################################################################################

fig_folder = os.path.join(parent_fig_folder, 'metric_vs_leadtime')
if not os.path.exists(fig_folder):
    os.makedirs(fig_folder)

for metric in metric_list:
    print(metric)

    fontsize = 10
    context = {
        'font.size': fontsize,
        'axes.labelsize': fontsize,
        'axes.titlesize': fontsize,
        'ytick.labelsize': fontsize,
        'xtick.labelsize': fontsize
    }
    with plt.rc_context(context):

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(
            x='Leadtime',
            y=metric,
            ci=None,
            hue='Model',
            # legend=False,
            data=results_df,
            ax=ax
        )
        plt.legend(fontsize=fontsize, loc='lower left', bbox_to_anchor=(0, 1.1))
        # ax.legend(model_list, fontsize=fontsize, loc='lower left', bbox_to_anchor=(0, 1.1))
        ax.set_ylabel(metric_dict[metric]['label'])
        ax.set_xticks(np.arange(30, n_forecast_days, 30))
        ax.set_xticklabels(np.arange(30, n_forecast_days, 30))
        plt.tight_layout()
        fname = '{}_vs_leadtime.png'.format(metric.lower())
        plt.savefig(os.path.join(fig_folder, fname))
        plt.close()

        valid_seas5_forecasts = results_df.loc['SEAS5_noBC'].dropna().index.values
        results_df_valid_seas5 = results_df.reset_index('Model').loc[valid_seas5_forecasts].\
            set_index('Model', 'Leadtime', 'Forecast date')

        temp_model_list = [icenet2_name, 'SEAS5_noBC', 'SEAS5', 'Day_persistence', 'Year_persistence']
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(
            x='Leadtime',
            y=metric,
            ci=None,
            # ci=95,
            hue='Model',
            # legend=False,
            data=results_df_valid_seas5.loc[:, :, temp_model_list],
            ax=ax
        )
        plt.legend(fontsize=fontsize, loc='lower left', bbox_to_anchor=(0, 1.1))
        ax.set_ylabel(metric_dict[metric]['label'])
        ax.set_xticks(np.arange(30, n_forecast_days, 30))
        ax.set_xticklabels(np.arange(30, n_forecast_days, 30))
        plt.tight_layout()
        fname = '{}_vs_leadtime_valid_seas5.png'.format(metric.lower())
        plt.savefig(os.path.join(fig_folder, fname))
        plt.close()

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(
            x='Leadtime',
            y=metric,
            hue='Model',
            # legend=False,
            data=results_df.loc[:, :, datetime(2012,9,15)],
            ax=ax
        )
        ax.set_ylabel(metric_dict[metric]['label'])
        plt.legend(fontsize=fontsize, loc='lower left', bbox_to_anchor=(0, 1.1))
        ax.set_xticks(np.arange(30, n_forecast_days, 30))
        ax.set_xticklabels(np.arange(30, n_forecast_days, 30))
        plt.tight_layout()
        fname = '{}_vs_leadtime_2012_09_15.png'.format(metric.lower())
        plt.savefig(os.path.join(fig_folder, fname))
        plt.close()

dates = pd.date_range(
    start='2012-09-15',
    periods=6,
    freq=pd.DateOffset(years=1)
)

fig, ax = plt.subplots()
sns.lineplot(
    data=results_df.loc[[icenet2_name, 'Year_persistence'], np.arange(1, 94), dates],
    y='MAE',
    x='Leadtime',
    hue='Forecast date',
    style='Model',
)
ax.set_xlabel('Leadtime (days)')
plt.legend(fontsize=5, loc='lower left', bbox_to_anchor=(0, 1))
plt.subplots_adjust(top=0.7)
fname = 'septembers.png'
plt.savefig(os.path.join(fig_folder, fname))
plt.close()

g = sns.relplot(
    data=results_df.loc[[icenet2_name, 'Year_persistence'], np.arange(1, 94), dates],
    y='MAE',
    kind='line',
    x='Leadtime',
    col='Forecast date',
    height=4,
    aspect=1,
    legend=False,
    color='k',
    col_wrap=3,
    style='Model',
)
# leg = g._legend
# leg.set_bbox_to_anchor([0.0, 0.9])  # coordinates of lower left of bounding box
# leg._loc = 2  # if required you can set the loc
# g.fig.tight_layout()

fname = 'septembers_multipanel.png'
plt.savefig(os.path.join(fig_folder, fname))
plt.close()

### Metric vs. forecast date
################################################################################

fig_folder = os.path.join(parent_fig_folder, 'metric_vs_forecast_date')
if not os.path.exists(fig_folder):
    os.makedirs(fig_folder)

fig, ax = plt.subplots()
sns.lineplot(
    x='Forecast date',
    y='MAE',
    data=results_df.loc[icenet2_name, [1, 30, 60, 90], :],
    hue='Leadtime',
    ax=ax
)
ax.set_ylabel('MAE (%)')
ax.set_xlabel('Forecast date')
plt.tight_layout()
fname = 'mae_vs_forecast_date.png'
plt.savefig(os.path.join(fig_folder, fname))
plt.close()

# Predictability maps
####################################################################

fig_folder = os.path.join(parent_fig_folder, 'predictability_maps')
if not os.path.exists(fig_folder):
    os.makedirs(fig_folder)

plot_dict = {
    0: {
        'target_date': '12-15',
        'thresh': 0.05,
    },
    1: {
        'target_date': '12-15',
        'thresh': 0.1,
    },
    2: {
        'target_date': '09-15',
        'thresh': 0.1,
    },
    3: {
        'target_date': '10-30',
        'thresh': 0.1,
    },
}

# Load true SIC data
true_sic_fpath = os.path.join('data', 'nh', 'siconca', 'siconca_all_interp.nc')
true_sic_da = xr.open_dataarray(true_sic_fpath)
# Replace 12:00 hour with 00:00 hour by convention
dates = [pd.Timestamp(date).to_pydatetime().replace(hour=0) for date in true_sic_da.time.values]
true_sic_da = true_sic_da.assign_coords(dict(time=dates))
true_sic_da = true_sic_da.sel(time=dates)
true_sic_da.load()

for fig_i, fig_dict in plot_dict.items():
    print(fig_i)

    thresh = fig_dict['thresh']
    target_date_str = fig_dict['target_date']

    target_date = pd.Timestamp('2012-{}'.format(target_date_str))
    dates = pd.date_range(
        start=target_date,
        periods=6,
        freq=pd.DateOffset(years=1)
    )
    forecast_da_target = forecast_da.sel(time=dates)

    abs_err_da = xr.ufuncs.fabs(forecast_da_target - true_sic_da)

    abs_err_thresh_da = abs_err_da < thresh
    abs_err_thresh_da = abs_err_thresh_da.compute()

    # Find the lead time at which the error first grows above thresh
    abs_err_diff_thresh_da = abs_err_thresh_da.astype(int).diff(dim='leadtime')

    # (time, x, y, leadtime) indexes of -1 transitions
    idxs = np.argwhere(abs_err_diff_thresh_da.data == -1)

    # Sort by descending lead time so that the longest lead times are the last to be
    #    written to the leadtime array
    idxs = idxs[np.argsort(idxs[:, 3])[::-1]]
    shortest_lt_within_error = np.full((6, 432, 432), np.nan)
    for i in range(idxs.shape[0]):
        idx_leadtime = idxs[i, :]

        idx = list(idx_leadtime[:3])
        leadtime = idx_leadtime[3]

        shortest_lt_within_error[idx[0], idx[1], idx[2]] = leadtime

    # OTHER CASES:
    # - Find points where abserr < thresh at all lead times
    # (no match because True at all lead times - great!)
    err_always_below_thresh = np.sum(abs_err_thresh_da.data, axis=-1) == abs_err_thresh_da.shape[-1]
    shortest_lt_within_error[err_always_below_thresh] = abs_err_thresh_da.shape[-1]

    # - Find points where err > thresh at all lead times
    # (no match because False at all lead times - sad!)
    err_always_above_thresh = np.sum(abs_err_thresh_da.data, axis=-1) == 0
    shortest_lt_within_error[err_always_above_thresh] = 0.

    # - Find points where only transition is err > thresh
    # (no match because within thresh then exceed thresh up til target date - sad!)
    err_below_then_above_thresh = (
        (np.sum(abs_err_diff_thresh_da.data, axis=-1) == 1) & \
        (np.sum(np.abs(abs_err_diff_thresh_da.data), axis=-1) == 1)
    )
    shortest_lt_within_error[err_below_then_above_thresh] = 0.

    # - Find points where err > thresh after (in init time) err < thresh
    idxs = np.argwhere(abs_err_diff_thresh_da.data == +1)
    idxs = idxs[np.argsort(idxs[:, 3])[::-1]]
    shortest_lt_exceed_error = np.full((6, 432, 432), np.nan)
    for i in range(idxs.shape[0]):
        idx_leadtime = idxs[i, :]

        idx = list(idx_leadtime[:3])
        leadtime = idx_leadtime[3]

        shortest_lt_exceed_error[idx[0], idx[1], idx[2]] = leadtime

    # Only shows True if neither grid cells are np.nan (i.e. there is both a +1 and -1)
    exceed_error_after_final_within = shortest_lt_exceed_error < shortest_lt_within_error
    shortest_lt_within_error[exceed_error_after_final_within] = 0.

    min_leadtimes = np.nanmean(shortest_lt_within_error, axis=0)
    min_leadtimes /= 7 # Convert to weeks

    # fig, ax = plt.subplots()
    land_mask_fpath = os.path.join('data', 'nh', 'masks', 'land_mask.npy')
    land_mask = np.load(land_mask_fpath)

    # im = ax.imshow(min_leadtimes, cmap=cmap, norm=norm)
    # ax.contourf(land_mask, levels=[0.5, 1], colors=[mpl.cm.gray(123)])
    #
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes('right', size='5%', pad=0.05)
    # plt.colorbar(im, cax)
    # ax.patch.set_visible(False)
    # ax.spines['top'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    # ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

    date_str = month_names[target_date.month-1] + ' {}th'.format(target_date.day)
    # ax.set_title('average minimum leadtime for absolute error below {:.1f}%\nfor {} forecasts'.format(100*thresh, date_str))
    # plt.tight_layout()

    #
    # land_da = true_sic_da.isel(time=0).copy()
    # land_da.data = land_mask
    # land_da_slice = land_da.where(
    #     (land_da.lat > latmin) &
    #     (land_da.lat < latmax) &
    #     (land_da.lon > lonmin) &
    #     (land_da.lon < lonmax),
    #     drop=True
    # ).astype(int)

    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    latmax = 90
    latmin = 0
    lonmin = -180
    lonmax = 180
    # Hudson Bay:
    # latmax = 75
    # latmin = 50
    # lonmin = -100
    # lonmax = -60

    # We just need to copy a DataArray with lat and lon coordinates for cartopy to use
    min_leadtimes_da = true_sic_da.isel(time=0).copy()
    min_leadtimes_da.data = min_leadtimes
    min_leadtimes_da_slice = min_leadtimes_da.where(
        (min_leadtimes_da.lat > latmin) &
        (min_leadtimes_da.lat < latmax) &
        (min_leadtimes_da.lon > lonmin) &
        (min_leadtimes_da.lon < lonmax),
        drop=True
    )

    lon = min_leadtimes_da_slice.lon.values
    lat = min_leadtimes_da_slice.lat.values.T
    proj = ccrs.PlateCarree()
    # newproj = ccrs.NearsidePerspective(-80, 60, 1e6)
    # newproj = ccrs.NearsidePerspective(-80, 60, 30e6)
    newproj = ccrs.NearsidePerspective(-80, 75, 5e6)
    # newproj = ccrs.NearsidePerspective(-0, 85, 7e6)
    # newproj = ccrs.Orthographic(-80, 60)
    # newproj = ccrs.Orthographic(0, 90)
    # newproj = ccrs.NorthPolarStereo(0, 90)
    # newproj = ccrs.LambertAzimuthalEqualArea(0, 90)

    # bounds = np.arange(0, 93/7, 2)
    bounds = np.arange(0, 93/7, 1)
    cmap = mpl.cm.get_cmap('cubehelix')
    # cmap.from_list('foo', cmap(bounds), len(bounds))
    n_bounds = bounds.size
    rgba_cmap_list = cmap(np.linspace(0, 1, n_bounds))
    rgba_cmap_list[-4:-1] = rgba_cmap_list[-1]  # Make long predictability white
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'foo', rgba_cmap_list, len(bounds))
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    ax = plt.axes(projection=newproj)
    im = ax.pcolormesh(lon, lat, min_leadtimes_da_slice, transform=proj, cmap=cmap, norm=norm)
    # ax.contourf(lon, lat, land_da_slice, transform=proj, levels=[0.5, 1], colors=[mpl.cm.gray(123)])
    ax.add_feature(cfeature.LAND, zorder=1)
    ax.coastlines(resolution='50m', linewidth=0.3)
    # ax.add_feature(cfeature.COASTLINE)
    plt.colorbar(im, label='# of weeks')
    ax.set_title('average minimum leadtime for absolute error below {:.1f}%\nfor {} forecasts'.format(100*thresh, date_str))
    fname = 'predictability_map_date_{}_thresh_{:.1f}.png'.format(target_date_str, thresh)
    plt.savefig(os.path.join(fig_folder, fname))
    plt.close()

print('Done.')
