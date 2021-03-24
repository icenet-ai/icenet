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

####################################################################

network_name = 'unet_batchnorm'
dataloader_name = '2021_03_03_1928_icenet2_init'

# Format for storing different IceNet2 results in one dataframe
icenet2_name = 'IceNet2__{}__{}'.format(dataloader_name, network_name)

results_df_fpath = os.path.join(
    config.folders['results'], 'results.csv'
)

results_df = pd.read_csv(results_df_fpath)

fig_folder = os.path.join(
    config.folders['results'], dataloader_name, network_name, 'validation'
)

####################################################################

model_list = results_df.Model.unique()
n_forecast_days = max(results_df.Leadtime)
results_df['Forecast date'] = [pd.Timestamp(date) for date in results_df['Forecast date']]
results_df = results_df.set_index(['Model', 'Leadtime', 'Forecast date'])
metric_list = results_df.columns
results_df['dayofyear'] = results_df.index.get_level_values(2).dayofyear

for metric in metric_list:
    if metric == 'MAE':
        cmap = 'seismic_r'
    else:
        cmap = 'seismic'

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
            cbar_kws=dict(label=metric)
        )
        ax.yaxis.set_major_locator(matplotlib.dates.DayLocator(bymonthday=15))
        ax.tick_params(axis='y', which='major',length=0)
        ax.yaxis.set_major_formatter(matplotlib.dates.DateFormatter('%m'))
        ax.yaxis.set_minor_locator(matplotlib.dates.DayLocator(bymonthday=1))
        ax.set_xticks(np.arange(30, n_forecast_days, 30))
        ax.set_xticklabels(np.arange(30, n_forecast_days, 30))
        ax.set_title('{} {}'.format(model, metric))
        ax.set_ylabel('Calendar month')
        ax.set_xlabel('Lead time (days)')
        plt.tight_layout()
        plt.savefig(os.path.join(fig_folder, '{}_heatmap_{}.png'.format(metric.lower(), model.lower())))
        plt.close()

    for model in ['Day_persistence', 'Year_persistence']:

        heatmap_df_diff = heatmap_dfs[icenet2_name] - heatmap_dfs[model]
        max = np.max(np.abs(heatmap_df_diff.values))

        fig, ax = plt.subplots(figsize=(6, 6))
        sns.heatmap(
            data=heatmap_df_diff,
            cmap='seismic',
            ax=ax,
            vmax=max,
            vmin=-max,
            cbar_kws=dict(label='{}'.format(metric))
        )

        ax.yaxis.set_major_locator(matplotlib.dates.DayLocator(bymonthday=15))
        ax.tick_params(axis='y', which='major',length=0)
        ax.yaxis.set_major_formatter(matplotlib.dates.DateFormatter('%m'))
        ax.yaxis.set_minor_locator(matplotlib.dates.DayLocator(bymonthday=1))
        ax.set_xticks(np.arange(30, n_forecast_days, 30))
        ax.set_xticklabels(np.arange(30, n_forecast_days, 30))
        ax.set_title('IceNet2 {} improvement over {}'.format(metric, model))
        ax.set_ylabel('Calendar month')
        ax.set_xlabel('Lead time (days)')
        plt.tight_layout()
        plt.savefig(os.path.join(fig_folder, 'diff_{}_heatmap_{}.png'.format(metric.lower(), model.lower())))
        plt.close()

for metric in metric_list:
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.lineplot(
        x='Leadtime',
        y=metric,
        ci=None,
        hue='Model',
        data=results_df,
        ax=ax
    )
    # ax.set_ylabel('MAE (%)')
    ax.set_ylabel(metric)
    ax.set_xticks(np.arange(30, n_forecast_days, 30))
    ax.set_xticklabels(np.arange(30, n_forecast_days, 30))
    plt.tight_layout()
    plt.savefig(os.path.join(fig_folder, '{}_vs_leadtime.png'.format(metric.lower())))
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
plt.savefig(os.path.join(fig_folder, 'mae_vs_leadtime_2012_09_15.png'))
plt.close()

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
plt.savefig(os.path.join(fig_folder, 'mae_vs_forecast_date.png'))
plt.close()
