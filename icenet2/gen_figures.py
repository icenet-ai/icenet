import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), 'icenet2'))
from misc import StretchOutNormalize
import re
import numpy as np
import pandas as pd
import xarray as xr
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
mpl.rcParams.update({
    'figure.facecolor': 'w',
    'figure.dpi': 300
})

### Results dataset
###############################################################################

# results_df_fpath = 'results_monthly/2021_04_25_130015_results.csv'
results_df_fpath = 'results_monthly/2021_04_26_173002_results.csv'

results_df = pd.read_csv(results_df_fpath)
results_df['Forecast date'] = pd.to_datetime(results_df['Forecast date'])
month_names = np.array(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec'])
forecast_month_names = month_names[results_df['Forecast date'].dt.month.values - 1]
results_df['Forecast month'] = forecast_month_names
results_df = results_df.set_index(['Leadtime', 'Forecast date', 'Model'])

### Which figure to generate
###############################################################################

# OPTIONS:
#   - 'egu_heatmaps'

plotnames = [
    # 'egu_heatmaps',
    'egu_maps',
]

### Plotting logic
###############################################################################

for plotname in plotnames:

    print('Plotting {}... '.format(plotname))

    fig_folder = os.path.join('figures', plotname)
    if not os.path.exists(fig_folder):
        os.makedirs(fig_folder)

    if plotname == 'egu_heatmaps':
        models_to_plot = [
            'IceNet__icenet_nature_egu_baseline__unet_tempscale__ensemble',
            # 'IceNet__icenet_nature_thickness__unet_tempscale__ensemble',
            'IceNet__icenet_nature_thickness2__unet_tempscale__ensemble',
            # 'IceNet2__2021_04_08_1205_icenet2_nh_sh_thinned5_weeklyinput_wind_3month__unet_batchnorm__ensemble',
            'IceNet2__2021_04_25_1351_icenet2_nh_thinned7_weeklyinput_wind_3month__unet_batchnorm__ensemble',
            'SEAS5',
            'IceNet__icenet2_nature__unet_tempscale__ensemble'
        ]

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.lineplot(
            data=results_df.loc[:, :, models_to_plot[:3]],
            x='Leadtime',
            y='Binary_accuracy',
            ci=None,
            hue='Model'
        )
        plt.savefig(os.path.join(fig_folder, 'leadtime.png'))
        plt.close()

        # Binary accuracy heatmaps
        dfs = []
        for model in models_to_plot:
            df = results_df.loc[pd.IndexSlice[:, :, model], :].\
                groupby(['Forecast month', 'Leadtime']).mean().reset_index().\
                pivot('Forecast month', 'Leadtime', 'Binary_accuracy').reindex(month_names).T
            dfs.append(df)

        ###############################

        # Second element minus first
        heatmap_combs = [
            (dfs[1], dfs[0]),
            (dfs[2], dfs[1].loc[1:3, :]),
            (dfs[2], dfs[3].loc[1:3, :]),
            (dfs[2], dfs[4].loc[1:3, :]),
        ]

        fnames = [
            'thickness_improvement_over_baseline.png',
            'daily_improvement_over_baseline.png',
            'daily_improvement_over_seas5.png',
            'daily_improvement_over_icenet1.png',
        ]

        titles = [
            'Improvement from including sea ice thickness',
            'Improvement from daily timescale',
            'IceNet2 improvement over SEAS5',
            'IceNet2 improvement over IceNet1',
        ]

        for heatmap_comb, fname, title in zip(heatmap_combs, fnames, titles):

            df_2, df_1 = heatmap_comb

            min = np.min((df_2 - df_1).values)
            max = np.max((df_2 - df_1).values)
            max = np.max(np.abs((min, max)))
            max = 2

            norm = StretchOutNormalize(vmin=-max, vmax=max, low=-.05, up=.05)

            with plt.rc_context({'font.size': 14}):

                fig, ax = plt.subplots()
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right', size='5%', pad=0.05)
                cbar_kws = dict(extend='both', fraction=0.05, aspect=8)
                cmap = 'PRGn'
                fmt = '+.1f'
                cbar_kws['label'] = 'Difference (%)'
                cbar_kws['format'] = FuncFormatter(lambda x, p: format(x, '+.0f'))
                cbar_kws['ticks'] = np.arange(-2, 2+1, 1)
                sns.heatmap(
                    data=df_2 - df_1,
                    # annot=True,
                    annot=False,
                    linewidths=0.5,
                    linecolor='k',
                    fmt=fmt,
                    vmin=-max,
                    vmax=max,
                    square=True,
                    cbar_kws=cbar_kws,
                    cmap=cmap,
                    norm=norm,
                    cbar_ax=cax,
                    ax=ax)
                ax.set_xlabel('')
                ax.tick_params(axis='y', rotation=0)
                ax.tick_params(axis='x', rotation=50)
                ax.set_ylabel('Lead time (months)')
                ax.set_title(title)
                plt.tight_layout()
                plt.savefig(os.path.join(fig_folder, fname), transparent=True)
                plt.close()

    if plotname == 'egu_maps':

        def arr_to_ice_edge_rgba_arr(arr, thresh, rgb):

            X, Y = np.meshgrid(np.arange(arr.shape[0]), np.arange(arr.shape[1]))
            X = X.T
            Y = Y.T

            cs = plt.contour(X, Y, arr, [thresh], alpha=0)  # Do not plot on any axes
            x = []
            y = []
            for p in cs.collections[0].get_paths():
                x_i, y_i = p.vertices.T
                x.extend(np.round(x_i))
                y.extend(np.round(y_i))
            x = np.array(x, int)
            y = np.array(y, int)
            ice_edge_arr = np.zeros(arr.shape, dtype=bool)
            ice_edge_arr[x, y] = True
            # Mask out ice edge contour that hugs the coastline
            ice_edge_arr[land_mask] = False
            ice_edge_arr[region_mask == 13] = False

            # Contour pixels -> alpha=1, alpha=0 elsewhere
            ice_edge_rgba_arr = np.zeros((*arr.shape, 4))
            ice_edge_rgba_arr[:, :, 3] = ice_edge_arr
            ice_edge_rgba_arr[:, :, :3] = rgb

            return ice_edge_rgba_arr

        icenet2_monthly_forecast_fpath = os.path.join(
            'data', 'forecasts_monthly', 'icenet2',
            '2021_04_25_1351_icenet2_nh_thinned7_weeklyinput_wind_3month',
            'unet_batchnorm', 'ensemble', 'monthly_forecasts.nc'
        )

        icenet_baseline_forecast_fpath = \
            '/data/hpcdata/users/tomand/code/icenet/data/forecasts/icenet/icenet_nature_egu_baseline/unet_tempscale/ensemble/monthly_forecasts.nc'

        icenet_thickness_forecast_fpath = \
            '/data/hpcdata/users/tomand/code/icenet/data/forecasts/icenet/icenet_nature_thickness2/unet_tempscale/ensemble/monthly_forecasts.nc'

        ground_truth_fpath = os.path.join(
            'data', 'nh', 'siconca', 'siconca_all_interp_monthly.nc'
        )

        land_mask_fpath = os.path.join(
            'data', 'nh', 'masks', 'land_mask.npy'
        )

        region_mask_fpath = \
            '/data/hpcdata/users/tomand/code/icenet/data/masks/region_mask.npy'

        ground_truth_da = xr.open_dataarray(ground_truth_fpath)
        icenet_baseline_da = xr.open_dataarray(icenet_baseline_forecast_fpath).mean('seed')
        icenet_thickness_da = xr.open_dataarray(icenet_thickness_forecast_fpath).mean('seed')
        icenet2_da = xr.open_dataarray(icenet2_monthly_forecast_fpath)

        land_mask = np.load(land_mask_fpath)
        region_mask = np.load(region_mask_fpath)

        target_date = pd.Timestamp('2020-09-01')
        leadtime = 1

        true_sic = ground_truth_da.sel(time=target_date).data
        true = true_sic >= .15

        #######################################################################

        pred_ice_edge_rgb = sns.color_palette('tab10')[2]
        true_ice_edge_rgb = [0, 0, 0]

        top = 130
        bot = 300
        left = 100
        right = 300

        fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(3, 9))
        axes = axes.ravel()
        for i, ax in enumerate(axes):
            if i == 0:
                forecast_raw = icenet_baseline_da.sel(time=target_date, leadtime=leadtime).data
                forecast = forecast_raw >= 0.5
                model = 'IceNet__icenet_nature_egu_baseline__unet_tempscale__ensemble'
                acc = results_df.loc[leadtime, target_date, model].Binary_accuracy.values[0]
                ax.set_title('Baseline.')# Binary accuracy = {:.1f}%.'.format(acc))
            elif i == 1:
                forecast_raw = icenet_thickness_da.sel(time=target_date, leadtime=leadtime).data
                forecast = forecast_raw >= 0.5
                model = 'IceNet__icenet_nature_thickness2__unet_tempscale__ensemble'
                ax.set_title('Baseline+thickness.')# Binary accuracy = {:.1f}%.'.format(acc))
                acc = results_df.loc[leadtime, target_date, model].Binary_accuracy.values[0]
            elif i == 2:
                forecast_raw = icenet2_da.sel(time=target_date, leadtime=leadtime).data
                forecast = forecast_raw >= 0.15
                model = 'IceNet2__2021_04_25_1351_icenet2_nh_thinned7_weeklyinput_wind_3month__unet_batchnorm__ensemble',
                ax.set_title('Daily prediction.')# Binary accuracy = {:.1f}%.'.format(acc))
                acc = results_df.loc[leadtime, target_date, model].Binary_accuracy.values[0]

            pred_ice_edge_rgba_arr = arr_to_ice_edge_rgba_arr(forecast, 1, pred_ice_edge_rgb)
            ice_edge_rgba_arr = arr_to_ice_edge_rgba_arr(true, 1, true_ice_edge_rgb)

            if i != 3:
                ax.imshow(forecast_raw[top:bot, left:right], 'Blues_r', clim=[0, 1])
                ax.contourf(land_mask[top:bot, left:right], levels=[0.5, 1], colors=[mpl.cm.gray(123)])

                err = forecast != true

                arr_err_rgba = np.zeros((*err.shape, 4))  # RGBA array
                color_rgba = np.array(mpl.cm.Oranges(180)).reshape(1, 1, 4)
                idx_arrs = np.where(np.abs(err) == 1)
                arr_err_rgba[idx_arrs[0], idx_arrs[1], :] = color_rgba
                idx_arrs = np.where(np.abs(err) == 0)
                arr_err_rgba[idx_arrs[0], idx_arrs[1], 3] = 0  # Alpha = 0 where no error is made

                ax.imshow(arr_err_rgba[top:bot, left:right])
                ax.imshow(ice_edge_rgba_arr[top:bot, left:right, :])
                ax.imshow(pred_ice_edge_rgba_arr[top:bot, left:right, :])

                t = ax.text(s='Binary acc: {:.1f}%'.format(acc),
                            x=.0, y=.0, fontsize=12, transform=ax.transAxes,
                            horizontalalignment='left')
                t.set_bbox(dict(facecolor='white', alpha=.9, edgecolor='k', pad=0.1))

            elif i == 3:
                proxy = [plt.Line2D([0], [1], color=true_ice_edge_rgb),
                         plt.Line2D([0], [1], color=pred_ice_edge_rgb),
                         plt.Rectangle((0,0),1,1,fc=mpl.cm.Oranges(180))]

                ax.legend(proxy, ['Observed ice edge', 'Predicted ice edge', 'Ice edge error'],
                          loc='upper center', fontsize=12)

        for ax in axes:
            ax.patch.set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
        plt.tight_layout()
        fname = 'all3.png'
        plt.savefig(os.path.join(fig_folder, fname), transparent=True)
        plt.close()

        ##########################
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6, 4))

        ax = axes[0]

        forecast_raw = icenet2_da.sel(time=target_date, leadtime=leadtime).data
        forecast = forecast_raw >= 0.15
        model = 'IceNet2__2021_04_25_1351_icenet2_nh_thinned7_weeklyinput_wind_3month__unet_batchnorm__ensemble',
        ax.set_title('{} 1-month forecast (IceNet2).'.format(target_date.strftime('%Y/%m')), fontsize=10)
        acc = results_df.loc[leadtime, target_date, model].Binary_accuracy.values[0]

        pred_ice_edge_rgba_arr = arr_to_ice_edge_rgba_arr(forecast, 1, pred_ice_edge_rgb)
        ice_edge_rgba_arr = arr_to_ice_edge_rgba_arr(true, 1, true_ice_edge_rgb)

        ax.imshow(forecast_raw[top:bot, left:right], 'Blues_r', clim=[0, 1])
        ax.contourf(land_mask[top:bot, left:right], levels=[0.5, 1], colors=[mpl.cm.gray(123)])

        err = forecast != true

        arr_err_rgba = np.zeros((*err.shape, 4))  # RGBA array
        color_rgba = np.array(mpl.cm.Oranges(180)).reshape(1, 1, 4)
        idx_arrs = np.where(np.abs(err) == 1)
        arr_err_rgba[idx_arrs[0], idx_arrs[1], :] = color_rgba
        idx_arrs = np.where(np.abs(err) == 0)
        arr_err_rgba[idx_arrs[0], idx_arrs[1], 3] = 0  # Alpha = 0 where no error is made

        ax.imshow(arr_err_rgba[top:bot, left:right])
        ax.imshow(ice_edge_rgba_arr[top:bot, left:right, :])
        ax.imshow(pred_ice_edge_rgba_arr[top:bot, left:right, :])

        t = ax.text(s='Binary accuracy: {:.1f}%'.format(acc),
                    x=.98, y=.02, fontsize=10, transform=ax.transAxes,
                    horizontalalignment='right')
        t.set_bbox(dict(facecolor='white', alpha=.9, edgecolor='k', pad=3))

        proxy = [plt.Line2D([0], [1], color=true_ice_edge_rgb),
                 plt.Line2D([0], [1], color=pred_ice_edge_rgb),
                 plt.Rectangle((0,0),1,1,fc=mpl.cm.Oranges(180))]

        axes[1].legend(proxy, ['Observed ice\nedge', 'Predicted ice\nedge', 'Ice edge error'],
                  loc='center left', fontsize=8)

        for ax in axes:
            ax.patch.set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

        plt.tight_layout()
        fname = 'icenet2only.png'
        fig.subplots_adjust(wspace=0.)
        plt.savefig(os.path.join(fig_folder, fname), transparent=True, bbox_inches='tight', pad_inches=0)
        plt.close()

    print('Plotting completed.')

# regex = re.compile('IceNet__icenet_nature_egu_baseline__unet_tempscale__(.*)')
# models = sorted([model for model in set(results_df.reset_index().Model) if regex.match(model)])
# print(models)
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(figsize=(5, 6))
# sns.lineplot(
#     data=results_df.loc[pd.IndexSlice[:, :, models], :],
#     # data=results_df,
#     y='Binary_accuracy',
#     x='Leadtime',
#     ci=None,
#     ax=ax,
#     hue='Model',
#     palette="Blues",
# )
#
# regex = re.compile('IceNet__icenet_nature_thickness__unet_tempscale__(.*)')
# models = sorted([model for model in set(results_df.reset_index().Model) if regex.match(model)])
# print(models)
# sns.lineplot(
#     data=results_df.loc[pd.IndexSlice[:, :, models], :],
#     # data=results_df,
#     y='Binary_accuracy',
#     x='Leadtime',
#     ci=None,
#     ax=ax,
#     hue='Model',
#     palette="Reds",
# )
# plt.legend(fontsize=5, loc='lower left', bbox_to_anchor=(0, 1.1))
# plt.tight_layout()


# TEMP TODO move to plotting script
# models_to_plot = [
#     'IceNet2__2021_04_08_1205_icenet2_nh_sh_thinned5_weeklyinput_wind_3month__unet_batchnorm__ensemble',
#     'IceNet',
#     'SEAS5',
#     'Linear trend'
# ]

# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(figsize=(5, 3.5))
# fig = sns.lineplot(
#     data=results_df.loc[:, :, models_to_plot],
#     # data=results_df,
#     y='Binary_accuracy',
#     x='Leadtime',
#     ci=None,
#     ax=ax,
#     hue='Model',
# ).get_figure()
# plt.legend(fontsize=5, loc='lower left', bbox_to_anchor=(0, 1.1))
# plt.tight_layout()
# # plt.subplots_adjust(bottom=-0.0)
# ax.set_xlabel('Lead time (months)')
# fig.savefig('temp.png', dpi=300)
#
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(figsize=(5, 3.5))
# fig = sns.lineplot(
#     data=results_df,
#     y='RMSE',
#     x='Leadtime',
#     ci=None,
#     ax=ax,
#     hue='Model',
# ).get_figure()
# plt.legend(fontsize=5, loc='lower left', bbox_to_anchor=(0, 1.1))
# plt.tight_layout()
# # plt.subplots_adjust(bottom=-0.0)
# ax.set_xlabel('Lead time (months)')
# fig.savefig('temp_rmse.png', dpi=300)


# sns.relplot(
#     data=results_df,
#     col_wrap=1,
#     kind='scatter',
#     y='Binary_accuracy',
#     x='Forecast date',
#     height=3,
#     aspect=2,
#     col='Leadtime',
# )
