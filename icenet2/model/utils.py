import os
import sys
import numpy as np
import re
import xarray as xr
import pandas as pd
from dateutil.relativedelta import relativedelta
import iris
import cartopy.crs as ccrs
import matplotlib.pyplot as plt


################## MISC FUNCTIONS
################################################################################


def make_varname_verbose(varname, leadtime, fc_month_idx):

    '''
    Takes IceNet short variable name (e.g. siconca_abs_3) and converts it to a
    long name for a given forecast calendar month and lead time (e.g.
    'Feb SIC').

    Inputs:
    varname: Short variable name.
    leadtime: Lead time of the forecast.
    fc_month_index: Mod-12 calendar month index for the month being forecast
        (e.g. 8 for September)

    Returns:
    verbose_varname: Long variable name.
    '''

    month_names = np.array(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec'])

    varname_regex = re.compile('^(.*)_(abs|anom|linear_trend)_([0-9]+)$')

    var_lookup_table = {
        'siconca': 'SIC',
        'tas': '2m air temperature',
        'ta500': '500 hPa air temperature',
        'tos': 'sea surface temperature',
        'rsds': 'downwelling solar radiation',
        'rsus': 'upwelling solar radiation',
        'psl': 'sea level pressure',
        'zg500': '500 hPa geopotential height',
        'zg250': '250 hPa geopotential height',
        'ua10': '10 hPa zonal wind speed',
        'uas': 'x-direction wind',
        'vas': 'y-direction wind'
    }

    initialisation_month_idx = (fc_month_idx - leadtime) % 12

    varname_match = varname_regex.match(varname)

    field = varname_match[1]
    data_format = varname_match[2]
    lead_or_lag = int(varname_match[3])

    verbose_varname = ''

    month_suffix = ' '
    month_prefix = ''
    if data_format != 'linear_trend':
        # Read back from initialisation month to get input lag month
        lag = lead_or_lag  # In no of months
        input_month_name = month_names[(initialisation_month_idx - lag + 1) % 12]

        if (initialisation_month_idx - lag + 1) // 12 == -1:
            # Previous calendar year
            month_prefix = 'Previous '

    elif data_format == 'linear_trend':
        # Read forward from initialisation month to get linear trend forecast month
        lead = lead_or_lag  # In no of months
        input_month_name = month_names[(initialisation_month_idx + lead) % 12]

        if (initialisation_month_idx + lead) // 12 == 1:
            # Next calendar year
            month_prefix = 'Next '

    # Month the input corresponds to
    verbose_varname += month_prefix + input_month_name + month_suffix

    # verbose variable name
    if data_format != 'linear_trend':
        verbose_varname += var_lookup_table[field]
        if data_format == 'anom':
            verbose_varname += ' anomaly'
    elif data_format == 'linear_trend':
        verbose_varname += 'linear trend SIC forecast'

    return verbose_varname


def make_varname_verbose_any_leadtime(varname):

    ''' As above, but agnostic to what the target month or lead time is. E.g.
    "SIC (1)" for sea ice concentration at a lag of 1 month. '''

    varname_regex = re.compile('^(.*)_(abs|anom|linear_trend)_([0-9]+)$')

    var_lookup_table = {
        'siconca': 'SIC',
        'tas': '2m air temperature',
        'ta500': '500 hPa air temperature',
        'tos': 'sea surface temperature',
        'rsds': 'downwelling solar radiation',
        'rsus': 'upwelling solar radiation',
        'psl': 'sea level pressure',
        'zg500': '500 hPa geopotential height',
        'zg250': '250 hPa geopotential height',
        'ua10': '10 hPa zonal wind speed',
        'uas': 'x-direction wind',
        'vas': 'y-direction wind',
        'land': 'land mask',
        'cos(month)': 'cos(init month)',
        'sin(month)': 'sin(init month)',
    }

    exception_vars = ['cos(month)', 'sin(month)', 'land']

    if varname in exception_vars:
        return var_lookup_table[varname]
    else:
        varname_match = varname_regex.match(varname)

        field = varname_match[1]
        data_format = varname_match[2]
        lead_or_lag = int(varname_match[3])

        # verbose variable name
        if data_format != 'linear_trend':
            verbose_varname = var_lookup_table[field]
            if data_format == 'anom':
                verbose_varname += ' anomaly'
        elif data_format == 'linear_trend':
            verbose_varname = 'Linear trend SIC forecast'

        verbose_varname += ' ({:.0f})'.format(lead_or_lag)

        return verbose_varname


################################################################################
################## FUNCTIONS
################################################################################

def fix_near_real_time_era5_func(latlon_path):

    '''
    Near-real-time ERA5 data is classed as a different dataset called 'ERA5T'.
    This results in a spurious 'expver' dimension. This method detects
    whether that dim is present and removes it, concatenating into one array
    '''

    ds = xr.open_dataarray(latlon_path)

    if len(ds.data.shape) == 4:
        print('Fixing spurious ERA5 "expver dimension for {}.'.format(latlon_path))

        arr = xr.open_dataarray(latlon_path).data
        arr = ds.data
        # Expver 1 (ERA5)
        era5_months = ~np.isnan(arr[:, 0, :, :]).all(axis=(1, 2))

        # Expver 2 (ERA5T - near real time)
        era5t_months = ~np.isnan(arr[:, 1, :, :]).all(axis=(1, 2))

        ds = xr.concat((ds[era5_months, 0, :], ds[era5t_months, 1, :]), dim='time')

        ds = ds.reset_coords('expver', drop=True)

        os.remove(latlon_path)
        ds.load().to_netcdf(latlon_path)


###############################################################################
############### LEARNING RATE SCHEDULER
###############################################################################


def make_exp_decay_lr_schedule(rate, start_epoch=1, end_epoch=np.inf, verbose=False):

    ''' Returns an exponential learning rate function that multiplies by
    exp(-rate) each epoch after `start_epoch`. '''

    def lr_scheduler_exp_decay(epoch, lr):
        ''' Learning rate scheduler for fine tuning.
        Exponential decrease after start_epoch until end_epoch. '''

        if epoch >= start_epoch and epoch < end_epoch:
            lr = lr * np.math.exp(-rate)

        if verbose:
            print('\nSetting learning rate to: {}\n'.format(lr))

        return lr

    return lr_scheduler_exp_decay

###############################################################################
############### PLOTTING
###############################################################################


def compute_heatmap(results_df, model, seed='NA', metric='Binary accuracy'):
    '''
    Returns a binary accuracy heatmap of lead time vs. calendar month
    for a given model.
    '''

    month_names = np.array(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                            'Jul', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec'])

    # Mean over calendar month
    mean_df = results_df.loc[model, seed].reset_index().\
        groupby(['Calendar month', 'Leadtime']).mean()

    # Pivot
    heatmap_df = mean_df.reset_index().\
        pivot('Calendar month', 'Leadtime', metric).reindex(month_names)

    return heatmap_df


def arr_to_ice_edge_arr(arr, thresh, land_mask, region_mask):

    '''
    Compute a boolean mask with True over ice edge contour grid cells using
    matplotlib.pyplot.contour and an input threshold to define the ice edge
    (e.g. 0.15 for the 15% SIC ice edge or 0.5 for SIP forecasts). The contour
    along the coastline is removed using the region mask.
    '''

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

    return ice_edge_arr


def arr_to_ice_edge_rgba_arr(arr, thresh, land_mask, region_mask, rgb):

    ice_edge_arr = arr_to_ice_edge_arr(arr, thresh, land_mask, region_mask)

    # Contour pixels -> alpha=1, alpha=0 elsewhere
    ice_edge_rgba_arr = np.zeros((*arr.shape, 4))
    ice_edge_rgba_arr[:, :, 3] = ice_edge_arr
    ice_edge_rgba_arr[:, :, :3] = rgb

    return ice_edge_rgba_arr
