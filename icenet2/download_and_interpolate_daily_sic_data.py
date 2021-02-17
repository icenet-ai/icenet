import sys
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import imageio
import xarray as xr
import pandas as pd
import shutil
import re
import time
from datetime import datetime, timedelta
# import plotly.express as px
from scipy import interpolate
# sys.path.insert(0, os.path.join(os.getcwd(), 'icenet'))
sys.path.insert(0, os.path.join(os.getcwd(), 'icenet2'))  # if using jupyter kernel
# sys.path.insert(0, os.path.abspath(os.path.join(__file__, os.pardir)))
import config
import icenet2_utils

'''

Script to download daily OSI-SAF sea ice concentration (SIC) data from 1979-2020
and save in yearly NetCDF files (if do_download == True), and interpolate missing days,
the polar hole, and NaN values and save as a final NetCDF (if do_interp == True).

The full download took me around 14 hours - this may differ depending on download
speed. The interpolation and saving takes around 10 minutes.

'''

###############################################################################

do_download = False  # True: download the raw daily data
do_preproc = True  # True: also preprocess the raw daily data and save in yearly NetCDF files
delete_raw_daily_data = True  # True: delete the raw daily SIC data after preprocessing
do_fill_missing_months = True  # True: create NetCDFs for the missing months with NaN data

do_interp = True  # True: interpolate missing days/values in the downloaded dataset
gen_interp_video = True  # True: generate video of interpolated dataset (takes ~30 mins)

regex = re.compile('^.*\.nc$')

land_mask = np.load(os.path.join(config.folders['masks'], config.fnames['land_mask']))

active_grid_cell_masks = {}
for month in np.arange(1, 13):
    month_str = '{:02d}'.format(month)
    active_grid_cell_masks[month_str] = np.load(os.path.join(config.folders['masks'],
                                                config.formats['active_grid_cell_mask'].format(month_str)))

da_year_folder = os.path.join(config.folders['siconca'], 'raw_yearly_data')

if do_download:

    var_remove_list = ['time_bnds', 'raw_ice_conc_values', 'total_standard_error',
                       'smearing_standard_error', 'algorithm_standard_error',
                       'status_flag', 'Lambert_Azimuthal_Grid']

    # Get the daily OSI 450 data of SIC from 01/01/1979 to 31/12/2015 in NetCDF
    #   format (all measurements at 12.00pm)

    # options: mirror, no host directory name, cut the first 4 directories
    #   from the path, output root is "raw-data/sea-ice"

    retrieve_cmd_template_osi450 = 'wget -m -nH -nv --cut-dirs=4 -P {} ' \
        'ftp://osisaf.met.no/reprocessed/ice/conc/v2p0/{:04d}/{:02d}/ice_conc_nh*'

    retrieve_cmd_template_osi430b = 'wget -m -nH -nv --cut-dirs=4 -P {} '\
        'ftp://osisaf.met.no/reprocessed/ice/conc-cont-reproc/v2p0/{:04d}/{:02d}/ice_conc_nh*'

    daily_nan_sic_fig_folder = os.path.join(config.folders['figures'], 'daily_sic_nans')
    if not os.path.exists(daily_nan_sic_fig_folder):
        os.makedirs(daily_nan_sic_fig_folder)

    if not os.path.exists(da_year_folder):
        os.makedirs(da_year_folder)

    tic = time.time()

    for year_i, year in enumerate(range(1979, 2021)):

        # Path to xarray.DataArray storing year of daily SIC data already downloaded
        da_year_path = os.path.join(da_year_folder, 'siconca_{:04d}.nc'.format(year))

        for month_i, month in enumerate(range(1, 13)):

            year_str = '{:04d}'.format(year)
            month_str = '{:02d}'.format(month)
            date = datetime(year, month, 1)

            print("{}/{}, ".format(year, month), end='', flush=True)

            if any([datetime(year, month, 1) == missing_month for missing_month in config.missing_sic_months]):
                print('Skipping missing month.')
                continue

            # Download the data if not already downloaded
            if do_download:
                if year <= 2015:
                    os.system(retrieve_cmd_template_osi450.format(config.folders['siconca'], year, month))
                else:
                    os.system(retrieve_cmd_template_osi430b.format(config.folders['siconca'], year, month))

            if do_download or do_preproc:
                # Folder the daily data was downloaded to
                month_data_folder = os.path.join(config.folders['siconca'], year_str, month_str)

                filenames_downloaded = sorted(os.listdir(month_data_folder))  # List of files in month folder
                filenames_downloaded = [filename for filename in filenames_downloaded if regex.match(filename)]
                paths_downloaded = [os.path.join(month_data_folder, filename) for filename in filenames_downloaded]

                filenames_cleaned = []
                for filename in filenames_downloaded:
                    # Extract the year, month, and day
                    match = re.match('.*_([0-9]{4})([0-9]{2})([0-9]{2}).*', filename)
                    # Save cleaned files in year_month_day format
                    filenames_cleaned.append('{}_{}_{}.nc'.format(match[1], match[2], match[3]))

                paths_after = [os.path.join(config.folders['siconca'], filename_a) for filename_a in filenames_cleaned]

            if do_preproc:
                print("Preprocessing {}/{}... ".format(year, month), end='', flush=True)
                tic_preproc = time.time()

                for day_i, path in enumerate(paths_downloaded):
                    with xr.open_dataset(path) as ds:
                        ds_day = xr.open_dataset(path)

                        # Remove unneeded variables from the NetCDF
                        ds_day = ds_day.drop(var_remove_list)

                        da_day = ds_day['ice_conc']

                        da_day.data = np.array(da_day.data, dtype=np.float32)

                        # Divide values to fit in range 0-1
                        da_day.data = da_day.data / 100.

                        da_day.data[0, ~active_grid_cell_masks[month_str]] = 0.

                        # TEMP: plotting any missing NaN values in figures/ folder while investigating them
                        if np.sum(np.isnan(da_day.data)) > 0:
                            fig, ax = plt.subplots()
                            ax.imshow(np.isnan(da_day.data)[0, :], cmap='gray')
                            ax.contour(land_mask, colors='white', alpha=.5, linewidths=.1)
                            ax.axes.xaxis.set_visible(False)
                            ax.axes.yaxis.set_visible(False)

                            date_ts = pd.Timestamp(da_day.time.values[0])  # np.datetime64 --> pd.Timestamp
                            plt.savefig(os.path.join(daily_nan_sic_fig_folder,
                                                     '{:04d}_{:02d}_{:02d}.png'.format(date_ts.year, date_ts.month, date_ts.day)),
                                        dpi=300)
                            plt.close()

                            print('Found NaNs in SIC day: {:04d}/{:02d}/{:02d}'.format(date_ts.year, date_ts.month, date_ts.day))

                            # TODO: how to deal with the missing values?
                            # da_day.data[np.isnan(da_day.data)] = 0.

                        # TODO: interpolate polar hole, or do this after downloading?

                        # Concat into one xr.DataArray for each year
                        if (month_i == 0 and day_i == 0):
                            # First month of SIC to be downloaded from this year
                            da_year = da_day
                        else:
                            # Not first month of SIC to be downloaded: `da_year` already in memory
                            da_year = xr.concat([da_year, da_day], dim='time')

                print('Done preprocessing month in {:.0f}s.\n\n'.format(time.time()-tic_preproc))

            if delete_raw_daily_data:
                shutil.rmtree(month_data_folder, ignore_errors=True)
                # os.remove(month_data_folder)

            if do_download:
                # Remove year folder once the year is fully processed
                if month == 12 or [year, month] == [1987, 11]:
                    year_dir = os.path.join(config.folders['siconca'], '{}'.format(year))
                    if len(os.listdir(year_dir)) == 0:
                        os.rmdir(year_dir)

        print('\n\n\nSaving year file... ', end='', flush=True)
        tic_save = time.time()
        da_year.to_netcdf(da_year_path, mode='w')
        print('Done in {:.0f}s.\n\n\n'.format(time.time()-tic_save))

    toc = time.time()
    dur = toc - tic
    print("\n\n\nCOMPLETED OSI-SAF SEA ICE DOWNLOAD: Total time taken - {:.0f}m:{:.0f}s".
          format(np.floor(dur / 60), dur % 60))

###############################################################################

if do_interp:

    # Open the downloaded data (stored in yearly files)
    p = da_year_folder
    siconca_year_fpaths = [os.path.join(p, f) for f in os.listdir(p) if regex.match(f)]

    print('\nLoading daily SIC dataset... ', end='', flush=True)
    da = xr.open_mfdataset(siconca_year_fpaths, combine='by_coords')['ice_conc']
    print('Done.')

    # Temporary fix for corrupt latitude field
    lat_vals = da.lat[0]
    da = da.assign_coords(lat=(('xc', 'yc'), lat_vals))

    da = da.astype(np.float32)

    # Remove corrupt days with artefacts
    # da = da.drop_sel(time=datetime(1984, 9, 14, 12))
    da = da.drop_sel(time=config.corrupt_sic_days)

    # Fill missing 1st Jan 1979 with observed 2nd Jan 1979 for continuity
    da_1979_01_01 = da.sel(time=[datetime(1979, 1, 2, 12)]).copy().assign_coords({'time': [datetime(1979, 1, 1, 12)]})
    da = xr.concat([da, da_1979_01_01], dim='time')
    da = da.sortby('time')

    # da_arts = da.loc['1979-07-28':'1979-08-28']
    # for date in tqdm(da_arts.time.values):
    #     date = pd.Timestamp(date)
    #     fig, ax = plt.subplots(figsize=(20, 20))
    #     ax.imshow(da_arts.sel(time=date))
    #     fname = date.strftime('%Y_%m_%d') + '.png'
    #     plt.savefig('figures/sic_artefacts/{}'.format(fname))
    #     plt.close()

    # ---------------- Find missing dates and save CSV of missing gap > thresh # days

    dates_obs = [pd.Timestamp(date).to_pydatetime() for date in da.time.values]

    dates_all = icenet2_utils.filled_daily_dates(datetime(1979, 1, 1, 12), datetime(2021, 1, 1, 12))

    dates_missing = []
    for date in dates_all:
        if date not in dates_obs:
            dates_missing.append(date)

    dates_obs_df = pd.DataFrame(dates_obs, columns=['date'])
    gaps_df = (dates_obs_df.diff() - timedelta(days=1)).rename(columns={'date': 'gap_from_prev'})
    gaps_df = pd.concat((dates_obs_df, gaps_df), axis=1)
    gaps_thresh_df = gaps_df[gaps_df.gap_from_prev >= timedelta(days=5)]

    end = gaps_thresh_df['date'] - timedelta(days=1)
    start = [row.date - row.gap_from_prev for date, row in gaps_thresh_df.iterrows()]
    start_end_gap_df = pd.DataFrame({'start': start, 'end': end, 'gap': gaps_thresh_df.gap_from_prev})
    start_end_gap_df = start_end_gap_df.reset_index().drop(columns='index')
    start_end_gap_df.to_csv(os.path.join(config.folders['data'], config.fnames['missing_sic_days']))

    # ---------------- Fill missing dates

    print('Interpolating {} missing days.\n'.format(len(dates_missing)))
    da_interp = da.copy()
    for date in tqdm(dates_missing):
        da_interp = xr.concat([da_interp, da.interp(time=date)], dim='time')
    da_interp = da_interp.sortby('time')
    print('\nDone.\n')

    # ---------------- Fill polar hole and NaN values

    print("Realising array for interpolation... ", end='', flush=True)
    da_interp.data = np.array(da_interp.data, dtype=np.float32)
    print('Done.')

    print("Bilinearly interpolating polar hole and any NaNs...")
    x = da_interp['xc'].data
    y = da_interp['yc'].data

    xx, yy = np.meshgrid(np.arange(432), np.arange(432))

    for date in tqdm(dates_all):

        skip_interp = False
        if date <= config.polarhole1_final_date:
            polarhole_mask = np.load(os.path.join(config.folders['masks'], config.fnames['polarhole1']))
        elif date <= config.polarhole2_final_date:
            polarhole_mask = np.load(os.path.join(config.folders['masks'], config.fnames['polarhole2']))
        else:
            skip_interp = True

        if not skip_interp:

            da_day = da_interp.sel(time=date)

            # Grid cells outside of polar hole or NaN regions
            valid = ~np.isnan(da_day.data)
            valid = valid & ~polarhole_mask

            ### Find grid cell locations surrounding NaN regions for bilinear interpolation
            nan_mask = np.ma.masked_array(np.full((432, 432), 0.))
            nan_mask[~valid] = np.ma.masked

            nan_neighbour_arrs = {}
            for direction in ('horiz', 'vertic'):

                # C-style indexing for horizontal raveling; F-style for vertical raveling
                if direction == 'horiz':
                    order = 'C'  # Scan columns fastest
                elif direction == 'vertic':
                    order = 'F'  # Scan rows fastest

                # Tuples with starts and ends indexes of masked element chunks
                slice_ends = np.ma.clump_masked(nan_mask.ravel(order=order))

                nan_neighbour_idxs = []
                nan_neighbour_idxs.extend([s.start - 1 for s in slice_ends])
                nan_neighbour_idxs.extend([s.stop for s in slice_ends])

                nan_neighbour_arr_i = np.array(np.full((432, 432), False), order=order)
                nan_neighbour_arr_i.ravel(order=order)[nan_neighbour_idxs] = True
                nan_neighbour_arrs[direction] = nan_neighbour_arr_i

            nan_neighbour_arr = nan_neighbour_arrs['horiz'] + nan_neighbour_arrs['vertic']
            # Remove artefacts along edge of the grid
            nan_neighbour_arr[:, 0] = nan_neighbour_arr[0, :] = nan_neighbour_arr[:, -1] = nan_neighbour_arr[-1, :] = False

            ### Perform bilinear interpolation
            x_valid = xx[nan_neighbour_arr]
            y_valid = yy[nan_neighbour_arr]
            values = da_day.data[nan_neighbour_arr]

            x_interp = xx[~valid]
            y_interp = yy[~valid]

            da_day.data[~valid] = interpolate.griddata((x_valid, y_valid), values, (x_interp, y_interp), method='linear')

            da_interp.loc[date, :] = da_day

    print('Done.')

    # ---------------- Save

    print('\nSaving interpolated dataset... ', end='', flush=True)
    tic_save = time.time()
    da_interp_path = os.path.join(config.folders['siconca'], 'siconca_all_interp.nc')
    if os.path.exists(da_interp_path):
        os.remove(da_interp_path)
    da_interp.to_netcdf(da_interp_path, mode='w')
    print('Done in {:.0f}s.\n\n\n'.format(time.time()-tic_save))

    print('Download/processing of SIC dataset containing {} days completed.\n'.format(len(dates_all)))

    # ---------------- Video

    # da_interp = xr.open_dataarray(da_interp_path)
    # for date in tqdm(dates_all):
    #     fig,ax=plt.subplots(figsize=(20,20))
    #     ax.imshow(da_interp.sel(time=date).data,cmap='Blues_r')
    #     ax.contourf(land_mask, levels=[.5, 1], colors='k')
    #     ax.axes.xaxis.set_visible(False)
    #     ax.axes.yaxis.set_visible(False)
    #     plt.savefig(date.strftime('figures/all_siconca_filled/%Y_%m_%d.png'))
    #     plt.close()
    #
    # for date in tqdm(dates_obs):
    #     fig,ax=plt.subplots(figsize=(20,20))
    #     ax.imshow(da.sel(time=date).data,cmap='Blues_r')
    #     ax.contourf(land_mask, levels=[.5, 1], colors='k')
    #     ax.axes.xaxis.set_visible(False)
    #     ax.axes.yaxis.set_visible(False)
    #     plt.savefig(date.strftime('figures/all_siconca_obs/%Y_%m_%d.png'))
    #     plt.close()

    if gen_interp_video:

        print('Making video of interpolated data...')
        video_path = os.path.join('videos', 'video_all_interp_spatial.mp4')
        icenet2_utils.xarray_to_video(
            da_interp, video_path, mask=land_mask, mask_type='contourf',
            fps=15, cmap='Blues_r', figsize=15
        )
        print('Done.')
