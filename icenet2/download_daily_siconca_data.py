import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import imageio
import xarray as xr
import pandas as pd
import shutil
import re
import time
from datetime import datetime
# import plotly.express as px
from scipy import interpolate
# sys.path.insert(0, os.path.join(os.getcwd(), 'icenet'))
sys.path.insert(0, os.path.join(os.getcwd(), 'icenet2'))  # if using jupyter kernel
# sys.path.insert(0, os.path.abspath(os.path.join(__file__, os.pardir)))
import config
import icenet2_utils

###############################################################################

do_download = False  # True: download the raw daily data
do_preproc = True  # True: also preprocess the raw daily data and save in yearly NetCDF files
delete_raw_daily_data = True  # True: delete the raw daily SIC data after preprocessing
do_fill_missing_months = True  # True: create NetCDFs for the missing months with NaN data

do_interp = True  # True: interpolate missing days/values in the downloaded dataset

# Get the daily OSI 450 data of SIC from 01/01/1979 to 31/12/2015 in NetCDF
#   format (all measurements at 12.00pm)
# options: mirror, no host directory name, cut the first 4 directories #   from the path, output root is "raw-data/sea-ice"
regex = re.compile('^.*\.nc$')

land_mask = np.load(os.path.join(config.folders['masks'], config.fnames['land_mask']))

active_grid_cell_masks = {}
for month in np.arange(1, 13):
    month_str = '{:02d}'.format(month)
    active_grid_cell_masks[month_str] = np.load(os.path.join(config.folders['masks'],
                                                config.formats['active_grid_cell_mask'].format(month_str)))
if do_download:

    var_remove_list = ['time_bnds', 'raw_ice_conc_values', 'total_standard_error',
                       'smearing_standard_error', 'algorithm_standard_error',
                       'status_flag', 'Lambert_Azimuthal_Grid']

    retrieve_cmd_template_osi450 = 'wget -m -nH -nv --cut-dirs=4 -P {} ' \
        'ftp://osisaf.met.no/reprocessed/ice/conc/v2p0/{:04d}/{:02d}/ice_conc_nh*'

    retrieve_cmd_template_osi430b = 'wget -m -nH -nv --cut-dirs=4 -P {} '\
        'ftp://osisaf.met.no/reprocessed/ice/conc-cont-reproc/v2p0/{:04d}/{:02d}/ice_conc_nh*'

    daily_nan_sic_fig_folder = os.path.join(config.folders['figures'], 'daily_sic_nans')
    if not os.path.exists(daily_nan_sic_fig_folder):
        os.makedirs(daily_nan_sic_fig_folder)


    tic = time.time()

    for year_i, year in enumerate(range(1979, 2021)):

        # Path to xarray.DataArray storing year of daily SIC data already downloaded
        da_year_path = os.path.join(config.folders['siconca'], 'siconca_{:04d}.nc'.format(year))

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

    p = config.folders['siconca']
    siconca_year_fpaths = [os.path.join(p, f) for f in os.listdir(p) if regex.match(f)]

    print('\nLoading daily SIC dataset... ', end='', flush=True)
    da = xr.open_mfdataset(siconca_year_fpaths, combine='by_coords')['ice_conc']
    print('Done.')

    # ---------------- Fill missing dates
    print('\nInterpolating missing days.')

    dates_obs = [pd.Timestamp(date).to_pydatetime() for date in da.time.values]

    dates_all = icenet2_utils.filled_daily_dates(datetime(1979, 1, 1, 12), datetime(2021, 1, 1, 12))

    dates_missing = []
    for date in dates_all:
        if date not in dates_obs:
            dates_missing.append(date)

    print('Found {} missing days.'.format(len(dates_missing)))
    da_interp = da.copy()
    for i, date in enumerate(dates_missing):
        print('Fraction completed: {:.0f}%'.format(100*i/len(dates_missing)))
        sys.stdout.write("\033[F")  # Cursor up one line
        da_interp = xr.concat([da_interp, da.interp(time=date)], dim='time')
    da_interp = da_interp.sortby('time')
    print('Done.\n')

    def make_frame(date, i):
        print('Fraction completed: {:.0f}%\r'.format(100*i/len(dates_missing)))
        sys.stdout.write("\033[F")  # Cursor up one line
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(da_interp.sel(time=date))
        ax.contourf(land_mask, levels=[.5, 1], colors='k')
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)

        ax.set_title('{:04d}/{:02d}/{:02d}'.format(date.year, date.month, date.day), fontsize=30)

        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        plt.close()
        return image

    print('Making video of interpolated data...')
    imageio.mimsave('video_all_interp.mp4',
                    [make_frame(date, i) for i, date in enumerate(dates_all)],
                    fps=15)
    print('Done.')
