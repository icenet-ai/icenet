import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
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

# --------------------- SEA ICE CONCENTRATION

do_download = True  # True: download the raw daily data
do_preproc = True  # True: preprocess the raw daily data
delete_raw_daily_data = True  # True: delete the raw daily SIC data after preprocessing
do_fill_missing_months = True  # True: create NetCDFs for the missing months with NaN data

# Get the daily OSI 450 data of SIC from 01/01/1979 to 31/12/2015 in NetCDF
#   format (all measurements at 12.00pm)
# options: mirror, no host directory name, cut the first 4 directories #   from the path, output root is "raw-data/sea-ice"
regex = re.compile('^.*\.nc$')

var_remove_list = ['time_bnds', 'raw_ice_conc_values', 'total_standard_error',
                   'smearing_standard_error', 'algorithm_standard_error',
                   'status_flag', 'Lambert_Azimuthal_Grid']

retrieve_cmd_template_osi450 = 'wget -m -nH --cut-dirs=4 -P {} ' \
    'ftp://osisaf.met.no/reprocessed/ice/conc/v2p0/{:04d}/{:02d}/ice_conc_nh*'

retrieve_cmd_template_osi430b = 'wget -m -nH --cut-dirs=4 -P {} '\
    'ftp://osisaf.met.no/reprocessed/ice/conc-cont-reproc/v2p0/{:04d}/{:02d}/ice_conc_nh*'

daily_nan_sic_fig_folder = os.path.join(config.folders['figures'], 'daily_sic_nans')
if not os.path.exists(daily_nan_sic_fig_folder):
    os.makedirs(daily_nan_sic_fig_folder)

land_mask = np.load(os.path.join(config.folders['masks'], config.fnames['land_mask']))

active_grid_cell_masks = {}
for month in np.arange(1, 13):
    month_str = '{:02d}'.format(month)
    active_grid_cell_masks[month_str] = np.load(os.path.join(config.folders['masks'],
                                                config.formats['active_grid_cell_mask'].format(month_str)))

tic = time.time()

for year in range(1979, 2021):

    for month in range(1, 13):

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

            for i, path in enumerate(paths_downloaded):
                with xr.open_dataset(path) as ds:
                    ds = xr.open_dataset(path)

                    # Remove unneeded variables from the NetCDF
                    ds = ds.drop(var_remove_list)

                    da = ds['ice_conc']

                    da.data = np.array(da.data, dtype=np.float32)

                    # Divide values to fit in range 0-1
                    da.data = da.data / 100.

                    da.data[0, ~active_grid_cell_masks[month_str]] = 0.

                    # TEMP: plotting any missing NaN values in figures/ folder while investigating them
                    if np.sum(np.isnan(da.data)) > 0:
                        import pandas as pd
                        fig, ax = plt.subplots()
                        ax.imshow(np.isnan(da.data)[0, :], cmap='gray')
                        ax.contour(land_mask, colors='white', alpha=.5, linewidths=.1)
                        ax.axes.xaxis.set_visible(False)
                        ax.axes.yaxis.set_visible(False)

                        date_ts = pd.Timestamp(da.time.values[0])  # np.datetime64 --> pd.Timestamp
                        plt.savefig(os.path.join(daily_nan_sic_fig_folder,
                                                 '{:04d}_{:02d}_{:02d}.png'.format(date_ts.year, date_ts.month, date_ts.day)),
                                    dpi=300)
                        plt.close()

                        print('Found NaNs in SIC day: {:04d}/{:02d}/{:02d}'.format(date_ts.year, date_ts.month, date_ts.day))

                        # TODO: how to deal with the missing values?
                        # da.data[np.isnan(da.data)] = 0.

                    # TODO: interpolate polar hole

                    # Write to new NetCDF
                    ds.to_netcdf(paths_after[i], mode='w')
                    os.remove(path)

            print("Done.")

        if delete_raw_daily_data:
            shutil.rmtree(month_data_folder, ignore_errors=True)
            # os.remove(month_data_folder)

        if do_download:
            # Remove year folder once the year is fully processed
            if month == 12 or [year, month] == [1987, 11]:
                year_dir = os.path.join(config.folders['siconca'], '{}'.format(year))
                if len(os.listdir(year_dir)) == 0:
                    os.rmdir(year_dir)

# Write NetCDF files for the missing months with all NaN data for continuity
# if do_fill_missing_months:
#     # Template NetCDF file for missing months
#     nan_data = np.full((1, 432, 432), np.nan)
#     ds = xr.open_dataset(os.path.join(config.ice_data_folder, "avg_sic_2019_01.nc"))
#     ds['ice_conc'].data = nan_data
#
#     for missing_month_date in config.missing_dates:
#         year_str = '{:04d}'.format(missing_month_date.year)
#         month_str = '{:02d}'.format(missing_month_date.month)
#
#         ds = ds.assign_coords({'time': [missing_month_date]})
#
#         ds.to_netcdf(os.path.join(config.ice_data_folder,
#                                   config.sic_monthly_avg_template.format(year_str, month_str)))

toc = time.time()
dur = toc - tic
print("\n\n\nCOMPLETED OSI-SAF SEA ICE DOWNLOAD: Total time taken - {:.0f}m:{:.0f}s".
      format(np.floor(dur / 60), dur % 60))
