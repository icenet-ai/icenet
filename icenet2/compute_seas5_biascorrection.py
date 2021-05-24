import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), 'icenet2'))
import config
import misc
from tqdm import tqdm
from datetime import datetime
import numpy as np
import regex as re
import xarray as xr
import pandas as pd
from time import time

calibration_folder = os.path.join('data', 'forecasts', 'seas5_calibration', 'EASE')
calibration_fpaths = [
    os.path.join(calibration_folder, f) for f
    in sorted(os.listdir(calibration_folder))
]

biascorrection_folder = os.path.join('data', 'forecasts', 'seas5_biascorrection')
if not os.path.exists(biascorrection_folder):
    os.makedirs(biascorrection_folder)

gen_videos = False  # Significantly slows script (e.g. 15 seconds vs 20 mins)

if gen_videos:
    video_folder = os.path.join('videos', 'forecast_videos', 'seas5_biascorrection')
    if not os.path.exists(video_folder):
        os.makedirs(video_folder)

### Load ground truth data
####################################################################

true_sic_fpath = os.path.join('data', 'nh', 'siconca', 'siconca_all_interp.nc')
true_sic_da = xr.open_dataarray(true_sic_fpath, chunks=dict(time=20))

# Replace 12:00 hour with 00:00 hour by convention
dates = [pd.Timestamp(date).to_pydatetime().replace(hour=0) for date in true_sic_da.time.values]
true_sic_da = true_sic_da.assign_coords(dict(time=dates))

true_sic_da.load()

### Compute bias correction fields for each forecast initialisation
####################################################################

for init_month in tqdm(np.arange(1, 12+1)):

    fname_regex = '.*ease_seas5_([0-9]*)_{:02d}_01\.nc$'.format(init_month)
    month_fpaths = \
        [fpath for fpath in calibration_fpaths if re.compile(fname_regex).match(fpath)]

    years = \
        [int(re.compile(fname_regex).match(fpath)[1]) for fpath in month_fpaths]

    err_da_list = []
    for forecast_fpath, year in zip(month_fpaths, years):

        forecast = xr.open_dataset(forecast_fpath)['siconc']

        # Assume SEAS5 forecast for 24:00 is close to daily average forecast for that day,
        #   and use coordinate convention of 00:00 for daily average
        dates = [pd.Timestamp(date) - pd.DateOffset(1) for date
                 in forecast.time.values]
        forecast = forecast.assign_coords(dict(time=dates))

        # Convert spatial coords to km
        forecast = forecast.assign_coords(dict(xc=forecast.xc/1e3, yc=forecast.yc/1e3))

        # Remove initialisation state
        forecast = forecast.loc[forecast.time > forecast.time[0]]

        err_da = forecast - true_sic_da

        err_da = err_da.assign_coords(dict(time=np.arange(1, 93+1))).rename(dict(time='leadtime'))
        err_da = err_da.expand_dims(dim={'year': [year]})
        err_da_list.append(err_da)

    biascorrection_field = xr.concat(err_da_list, dim='year').mean('year')

    biascorrection_fname = '{:02d}_01.nc'.format(init_month)
    biascorrection_fpath = os.path.join(biascorrection_folder, biascorrection_fname)
    if os.path.exists(biascorrection_fpath):
        os.remove(biascorrection_fpath)
    biascorrection_field.to_netcdf(biascorrection_fpath)

    if gen_videos:
        from misc import xarray_to_video
        land_mask = np.load('data/nh/masks/land_mask.npy')

        video_fpath = os.path.join(video_folder, '{:02d}_01.mp4'.format(init_month))

        # Convert to SIC (%) and time format expected by method
        biascorrection_field *= 100
        biascorrection_field = biascorrection_field.rename({'leadtime': 'time'})
        biascorrection_field = biascorrection_field.assign_coords({'time': dates[1:]})

        xarray_to_video(
            biascorrection_field, video_fpath, mask=land_mask,
            cmap='seismic', clim=(-100, 100), fps=5
        )
