#!/bin/bash

python3 -u icenet2/download_and_regrid_daily_era5_data.py --var tas > era5_download_logs/nh/tas.txt 2>&1 &
python3 -u icenet2/download_and_regrid_daily_era5_data.py --var ta500 > era5_download_logs/nh/ta500.txt 2>&1 &
python3 -u icenet2/download_and_regrid_daily_era5_data.py --var tos > era5_download_logs/nh/tos.txt 2>&1 &
python3 -u icenet2/download_and_regrid_daily_era5_data.py --var psl > era5_download_logs/nh/psl.txt 2>&1 &
python3 -u icenet2/download_and_regrid_daily_era5_data.py --var zg500 > era5_download_logs/nh/zg500.txt 2>&1 &
python3 -u icenet2/download_and_regrid_daily_era5_data.py --var zg250 > era5_download_logs/nh/zg250.txt 2>&1 &
