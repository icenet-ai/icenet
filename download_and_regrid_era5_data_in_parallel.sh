#!/bin/bash

python3 icenet2/download_and_regrid_daily_era5_data.py --var tas > era5_download_logs/tas.txt 2>&1 &
python3 icenet2/download_and_regrid_daily_era5_data.py --var ta500 > era5_download_logs/ta500.txt 2>&1 &
python3 icenet2/download_and_regrid_daily_era5_data.py --var tos > era5_download_logs/tos.txt 2>&1 &
python3 icenet2/download_and_regrid_daily_era5_data.py --var psl > era5_download_logs/psl.txt 2>&1 &
python3 icenet2/download_and_regrid_daily_era5_data.py --var zg500 > era5_download_logs/zg500.txt 2>&1 &
python3 icenet2/download_and_regrid_daily_era5_data.py --var zg250 > era5_download_logs/zg250.txt 2>&1 &
