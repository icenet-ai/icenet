# IceNet2: A daily to seasonal sea ice forecasting AI

TODO: add some plots.

Codebase to train IceNet2, an ensemble of `M` U-Net neural networks for forecasting maps of daily-averaged Arctic sea ice `1, 2, ..., N` days into the future.
A flexible data loader class is provided to dictate which map variables are input to the networks (e.g. past sea ice and other climate variables), how far they look back into the past, and how far ahead to forecast.

This is an extension of the paper [Seasonal Arctic sea ice forecasting with probabilistic deep learning](https://doi.org/10.31223/X5430P) to operate on a daily timescale (rather than monthly) and perform probabilistic regression (rather than probabilistic classification).

![](architecture.png)

The guidelines below assume you're working on a Unix-like machine with a GPU.

#### Example folder structure

```
.
|-icenet2/ (code folder)

|-data/
|---forecasts/ (daily validation forecast NetCDFs in model subfolders)
|-----Day_persistence/
|-----icenet2/
|-------2021_04_25_1351_icenet2_nh_thinned7_weeklyinput_wind_3month/
|---------unet_batchnorm/
|-----------ensemble/
|-----seas5/

|---forecasts_monthly/ (as above but for monthly averages)
|---nh/ (northern hemisphere data)
|---sh/ (sourthern hemisphere data)
|---network_datasets/ (normalised network datasets in daily .npy file format)

|-dataloader_configs/ (.json files specifying icenet2 input-output data and training config options)

|-results/ (.csv files of daily validation results and subfolders for icenet2 models containing trained models and validation plots)
|---2021_04_28_141515_results.csv
|---2021_04_25_1351_icenet2_nh_thinned7_weeklyinput_wind_3month/
|-----unet_batchnorm/
|-------networks/
|-------validation/

|-results_monthly/ (.csv files of monthly validation results)
|---2021_04_26_173002_results.csv

|-figures/

|-videos/
|---forecast_videos/
|-----2021_04_25_1351_icenet2_nh_thinned7_weeklyinput_wind_3month/
```

#### Preliminary setup

The following instructions assume you have conda installed. If you don't yet have conda, you can download it [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html).

For the script to download ERA5 data to work, you must first set up a CDS account and populate your `cdsapirc` file.
Follow the 'Install the CDS API key' instructions available [here](https://cds.climate.copernicus.eu/api-how-to#install-the-cds-api-key).
This shouldn't take more than a few minutes.

Optional: I use `tmux` while SSH'd into BAS's HPC to keep my commands running after disconnecting: `tmux new -s icenet` to create a new `tmux` session and `tmux attach -t icenet` when attaching.

### 1) Set up Conda environment

After cloning the repo, run the commands below in the root of the repository to set up the conda environment:

- `conda env create -f environment.yml -n icenet2`
- `conda activate icenet2`

### 2) Download data

- `python3 icenet2/gen_masks.py`. This obtains masks for land, the polar holes, and monthly maximum ice extent.

- `python3 icenet2/download_and_interpolate_daily_sic_data.py`. Gets OSI-SAF SIC data.
This downloads daily SIC data, linearly interpolates missing days, and bilinearly interpolates missing grid cells (e.g. polar hole).
Probably best to run overnight.

- `./download_and_regrid_era5_data_in_parallel.sh`. Gets ERA5 reanalysis data.
This runs multiple `python3 icenet2/download_and_regrid_daily_era5_data.py` commands to acquire multiple variables in parallel.
This downloads the raw hourly-averaged ERA5 data in global latitude-longitude format, computes daily averages, and regrids to the EASE grid that OSI-SAF SIC data lies on.
To rotate and regrid ERA5 surface wind vector fields, use `icenet2/rotate_and_regrid_era5_wind_vector_data.py`.

### 3) Normalise data and set up data loader configuration

- `python3 icenet2/preproc_icenet2_data.py`. Normalises the data and save in daily NumPy files.

- `python3 icenet2/gen_data_loader_config.py`. Sets up data loader.

### 4) Train IceNet2

- `python3 icenet2/train_icenet2.py`, or submit via SLURM with `sbatch slurm_train_script.sh`. Both take hyperameter settings as command line inputs. Run multiple times with different settings of the `seed` hyperparameter to train an ensemble.

### 5) Run validation

- `python3 icenet2/predict_validation.py`. Use `xarray` to save daily predictions in yearly NetCDFs for IceNet2 and benchmarks with dimensions `(target date, x, y, lead time)`.
- `python3 icenet2/analyse_validation.py`. Load the forecast data and compute forecast metrics, storing results in a global `pandas` DataFrame with MultiIndex `(target date, lead time, model)` and columns for each metric. Optionally use `dask` to avoid loading the entire forecast datasets to memorry, and process chunks in parallel.
- `python3 icenet2/plot_validation.py`. Plot results using seaborn.

### Misc

- `icenet2/models.py` defines network architectures.
- `icenet2/config.py` defines globals.
- `icenet2/losses.py` defines loss functions.
- `icenet2/callbacks.py` defines training callbacks.
- `icenet2/metrics.py` defines training metrics.
- `icenet2/utils.py` defines IceNet2 utility functions like the data preprocessor, data loader, and regridding methods.
- `icenet2/misc.py` defines miscellaneous methods like a method to generate videos of IceNet2 forecasts.
- `icenet2/gen_figures.py` code for reproducing various figures.
