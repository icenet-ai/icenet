# IceNet2: A daily to seasonal sea ice forecasting AI

Codebase to train IceNet2, an ensemble of `M` U-Net neural networks for forecasting maps of daily-averaged Arctic sea ice `N` days into the future. A flexible data loader class is provided to dictate which map variables are input to the networks (e.g. past sea ice and other climate variables), how far they look back into the past, and how far ahead to forecast.

This is an extension of the paper [Seasonal Arctic sea ice forecasting with probabilistic deep learning](https://doi.org/10.31223/X5430P) to operate on a daily timescale (rather than monthly) and perform probabilistic regression (rather than probabilistic classification).

The guidelines below assume you're working on a Unix-like machine with a GPU.

#### Preliminary setup

The following instructions assume you have conda installed. If you don't yet have conda, you can download it [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html).

For the script to download ERA5 data to work, you must first set up a CDS account and populate your `cdsapirc` file. Follow the 'Install the CDS API key' instructions available [here](https://cds.climate.copernicus.eu/api-how-to#install-the-cds-api-key). This shouldn't take more than a few minutes.

Optional: I use `tmux` while SSH'd into BAS's HPC to keep my commands running after disconnecting: `tmux new -s icenet` to create a new `tmux` session and `tmux attach -t icenet` when attaching.

### 1) Set up Conda environment

Running the commands below in the root of the repository will set up the conda environment:

- `conda env create -f environment.yml`
- `conda activate icenet2`

### 2) Download data

- Get OSI-SAF SIC data: `python3 icenet2/download_and_interpolate_daily_sic_data.py`. This downloads daily SIC data, linearly interpolates missing days, and bilinearly interpolates missing grid cells (e.g. polar hole). Probably best to run overnight.

- Get ERA5 reanalysis data: `./download_and_regrid_era5_data_in_parallel.sh`. This runs multiple `python3 icenet2/download_and_regrid_daily_era5_data.py` commands to acquire multiple variables in parallel. This downloads the raw hourly-averaged ERA5 data in global latitude-longitude format, computes daily averages, and regrids to the EASE grid that OSI-SAF SIC data lies on.

- Generate masks: `python3 icenet2/gen_masks.py`. This obtains masks for land, the polar holes, and monthly maximum ice extent.

### 3) Normalise data and set up data loader configuration

- Normalise the data and save in daily NumPy files: `python3 icenet2/preproc_icenet2_data.py`

- Set up data loader: `python3 icenet2/gen_data_loader_config.py`

### 4) Train IceNet2

- `python3 icenet2/train_icenet2`

### 5) Run validation

- `python3 icenet2/validate_icenet2`

## Repo TODO list

##### Misc
* [x] Config script

##### Downloading data
* [x] Script to download daily OSI-SAF SIC data and fill missing days appropriately
* [x] Script to download hourly era5 data, compute daily averages, and regrid to EASE grid

##### Preprocessing data
* [x] Class + script to preprocess ERA5 + SIC data into .npy files
* [x] Class for daily data loader

##### Training IceNet2
* [x] Script to define loss function, validation metric, and IceNet2 architecture
* [x] Script to train IceNet2

##### Validating IceNet2
* [ ] Script to validate IceNet2 and produce plots
