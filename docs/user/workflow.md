# Workflow
## Overview

The workflow for machine learning is largely always the same, most of your time will be spent preparing data.

## Processing steps

![Pipeline Diagram](../images/pipeline.png)
/// caption
**Figure 1:** IceNet processing pipeline
///

1. Datasets are downloaded to a data store
2. Datasets are infilled or interpolated to fill gaps
3. Datasets are manipulated (e.g. reprojection) to make their represenation consistent enough to use
4. Preprocessing takes place to create a data loader 
  a. Datasets are processed to select and filter dates based on model requirements
  b. Creation and addition of masks suitable for data takes place
  c. Synthesized data is generated and added
5. AI-ready dataset configurations are produced
  a. Caching might take place

## Particulars

There are various aspects of processing that can be unintuitive to understand. Most of these arise from behaviours in the environmental forecasting toolboxes, but are described here based on the IceNet implementation (TODO: move when suitable!)

### Setting dates 

Downloading `download_*` (step 1) commands simply give bounds for dates and download what they can. 

Preprocessing then accounts for the eventual splitting of data. All `preprocess` commands thus take splits in their arguments, allowing them to account for what will eventually be the bounds of splits, as opposed to the dataset itself. This allows optimisation of data processing (e.g. copying 10-years data for 10-years usage, rather than cloning datasets.)

#### Initialisation, lag and lead

Forecast initialisation date is the date to start the forecast.
Lag is the number of _**additional**_ channels prior to `forecast_init_date - 1` that will be included.

```
     .-.-.-.-.-.-.-.-.-.-.-.-.
LEAD | | | | |0|1|2|3|4|5|6|7|
     :-:-:-:-:-:-:-:-:-:-:-:-:
 LAG | |2|1|0| | | | | | | | |
     '-'-'-'-'-'-'-'-'-'-'-'-'
              ^
      FORECAST_INIT_DATE
```

`preprocess_*` commands accept splits in order to constrain the amount of data copied from the fully downloaded data stores.

### Missing or invalid data

[Ref](https://github.com/environmental-forecasting/download-toolbox/issues/)

Known (static) invalid or missing data from is set against the `DatasetConfig` implementation for the dataset in question. These dates are stored in the `data.*.json` files for downstream processes to consider.  

#### Spatially

`preprocess_missing_spatial` calls `spatial_interpolation` in `preprocess_toolbox/dataset/spatial.py`, ignoring the masks and interpolating the nans that remain in the domain.

__TODO: need to review the spatial interpolation, it's probably not being used the way we want__ 

#### Temporally

Invalid dates are further identified with `preprocess_missing_time` which interpolates the data across that dimension, adds detected dates to the configuration. The rendering of linear trends and other processed datasets can then transparently use the data, but the registration as a missing date means that sample rendering for the network can account for it in sample weights (by invalidating it). 

#### Temporary anomolies

ERA5T and OSISAF
CMIP node unreliability

### Linear trends

### Masking

### Sample generation

#### Forecast target

A loader configuration may select one processed variable as its forecast target:

```json
{
  "target": {
    "source": "osisaf",
    "variable": "siconca_abs",
    "mask": "land",
    "weight_mask": "active_grid_cell"
  }
}
```

`source` must identify an entry in `sources`, and `variable` a processed
variable in that source's `processed_files` mapping.

The target is opened from its own files, separately from the predictor
channels, and each step is selected by timestamp. It therefore does not need to
be an input channel, its files need not cover the lag history, and its time axis
need not line up with the predictors'.

The target window follows the diagram in [Initialisation, lag and
lead](#initialisation-lag-and-lead): it opens on `forecast_init_date`, while the
lag channels close on `forecast_init_date - 1`. The two windows never overlap,
so the target cannot be handed back to the network as one of its own inputs.
A target step with no data is dropped from the dataset rather than being filled
from a neighbouring date.

`mask` names an optional static mask of cells that are never valid for this
target, and `weight_mask` an optional sample weighting field, which may carry a
`month` dimension to vary by target step. Either may be `null`, which is the
default for an explicitly configured target: `land` and `active_grid_cell`
carry sea ice meaning and are not applied to, say, a temperature field unless
asked for. Configurations with no `target` block keep the legacy `siconca_abs`
target, along with both legacy masks, where it can be inferred unambiguously.

The current TensorFlow network uses a sigmoid output and a binary accuracy
metric, so targets used by that training pipeline must be preprocessed to the
range [0, 1]. Only one forecast target is supported per network dataset.

#### Sample weights