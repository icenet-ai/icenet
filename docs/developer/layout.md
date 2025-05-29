# Library Layout
Firstly, detailed documentation generated from the docstrings are available under `API/`. 

This page intends to give an overview beyond the `API/` of how different components of `icenet`
contribute to enable sea-ice forecasting alongside external data and preprocessing tooling, for
which we generally use [`download-toolbox`](https://pypi.org/project/download-toolbox) and 
[`preprocess-toolbox`](https://pypi.org/project/preprocess-toolbox).

## icenet

### icenet.data
#### icenet.data.datasets
##### icenet.data.datasets.splitting
##### icenet.data.datasets.utils
#### icenet.data.loaders
##### icenet.data.loaders.stdlib
##### icenet.data.loaders.utils
##### icenet.data.loaders.base
##### icenet.data.loaders.dask
#### icenet.data.processors
##### icenet.data.processors.amsr
##### icenet.data.processors.cds
##### icenet.data.processors.cmip
##### icenet.data.processors.osisaf
#### icenet.data.masks
##### icenet.data.masks.nsidc
##### icenet.data.masks.osisaf
#### icenet.data.references
##### icenet.data.references.osisaf
#### icenet.data.loader
#### icenet.data.meta
#### icenet.data.network_dataset
### icenet.model
#### icenet.model.callbacks
#### icenet.model.losses
#### icenet.model.handlers
##### icenet.model.handlers.wandb
#### icenet.model.networks
##### icenet.model.networks.base
##### icenet.model.networks.tensorflow
#### icenet.model.cli
#### icenet.model.predict
#### icenet.model.train
#### icenet.model.utils
#### icenet.model.metrics
### icenet.plotting
#### icenet.plotting.trend
#### icenet.plotting.data
#### icenet.plotting.forecast
#### icenet.plotting.utils
#### icenet.plotting.video
### icenet.process
#### icenet.process.azure
#### icenet.process.local
#### icenet.process.train
#### icenet.process.utils
#### icenet.process.forecasts
#### icenet.process.predict
### icenet.results
#### icenet.results.metrics
#### icenet.results.threshold
### icenet.tests
#### icenet.tests.test_entry_points
#### icenet.tests.test_mod
### icenet.cli
### icenet.exceptions
### icenet.utils
