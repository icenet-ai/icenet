# IceNet2

### Plan for the repo

##### Misc
* [ ] Config script

##### Downloading data
* [ ] Script to download hourly era5 data, compute daily averages, and regrid to EASE grid
* [ ] Script to download daily OSI-SAF SIC data and fill missing days appropriately

##### Preprocessing data
* [ ] Class + script to preprocess ERA5 + SIC data into .npy files
* [ ] Class + pickling script to instantiate daily data loader

##### Training IceNet2
* [ ] Script to define loss function, validation metric, and IceNet2 architecture
* [ ] Script to train IceNet2

##### Validating IceNet2
* [ ] Script to validate IceNet2 and produce plots
