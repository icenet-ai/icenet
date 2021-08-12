import inspect
import logging
import os

from datetime import datetime
from inspect import Parameter, Signature

import icenet2.constants as constants
import icenet2.data.loader as loader
import icenet2.model.models as models

import numpy as np
import tensorflow
import tqdm
import xarray as xr

from tensorflow.keras.models import load_model


def predict_forecast(
    input_path,
    output_path,
    data_configuration,
    network_path=None,
    model_func=models.unet_batchnorm,
    seed=42,
    n_forecast_days=93,
    start_dates=tuple([datetime.now().date()])
):
    # TODO: generic predict functions for the different models
    #  that take init date as input?

    icenet2_name = model_func.__name__

    # TODO: network fpath
    network_path = os.path.join(constants.FOLDERS['results'],
                                 "network.{}.h5".format(seed))

    # TODO: custom objects
    # TODO: dynamic num of forecast days
    dataloader = loader.get_loader(
        loader.get_configuration(data_configuration))

    if network_path and os.path.exists(network_path):
        logging.info("Loading model from {}...".format(network_path))
        network = load_model(
            network_path,
        )
    else:
        logging.warning("No network exists, creating untrained model")
        network = model_func(
            (),
            [],
            [],
            n_forecast_days=dataloader.n_forecast_days
        )

#     da_with_coords = xr.open_dataarray(
#         'data/nh/siconca/raw_yearly_data/siconca_1979.nc')
# 
#     shape = (len(start_dates),
#              *dataloader.config['raw_data_shape'],
#              n_forecast_days)
# 
#     forecasts = xr.DataArray(
#         data=np.zeros(shape, dtype=np.float32),
#         dims=('time', 'yc', 'xc', 'leadtime'),
#         coords={
#             'time': start_dates,
#             'yc': da_with_coords.coords['yc'],
#             'xc': da_with_coords.coords['xc'],
#             'leadtime': np.arange(1, n_forecast_days + 1)
#         }
#     )

    # forecast_start_date = all_forecast_start_dates[0]
    batch = []
    for forecast_start_date in tqdm.tqdm(start_dates):
        X, y = dataloader.data_generation([('nh', forecast_start_date)])
        batch.append((X, y))

    pred = network(batch, training=False)
    return pred


# TODO: better method via click via single 'icenet' entry point
def cli():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input_path")
    parser.add_argument("output_path")
    parser.add_argument("data_configuration")
    parser.add_argument("-n", "--network_path", default=None)
    # TODO: mechanism for dynamic lookup via importlib
    parser.add_argument("-m", "--model_func", default=models.unet_batchnorm)
    parser.add_argument("-s", "--seed", default=42, type=int)
    parser.add_argument("-f", "--n_forecast_days", default=93, type=int)
    # TODO: mechanism
    parser.add_argument("-d", "--start_dates", default=tuple([datetime.now()
                                                             .date()]))
    args = parser.parse_args()

    logging.info("Prediction")
    predict_forecast(**vars(args))


if __name__ == "__main__":
    cli()
