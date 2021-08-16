import logging
import os

from datetime import datetime

import icenet2.model.models as models

from icenet2.data.loader import IceNetDataSet

from tensorflow.keras.models import load_model


def predict_forecast(
    data_configuration,
    model_func=models.unet_batchnorm,
    start_dates=tuple([datetime.now().date()]),
    seed=42,
    network_folder=os.path.join(".", "results", "networks")
):
    # TODO: generic predict functions for the different models
    #  that take init date as input?
    ds = IceNetDataSet(data_configuration)
    dl = ds.get_data_loader()

    forecast_inputs = {date: dl.generate_sample(date) for date in start_dates}

    network_path = os.path.join(network_folder,
                                "network_{}.{}.h5".format(ds.identifier, seed))

    if network_path and os.path.exists(network_path):
        logging.info("Loading model from {}...".format(network_path))
        network = load_model(network_path)
    else:
        logging.warning("No network exists, creating untrained model")
        network = model_func(
            (*ds.shape, dl.num_channels),
            [],
            [],
            n_forecast_days=ds.n_forecast_days
        )

    pred = network([forecast_inputs.values()], training=False)
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
