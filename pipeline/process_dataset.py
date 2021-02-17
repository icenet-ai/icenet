import argparse
import glob
import logging
import os
import pickle
import sys

from datetime import datetime
from dateutil.relativedelta import relativedelta

import numpy as np
import tensorflow as tf

import icenet.config as config
from icenet.data import CachingProcessor
from icenet.utils import filled_datetime_array


def get_args():
    a = argparse.ArgumentParser()
    a.add_argument("-g", "--generate", help="Generate data", default=False, action="store_true")
    a.add_argument("-d", "--num-forecast-months", help="Number of forecast months", default=6, type=int)
    a.add_argument("input", help="Input directory")
    a.add_argument("output", help="Output directory")
    return a.parse_args()


def generate_data():
    rng = np.random.default_rng(42)

    for i in range(10):
        logging.info("Generating Record {}".format(i))

        with tf.io.TFRecordWriter(os.path.join(args.output, "{}.tfrecord".format(i))) as writer:
            x = rng.random((432, 432, 57))
            y = rng.random((432, 432, 4, 6))

            output = tf.train.Example(features=tf.train.Features(feature={
                "x": tf.train.Feature(float_list=tf.train.FloatList(value=x.reshape(-1))),
                "y": tf.train.Feature(float_list=tf.train.FloatList(value=y.reshape(-1))),
            })).SerializeToString()

            writer.write(output)


def decode_item(proto):
    features = {
        "x": tf.io.FixedLenFeature([432, 432, 57], tf.float32),
        "y": tf.io.FixedLenFeature([432, 432, 4, 6], tf.float32),
    }

    return tf.io.parse_single_example(proto, features)


def example_rw():
    ### BEGIN: Load demo
    fns = glob.glob("{}/*.tfrecord".format(args.output))

    if args.generate and len(fns) == 0:
        logging.info("Generating data")
        generate_data()

    ds = tf.data.TFRecordDataset(fns)
    ps = ds.map(decode_item)

    for pr in ps:
        print("{}".format(pr['x'].shape))
        print("{}".format(pr['y'].shape))

    ### END: Load demo

def process_era_dataset():
    # TODO: Move to YAML configuration
    # TODO: Include/exclude by ommission, it makes code clearer
    input_data = {
        "siconca":
            {"abs": {'lookbacks': np.arange(0, 12)},
             "anom": {'lookbacks': np.arange(0, 3)},
             "linear_trend": {"include": True}},
        "tas":
            {  # "abs": {"include": False, 'lookbacks': np.arange(0, 3)},
                "anom": {'lookbacks': np.arange(0, 3)}},
        "rsds": {
            "abs": {'lookbacks': np.arange(0, 3)},
            # "anom": {"include": False, 'lookbacks': np.arange(0, 3)}
        },
        "rsus":
            {"abs": {'lookbacks': np.arange(0, 3)},
             # "anom": {"include": False, 'lookbacks': np.arange(0, 3)}
             },
        "tos":
            {  # "abs": {"include": False, 'lookbacks': np.arange(0, 3)},
                "anom": {'lookbacks': np.arange(0, 3)}},
        "psl":
            {  # "abs": {"include": False, 'lookbacks': np.arange(0, 3)},
                "anom": {'lookbacks': np.arange(0, 3)}},
        "zg500":
            {  # "abs": {"include": False, 'lookbacks': np.arange(0, 3)},
                "anom": {'lookbacks': np.arange(0, 3)}},
        "zg250":
            {  # "abs": {"include": False, 'lookbacks': np.arange(0, 3)},
                "anom": {'lookbacks': np.arange(0, 3)}},
        "ua10":
            {"abs": {'lookbacks': np.arange(0, 3)},
             # "anom": {"include": False, 'lookbacks': np.arange(0, 3)}
             },
        "uas":
            {"abs": {'lookbacks': np.arange(0, 3)},
             # "anom": {"include": False, 'lookbacks': np.arange(0, 3)}
             },
        "vas":
            {"abs": {'lookbacks': np.arange(0, 3)},
             # "anom": {"include": False, 'lookbacks': np.arange(0, 3)}
             },
        "sfcWind":
            {"abs": {'lookbacks': np.arange(0, 3)},
             # "anom": {"include": False, 'lookbacks': np.arange(0, 3)}
             },
        "land":
            {"metadata": True, },
        "circmonth":
            {"metadata": True, },
    }

    # TODO: There were some potential oddities to check with Tom about the boundary calculations for periods (+13? -6?)
    #  Surely these are to be contiguous blocks, but I might be missing something. Regardless, explicit definition
    #  is best
    obs_train_dates = filled_datetime_array(datetime(1979, 1, 1), datetime(2011, 12, 31))
    obs_val_dates = filled_datetime_array(datetime(2012, 1, 1), datetime(2017, 12, 31))
    obs_test_dates = filled_datetime_array(datetime(2018, 1, 1), datetime(2020, 1, 1))

    combined_dates = np.hstack((obs_train_dates, obs_val_dates, obs_test_dates))
    all_obs_dates = filled_datetime_array(combined_dates.min(), combined_dates.max())

    logging.info("Creating loader")

    cp = IcenetERAPreprocessor(source=,
    #                                    name='dataset2',
    #                                    batch_size=4,
    #                                    shuffle=True,
    #                                    num_forecast_months=args.num_forecast_months,
    #                                    obs_train_dates=obs_train_dates,
    #                                    obs_val_dates=obs_val_dates,
    #                                    obs_test_dates=obs_test_dates,
    #                                    verbose_level=2,
    #                                    raw_data_shape=(432, 432),
    #                                    default_seed=42,
    #                                    dtype=np.float32,
    #                                    loss_weight_months=True,
    #                                    loss_weight_classes=False,
    #                                    cmip6_transfer_train_dict=cmip6_transfer_train_dict,
    #                                    cmip6_transfer_val_dict=cmip6_transfer_val_dict,
    #                                    convlstm=False,
    #                                    n_convlstm_input_months=12,
    #                                    cache_path=os.path.join(dataset_folder, "cache"))


def process_cmip_dataset():
    pass

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    args = get_args()

    process_era_dataset()
    process_cmip_dataset()


#     dataset_folder = os.path.join(config.results_folder, 'icenet2_linear_trend_input_6runs_absrad')
#     dataloader = CachingDataLoader(input_data=input_data,
#                                    name='dataset2',
#                                    batch_size=4,
#                                    shuffle=True,
#                                    num_forecast_months=args.num_forecast_months,
#                                    obs_train_dates=obs_train_dates,
#                                    obs_val_dates=obs_val_dates,
#                                    obs_test_dates=obs_test_dates,
#                                    verbose_level=2,
#                                    raw_data_shape=(432, 432),
#                                    default_seed=42,
#                                    dtype=np.float32,
#                                    loss_weight_months=True,
#                                    loss_weight_classes=False,
#                                    cmip6_transfer_train_dict=cmip6_transfer_train_dict,
#                                    cmip6_transfer_val_dict=cmip6_transfer_val_dict,
#                                    convlstm=False,
#                                    n_convlstm_input_months=12,
#                                    cache_path=os.path.join(dataset_folder, "cache"))


    # TODO: We will pickle the DATASET(S), not the loader
    #pickle_path = os.path.join(dataset_folder, 'data_loader.pickle')

    #with open(pickle_path, 'wb') as writefile:
    #    pickle.dump(dataloader, writefile)
