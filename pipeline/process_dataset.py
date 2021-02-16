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
from icenet.data import CachingDataLoader
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


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    args = get_args()

    # TODO: Move to YAML configuration
    input_data = {
        "siconca":
            {"abs": {"include": True, 'lookbacks': np.arange(0, 12)},
             "anom": {"include": True, 'lookbacks': np.arange(0, 3)},
             "linear_trend": {"include": True}},
        "tas":
            {"abs": {"include": False, 'lookbacks': np.arange(0, 3)},
             "anom": {"include": True, 'lookbacks': np.arange(0, 3)}},
        "rsds":
            {"abs": {"include": True, 'lookbacks': np.arange(0, 3)},
             "anom": {"include": False, 'lookbacks': np.arange(0, 3)}},
        "rsus":
            {"abs": {"include": True, 'lookbacks': np.arange(0, 3)},
             "anom": {"include": False, 'lookbacks': np.arange(0, 3)}},
        "tos":
            {"abs": {"include": False, 'lookbacks': np.arange(0, 3)},
             "anom": {"include": True, 'lookbacks': np.arange(0, 3)}},
        "psl":
            {"abs": {"include": False, 'lookbacks': np.arange(0, 3)},
             "anom": {"include": True, 'lookbacks': np.arange(0, 3)}},
        "zg500":
            {"abs": {"include": False, 'lookbacks': np.arange(0, 3)},
             "anom": {"include": True, 'lookbacks': np.arange(0, 3)}},
        "zg250":
            {"abs": {"include": False, 'lookbacks': np.arange(0, 3)},
             "anom": {"include": True, 'lookbacks': np.arange(0, 3)}},
        "ua10":
            {"abs": {"include": True, 'lookbacks': np.arange(0, 3)},
             "anom": {"include": False, 'lookbacks': np.arange(0, 3)}},
        "uas":
            {"abs": {"include": True, 'lookbacks': np.arange(0, 3)},
             "anom": {"include": False, 'lookbacks': np.arange(0, 3)}},
        "vas":
            {"abs": {"include": True, 'lookbacks': np.arange(0, 3)},
             "anom": {"include": False, 'lookbacks': np.arange(0, 3)}},
        "sfcWind":
            {"abs": {"include": True, 'lookbacks': np.arange(0, 3)},
             "anom": {"include": False, 'lookbacks': np.arange(0, 3)}},
        "land":
            {"metadata": True,
             "include": True},
        "circmonth":
            {"metadata": True,
             "include": True},
    }



    # TODO: There were some potential oddities to check with Tom about the boundary calculations for periods (+13? -6?)
    #  Surely these are to be contiguous blocks, but I might be missing something. Regardless, explicit definition
    #  is best
    obs_train_dates = filled_datetime_array(datetime(1979, 1, 1), datetime(2011, 12, 31))
    obs_val_dates = filled_datetime_array(datetime(2012, 1, 1), datetime(2017, 12, 31))
    obs_test_dates = filled_datetime_array(datetime(2018, 1, 1), datetime(2020, 1, 1))

    combined_dates = np.hstack((obs_train_dates, obs_val_dates, obs_test_dates))
    all_obs_dates = filled_datetime_array(combined_dates.min(), combined_dates.max())

    # TRANSFER LEARNING
    cmip6_model_names = ['EC-Earth3', 'MRI-ESM2-0']

    # for r1i1p1f1
    cmip6_start_date_250years = datetime(1850, 1, 1)
    cmip6_end_date_250years = datetime(2100, 12, 1) - relativedelta(months=args.num_forecast_months)
    start_date = cmip6_start_date_250years + relativedelta(months=13)
    all_cmip6_forecast_dates_250years = filled_datetime_array(start_date, cmip6_end_date_250years)

    cmip6_start_date_180years = datetime(1850, 1, 1)
    cmip6_end_date_180years = datetime(2030, 12, 1) - relativedelta(months=args.num_forecast_months)
    start_date = cmip6_start_date_180years + relativedelta(months=13)
    all_cmip6_forecast_dates_180years = filled_datetime_array(start_date, cmip6_end_date_180years)

    cmip6_run_dict = {
        'EC-Earth3': ('r17i1p1f1', 'r18i1p1f1', 'r19i1p1f1'),
        'MRI-ESM2-0': ('r2i1p1f1', 'r3i1p1f1', 'r4i1p1f1')
    }

    cmip6_transfer_train_dict = {}
    cmip6_transfer_val_dict = {}

    # Set up nested dicts
    for cmip6_model_name in cmip6_model_names:
        cmip6_transfer_train_dict[cmip6_model_name] = {}
        cmip6_transfer_val_dict[cmip6_model_name] = {}

    # No validation:
    for source_id, member_ids in cmip6_run_dict.items():
        for member_id in member_ids:
            if source_id == 'EC-Earth3' or (source_id == 'MRI-ESM2-0' and member_id == 'r1i1p1f1'):
                cmip6_transfer_train_dict[source_id][member_id] = all_cmip6_forecast_dates_250years
            else:
                cmip6_transfer_train_dict[source_id][member_id] = all_cmip6_forecast_dates_180years

            cmip6_transfer_val_dict[source_id][member_id] = []

    logging.info("Creating loader")

    dataset_folder = os.path.join(config.results_folder, 'icenet2_linear_trend_input_6runs_absrad')
    dataloader = CachingDataLoader(input_data=input_data,
                                   dataset_name='dataset2',
                                   batch_size=4,
                                   shuffle=True,
                                   args.num_forecast_months=args.num_forecast_months,
                                   obs_train_dates=obs_train_dates,
                                   obs_val_dates=obs_val_dates,
                                   obs_test_dates=obs_test_dates,
                                   verbose_level=2,
                                   raw_data_shape=(432, 432),
                                   default_seed=42,
                                   dtype=np.float32,
                                   loss_weight_months=True,
                                   loss_weight_classes=False,
                                   cmip6_transfer_train_dict=cmip6_transfer_train_dict,
                                   cmip6_transfer_val_dict=cmip6_transfer_val_dict,
                                   convlstm=False,
                                   n_convlstm_input_months=12,
                                   cache_path=os.path.join(dataset_folder, "cache"))


    # TODO: We will pickle the DATASET(S), not the loader
    #pickle_path = os.path.join(dataset_folder, 'data_loader.pickle')

    #with open(pickle_path, 'wb') as writefile:
    #    pickle.dump(dataloader, writefile)
