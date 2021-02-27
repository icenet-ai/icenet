import argparse
import glob
import logging

import numpy as np
import tensorflow as tf

from icenet.architectures import unet_batchnorm

from icenet.utils import construct_custom_categorical_accuracy, categorical_focal_loss

# 8/8 [==============================] - 480s 59s/step - loss: 0.1547 - val_loss: 0.4225

# Post strategy on 4x GPU sys, batch size 8
# 100/100 [==============================] - 259s 3s/step - loss: 0.0317 - val_loss: 0.0518


def get_args():
    a = argparse.ArgumentParser()
    a.add_argument("-s", "--seed", default=42, type=int,
                   help='Random seed to use for training IceNet (defaults to 42).')
    a.add_argument("-sc", "--central-storage-strategy", dest="strategy_cs",
                   help="Use CentralStorageStrategy", action="store_true", default=False)
    a.add_argument("-b", "--batch", default=1, type=int)
    a.add_argument("-e", "--epoch", default=1, type=int)

    a.add_argument("input", help="Input directory")

    return a.parse_args()


def init():
    np.random.seed(args.seed)
    tf.random.set_seed = args.seed


def config():
    # TODO: Source from YAML - can be generated for run experimentation via init
    return dict(
        learning_rate=5e-4,
        filter_size=3,
        n_filters_factor=2.,
        n_forecast_months=6,
        weight_decay=0.,
        batch_size=4,
        dropout_rate=0.5
    )


@tf.function
def decode_item(proto):
    features = {
        "x": tf.io.FixedLenFeature([432, 432, 57], tf.float32),
        "y": tf.io.FixedLenFeature([432, 432, 4, 6], tf.float32),
    }

    item = tf.io.parse_single_example(proto, features)
    return item['x'], item['y']
#    x = np.random.random((432, 432, 57))
#    y = np.random.random((432, 432, 4, 6))
#    return x, y


def load_dataset(input, batch_size=1):
    fns = glob.glob("{}/*.tfrecord".format(input))
    train_fns, test_fns, val_fns = \
        fns[:round(len(fns) * 0.8)], \
        fns[round(len(fns) * 0.8) + 1:round(len(fns) * 0.9)], \
        fns[round(len(fns) * 0.9) + 1:]

    counts = {
        'train': len(train_fns),
        'test': len(test_fns),
        'val': len(val_fns),
    }

    train_ds, test_ds, val_ds = \
        tf.data.TFRecordDataset(train_fns), \
        tf.data.TFRecordDataset(test_fns), \
        tf.data.TFRecordDataset(val_fns)

    #TODO: Further optimisations available, input pipeline again the bottleneck with DS in place
    #TODO: Comparitive/profiling runs
    #TODO: parallel for batch size while that's small
    train_ds = train_ds.map(decode_item, num_parallel_calls=batch_size).batch(batch_size)# .shuffle(batch_size)
    test_ds = test_ds.map(decode_item, num_parallel_calls=batch_size).batch(batch_size)
    val_ds = val_ds.map(decode_item, num_parallel_calls=batch_size).batch(batch_size)

    return train_ds, test_ds, val_ds, counts


def train(cfg, train, val, counts, batch_size=1, epochs=1):
    strategy = tf.distribute.experimental.CentralStorageStrategy()

    # TODO: We'll have to diverge here

    with strategy.scope():
        network = unet_batchnorm(input_shape=[432, 432, 57],
                                 filter_size=cfg['filter_size'],
                                 n_filters_factor=cfg['n_filters_factor'],
                                 n_forecast_months=cfg['n_forecast_months'])

        network.summary()

        logging.info("Compiling network")

        # TODO: Custom training for distributing loss calculations
        # TODO: Recode categorical_focal_loss(gamma=2.)
        # TODO: Recode construct_custom_categorical_accuracy(use_all_forecast_months=True|False. single_forecast_leadtime_idx=0)
        network.compile(
            optimizer=tf.keras.optimizers.Adam(lr=cfg['learning_rate']),
            loss='MeanSquaredError',
            metrics=[])

        model_history = network.fit(train,
                                    epochs=epochs,
                                    steps_per_epoch=counts['train']/batch_size,
                                    validation_steps=counts['val']/batch_size,
                                    validation_data=val,)


if __name__ == "__main__":
    args = get_args()

    logging.getLogger().setLevel(logging.DEBUG)
    logging.info("Training UNET")

    init()
    #TODO: Clean these up
    train_ds, test_ds, val_ds, counts = load_dataset(args.input, batch_size=args.batch)
    train(config(), train_ds, val_ds, counts, batch_size=args.batch, epochs=args.epoch)


