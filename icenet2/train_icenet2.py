import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), 'icenet2'))
import pprint
import config
import utils
import models
import losses
import metrics
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import argparse
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
import matplotlib.pyplot as plt

#### Commandline args
####################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--filter_size', default=3, type=int)
parser.add_argument('--n_filters_factor', default=2, type=float)
parser.add_argument('--lr_10e_decay_fac', default=0.5, type=float)
parser.add_argument('--lr_decay_start', default=10, type=int)
parser.add_argument('--lr_decay_end', default=30, type=int)
parser.add_argument("--nowandb", help="Don't use Weights and Biases", default=False, action="store_true")
args = parser.parse_args()

#### Weights and Biases
####################################################################

os.environ['WANDB_DIR'] = '/data/hpcdata/users/tomand/code'
os.environ['WANDB_CONFIG_DIR'] = '/data/hpcdata/users/tomand/code'
os.environ['WANDB_API_KEY'] = '812e7529d63e68066174e3ce97fffe94e1887893'

if args.nowandb:
    wandb_mode = 'disabled'
else:
    wandb_mode = 'online'

import wandb
import callbacks  # Uses wandb so env variables must be set
from wandb.keras import WandbCallback

defaults = dict(
    seed=args.seed,
    lr=args.lr,
    filter_size=args.filter_size,
    n_filters_factor=args.n_filters_factor,
    lr_10e_decay_fac=args.lr_10e_decay_fac,
    lr_decay_start=args.lr_decay_start,
    lr_decay_end=args.lr_decay_end,
)

wandb.init(
    project='icenet2',
    entity='tomandersson',
    dir='/data/hpcdata/users/tomand/code/icenet2',
    config=defaults,
    allow_val_change=True,
    mode=wandb_mode,
)

wandb.config.update({'lr_decay': -0.1*np.log(wandb.config.lr_10e_decay_fac)})

print('SEED: {}\n\n'.format(wandb.config.seed))

print('\n\nHyperparams:')
pprint.pprint(dict(wandb.config))
print('\n\n')

#### User input
####################################################################

icenet2_name = 'unet_batchnorm'
# icenet2_name = 'unet_batchnorm_large'
# dataloader_name = '2021_03_03_1928_icenet2_init'
# dataloader_name = '2021_04_06_1709_icenet2_nh_sh_thinned5_weeklyinput_wind'
# dataloader_name = '2021_04_08_1205_icenet2_nh_sh_thinned5_weeklyinput_wind_3month'
dataloader_name = '2021_04_25_1351_icenet2_nh_thinned7_weeklyinput_wind_3month'

pre_load_network = False
pre_load_network_fname = 'network_{}.h5'.format(wandb.config.seed)  # From transfer learning
custom_objects = {
    'weighted_MAE_corrected': metrics.weighted_MAE_corrected,
    'weighted_RMSE_corrected': metrics.weighted_RMSE_corrected,
    'weighted_MSE_corrected': losses.weighted_MSE_corrected,
}
# prev_best = 11.05  # Baseline 'monitor' value for EarlyStopping callback
prev_best = None

n_epochs = 35

# early_stopping_patience = 5
early_stopping_patience = n_epochs  # No of epochs without improvement before training is aborted

# checkpoint_monitor = 'val_weighted_RMSE'
checkpoint_monitor = 'val_weighted_MAE_corrected'
checkpoint_mode = 'min'

# Miscellaneous callback inputs
# callback_batch_frequency = 2797  # One validation per epoch
# sample_callbacks_at_zero = True
# wandb_log_figure = True
# wandb_log_weights = True

max_queue_size = 3
workers = 5
use_multiprocessing = True

training_verbosity = 2

# Data loaders; set up paths
####################################################################

network_folder = os.path.join(config.folders['results'], dataloader_name, icenet2_name, 'networks')
if not os.path.exists(network_folder):
    os.makedirs(network_folder)
network_fpath = os.path.join(network_folder, 'network_{}.h5'.format(wandb.config.seed))
network_path_preload = os.path.join(network_folder, pre_load_network_fname)

dataloader_config_fpath = os.path.join('dataloader_configs', dataloader_name+'.json')

dataloader = utils.IceNet2DataLoader(dataloader_config_fpath, wandb.config.seed)

val_dataloader = utils.IceNet2DataLoader(dataloader_config_fpath)
val_dataloader.convert_to_validation_data_loader()

input_shape = (*dataloader.config['raw_data_shape'], dataloader.tot_num_channels)

print('\n\nNUM TRAINING SAMPLES: {}'.format(len(dataloader.all_forecast_IDs)))
print('NUM VALIDATION SAMPLES: {}\n\n'.format(len(val_dataloader.all_forecast_IDs)))
print('NUM INPUT CHANNELS: {}\n\n'.format(dataloader.tot_num_channels))

#### Loss, metrics, and callbacks
####################################################################

# Loss
loss = losses.weighted_MSE_corrected
# loss = losses.weighted_MSE

# Metrics
metrics_list = [
    # metrics.weighted_MAE,
    metrics.weighted_MAE_corrected,
    metrics.weighted_RMSE_corrected,
    losses.weighted_MSE_corrected
]

# Callbacks
callbacks_list = []

# Run validation every N batches
# callbacks_list.append(
#     callbacks.IceNetPreTrainingEvaluator(
#         validation_frequency=callback_batch_frequency,
#         val_dataloader=val_dataloader,
#         sample_at_zero=sample_callbacks_at_zero
#     ))

# Checkpoint the model weights when a validation metric is improved
callbacks_list.append(
    ModelCheckpoint(
        filepath=network_fpath,
        monitor=checkpoint_monitor,
        verbose=1,
        mode=checkpoint_mode,
        save_best_only=True
    ))

# Abort training when validation performance stops improving
callbacks_list.append(
    EarlyStopping(
        monitor=checkpoint_monitor,
        mode=checkpoint_mode,
        verbose=1,
        patience=early_stopping_patience,
        baseline=prev_best
    ))

# Checkpoint the model every N batches (must sync with batchwise validation)
# callbacks_list.append(
#     callbacks.BatchwiseModelCheckpoint(
#         save_frequency=callback_batch_frequency,
#         model_path=network_fpath,
#         mode=checkpoint_mode,
#         monitor=checkpoint_monitor,
#         prev_best=None,
#         sample_at_zero=sample_callbacks_at_zero
#     ))

# Log a figure to wandb each epoch illustrating IceNet2 improvement over time
# callbacks_list.append(
#     callbacks.BatchwiseWandbLogger(
#         batch_frequency=len(dataloader),
#         log_metrics=False,
#         log_weights=False,
#         log_figure=True,
#         dataloader=dataloader,
#         sample_at_zero=False
#     ))

# Log training metrics to wandb each epoch
callbacks_list.append(
    WandbCallback(
        monitor=checkpoint_monitor,
        mode=checkpoint_mode,
        save_model=False
    ))

# Learning rate schedule with exponential decay
callbacks_list.append(
    LearningRateScheduler(
        utils.make_exp_decay_lr_schedule(
            rate=wandb.config.lr_decay,
            start_epoch=wandb.config.lr_decay_start,
            end_epoch=wandb.config.lr_decay_end,
        )))

#### Define model
####################################################################

if pre_load_network:
    print("\nLoading network from {}... ".format(network_path_preload), end='', flush=True)
    network = load_model(network_path_preload, custom_objects=custom_objects)
    print('Done.\n')

else:

    network = models.unet_batchnorm(
        input_shape=input_shape,
        loss=loss,
        metrics=metrics_list,
        learning_rate=wandb.config.lr,
        filter_size=wandb.config.filter_size,
        n_filters_factor=wandb.config.n_filters_factor,
        n_forecast_days=dataloader.config['n_forecast_days'],
    )

    print('\nNETWORK ARCH:')
    print(network.summary())
    print('\n\n\n')

np.random.seed(wandb.config.seed)
tf.random.set_seed = wandb.config.seed
dataloader.set_seed(wandb.config.seed)
dataloader.on_epoch_end()  # Randomly shuffle training samples

#### Train
####################################################################

# tf.config.experimental_run_functions_eagerly(True)

print('\n\nTraining IceNet2:\n')
history = network.fit(
    dataloader,
    epochs=n_epochs,
    verbose=training_verbosity,
    callbacks=callbacks_list,
    validation_data=val_dataloader,
    max_queue_size=max_queue_size,
    workers=workers,
    use_multiprocessing=use_multiprocessing
)

fig, ax = plt.subplots()
ax.plot(history.history['val_weighted_MAE_corrected'], label='val')
ax.plot(history.history['weighted_MAE_corrected'], label='train')
ax.legend(loc='best')
plt.savefig(os.path.join(network_folder, 'network_{}_history.png'.format(wandb.config.seed)))
