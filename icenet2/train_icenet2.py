import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), 'icenet2'))
import config
import utils
import models
import losses
import metrics
import tensorflow as tf
import numpy as np
import argparse
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
import matplotlib.pyplot as plt

#### Commandline args
####################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=42, type=int)
args = parser.parse_args()

seed = args.seed

print('SEED: {}\n\n'.format(seed))

#### Weights and Biases
####################################################################

os.environ['WANDB_DIR'] = '/data/hpcdata/users/tomand/code'
os.environ['WANDB_CONFIG_DIR'] = '/data/hpcdata/users/tomand/code'
os.environ['WANDB_API_KEY'] = '812e7529d63e68066174e3ce97fffe94e1887893'

import wandb
import callbacks  # Uses wandb so env variables must be set
from wandb.keras import WandbCallback

wandb.init(project='icenet2',
           entity='tomandersson',
           dir='/data/hpcdata/users/tomand/code/icenet2')  #, config=defaults)

#### User input
####################################################################

icenet2_name = 'unet_batchnorm'
# dataloader_name = '2021_03_03_1928_icenet2_init'
dataloader_name = '2021_03_29_1437_icenet2_nh_sh'

n_epochs = 40

early_stopping_patience = 5  # No of epochs without improvement before training is aborted

# checkpoint_monitor = 'val_weighted_RMSE'
checkpoint_monitor = 'val_weighted_MAE'
checkpoint_mode = 'min'

# Miscellaneous callback inputs
# callback_batch_frequency = 2797  # One validation per epoch
# sample_callbacks_at_zero = True
# wandb_log_figure = True
# wandb_log_weights = True

max_queue_size = 5
workers = 5
use_multiprocessing = False

training_verbosity = 0

# Data loaders; set up paths
####################################################################

network_folder = os.path.join(config.folders['results'], dataloader_name, icenet2_name, 'networks')
if not os.path.exists(network_folder):
    os.makedirs(network_folder)
network_fpath = os.path.join(network_folder, 'network_{}.h5'.format(seed))

dataloader_config_fpath = os.path.join('dataloader_configs', dataloader_name+'.json')

dataloader = utils.IceNet2DataLoader(dataloader_config_fpath)

val_dataloader = utils.IceNet2DataLoader(dataloader_config_fpath)
val_dataloader.convert_to_validation_data_loader()

input_shape = (*dataloader.config['raw_data_shape'], dataloader.tot_num_channels)

#### Loss, metrics, and callbacks
####################################################################

# Loss
loss = losses.weighted_MSE

# Metrics
metrics = [metrics.weighted_MAE, metrics.weighted_RMSE, losses.weighted_MSE]

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
        patience=early_stopping_patience
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
callbacks_list.append(
    callbacks.BatchwiseWandbLogger(
        batch_frequency=len(dataloader),
        log_metrics=False,
        log_weights=False,
        log_figure=True,
        dataloader=dataloader,
        sample_at_zero=True
    ))

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
        utils.make_exp_decay_lr_schedule(rate=0.15)
    ))

#### Define model
####################################################################

network = models.unet_batchnorm(
    input_shape=input_shape,
    loss=loss,
    metrics=metrics,
    learning_rate=1e-4,
    filter_size=3,
    n_filters_factor=1,
    n_forecast_days=dataloader.config['n_forecast_days'],
)

np.random.seed(seed)
tf.random.set_seed = seed
dataloader.set_seed(seed)
dataloader.on_epoch_end()  # Randomly shuffle training samples

#### Train
####################################################################

tf.config.experimental_run_functions_eagerly(True)

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
ax.plot(history.history['val_weighted_MAE'], label='val')
ax.plot(history.history['weighted_MAE'], label='train')
ax.legend(loc='best')
plt.savefig(os.path.join(network_folder, 'network_{}_history.png'.format(seed)))
