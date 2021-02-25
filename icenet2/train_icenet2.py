import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), 'icenet2'))
import icenet2_utils
import config
import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
import wandb

wandb.init(project="icenet2")#, config=defaults)

####################################################################

icenet2_name = 'unet_batchnorm'
dataloader_name = '2021_02_25_1444_icenet2_init'

seed = 42

callback_batch_frequency = 365
sample_callbacks_at_zero = True
wandb_log_figure = True
wandb_log_weights = True
checkpoint_monitor = 'val_RMSE'
checkpoint_mode = 'min'

# Data loaders; set up paths
####################################################################

network_folder = os.path.join(config.folders['results'], dataloader_name, icenet2_name, 'networks')
if not os.path.exists(network_folder):
    os.makedirs(network_folder)
network_fpath = os.path.join(network_folder, 'network_{}.h5'.format(seed))

dataloader_fpath = os.path.join('dataloader_configs', dataloader_name+'.json')

dataloader = icenet2_utils.IceNet2DataLoader(dataloader_fpath)

val_dataloader = icenet2_utils.IceNet2DataLoader(dataloader_fpath)
val_dataloader.convert_to_validation_data_loader()

input_shape = (*dataloader.config['raw_data_shape'], dataloader.tot_num_channels)

#### Loss, metrics, and callbacks
####################################################################

# TODO: early stopping

# Loss
loss = icenet2_utils.RMSE

# Metrics
metrics = [icenet2_utils.RMSE]

# Callbacks
callbacks = []
callbacks.append(icenet2_utils.IceNetPreTrainingEvaluator(
    validation_frequency=callback_batch_frequency,
    val_dataloader=val_dataloader,
    sample_at_zero=sample_callbacks_at_zero
))

callbacks.append(icenet2_utils.BatchwiseModelCheckpoint(
    save_frequency=callback_batch_frequency,
    model_path=network_fpath,
    mode=checkpoint_mode,
    monitor=checkpoint_monitor,
    prev_best=None,
    sample_at_zero=sample_callbacks_at_zero
))

callbacks.append(icenet2_utils.BatchwiseWandbLogger(
    batch_frequency=callback_batch_frequency,
    log_weights=wandb_log_weights,
    log_figure=wandb_log_figure,
    dataloader=dataloader,
    sample_at_zero=sample_callbacks_at_zero
))

####################################################################

network = icenet2_utils.unet_batchnorm(
    input_shape=input_shape,
    loss=loss,
    metrics=metrics,
    learning_rate=1e-5,
    filter_size=3,
    # n_filters_factor=1,
    n_filters_factor=1,
    n_forecast_days=dataloader.config['n_forecast_days'],
)

np.random.seed(seed)
tf.random.set_seed = seed
dataloader.set_seed(seed)
dataloader.on_epoch_end()  # Randomly shuffle training samples

tf.config.experimental_run_functions_eagerly(True)

# TEMP reduce training and validation set size
# dataloader.all_forecast_start_dates = dataloader.all_forecast_start_dates
val_dataloader.all_forecast_start_dates = val_dataloader.all_forecast_start_dates[0:365]

print('\n\nTraining IceNet2:\n')
history = network.fit(
    dataloader,
    epochs=10,
    verbose=1,
    callbacks=callbacks,
    validation_data=val_dataloader,
    max_queue_size=5,
    workers=5,
    use_multiprocessing=True
)
