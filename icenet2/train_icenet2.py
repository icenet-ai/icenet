import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), 'icenet2'))
import icenet2_utils
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd


# TEMP
import os
import sys
import warnings
import numpy as np
from datetime import datetime
sys.path.insert(0, os.path.join(os.getcwd(), 'icenet2'))  # if using jupyter kernel
import config
import icenet2_utils
from dateutil.relativedelta import relativedelta
from mpl_toolkits.axes_grid1 import make_axes_locatable
import tensorflow as tf
import xarray as xr
import pandas as pd
import regex as re
import json
import imageio
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

dataloader = icenet2_utils.IceNet2DataLoader('dataloader_configs/2021_02_24_1352_icenet2_init.json')

# TODO loss
# TODO metrics
# TODO callbacks/early stopping

network = icenet2_utils.unet_batchnorm(
    input_shape=dataloader.config['raw_data_shape'],
    loss=,
    metrics=,
    learning_rate=1e-4,
    filter_size=3,
    n_filters_factor=1,
    n_forecast_days=1,
)

# TODO training with dataloader
