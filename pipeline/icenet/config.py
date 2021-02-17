import os

from datetime import datetime

# TODO: Not ideal to have configuration stored like this, assess and redistribute

data_folder = 'data'
ice_data_folder = os.path.join(data_folder, 'siconca_monthly_averages')
temp_ice_data_folder = os.path.join(data_folder, 'temp-sea-ice')
era5_data_folder = os.path.join(data_folder, 'era5_raw_data')
mask_data_folder = os.path.join(data_folder, 'masks')
network_datasets_folder = os.path.join(data_folder, 'network-datasets')

results_folder = 'results'
validation_predictions_foldername = 'all-validation-predictions'
summary_figures_foldername = 'validation-summary-figures'
video_folder = 'videos'

active_grid_cell_file_format = 'active_grid_cell_mask_{}.npy'
land_mask_filename = 'land_mask.npy'
region_mask_filename = 'region_mask.npy'

data_start_date = datetime(1979, 1, 1)  # Month that data begins
data_end_date = datetime(2020, 1, 1)  # Final month of data to use + 1 month

# Pre-defined polar hole radii (in number of 25km x 25km grid cells)
polarhole1_radius = 28
polarhole2_radius = 11
polarhole3_radius = 3

# Whether or not to mask out the 3rd polar hole mask from
# Nov 2005 to Dec 2015 with a radius of only 3 grid cells. Including it creates
# some complications when analysing performance on a validation set that
# overlaps with the 3rd polar hole period.
use_polarhole3 = False

# Final month that each of the polar holes apply
# NOTE: 1st of the month chosen arbitrarily throughout as always working wit
#   monthly averages
polarhole1_final_date = datetime(1987, 6, 1)  # 1987 June
polarhole2_final_date = datetime(2005, 10, 1)  # 2005 Oct
polarhole3_final_date = datetime(2015, 12, 1)  # 2015 Dec

# missing_dates = [datetime(1986, 4, 1), datetime(1986, 5, 1),
#                  datetime(1986, 6, 1), datetime(1987, 12, 1)]

# Now using NASA Team data with filled in missing months
missing_dates = []

sic_monthly_avg_template = 'avg_sic_{}_{}.nc'
t2m_monthly_avg_template = 'avg_t2m_{}_{}.nc'
msl_monthly_avg_template = 'avg_msl_{}_{}.nc'
u10_monthly_avg_template = 'avg_u10_{}_{}.nc'
v10_monthly_avg_template = 'avg_v10_{}_{}.nc'

# .np or .npz appended to these in build_train_val_test_sets.py based on whether
# the input-output arrays are to be compressed or not
train_inputs_filename = 'train_inputs'
train_outputs_filename = 'train_outputs'
val_inputs_filename = 'val_inputs'
val_outputs_filename = 'val_outputs'
test_inputs_filename = 'test_inputs'
test_outputs_filename = 'test_outputs'

icenet_time_json_fname = 'network_time_signature.json'

network_filename = 'network'

summary_figure_format = '{}_summary_figure'
actual_vs_predicted_icenet_format = 'actual_vs_icenet_sic'
residual_vs_predicted_icenet_format = 'residual_vs_icenet_sic'
actual_vs_predicted_clim_format = 'actual_vs_clim_sic'
residual_vs_predicted_clim_format = 'residual_vs_clim_sic'
actual_given_predicted_icenet_format = 'actual_given_icenet_sic'
residual_given_predicted_icenet_format = 'residual_given_icenet_sic'
actual_given_predicted_clim_format = 'actual_given_clim_sic'
residual_given_predicted_clim_format = 'residual_given_clim_sic'
training_logs_filename = 'training_logs'

ground_truth_sic_map_format = '{}_{}_ground_truth_sic'
icenet_pred_sic_map_format = '{}_{}_pred_icenet'
ensemble_uncertainty_map_format = '{}_{}_ensemble_uncertainty'
ensemble_surprise_map_format = '{}_{}_ensemble_surprise'
icenet_clim_correction_map_format = '{}_{}_icenet_clim_correction'
clim_pred_sic_map_format = '{}_{}_pred_clim'
icenet_error_sic_map_format = '{}_{}_error_icenet'
clim_error_sic_map_format = '{}_{}_error_clim'
