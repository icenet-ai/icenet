import numpy as np
import xarray as xr
import os
import shutil
import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), 'icenet2'))  # if using jupyter kernel
import config
import matplotlib.pyplot as plt
import argparse

################################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--hemisphere', default='nh', type=str)
args = parser.parse_args()

print('HEMISPHERE: {}\n\n'.format(args.hemisphere.upper()))

###############################################################################

save_land_mask = True  # Save the land mask (constant across months)
save_polarhole_masks = False  # Save the polarhole masks
save_figures = True  # Figures of the max extent masks

if save_figures:
    fig_folder = os.path.join('figures', 'max_extent_masks', args.hemisphere)
    if not os.path.exists(fig_folder):
        os.makedirs(fig_folder)

siconca_data_folder = os.path.join('data', args.hemisphere, 'siconca')
if not os.path.exists(siconca_data_folder):
    os.makedirs(siconca_data_folder)

mask_folder = os.path.join('data', args.hemisphere, 'masks')
if not os.path.exists(mask_folder):
    os.makedirs(mask_folder)

retrieve_cmd_template_osi450 = 'wget -m -nH --cut-dirs=4 -P ' + siconca_data_folder + \
    ' ftp://osisaf.met.no/reprocessed/ice/conc/v2p0/{:04d}/{:02d}/' + '{}'
filename_template_osi450 = 'ice_conc_{}_ease2-250_cdr-v2p0_{:04d}{:02d}021200.nc'

###############################################################################

#### Generate the land-lake-sea mask using the second day from each month of
#### the year 2000 (chosen arbitrarily as the mask is fixed within month)

year = 2000

for month in range(1, 13):

    # Download the data if not already downloaded
    filename_osi450 = filename_template_osi450.format(args.hemisphere, year, month)
    os.system(retrieve_cmd_template_osi450.format(year, month, filename_osi450))

    year_str = '{:04d}'.format(year)
    month_str = '{:02d}'.format(month)
    month_folder = os.path.join(siconca_data_folder, year_str, month_str)

    day_path = os.path.join(month_folder, filename_osi450)

    with xr.open_dataset(day_path) as ds:
        status_flag = ds['status_flag']
        status_flag = np.array(status_flag.data).astype(np.uint8)
        status_flag = status_flag.reshape(432, 432)

        binary = np.unpackbits(status_flag, axis=1).reshape(432, 432, 8)

        # Mask out: land, lake, and 'outside max climatology' (i.e. open sea)
        max_extent_mask = np.sum(binary[:, :, [7, 6, 0]], axis=2).reshape(432, 432) >= 1
        max_extent_mask = ~max_extent_mask  # False outside of max extent
        max_extent_mask[325:386, 317:380] = False  # Remove Caspian and Black seas

    mask_filename = config.formats['active_grid_cell_mask'].format(month_str)
    mask_path = os.path.join(mask_folder, mask_filename)
    np.save(mask_path, max_extent_mask)

    if save_land_mask and month == 1:
        land_mask = np.sum(binary[:, :, [7, 6]], axis=2).reshape(432, 432) >= 1

        land_mask_path = os.path.join(mask_folder, config.fnames['land_mask'])
        np.save(land_mask_path, land_mask)

    if save_figures:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(max_extent_mask, cmap='Blues_r')
        ax.contour(land_mask, colors='k', linewidths=0.3)
        plt.savefig(os.path.join(fig_folder, month_str + '.png'))
        plt.close()

# Delete the data/siconca/2000 folder holding the temporary daily files
shutil.rmtree(os.path.join(siconca_data_folder, year_str))

if save_polarhole_masks:
    #### Generate the polar hole masks
    x = np.tile(np.arange(0, 432).reshape(432, 1), (1, 432)).astype(np.float32) - 215.5
    y = np.tile(np.arange(0, 432).reshape(1, 432), (432, 1)).astype(np.float32) - 215.5
    squaresum = np.square(x) + np.square(y)

    # Jan 1979 - June 1987
    polarhole1 = np.full((432, 432), False)
    polarhole1[squaresum < config.polarhole1_radius**2] = True
    np.save(os.path.join(config.folders['masks'], config.fnames['polarhole1']), polarhole1)

    # July 1987 - Oct 2005
    polarhole2 = np.full((432, 432), False)
    polarhole2[squaresum < config.polarhole2_radius**2] = True
    np.save(os.path.join(config.folders['masks'], config.fnames['polarhole2']), polarhole2)

    # Nov 2005 - Dec 2015
    polarhole3 = np.full((432, 432), False)
    polarhole3[squaresum < config.polarhole3_radius**2] = True
    np.save(os.path.join(config.folders['masks'], config.fnames['polarhole3']), polarhole3)
