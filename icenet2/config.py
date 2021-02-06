import os
from datetime import datetime

folders = {
    'data': 'data',
    'masks': os.path.join('data', 'masks'),
    'siconca': os.path.join('data', 'siconca'),
    'figures': 'figures',
}

for folder in folders.values():
    if not os.path.exists(folder):
        os.makedirs(folder)

fnames = {
    'land_mask': 'land_mask.npy',
    'polarhole1': 'polarhole1_mask.npy',
    'polarhole2': 'polarhole2_mask.npy',
    'polarhole3': 'polarhole3_mask.npy'
}

formats = {
    'active_grid_cell_mask': 'active_grid_cell_mask_{}.npy',
}

polarhole1_final_date = datetime(1987, 6, 1)  # 1987 June
polarhole2_final_date = datetime(2005, 10, 1)  # 2005 Oct
polarhole3_final_date = datetime(2015, 12, 1)  # 2015 Dec

# Pre-defined polar hole radii (in number of 25km x 25km grid cells)
polarhole1_radius = 28
polarhole2_radius = 11
polarhole3_radius = 3

missing_sic_months = [datetime(1986, 4, 1), datetime(1986, 5, 1),
                      datetime(1986, 6, 1), datetime(1987, 12, 1)]
