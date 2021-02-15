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
    'missing_sic_days': 'missing_sic_days.csv',
    'polarhole1': 'polarhole1_mask.npy',
    'polarhole2': 'polarhole2_mask.npy',
}

formats = {
    'active_grid_cell_mask': 'active_grid_cell_mask_{}.npy',
}

polarhole1_final_date = datetime(1987, 6, 1)  # 1987 June
polarhole2_final_date = datetime(2005, 10, 1)  # 2005 Oct

# Pre-defined polar hole radii (in number of 25km x 25km grid cells)
polarhole1_radius = 28
polarhole2_radius = 11

missing_sic_months = [datetime(1986, 4, 1), datetime(1986, 5, 1),
                      datetime(1986, 6, 1), datetime(1987, 12, 1)]
corrupt_sic_days = [
    datetime(1979, 5, 28, 12),
    datetime(1979, 5, 30, 12),
    datetime(1979, 6, 1, 12),
    datetime(1979, 6, 3, 12),
    datetime(1979, 6, 11, 12),
    datetime(1979, 6, 13, 12),
    datetime(1979, 6, 15, 12),
    datetime(1979, 6, 17, 12),
    datetime(1979, 6, 19, 12),
    datetime(1979, 6, 21, 12),
    datetime(1979, 6, 23, 12),
    datetime(1979, 6, 25, 12),
    datetime(1979, 7, 1, 12),
    datetime(1979, 7, 25, 12),
    datetime(1979, 7, 27, 12),
    datetime(1984, 9, 14, 12),
    datetime(1987, 1, 16, 12),
    datetime(1987, 1, 18, 12),
    datetime(1987, 1, 30, 12),
    datetime(1987, 2, 1, 12),
    datetime(1987, 2, 23, 12),
    datetime(1987, 2, 27, 12),
    datetime(1987, 3, 1, 12),
    datetime(1987, 3, 13, 12),
    datetime(1987, 3, 23, 12),
    datetime(1987, 3, 25, 12),
    datetime(1987, 4, 4, 12),
    datetime(1987, 4, 6, 12),
    datetime(1987, 4, 10, 12),
    datetime(1987, 4, 12, 12),
    datetime(1987, 4, 14, 12),
    datetime(1987, 4, 16, 12),
    datetime(1987, 4, 4, 12),
    datetime(1990, 1, 26, 12)
]
