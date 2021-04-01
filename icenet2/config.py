import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), 'icenet2'))
import misc
from datetime import datetime

folders = {
    'data': 'data',
    'results': 'results',
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
corrupt_sic_days = {
    'nh': [
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
    ],
    'sh': [
        datetime(1979, 2, 5, 12),
        datetime(1979, 2, 25, 12),
        datetime(1979, 3, 23, 12),
        datetime(1979, 3, 27, 12),
        datetime(1979, 3, 29, 12),
        datetime(1979, 4, 12, 12),
        datetime(1979, 5, 16, 12),
        datetime(1979, 7, 11, 12),
        datetime(1979, 7, 13, 12),
        datetime(1979, 7, 15, 12),
        datetime(1979, 7, 17, 12),
        datetime(1979, 8, 10, 12),
        datetime(1979, 9, 3, 12),
        datetime(1980, 2, 16, 12),
        datetime(1980, 3, 15, 12),
        datetime(1980, 3, 31, 12),
        datetime(1980, 4, 22, 12),
        datetime(1981, 6, 10, 12),
        datetime(1982, 8, 6, 12),
        datetime(1983, 7, 8, 12),
        datetime(1983, 7, 10, 12),
        datetime(1983, 7, 22, 12),
        datetime(1984, 6, 12, 12),
        datetime(1984, 9, 14, 12),
        datetime(1984, 9, 16, 12),
        datetime(1984, 10, 4, 12),
        datetime(1984, 10, 6, 12),
        datetime(1984, 10, 8, 12),
        datetime(1984, 11, 19, 12),
        datetime(1984, 11, 21, 12),
        datetime(1985, 7, 23, 12),
        *misc.filled_daily_dates(datetime(1986, 7, 2, 12), datetime(1986, 11, 2)),
        datetime(1990, 8, 14, 12),
        datetime(1990, 8, 15, 12),
        datetime(1990, 8, 24, 12)
    ]
}
