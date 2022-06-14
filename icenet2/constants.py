import os

"""

# TODO: any actual usages left?
"""

FOLDERS = {
    # TODO: add cache - data is not really the dataset for the model
    'data': 'data',
    'results': 'results',
    'masks': os.path.join('data', 'masks'),
    'siconca': os.path.join('data', 'siconca'),
    'figures': 'figures',
}

FILENAMES = {
    'land_mask': 'land_mask.npy',
    'missing_sic_days': 'missing_sic_days.csv',
}


__all__ = [
    "FILENAMES", "FOLDERS"
]
