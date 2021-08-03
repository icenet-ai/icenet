import argparse
import logging

import icenet2.constants as constants

def download(
        source, member, experiment, ):

def config(filename=None):
    if filename:
        raise NotImplementedError("Not retrieving from filename yet")
    # TODO:
    return [{
        "source": {
            "id": "MRI-ESM2-0",
            "servers": ["esgf-data2.diasjp.net"],
        },
        "frequency": 'd'
        "period"
    }]

        : {
            'experiment_ids': ['historical', 'ssp245'],
            'variable_dict': {
                'siconca': {
                    'include': True,
                    'table_id': 'SImon',
                    'plevels': None
                },
                'tas': {
                    'include': True,
                    'table_id': 'Amon',
                    'plevels': None
                },
                'ta': {
                    'include': True,
                    'table_id': 'Amon',
                    'plevels': [500_00]
                },
                'tos': {
                    'include': True,
                    'table_id': 'Omon',
                    'plevels': None,
                    'ocean_variable': True
                },
                'psl': {
                    'include': True,
                    'table_id': 'Amon',
                    'plevels': None
                },
                'rsus': {
                    'include': True,
                    'table_id': 'Amon',
                    'plevels': None
                },
                'rsds': {
                    'include': True,
                    'table_id': 'Amon',
                    'plevels': None
                },
                'zg': {
                    'include': True,
                    'table_id': 'Amon',
                    'plevels': [500_00, 250_00]
                },
                'uas': {
                    'include': True,
                    'table_id': 'Amon',
                    'plevels': None
                },
                'vas': {
                    'include': True,
                    'table_id': 'Amon',
                    'plevels': None
                },
                'ua': {
                    'include': True,
                    'table_id': 'Amon',
                    'plevels': [10_00]
                },
            }
        },
        'EC-Earth3': {
            'experiment_ids': ['historical', 'ssp245'],
            'data_nodes': ['esgf.bsc.es'],
            'frequency': 'mon',
            'variable_dict': {
                'siconca': {
                    'include': True,
                    'table_id': 'SImon',
                    'plevels': None
                },
                'tas': {
                    'include': True,
                    'table_id': 'Amon',
                    'plevels': None
                },
                'ta': {
                    'include': True,
                    'table_id': 'Amon',
                    'plevels': [500_00]
                },
                'tos': {
                    'include': True,
                    'table_id': 'Omon',
                    'plevels': None,
                    'ocean_variable': True
                },
                'psl': {
                    'include': True,
                    'table_id': 'Amon',
                    'plevels': None
                },
                'rsus': {
                    'include': True,
                    'table_id': 'Amon',
                    'plevels': None
                },
                'rsds': {
                    'include': True,
                    'table_id': 'Amon',
                    'plevels': None
                },
                'zg': {
                    'include': True,
                    'table_id': 'Amon',
                    'plevels': [500_00, 250_00]
                },
                'uas': {
                    'include': True,
                    'table_id': 'Amon',
                    'plevels': None
                },
                'vas': {
                    'include': True,
                    'table_id': 'Amon',
                    'plevels': None
                },
                'ua': {
                    'include': True,
                    'table_id': 'Amon',
                    'plevels': [10_00]
                },
            }
        }
    }


def cli():
    parser = argparse.ArgumentParser()

    args = parser.parse_args()
