#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 10:41:08 2024

@author: anais

Functions allowing to extract the processed data from folder `data/processed/`.

If the folder is empty, or if parameters where changed, run
`makeFile/processed_dataset.py` to update it.

"""

import numpy as np
import os.path


ABSOLUTE_PATH = os.path.abspath(__file__)  # Path to extract_processed_dataset.
CURRENT_DIR = os.path.dirname(ABSOLUTE_PATH)  # Path to auxiliary directory.
PARENT_DIR = os.path.dirname(CURRENT_DIR)  # Path to telomeres directory.
PROJECT_DIR = os.path.dirname(PARENT_DIR)

DIR = os.path.join(PROJECT_DIR, 'data', 'processed')


def write_parameters_linit(par_linit):
    if isinstance(par_linit, type(None)):
        return 'linit_variable'
    ltrans, l0, l1 = np.round(par_linit).astype(int)
    return f'linit{ltrans}-{l0}-{l1}'


def extract_postreat_lineages(strain='TetO2-TLC1'):
    path = os.path.join(DIR, f'cycles_{strain}', 'LINEAGES_POSTREATED.npy')
    return tuple(np.load(path, allow_pickle='TRUE'))


def extract_distribution_telomeres_init(par_l_init=[0, 0, 0]):
    if np.all(np.array(par_l_init) == np.zeros(3)):
        path = os.path.join(DIR, 'telomeres_initial_distribution',
                            'original.csv')
    else:
        msg = write_parameters_linit(par_l_init)
        path = os.path.join(DIR, 'telomeres_initial_distribution',
                            f'modified_{msg}.csv')
    return np.transpose(np.genfromtxt(path, delimiter=',', skip_header=1))


# def extract_distribution_telomeres_init_full(par_l_init=[0, 0, 0]):
#     if np.all(np.array(par_l_init) == np.zeros(3)):
#         path = os.path.join(DIR, 'telomeres_initial_distribution',
#                             'original_full.csv')
#     else:
#         msg = write_parameters_linit(par_l_init)
#         path = os.path.join(DIR, 'telomeres_initial_distribution',
#                             f'original_full_modified_{msg}.csv')
#     return np.transpose(np.genfromtxt(path, delimiter=',', skip_header=1))


def extract_distribution_cycles():
    path = os.path.join(DIR, 'cycles_TetO2-TLC1',
                        'EMPIRICAL_DISTRIBUTIONS.npy')
    cdts = np.load(path, allow_pickle='TRUE').item()
    return cdts


def extract_population_lmode():
    path = os.path.join(DIR, 'population_evolution',
                        'telomere_lengths_DOX+.csv')
    return np.genfromtxt(path, delimiter=',')


def extract_population_concentration_doxP():
    path = os.path.join(DIR, 'population_evolution',
                        'cell_concentration_DOX+.npy')
    return np.load(path, allow_pickle='TRUE').item()


def extract_population_concentration_doxM():
    path = os.path.join(DIR, 'population_evolution',
                        'cell_concentration_DOX-.npy')
    return np.load(path, allow_pickle='TRUE').item()


def extract_population_concentration_pol32():
    keys = ['tlc1', 'tlc1_pol32']
    return {key:
            np.load(os.path.join(DIR.replace('data', 'data_ignored'),
                                 f'cell_concentration_{key}.npy'),
                    allow_pickle='TRUE').item() for key in keys}

def extract_population_concentration_rad51():
    keys = ['RAD51', 'RAD51_sep', 'RAD51_oct']
    return {key:
            np.load(os.path.join(DIR.replace('data', 'data_ignored'),
                                 f'cell_concentration_{key}.npy'),
                    allow_pickle='TRUE').item() for key in keys}
