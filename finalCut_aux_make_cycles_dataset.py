#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 18:31:14 2023

@author: arat
"""

import os
import numpy as np
import pandas as pd
import scipy.io as sio

import aux_functions as af
import lineage_simulation as sim



def extract_cycles_dataset(data, gal_delay=30, dox_frame_idx=None,
                           sfolder=None):
    """ Extract and return from `data` the cell cycles, from strictly after
    DOX addition to the end of measurements, of all the lineages longer than
    `gen_count_by_lineage_min`. Besides, compute and return information on each
    lineage (generations at which event triggered and type).

    """
    lineage_count = len(data['endlineage'][0])

    # Array, for each lineage, of the time at which DOX has been added.
    dox_time = dox_frame_idx or 0
    gal_times = data['DOXaddition'][0].astype(int)
    nogal_times = gal_times + gal_delay

    # Extraction of cycles duration times after DOX addition saving some info.
    # > Initialization of lists:
    # cycles_dox = [] # `cycles[i]`: array of cycle durations of the ith lineage.
    cycles = {'dox': [], 'gal': [], 'raf': [], 'raf_dox': []}

    # > Iteration on all lineages.
    for i in range(lineage_count):
        # Extration of times of division/death (except last one) in ith lin.
        # div_times = data['track'][0, i].astype(int)[:, :2]
        # birth_times = np.append(np.zeros((len(div_times), 1)),
        #                         div_times[:, :-1], axis=1)

        birth_times = data['track'][0, i].astype(int)[:, 0]
        div_times = data['track'][0, i].astype(int)[:, 1]
        # We keep only times after Dox addition.
        # is_dox = div_times[:, 0] > dox_time
        is_dox = birth_times >= dox_time
        # div_times_dox = div_times[is_dox]
        # Or during Galactose addition.
        # is_gal = np.logical_and(div_times[:, 0] > gal_times[i],
        #                         birth_times[:, 0] < nogal_times[i])
        # # Option 1: Gal-cycle as soon as there is Gal in a part of the cycle.
        # is_gal = np.logical_or(np.logical_and(birth_times >= gal_times[i],
        #                                       birth_times < nogal_times[i]),
        #                        np.logical_and(div_times > gal_times[i],
        #                                       div_times <= nogal_times[i]))
        # Option 2: Gal-cycle cycles that starts with Gal.
        is_gal = np.logical_and(birth_times >= gal_times[i],
                                birth_times < nogal_times[i])

        # div_times_gal = div_times[is_gal, :]
        # Or when ther is no Galactose.
        # div_times_raf = div_times[~is_gal, :]
        is_raf_dox = np.logical_and(~is_gal, is_dox)
        # div_times_raf_dox = div_times[is_raf_dox, :]
        # We turn "times of division" to cycle duration times.
        # cycles['dox'].extend(div_times_dox[:, 1] - div_times_dox[:, 0])
        # cycles['gal'].extend(div_times_gal[:, 1] - div_times_gal[:, 0])
        # cycles['raf'].extend(div_times_raf[:, 1] - div_times_raf[:, 0])
        # cycles['raf_dox'].extend(div_times_raf_dox[:, 1] -
        #                            div_times_raf_dox[:, 0])
        cycles['dox'].extend(div_times[is_dox] - birth_times[is_dox])
        cycles['gal'].extend(div_times[is_gal] - birth_times[is_gal])
        cycles['raf'].extend(div_times[~is_gal] - birth_times[~is_gal])
        cycles['raf_dox'].extend(div_times[is_raf_dox] - birth_times[is_raf_dox])
        # print(div_times[:, 1] - div_times[:, 0])
        # print('dox', div_times_dox[:, 1] - div_times_dox[:, 0])
        # print('gal', div_times_gal[:, 1] - div_times_gal[:, 0])
        # print('raf', div_times_raf[:, 1] - div_times_raf[:, 0])
        # print('raf_dox', div_times_raf_dox[:,1] - div_times_raf_dox[:,0])

    # Saving.
    if not isinstance(sfolder, type(None)):
        if not os.path.exists(sfolder):
            os.makedirs(sfolder)
        for key, cdts in cycles.items():
            name = f'cycles_{key}.csv'
            pd.DataFrame(cdts).to_csv(sfolder + name, header=None, index=None)
    return cycles

def compute_cycle_threshold(cycles, is_printed=False, key=''):
    cst = 2
    if 'raf' in key:
        cst = 2
    threshold = np.mean(cycles) + cst * np.std(cycles)
    if is_printed:
        print("Threshold: ", threshold)
        print("Mean: ", np.mean(cycles))
        print("STD: ", np.std(cycles), '\n')
    return threshold

def compute_cycle_thresholds(cycles, is_printed=False):
    thresholds ={}
    for key, cdts in cycles.items():
        if is_printed:
            print(key)
        thresholds[key] = compute_cycle_threshold(cdts, is_printed, key)
    return thresholds

def gather_cycle_dataset(folders_to_gather, sfolder=None,
                         keys=['dox', 'gal', 'raf', 'raf_dox'],
                         is_printed=False):
    cycles = {key: [] for key in keys}
    # Extracting and concatening.
    for folder in folders_to_gather:
        for key in keys:
            cycles[key].extend(np.loadtxt(folder + f'cycles_{key}.csv'))
    if is_printed:
        for key in keys:
            if is_printed:
                print(key)
            compute_cycle_threshold(cycles[key], is_printed, key)
    # Saving.
    if not isinstance(sfolder, type(None)):
        if not os.path.exists(sfolder):
            os.makedirs(sfolder)
        for key, cdts in cycles.items():
            name = f'cycles_{key}.csv'
            pd.DataFrame(cdts).to_csv(sfolder + name, header=None, index=None)
    return cycles

def make_cycles_dataset(folder, thresholds=None, is_printed=False):
    cycles, ncycles, lcycles = {}, {}, {}
    if isinstance(thresholds, type(None)):
        thresholds = {'gal': None, 'raf_dox': None}

    # Extracting and concatening.
    for key in ['gal', 'raf_dox']:
        cycles[key] = np.loadtxt(folder + f'cycles_{key}.csv')
        if isinstance(thresholds[key], type(None)):
            thresholds[key] = compute_cycle_threshold(cycles[key], is_printed,
                                                      key)
        ncycles[key] = cycles[key][cycles[key] <= thresholds[key]]
        lcycles[key] = cycles[key][cycles[key] > thresholds[key]]

        # Saving.
        if not isinstance(folder, type(None)):
            name = f'cycles_{key}.csv'
            pd.DataFrame(ncycles[key]).to_csv(folder + 'n' + name, header=None,
                                              index=None)
            pd.DataFrame(lcycles[key]).to_csv(folder + 'l' + name, header=None,
                                              index=None)
    if is_printed:
        print(thresholds)
    return ncycles, lcycles

def gather_postreated_output(output_1, output_2):
    # ({'cycle': cycles}, gtrigs, lineage_types, is_unseen_htypes,
    #      is_accidental_deaths, lcycle_per_seq_counts)
    gen1 = len(output_1[0]['cycle'][0])
    gen2 = len(output_2[0]['cycle'][0])
    gen = max(gen1, gen2)
    cycles1 = af.reshape_with_nan(output_1[0]['cycle'], gen, axis=-1)
    cycles2 = af.reshape_with_nan(output_2[0]['cycle'], gen, axis=-1)
    output = [{'cycle': np.append(cycles1, cycles2, 0)}]

    gtrigs = {}
    for key in output_1[1]:
        gtrigs1 = output_1[1][key]
        gtrigs2 = output_2[1][key]
        if key == 'nta':
            nta_count = max(len(gtrigs1[0]), len(gtrigs2[0]))
            gtrigs1 = af.reshape_with_nan(gtrigs1, nta_count, axis=-1)
            gtrigs2 = af.reshape_with_nan(gtrigs2, nta_count, axis=-1)
        gtrigs[key] = np.append(gtrigs1, gtrigs2, 0)
    output.append(gtrigs)

    for i in range(2, 5):
        if isinstance(output_1[i], type(None)):
            output.append(None)
        else:
            output.append(np.append(output_1[i], output_2[i], 0))

    lc_per_seq = {}
    for key in output_1[5]:
        lc_per_seq1 = output_1[1][key]
        lc_per_seq2 = output_2[1][key]
        if key == 'nta':
            lc_per_seq1 = af.reshape_with_nan(lc_per_seq1, nta_count, axis=-1)
            lc_per_seq2 = af.reshape_with_nan(lc_per_seq2, nta_count, axis=-1)
        lc_per_seq[key] = np.append(lc_per_seq1, lc_per_seq2, 0)
    output.append(lc_per_seq)

    return output

def print_if_true(string_to_print, is_printed):
    if is_printed:
        print(string_to_print)


# Data information
# ----------------

# If True data is saved.
IS_SAVED = True
if __name__ == "__main__":
    IS_PRINTED = True
else:
    IS_PRINTED = False


# Folders containing the data.
SUBF = 'data_finalCut/uFluidicData_maths_23-11-30'
FOLDERS = {'noFc_noCas9_noDox': f'{SUBF}/noFc/noCas9/noDox/20220901_Gal82/',
           'noFc_1': f'{SUBF}/noFc/Cas9/Dox/20211202_CTL_Gal259_PB/',
           'noFc_2': f'{SUBF}/noFc/Cas9/Dox/20220223_CTL_GalDox103_PB/',
           'noFc_n2': f'{SUBF}/noFc/Cas9/Dox/sum_n2/',
           #
           'Fc0_1': f'{SUBF}/TG0/20211202_Fc0_Gal259_PB/',
           'Fc0_2': f'{SUBF}/TG0/20220223_Fc0_Gal103-134_PB/',
           'Fc0_n2': f'{SUBF}TG0/Fc0_sum_n2_Gal259/',
           #
           'Fc20_1': f'{SUBF}/TG20/Dox/20220223_Fc20_GalDox103-134/',
           'Fc20_2': f'{SUBF}/TG20/Dox/20220624_Fc20_Gal108-159/',
           'Fc20_n2': f'{SUBF}TG20/Dox/Fc20_sum2/',
           #
           'Fc30_1': f'{SUBF}/TG30/Dox/20200626_Fc30_Gal120-159_PB/',
           'Fc30_2': f'{SUBF}/TG30/Dox/20200709_Fc30_Gal122-159_PB/',
           'Fc30_n2': f'{SUBF}TG30/Dox/Fc30_sum2/',
           #
           'Fc40_1': f'{SUBF}/TG40/20201125_Fc40_Gal119_PB/',
           'Fc40_2': f'{SUBF}/TG40/20210806_Fc40_Gal129-160_PB/',
           'Fc40_n2': f'{SUBF}TG40/FC40_Sum2/',
           #
           'Fc50_1': f'{SUBF}/TG50/20200626_Fc50_Gal120_PB/',
           'Fc50_2': f'{SUBF}/TG50/20210806_Gal129_PB/',
           'Fc50_n2': f'{SUBF}TG50/TG50_n2/',
           #
           'Fc70_1': f'{SUBF}/TG70/20210806_Fc70_DoxGal129-160_PB/',
           'Fc70_2': f'{SUBF}/TG70/20230701_Fc70_DoxGal134-172/',
           'Fc70_n2': f'{SUBF}TG70/Fc70_n2/'
           }

# For each experiment: indexes of the frames (i.e. times [10 min]) at which Dox
# was added and Gal added and removed.
IDXS_CDT = {# [t_Dox, t_Gal, t_noGal]
            'noFc_noCas9_noDox': [None, 82, 121],
            'noFc_1': [222, 259, 298],
            'noFc_2': [71, 103, 134],
            'Fc0_1': [222, 259, 298],
            'Fc0_2': [71, 103, 134],
            'Fc20_1': [71, 103, 134],
            'Fc20_2': [74, 108, 159],
            'Fc30_1': [91, 120, 159],
            'Fc30_2': [93, 122, 159],
            'Fc40_1': [88, 119, 154],
            'Fc40_2': [89, 129, 160],
            'Fc50_1': [91, 120, 159],
            'Fc50_2': [89, 129, 160],
            'Fc70_1': [89, 129, 160],
            'Fc70_2': [100, 134, 172]
            }


# Threshold between normal/long cycles and associated datasets
# ------------------------------------------------------------

# Two experimental conditions:
# - Raffinose
# - Galactose

# For each condition we compute:
# 1) A threshold D between normal (<= D) and long (> D) cycles
# 2) A dataset of normal cycles and a dataset of long cycles
#    based on the data of experiment 'noFc_noCas9_noDox'
#    ('noFc_1' and 'noFc_2' for comparison only).
KEYS = ['noFc_1', 'noFc_2', 'noFc_noCas9_noDox']

# Extraction of CDTs in each dataset separately.
CYCLES_EXP, thresholds = {}, {}
for key in KEYS:
    print_if_true('\n' + key, IS_PRINTED)
    path = FOLDERS[key] + 'results_analysis.mat'
    out = sio.loadmat(path)['results_analysis']
    idx_dox, idx_gal, idx_nogal = IDXS_CDT[key]
    if IS_SAVED:
        sfolder = f'data_finalCut/{key}/'
    else:
        sfolder = None
    cycles = extract_cycles_dataset(out, gal_delay=idx_nogal - idx_gal,
                                    dox_frame_idx=idx_dox, sfolder=sfolder)
    CYCLES_EXP[key] = cycles
    # Print thresholds in each dataset for information.
    thresholds[key] = compute_cycle_thresholds(cycles, IS_PRINTED)
    print("Number of cycles in the dataset 'gal'", len(cycles['gal']))
    print("Number of cycles in the dataset 'raf_dox'", len(cycles['raf_dox']))


# Gather both datasets to make the dataset of 'noFC_n2'.
print_if_true('\n' + 'noFc_n2', IS_PRINTED)
out = gather_cycle_dataset(['data_finalCut/noFc_1/','data_finalCut/noFc_2/'],
                           'data_finalCut/noFc_n2/', is_printed=IS_PRINTED)

# # 1) Computation of the thresholds based on the dataset of 'noFC_n2'.
# THRESHOLDS = compute_cycle_thresholds(cycles, IS_PRINTED)

# 1) Computation of the thresholds based on the dataset of 'noFc_noCas9_noDox'.
THRESHOLDS = thresholds['noFc_noCas9_noDox']

# 2) Creation of the datasets of normal/long cycles based on the thresholds.
NCYCLES, LCYCLES = make_cycles_dataset('data_finalCut/noFc_n2/',
                                       thresholds=THRESHOLDS,
                                       is_printed=IS_PRINTED)


# Postreatment: generations of arrests (based on THRESHOLDS) and cycles
# ---------------------------------------------------------------------

DATA_EXP, DATA_EXP_SEN, GSEN_EXP, CDTS_EXP = {}, {}, {}, {}

# Iteration on all experiments with Dox.
for key in list(IDXS_CDT.keys()):
    if key != 'noFc_noCas9_noDox':
        # Postreat.
        print(key)
        path = FOLDERS[key] + "results_analysis.mat"
        DATA_EXP[key] = sim.postreat_experimental_lineages_from_path(path,
                            'results_analysis',  threshold=THRESHOLDS['raf'],
                            gcount_min=1,
                            par_multiple_thresholds=[THRESHOLDS['gal'],
                                                     IDXS_CDT[key][2]])
        # DATA_EXP[key] = out
        # # Save.
        # np.save(FOLDERS[key] + "postreat.npy", out)

# Gather postreated data of replicated experiments.
for key in ['noFc_n2', 'Fc0_n2', 'Fc20_n2', 'Fc30_n2', 'Fc40_n2', 'Fc50_n2',
            'Fc70_n2']:
    key_1 = key.replace('_n2', '_1')
    key_2 = key.replace('_n2', '_2')
    DATA_EXP[key] = gather_postreated_output(DATA_EXP[key_1], DATA_EXP[key_2])
    # np.save(FOLDERS[key] + "postreat.npy", DATA_EXP[key])
