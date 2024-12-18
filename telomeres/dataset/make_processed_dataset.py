#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 12:51:00 2024

@author: anais

Functions allowing to extract and possibly post-treat the data contained in
`data/raw` and save it, with standardized format, in `data/processed`.

This functions are called by `makeFile/processed_dataset.py`.

WARNING. Outputs of function `make_microfluidic_dataset` depend on the
parameters contained define here through `PAR_CYCLES_POSTREAT`. If one modify
this variable is modified, he/she will need to recreate the data of
`data/processed`.by running `makeFile/processed_dataset.py` to make the changes
effective. This has no effect on simulations, only on the experimental data.

"""

import os
import numpy as np
import pandas as pd
import scipy.io as sio

from telomeres.dataset.extract_processed_dataset import \
    write_parameters_linit, extract_distribution_telomeres_init
from telomeres.lineage.posttreat import postreat_experimental_lineages, \
    is_as_expected_lineages


# Parameters for the initial distribution of telomere lengths
# -----------------------------------------------------------
from telomeres.model.parameters import PAR_L_INIT


# Parameters for cycle duration times postreatment
# ------------------------------------------------

# Threshold between long (>) and normal (<=) cycles [10 min].
THRESHOLD = 18  # (Martin et al. 2021)

# Minimal duration of cell cycles in the postreated dataset [10 min].
CYCLE_MIN = 5

# Minimal number of generations per lineage in the postreated dataset.
GEN_COUNT_MIN = 2  # 1 or 2 does change 'RAD51' dataset but not 'TetO2-TLC1'.

# .............................................................................
# WARNING: if `PAR_CYCLES_POSTREAT` is updated, need to recreate the
#          processed dataset by running `makeFile/processed_dataset.py`.
PAR_CYCLES_POSTREAT = [THRESHOLD, CYCLE_MIN, GEN_COUNT_MIN]
# .............................................................................


DIR_DATA_RAW = os.path.join('..', 'data', 'raw')
DIR_DATA = DIR_DATA_RAW.replace('raw', 'processed')

# DIR_DATA_IGNORED = os.path.join('..', 'data_ignored')


def make_distributions_cycles(data, threshold, cdt_min, sfolder=None):
    """From the dataset of cycle duration times along lineages extract and
    return the cycle duration times [min] by categories:
        - `arrest`: all the long cycles (> `threshold` in 10min) except the
        last cycles of senescence.
        - `cycles_sen_last`: the last cycles of senescence.
        - `normal_atype`: all the normal (<= `threshold`) before a 1st
        arrest (non-terminal or terminal).
        - `normal_btype`: ... after a 1st arrest (necessarily nta).

    """
    (cycles, gtrigs, lineage_types, is_unseen_htypes, is_accidental_deaths,
     lcycle_per_seq_counts) = data
    cycles = cycles['cycle']

    lineage_count = len(cycles)

    # Long cycles: all long cycles except the one terminating a dead lineages,
    # saved separetely & normal cycles distinguishing before or after 1st nta.
    keys = ['arr', 'sen_last', 'nta', 'senA', 'senB', 'norA', 'norB']
    file_names = {'arr': 'cycles_arrest',
                  'nta': 'cycles_nta',
                  'sen': 'cycles_sen',
                  'sen_last': 'cycles_sen_last',
                  'senA': 'cycles_sen_atype',
                  'senB': 'cycles_sen_btype',
                  'norA': 'cycles_normal_atype',
                  'norB': 'cycles_normal_btype'}

    cdts_ = {key: np.array([]) for key in keys}
    # NB: `cdts_['l*']` (`norB`) arrays are first made of the sequences
    #     of cycles that contains the good long (normal) cycle but also normal
    #     (long) cycles.
    is_senescent_s = ~np.isnan(gtrigs['sen'])
    is_dead_s = is_as_expected_lineages(gtrigs, lineage_types,
                                        is_accidental_deaths, ['dead'])
    is_atype_s = is_as_expected_lineages(gtrigs, lineage_types,
                                         is_accidental_deaths, ['atype'])
    is_btype_s = is_as_expected_lineages(gtrigs, lineage_types,
                                         is_accidental_deaths, ['btype'])
    for lin in range(lineage_count):
        # Cycles of the current lineage (excluding NaN values).
        ctemp = cycles[lin][~np.isnan(cycles[lin])]
        if is_btype_s[lin]:  # If the lineage is type B.
            idx_nta1 = int(gtrigs['nta'][lin, 0])
            cdts_['norA'] = np.append(cdts_['norA'], ctemp[:idx_nta1])
            cdts_['norB'] = np.append(cdts_['norB'], ctemp[idx_nta1:])
            if is_senescent_s[lin]:
                idx_sen = int(gtrigs['sen'][lin])
                # If dead, then the last sequence of arrest can be interpreted
                #  as senescence (see `simulation_lineage.py`).
                if is_dead_s[lin]:
                    cdts_['senB'] = np.append(cdts_['senB'], ctemp[idx_sen:-1])
                    cdts_['sen_last'] = np.append(cdts_['sen_last'], ctemp[-1])
                    cdts_['arr'] = np.append(cdts_['arr'], ctemp[:-1])
                cycles_nta_temp = ctemp[idx_nta1:idx_sen]
            else:
                cycles_nta_temp = ctemp[idx_nta1:]
                cdts_['arr'] = np.append(cdts_['arr'], ctemp)
            cdts_['nta'] = np.append(cdts_['nta'], cycles_nta_temp)
        # Otherwise, if type A (necessarily senescent).
        elif is_atype_s[lin]:
            idx_sen = int(gtrigs['sen'][lin])
            cdts_['norA'] = np.append(cdts_['norA'], ctemp[:idx_sen])
            cdts_['senA'] = np.append(cdts_['senA'], ctemp[idx_sen:-1])
            cdts_['sen_last'] = np.append(cdts_['sen_last'], ctemp[-1])
            cdts_['arr'] = np.append(cdts_['arr'], ctemp[:-1])
        # Otherwise, NaN types need to account for them only for 'arr'.
        #    (see `pst.postreat_experimental_lineages`).
        else:
            ncycles_idxs = ctemp <= 10 * threshold
            cdts_['norA'] = np.append(cdts_['norA'], ctemp[ncycles_idxs])
            cdts_['arr'] = np.append(cdts_['arr'], ctemp)
    # We remove normal cycles for `lc*` arrays and long from `norA`.
    cdts_['arr'] = cdts_['arr'][cdts_['arr'] > 10 * threshold]
    cdts_['nta'] = cdts_['nta'][cdts_['nta'] > 10 * threshold]
    cdts_['norB'] = cdts_['norB'][cdts_['norB'] <= 10 * threshold]
    # In addition we remove abnormally short cdts_ from dataset.
    cdts_['norA'] = cdts_['norA'][cdts_['norA'] >= 10 * cdt_min]
    cdts_['norB'] = cdts_['norB'][cdts_['norB'] >= 10 * cdt_min]
    # We concatenate (and shuffle) senA and sen B to get sen cycles.
    cdts_['sen'] = np.append(cdts_['senA'], cdts_['senB'])
    np.random.shuffle(cdts_['sen'])

    # Saving.
    if not isinstance(sfolder, type(None)):
        for key, cdts in cdts_.items():
            path = os.path.join(sfolder, f'{file_names[key]}.csv')
            pd.DataFrame(cdts).to_csv(path, header=None, index=None)
        np.save(os.path.join(sfolder, 'EMPIRICAL_DISTRIBUTIONS.npy'), cdts_)

        # Save in csv for the `source data` file (not used by python).
        cdts_final = {k:v for k,v in cdts_.items() if k in
                      ['norA', 'norB', 'nta', 'sen']}
        df = pd.DataFrame.from_dict(cdts_final, orient='index').T
        df.to_csv(os.path.join(sfolder, 'Source Data Fig1d.csv'), index=False)
    return cdts_


def make_microfluidic_dataset(file_name, parameters=PAR_CYCLES_POSTREAT,
                              is_saved=True, strain=''):
    (threshold, cdt_min, gcount_min) = parameters

    data_path = os.path.join(DIR_DATA_RAW, file_name)
    data = sio.loadmat(data_path)
    data = data[list(data.keys())[-1]]
    if is_saved:
        folder = os.path.join(DIR_DATA, f'cycles_{strain}')
        if not os.path.exists(folder):
            os.makedirs(folder)
    else:
        folder = None
    data = postreat_experimental_lineages(data, threshold, gcount_min)
    path = os.path.join(folder, 'LINEAGES_POSTREATED.npy')
    np.save(path, np.asarray(data, dtype="object"), allow_pickle=True)
    # Save in csv to create the `source data` file easily (not used by python).
    df = pd.DataFrame(data[0]['cycle'][::-1],
                      index=np.arange(len(data[0]['cycle'])))
    df.to_csv(os.path.join(folder, 'Source Data Fig1c.csv'), index=False)
    return make_distributions_cycles(data, threshold, cdt_min, sfolder=folder)


def make_distribution_telomeres_init(file_name, is_saved=True):
    """Make initial telomere distribution in good format, from raw data saved
    in the file `file_name` (`etat_asymp_val_juillet` or an equivalent).

    NB: The file is assumed to be structured as an array `arr`s.t. P(L = i) is
        on the 2nd half of `raw` for all `i ` in the 1st half (`i` not
        neccessarily an integer).

    Parameters
    ----------
    file_name : string

    Returns
    -------
    lengths : ndarray
        x-axis of the distribution, with only integer lengths.
    densities : ndarray
        Corresponding y-axis

    """
    data_raw = np.loadtxt(os.path.join(DIR_DATA_RAW, file_name))

    # Separation of the file: full distribution (non zero probability to have
    # non-interger length).
    idx_middle = int(len(data_raw) / 2)
    # lengths_all = data_raw[:idx_middle]
    densities_all = data_raw[idx_middle:]

    # "Compressed" distribution: supported only on intergers.
    length_count = int(idx_middle / 2)
    lengths = np.arange(1, length_count + 1)
    densities = np.zeros(length_count)
    for i in range(length_count):  # We remove lengths that are not integers.
        densities[i] = densities_all[2 * i] + densities_all[2 * i + 1]
    # We make sure it is a probability by renormalizing.
    mass = np.sum(np.diff(np.append(0, lengths)) * densities)
    densities = densities / mass

    if is_saved:
        folder = os.path.join(DIR_DATA, 'telomeres_initial_distribution')
        if not os.path.exists(folder):
            os.makedirs(folder)
        df = pd.DataFrame({'x': lengths, 'y': densities})
        df.to_csv(os.path.join(folder, 'original.csv'), index=False)
        # df_all = pd.DataFrame({'x': lengths_all, 'y': densities_all})
        # df_all.to_csv(os.path.join(folder, 'original_full.csv'), index=False)
    return lengths, densities


def make_distribution_telomeres_init_fitted(is_saved=True,
                                            par_l_init=PAR_L_INIT):
    from telomeres.model.posttreat import transform_l_init
    distribution = extract_distribution_telomeres_init()
    distribution_new = transform_l_init(distribution, par_l_init=par_l_init)
    if is_saved:
        msg = write_parameters_linit(par_l_init)
        path = os.path.join(DIR_DATA, 'telomeres_initial_distribution',
                            f'modified_{msg}.csv')
        df = pd.DataFrame({'x': distribution_new[0], 'y': distribution_new[1]})
        df.to_csv(path, index=False)
    return distribution_new


def postreat_population_concentration():
    path_raw = os.path.join(DIR_DATA_RAW, 'senesence TetO2 tlc1.xlsx')
    data = {}  # Data.
    data['DOX+'] = np.array(pd.read_excel(path_raw, header=None,
                                          sheet_name='raw-DOX+_anais'))
    data['DOX-'] = np.array(pd.read_excel(path_raw, header=None,
                                          sheet_name='raw-DOX-_anais'))
    path_rad51 = os.path.join(DIR_DATA_RAW, '241025_Vero.xlsx')
    if os.path.exists(path_rad51):
        data_tmp = {}
        data_tmp = np.array(pd.read_excel(path_rad51, usecols="C:L"))
        data['RAD51'] = np.transpose(data_tmp)
        data['RAD51_oct'] = np.transpose(data_tmp[:4])
        data['RAD51_sep'] = np.transpose(data_tmp[4:])

    data_t = {}  # Data rescaled (log scale, ...).
    for key, d in data.items():
        # First transformation (Optical density to Population doublings (?)).
        data_t[key] = {'PD': 1e5 * 2 ** d}
        # Second transformation (Optical density to concentration).
        data_t[key]['c'] = 3e7 * d
        # No transformation (Optical density, OD 600 (?)).
        data_t[key]['OD'] = d

    path_pol32 = os.path.join(DIR_DATA_RAW, 'viability assays 1.xlsx')
    if os.path.exists(path_pol32):
        data_tmp = {}
        data_tmp = pd.read_excel(path_pol32).to_dict('split')
        data_tmp_sort = {}  # All the replicates and strains.
        for i in data_tmp['index']:
            data_tmp_sort[data_tmp['data'][i][0]] = data_tmp['data'][i][1:]
        keys_replicate = {  # All the replicates of the strains of interest.
            'tlc1': ['MATa tlc1-1c', 'MATa tlc1-2c', 'MATa tlc1-3b',
                     'MATa tlc1-4c'],
            'tlc1_pol32': ['MATa tlc1 pol32-3d', 'MATa tlc1 pol32-4b',
                           'MATa tlc1 pol32-5a', 'MATa tlc1 pol32-6c']}
        for k, keys in keys_replicate.items():
            data[k] = np.transpose([data_tmp_sort[key] for key in keys])
        for key in keys_replicate:
            # First transformation (Pop doublings to Optical density 600 (?)).
            data_t[key] = {'OD': np.log10(data[key])}  # np.log2(data[key] / 1e5)}
            # data_t[key] = {'OD': np.log10(data[key])}
            # Second transformation (Concentration to OD 600).
            data_t[key]['PD'] = data[key] / 3e7  # To check !!!!
            # No transformation (concentration cell/mL).
            data_t[key]['c'] = data[key]

    # Statistics on all experiments.
    stats = {}
    for key_strain, d in data_t.items():
        stats[key_strain] = {k: {'avg': np.mean(d[k], axis=1),
                                 'std': np.std(d[k], axis=1),
                                 'all': d[k]} for k in d}
    return stats


def extract_population_lmode_from_raw():
    path = os.path.join(DIR_DATA_RAW,
                        'BioRad_2012-04-26_18hr_36min_analysis.txt')
    evo_lmode = np.transpose(np.loadtxt(path))
    return evo_lmode


def make_population_dataset(is_saved=True):
    # Cell concentration.
    stats = postreat_population_concentration()
    # Mode of telomere distribution.
    evo_lmode = extract_population_lmode_from_raw()
    # Save.
    if is_saved:
        for key, stat in stats.items():
            # if 'DOX' in key:
            folder = os.path.join(DIR_DATA, 'population_evolution')
            folder_telo = folder
            # else:
            #     folder = os.path.join(DIR_DATA_IGNORED, 'processed')
            if not os.path.exists(folder):
                os.makedirs(folder)
            np.save(os.path.join(folder, f'cell_concentration_{key}.npy'),
                    stat)
            # Save in csv for the `source data` file (not used by python).
            if key == 'DOX-':
                data_article = stat['OD']
                df = pd.DataFrame.from_dict(data_article, orient='index').T
                df.to_csv(os.path.join(
                    folder, 'Source Data SFig3a.csv'), index=False)
            if key == 'DOX+':
                data_article = stat['OD']
                df = pd.DataFrame.from_dict(data_article, orient='index').T
                df.to_csv(os.path.join(
                    folder, 'Source Data Fig3b-exp.csv'), index=False)
            if key == 'tlc1_pol32':
                data_article = stat['c']
                df = pd.DataFrame.from_dict(data_article, orient='index').T
                df.to_csv(os.path.join(
                    folder, 'Source Data SFig3c-.csv'), index=False)
            if key == 'tlc1':
                data_article = stat['c']
                df = pd.DataFrame.from_dict(data_article, orient='index').T
                df.to_csv(os.path.join(
                    folder, 'Source Data SFig3c+.csv'), index=False)
            if key == 'RAD51':
                data_article = stat['OD']
                df = pd.DataFrame.from_dict(data_article, orient='index').T
                df.to_csv(os.path.join(
                    folder, 'Source Data SFig6d.csv'), index=False)

        path = os.path.join(folder_telo, 'telomere_lengths_DOX+.csv')
        pd.DataFrame(evo_lmode).to_csv(path, header=None, index=None)
    return stats, evo_lmode
