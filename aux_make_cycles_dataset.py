#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 16:30:23 2022

@author: arat
"""

import imp
import numpy as np
import os
import pandas as pd
import scipy.io as sio
# import aux_write_paths as wp

import lineage_simulation as sim
import parameters as par
imp.reload(par)

def write_path_directory_from_file(file_path, make_dir=False):
    """ Return the string corresponding to the path of the directory in which
    the file with path given as argument in saved.

    """
    idx = len(file_path) - file_path[::-1].find("/")
    print(file_path, file_path[:idx])
    if (not os.path.exists(file_path[:idx])) and make_dir:
        os.makedirs(file_path[:idx])
    return file_path[:idx]

def make_cycles_dataset(data, threshold=par.THRESHOLD, cycle_min=par.CYCLE_MIN,
                        sfolder=None):
    """ From the dataset of cycle duration times along lineages extract and
    return the cycle duration times by categories:
        - `lcycles`: all the long cycles (> `threshold` in 10min) except the
        last cycles of senescence.
        - `lcycles_last_sen`: the last cycles of senescence.
        - `ncycles_bf_arrest`: all the normal (<= `threshold`) before a 1st
        arrest (non-terminal or terminal).
        - `ncycles_af_arrest`: ... after a 1st arrest (necessarily nta).

    """
    (cycles, gtrigs, lineage_types, is_unseen_htypes, is_accidental_deaths,
     lcycle_per_seq_counts) = data
    cycles = cycles['cycle']

    lineage_count = len(cycles)

    # Long cycles: all long cycles except the one terminating a dead lineages,
    # stored separetely & normal cycles distinguishing before or after 1st nta.
    keys = ['lc', 'lc_last_sen', 'lc_nta', 'lc_senA', 'lc_senB',
            'nc_bf_arrest','nc_af_arrest']
    cdts_ = {key: np.array([]) for key in keys}
    # NB: `cdts_['l*']` (`nc_af_arrest`) arrays are first made of the sequences
    #     of cycles that contains the good long (normal) cycle but also normal
    #     (long) cycles.
    is_senescent_s = ~np.isnan(gtrigs['sen'])
    is_dead_s = sim.is_as_expected_lineages(gtrigs, lineage_types,
                                            is_accidental_deaths, ['dead'])
    is_atype_s = sim.is_as_expected_lineages(gtrigs, lineage_types,
                                             is_accidental_deaths, ['atype'])
    is_btype_s = sim.is_as_expected_lineages(gtrigs, lineage_types,
                                             is_accidental_deaths, ['btype'])
    for lin in range(lineage_count):
        # Cycles of the current lineage (excluding NaN values).
        ctemp = cycles[lin][~np.isnan(cycles[lin])]
        if is_btype_s[lin]: # If the lineage is type B.
            idx_nta1 = int(gtrigs['nta'][lin, 0])
            cdts_['nc_bf_arrest'] = np.append(cdts_['nc_bf_arrest'],
                                              ctemp[:idx_nta1])
            cdts_['nc_af_arrest'] = np.append(cdts_['nc_af_arrest'],
                                              ctemp[idx_nta1:])
            if is_senescent_s[lin]:
                idx_sen = int(gtrigs['sen'][lin])
                # If dead, then the last sequence of arrest can be interpreted
                #  as senescence (see `simulation_lineage.py`).
                if is_dead_s[lin]:
                    cdts_['lc_senB'] = np.append(cdts_['lc_senB'],
                                                 ctemp[idx_sen:-1])
                    cdts_['lc_last_sen'] = np.append(cdts_['lc_last_sen'],
                                                     ctemp[-1])
                    cdts_['lc'] = np.append(cdts_['lc'], ctemp[:-1])
                lcycles_nta_temp = ctemp[idx_nta1:idx_sen]
            else:
                lcycles_nta_temp = ctemp[idx_nta1:]
                cdts_['lc'] = np.append(cdts_['lc'], ctemp)
            cdts_['lc_nta'] = np.append(cdts_['lc_nta'], lcycles_nta_temp)
        # Otherwise, if type A (necessarily senescent).
        elif is_atype_s[lin]:
            idx_sen = int(gtrigs['sen'][lin])
            cdts_['nc_bf_arrest'] = np.append(cdts_['nc_bf_arrest'],
                                              ctemp[:idx_sen])
            cdts_['lc_senA'] = np.append(cdts_['lc_senA'], ctemp[idx_sen:-1])
            cdts_['lc_last_sen'] = np.append(cdts_['lc_last_sen'], ctemp[-1])
            cdts_['lc'] = np.append(cdts_['lc'], ctemp[:-1])
        # Otherwise, NaN types need to account for them only for 'lc'.
        #    (see `sim.postreat_experimental_lineages`).
        else:
            ncycles_idxs = ctemp <= 10 * threshold
            cdts_['nc_bf_arrest'] = np.append(cdts_['nc_bf_arrest'],
                                              ctemp[ncycles_idxs])
            cdts_['lc'] = np.append(cdts_['lc'], ctemp)
    # We remove normal cycles for `lc*` arrays and long from `nc_bf_arrest`.
    cdts_['lc'] = cdts_['lc'][cdts_['lc'] > 10 * threshold]
    cdts_['lc_nta'] = cdts_['lc_nta'][cdts_['lc_nta'] > 10 * threshold]
    cdts_['nc_af_arrest'] = cdts_['nc_af_arrest'][cdts_['nc_af_arrest']
                                                  <= 10 * threshold]
    # In addition we remove abnormally short cdts_ from dataset.
    cdts_['nc_bf_arrest'] = cdts_['nc_bf_arrest'][cdts_['nc_bf_arrest'] 
                                                  >= 10 * cycle_min]
    cdts_['nc_af_arrest'] = cdts_['nc_af_arrest'][cdts_['nc_af_arrest']
                                                  >= 10 * cycle_min]
    # We concatenate (and shuffle) senA and sen B to get sen cycles.
    cdts_['lc_sen'] = np.append(cdts_['lc_senA'], cdts_['lc_senB'])
    np.random.shuffle(cdts_['lc_sen'])

    # Saving.
    if not isinstance(sfolder, type(None)):
        for key, cdts in cdts_.items():
            name = key.replace('c', 'cycles') + '.csv'
            pd.DataFrame(cdts).to_csv(sfolder + name, header=None, index=None)
    return cdts_


def make_cycles_dataset_from_path(data_path, data_key, threshold=par.THRESHOLD,
                                  gcount_min=par.GEN_COUNT_BY_LINEAGE_MIN,
                                  cycle_min=par.CYCLE_MIN, sfolder=None,
                                  is_saved=True):
    data = sio.loadmat(data_path)
    data = data[data_key]
    if not is_saved:
        folder = None
    elif isinstance(sfolder, type(None)):
        folder = write_path_directory_from_file(data_path, make_dir=True)
    else:
        folder = sfolder
        if not os.path.exists(folder):
            os.makedirs(folder)

    data = sim.postreat_experimental_lineages(data, threshold, gcount_min)
    return make_cycles_dataset(data, threshold=threshold, cycle_min=cycle_min,
                               sfolder=folder)

# Create formated dataset.
if __name__ == "__main__":
    out = make_cycles_dataset_from_path('data/microfluidic/'
                                        'TelomeraseNegative.mat',
                                        'OrdtryT528total160831')

    out_mutant = make_cycles_dataset_from_path('data/microfluidic/rad51/'
                                               'TelomeraseNegMutantRAD51.mat',
                                               'OrdtrRAD51D')

