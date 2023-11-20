#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 16:30:23 2022

@author: arat
"""

import imp
import numpy as np
import pandas as pd
import scipy.io as sio

import lineage_simulation as sim
import parameters as par
imp.reload(par)


EXP_DATA = sio.loadmat('data/microfluidic/TelomeraseNegative.mat')
EXP_DATA = EXP_DATA['OrdtryT528total160831']
EXP_DATA = sim.postreat_experimental_lineages(EXP_DATA, par.THRESHOLD,
                                              par.GEN_COUNT_BY_LINEAGE_MIN)

EXP_DATA_MUTANT = sio.loadmat('data/microfluidic/TelomeraseNegMutantRAD51.mat')
EXP_DATA_MUTANT = EXP_DATA_MUTANT['OrdtrRAD51D']
EXP_DATA_MUTANT = sim.postreat_experimental_lineages(EXP_DATA_MUTANT,
                                   par.THRESHOLD, par.GEN_COUNT_BY_LINEAGE_MIN)


def make_cycles_dataset(data, threshold=par.THRESHOLD,
                        cycle_min=par.CYCLE_MIN, name=''):
    """ From the dataset of cycle duration times along lineages extract and
    return the cycle duration times by categories:
        - `lcycles`: all the long cycles (> `threshold` in 10min) except the
        last cycles of senescence.
        - `lcycles_last_sen`: the last cycles of senescence.
        - `ncycles_bf`: all the normal (<= `threshold`) before a 1st arrest
        (non-terminal or terminal).
        - `ncycles_af`: ... after a 1st arrest (necessarily nta).

    """
    (cycles, gtrigs, lineage_types, is_unseen_htypes, is_accidental_deaths,
     lcycle_per_seq_counts) = data
    cycles = cycles['cycle']

    lineage_count = len(cycles)

    # Long cycles: all long cycles except the one terminating a dead lineages,
    #   stored separetely.
    lcycles = np.array([])
    lcycles_nta = np.array([])
    lcycles_senA = np.array([])
    lcycles_senB = np.array([])
    lcycles_last_sen = np.array([])
    # Normal cycles distinguishing before or after a 1st non-terminal arrest.
    ncycles_bf = np.array([])
    ncycles_af = np.array([])
    # NB: `lcycles*` (`ncycles_af`) arrays are first made of the sequences of
    #     cycles that contains the good long (normal) cycle but also normal
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
        cycles_temp = cycles[lin][~np.isnan(cycles[lin])]
        if is_btype_s[lin]: # If the lineage is type B.
            idx_nta1 = int(gtrigs['nta'][lin, 0])
            ncycles_bf = np.append(ncycles_bf, cycles_temp[:idx_nta1])
            ncycles_af = np.append(ncycles_af, cycles_temp[idx_nta1:])
            if is_senescent_s[lin]:
                idx_sen = int(gtrigs['sen'][lin])
                # If dead, then the last sequence of arrest can be interpreted
                #  as senescence (see `simulation_lineage.py`).
                if is_dead_s[lin]:
                    lcycles_senB = np.append(lcycles_senB,
                                             cycles_temp[idx_sen:-1])
                    lcycles_last_sen = np.append(lcycles_last_sen,
                                                 cycles_temp[-1])
                    lcycles = np.append(lcycles, cycles_temp[:-1])
                lcycles_nta_temp = cycles_temp[idx_nta1:idx_sen]
            else:
                lcycles_nta_temp = cycles_temp[idx_nta1:]
                lcycles = np.append(lcycles, cycles_temp)
            lcycles_nta = np.append(lcycles_nta, lcycles_nta_temp)
        # Otherwise, if type A (necessarily senescent).
        elif is_atype_s[lin]:
            idx_sen = int(gtrigs['sen'][lin])
            ncycles_bf = np.append(ncycles_bf, cycles_temp[:idx_sen])
            lcycles_senA = np.append(lcycles_senA, cycles_temp[idx_sen:-1])
            lcycles_last_sen = np.append(lcycles_last_sen, cycles_temp[-1])
            lcycles = np.append(lcycles, cycles_temp[:-1])
        # Otherwise, NaN types need to account for them only for lcycles
        #    (see `sim.postreat_experimental_lineages`).
        else:
            ncycles_idxs = cycles_temp <= 10 * threshold
            ncycles_bf = np.append(ncycles_bf, cycles_temp[ncycles_idxs])
            lcycles = np.append(lcycles, cycles_temp)
    # We remove normal cycles for `lcycles*` arrays and long from `ncycles_bf`.
    lcycles = lcycles[lcycles > 10 * threshold]
    lcycles_nta = lcycles_nta[lcycles_nta > 10 * threshold]
    ncycles_af = ncycles_af[ncycles_af <= 10 * threshold]

    # In addition we remove abnormally short cycles from dataset.
    ncycles_bf = ncycles_bf[ncycles_bf >= 10 * cycle_min]
    ncycles_af = ncycles_af[ncycles_af >= 10 * cycle_min]

    # We concatenate (and shuffle) senA and sen B to get sen cycles.
    lcycles_sen = np.append(lcycles_senA, lcycles_senB)
    np.random.shuffle(lcycles_sen)

    # Saving.
    end = f"_{name}.csv"
    pd.DataFrame(lcycles).to_csv("data/microfluidic/lcycles" + end,
                                 header=None, index=None)
    pd.DataFrame(lcycles_sen).to_csv("data/microfluidic/lcycles_sen" + end,
                                     header=None, index=None)
    pd.DataFrame(lcycles_nta).to_csv("data/microfluidic/lcycles_nta" + end,
                                     header=None, index=None)
    pd.DataFrame(lcycles_senA).to_csv("data/microfluidic/lcycles_senA" + end,
                                      header=None, index=None)
    pd.DataFrame(lcycles_senB).to_csv("data/microfluidic/lcycles_senB" + end,
                                      header=None, index=None)
    pd.DataFrame(lcycles_last_sen).to_csv("data/microfluidic/lcycles_last_sen"
                                          + end, header=None, index=None)
    pd.DataFrame(ncycles_bf).to_csv("data/microfluidic/ncycles_bf_arrest"+ end,
                                    header=None, index=None)
    pd.DataFrame(ncycles_af).to_csv("data/microfluidic/ncycles_af_arrest"+ end,
                                    header=None, index=None)
    return (lcycles, lcycles_nta, lcycles_sen, lcycles_senA, lcycles_senB,
            lcycles_last_sen, ncycles_bf, ncycles_af)

out = make_cycles_dataset(EXP_DATA)
out_mutant = make_cycles_dataset(EXP_DATA_MUTANT, name='rad51')