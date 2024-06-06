#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 17:51:29 2022

@author: arat

Script to compute all the data necessary to plot the figures.
Ideally should be run using parallel computing through slrum (see
`cluster_lineage_compute.batch`) otherwise, the present script will run the
required simulations in serie, which is not recommanded because very long.

"""
import imp
import copy
import os
import scipy.io as sio
import numpy as np


import lineage_simulation as sim
import parameters as par
imp.reload(par)

# Reminder
# --------
# type_of_sort: 'gdeath', 'lmin', 'gnta1', 'gnta2', ..., 'gsen'.
# gtrig keys: 'nta', 'sen' 'death'.
# gtrig_to_compare: 'nta1', 'nta2', ..., 'sen' 'death'.
# characteristics: 'atype', btype', 'htype', 'arrested1', 'arrested2', ...,
#                  'senescent', 'dead', dead_accidentally', 'dead_naturally'.


# Definition of parameters (common to all jobs)
# ---------------------------------------------

SIMULATION_COUNT = 1000
PROC_COUNT = 11

CHARAC_S = [['senescent'],
            ['atype', 'senescent'],
            ['btype', 'senescent'],
            ['btype', 'arrested2', 'senescent'],
            ['btype']]
i1 = 2 * len(CHARAC_S)

CHARAC_S_2 = [['senescent'],
              ['atype', 'senescent'],
              ['btype', 'senescent'],
              ['btype']]


P_ACC_S = par.P_ACCIDENT * np.array([1., 10., 20, 30., 40., 50.])
P_ACC_S_TEST = np.array([0.054]) # par.P_ACCIDENT * np.array([25, 34.8, 30, 40])
P_ACC_S_TEST[0] = 0.054
L_TRANS_S = np.array([-20, -10, 0, 10, 20, 40])
L0_S = np.array([-40, -20, -10, 0, 10, 20])
L1_S = np.array([-168, -84, -42, 0, 42, 84])

POSTREAT_DT = 1  # Time step in 10 min.

i2 = len(P_ACC_S) * len(CHARAC_S_2)
i3, i4, i5 = np.array([2 * len(L_TRANS_S), 2 * len(L0_S),
                       len(L1_S)]) * len(CHARAC_S_2)
i6 = len(P_ACC_S_TEST)

# Computation
# -----------

# > NB Parameters for the job array. `idx` should ran from 0 to `job_count-1`.
job_count = i1 + i2 + i3 + i4 + i5 + i6 + 2
print('job_count', job_count)
is_run_in_parallel_from_slurm = "SLURM_ARRAY_TASK_ID" in os.environ.keys()

# > Experimental data: extraction and formatting.
EXP_DATA = sio.loadmat('data/microfluidic/TelomeraseNegative.mat')
EXP_DATA = EXP_DATA['OrdtryT528total160831']
EXP_DATA = sim.postreat_experimental_lineages(EXP_DATA, par.THRESHOLD,
                                              par.GEN_COUNT_BY_LINEAGE_MIN)

DATA_EXP_MUTANT = sio.loadmat('data/microfluidic/TelomeraseNegMutantRAD51.mat')
DATA_EXP_MUTANT = DATA_EXP_MUTANT['OrdtrRAD51D']
DATA_EXP_MUTANT = sim.postreat_experimental_lineages(DATA_EXP_MUTANT,
                                                     par.THRESHOLD, 2)


# If parallel computation run from sbacth command, only one idx computed.
if is_run_in_parallel_from_slurm:
    idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
    idxs = [idx]
# Otherwise computation in serie.
else:
    idxs = np.arange(job_count)

# Iteration on all jobs to run.
for run_idx in idxs:
    print(f'\n Simulation n° {run_idx + 1} / {job_count}')

    # Best-fit parameters.
    if run_idx < len(CHARAC_S):  # Htype seen, everything stored.
        print('1: ', run_idx)
        # characteristics = CHARAC_S[run_idx]
        # exp_data_selected = sim.select_exp_lineages(EXP_DATA, characteristics)
        # lineage_count = len(exp_data_selected[0]['cycle'])

        # sim.simulate_lineages_evolutions(SIMULATION_COUNT, lineage_count,
        #                                  characteristics, is_htype_seen=True,
        #                                  is_lcycle_counts=True, is_evos=True,
        #                                  proc_count=PROC_COUNT)

    elif run_idx < i1:  # Htype unseen, no evolution array.
        print('2: ', run_idx)
        # characteristics = CHARAC_S[run_idx - len(CHARAC_S)]
        # exp_data_selected = sim.select_exp_lineages(EXP_DATA, characteristics)
        # lineage_count = len(exp_data_selected[0]['cycle'])

        # sim.simulate_lineages_evolutions(SIMULATION_COUNT, lineage_count,
        #                                  characteristics, is_htype_seen=False,
        #                                  is_lcycle_counts=True,
        #                                  proc_count=PROC_COUNT)

    # Varying parameters.
    # i) Varying pdeath.
    elif run_idx < i1 + i2:
        print('3: ', run_idx)
        # idx_par, idx_char = np.divmod(run_idx - i1, len(CHARAC_S_2))
        # characteristics = CHARAC_S_2[idx_char]
        # exp_data_selected = sim.select_exp_lineages(EXP_DATA, characteristics)
        # lineage_count = len(exp_data_selected[0]['cycle'])
        # p_exit = copy.deepcopy(par.P_EXIT)
        # p_exit[0] = P_ACC_S[idx_par]

        # sim.simulate_lineages_evolutions(SIMULATION_COUNT, lineage_count,
        #                                  characteristics, is_htype_seen=False,
        #                                  is_lcycle_counts=False,
        #                                  proc_count=PROC_COUNT,
        #                                  p_exit=p_exit)

    # ii) Varying ltrans.
    elif run_idx < i1 + i2 + i3:
        print('4: ', run_idx)
        # p_exit = copy.deepcopy(par.P_EXIT)
        # if run_idx < i1 + i2 + i3 / 2: # Normal types.
        #     ridx = run_idx
        # else: # Mutants (RAD51 with pacc x5)
        #     ridx = run_idx - int(i3 / 2)
        #     p_exit[0] = par.P_ACCIDENT * 5
        # idx_par, idx_char = np.divmod(ridx - (i1 + i2), len(CHARAC_S_2))
        # characteristics = CHARAC_S_2[idx_char]
        # exp_data_selected = sim.select_exp_lineages(EXP_DATA, characteristics)
        # lineage_count = len(exp_data_selected[0]['cycle'])
        # parameters = copy.deepcopy(par.PAR)
        # parameters[2][0] += L_TRANS_S[idx_par]

        # sim.simulate_lineages_evolutions(SIMULATION_COUNT, lineage_count,
        #                                  characteristics, is_htype_seen=False,
        #                                  parameters=parameters,
        #                                  is_lcycle_counts=False,
        #                                  proc_count=PROC_COUNT, p_exit=p_exit)

    # iii) Varying l0.
    elif run_idx < i1 + i2 + i3 + i4:
        print('5: ', run_idx)
        # p_exit = copy.deepcopy(par.P_EXIT)
        # if run_idx < i1 + i2 + i3 / 2: # Normal types.
        #     ridx = run_idx
        # else: # Mutants (RAD51 with pacc x5).
        #     ridx = run_idx - int(i3 / 2)
        #     p_exit[0] = par.P_ACCIDENT * 5
        # idx_par, idx_char = np.divmod(ridx - (i1 + i2 + i3), len(CHARAC_S_2))
        # characteristics = CHARAC_S_2[idx_char]
        # exp_data_selected = sim.select_exp_lineages(EXP_DATA, characteristics)
        # lineage_count = len(exp_data_selected[0]['cycle'])
        # parameters = copy.deepcopy(par.PAR)
        # parameters[2][1] += L0_S[idx_par]

        # sim.simulate_lineages_evolutions(SIMULATION_COUNT, lineage_count,
        #                                  characteristics, is_htype_seen=False,
        #                                  parameters=parameters,
        #                                  is_lcycle_counts=False,
        #                                  proc_count=PROC_COUNT, p_exit=p_exit)

    # iv) Varying l1.
    elif run_idx < i1 + i2 + i3 + i4 + i5:
        print('6: ', run_idx)
    #     idx_par, idx_char = np.divmod(run_idx - (i1 + i2 + i3 + i4),
    #                                   len(CHARAC_S_2))
    #     characteristics = CHARAC_S_2[idx_char]
    #     exp_data_selected = sim.select_exp_lineages(EXP_DATA, characteristics)
    #     lineage_count = len(exp_data_selected[0]['cycle'])
    #     parameters = copy.deepcopy(par.PAR)
    #     parameters[2][2] += L1_S[idx_par]

    #     sim.simulate_lineages_evolutions(SIMULATION_COUNT, lineage_count,
    #                                      characteristics, is_htype_seen=False,
    #                                      parameters=parameters,
    #                                      is_lcycle_counts=False,
    #                                      proc_count=PROC_COUNT)
    elif run_idx < i1 + i2 + i3 + i4 + i5 + i6:
        print('7: ', run_idx)
        idx_par = run_idx - (i1 + i2 + i3 + i4 + i5)
        characteristics = ['senescent']
        exp_data_selected = sim.select_exp_lineages(DATA_EXP_MUTANT,
                                                    characteristics)
        lineage_count = len(exp_data_selected[0]['cycle'])
        p_exit = copy.deepcopy(par.P_EXIT)
        p_exit[0] = P_ACC_S_TEST[idx_par]

        sim.simulate_lineages_evolutions(SIMULATION_COUNT, lineage_count,
                                         characteristics, is_htype_seen=False,
                                         is_lcycle_counts=False,
                                         proc_count=PROC_COUNT,
                                         p_exit=p_exit)

    # else:  # time vs generation evo with best-fit parameters.
    #     if run_idx % 2 == 0:
    #         par_update = {}
    #         sim.compute_postreat_data(EXP_DATA, SIMULATION_COUNT,
    #                                   ['senescent'], POSTREAT_DT,
    #                                   proc_count=PROC_COUNT)
    #     else:
    #         sim.compute_postreat_data(EXP_DATA, SIMULATION_COUNT,
    #                                   ['senescent'], POSTREAT_DT,
    #                                   proc_count=PROC_COUNT)
