#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 14:36:55 2024

@author: anais
"""

import numpy as np
import os
import imp
import pandas as pd

import aux_functions as fct
import aux_write_paths as wp
import parameters as par
imp.reload(par)
import population_postreat as pps
import population_simulation as sim



# Parameters
# ----------

is_run_in_parallel_from_slurm = "SLURM_ARRAY_TASK_ID" in os.environ.keys()
idx = 1
if is_run_in_parallel_from_slurm:
    idx = int(os.environ["SLURM_ARRAY_TASK_ID"]) + 1

# Experiment design.
# ------------------
# > Experimental concentration initially / at dilution [cell].
CELL_COUNT = 300 # 250, 300. Cas test: 15
T_MAX = 9

# Computation parameters
# ----------------------
# > Number of parallelization.s per simulation.
#   S.t. PARA_COUNT[i] 1D array of the numbers of paralelizations to simulate
#   at concentration C_EXP[i].
PARA_COUNT = 1
# > Number of times the simulation is run.
SIMU_COUNT = 30 # 25, 20. Cas test: 3

TIMES, TIME_SAVED_IDXS, DIL_IDXS = par.TIMES, par.TIME_SAVED_IDXS, par.DIL_IDXS


# Printing of chosen parameters
# -----------------------------
print('P_ACCIDENTAL_DEATH: ', par.P_ACCIDENTAL_DEATH)
print('MAX_SEN_CYCLE_COUNT: ' + str(par.MAX_SEN_CYCLE_COUNT))
print('PROP_SAT: ', par.PROP_SAT)
print('HYBRID_CHOICE: ', par.HYBRID_CHOICE)
print('PAR_NTA: ', par.PAR_NTA)
print('PAR_SEN: ', par.PAR_SEN)
print('PAR_L_INIT: ', par.PAR_L_INIT)
print('CELL_COUNT: ' + str(CELL_COUNT) + ' cell/mL')
print('PARA_COUNT: ' + str(PARA_COUNT))
if not is_run_in_parallel_from_slurm:
    print('simu_count: ' + str(SIMU_COUNT))



def statistics_simus(folder, simu_count, t_max):
    """ Postreat saved data, computing and saving mean, std, min, max... on all
    of the `simu_count` first simulations present in the folder at path
    `folder`.

    """
    simus = np.arange(simu_count)
    # Load paths to all simulations in a list.
    s = [f'{folder}output_{i:02d}.npy' for i in simus + 1]
    s_postreat = [f'{folder}output_{i:02d}_p_from_c.npy' for i in simus + 1]
  
    # Genearal data.
    # > Paths to data.
    stat_data_path = wp.write_sim_pop_postreat_average(folder, simu_count)
    # > Times array (only up to `t_max`).
    times = np.load(stat_data_path, allow_pickle='TRUE').any().get('times')
    time_count = len(times)
    t_max = min(t_max, times[-1])
    times = times[times <= t_max]
    # > Days arrays.
    days_exp = np.arange(1 , len(par.Y[:, 0]) + 1)
    idxs_bf_dil = np.array([np.where(times == day)[0][0] - 1 for day in
                            days_exp[days_exp <= times[-1]]]).astype('int')
    # > For all key to postreat.
    for key in ['evo_c']:
        print(key)
        # We reshape to common shape for all simulations.
        evo_s = [np.load(s_postreat[i], allow_pickle='TRUE').any().get(
                 key) for i in simus]
        evo_s = [fct.reshape_with_nan(evo_s[i], time_count, 0)[idxs_bf_dil] for
                 i in simus]
        for i in simus:
            name = s[i].replace('.npy', '_cOD.csv')
        name = s[0].replace('01.npy', 'cOD.csv')
        pd.DataFrame(evo_s).to_csv(name, header=None, index=None)

    # > Time evolution of telomere lengths.
    # > Times array up to `t_max`.
    times = np.load(stat_data_path, allow_pickle='TRUE').any().get('times')
    t_max = min(t_max, times[-1])
    times = times[times <= t_max]
    # > Days arrays.
    days_exp = np.arange(len(par.EVO_L_EXP[0]))
    idxs_bf_dil = np.array([np.where(times == day)[0][0] for day in
                            days_exp[days_exp <= times[-1]]])
    day_max = min(len(days_exp), len(idxs_bf_dil))
    idxs_bf_dil = idxs_bf_dil[:day_max]
    for key in ['evo_lmode']:
        evo_s = [fct.reshape_with_nan(np.load(s[i], allow_pickle='TRUE'
                      ).any().get(key), time_count, 0)[idxs_bf_dil] for i in 
                 simus]
        for i in simus:
            name = s[i].replace('.npy', '_lmode.csv')
            pd.DataFrame(evo_s[i]).to_csv(name, header=None, index=None)
        name = s[0].replace('01.npy', 'lmode.csv')
        pd.DataFrame(evo_s).to_csv(name, header=None, index=None)
    return

# Simulation
# ----------
run_idx = 0
dir_path = wp.write_simu_pop_subdirectory(CELL_COUNT)
print(dir_path)
print('Number of parallelizations: ', PARA_COUNT)
sub_dir_path = wp.write_simu_pop_subdirectory(CELL_COUNT, PARA_COUNT)

for i in range(1, SIMU_COUNT + 1):
    np.random.seed(i)
    print(f'Simulation n° {i}/{SIMU_COUNT}')

    file_path = sub_dir_path + f'output_{i:02d}.npy'
    if not(os.path.exists(file_path)):
        outputs = sim.simu_parallel(TIMES, TIME_SAVED_IDXS, DIL_IDXS, 
                                    PARA_COUNT, CELL_COUNT)
        # Save dictionary 'output' and postreat.
        np.save(file_path, outputs)
        pps.postreat_from_evo_c(file_path)
        
statistics_simus(sub_dir_path, SIMU_COUNT, T_MAX)
