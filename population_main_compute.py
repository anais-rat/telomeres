#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 16:52:56 2022

@author: arat
"""

import imp
import numpy as np
import os
import time

import aux_write_paths as wp
imp.reload(wp)
import population_simulation_bfchange as sim
import parameters as par
imp.reload(par)
import population_postreat as pps


# Parameters
# ----------

IS_PLOT = True
if IS_PLOT:
    # .....if IS_PLOT True......
    import population_plot as pl

    IS_SAVED = False
    # ..............
    if IS_SAVED:
        FIG_DIRECTORY = "figures/manuscript"
    # ..............
    else:
        FIG_DIRECTORY = None

    TMAX_TO_PLOT = 8 # In days. math.inf to plot all simulated times.
    ANC_GROUP_COUNT = 10
    # ..........................

is_run_in_parallel_from_slurm = "SLURM_ARRAY_TASK_ID" in os.environ.keys()
idx = 1
if is_run_in_parallel_from_slurm:
    idx = int(os.environ["SLURM_ARRAY_TASK_ID"]) + 1

# Experiment design.
# ------------------
# > Experimental concentration initially / at dilution [cell].
C_EXP = np.array([300]) # 250, 300. Cas test: 15


# Computation parameters
# ----------------------
# > Number of parallelization.s per simulation.
#   S.t. PARA_COUNT[i] 1D array of the numbers of paralelizations to simulate
#   at concentration C_EXP[i].
PARA_COUNT = [np.array([1])]
# > Number of times the simulation is run.
SIMU_COUNTS = np.array([30]) # 25, 20. Cas test: 3

TIMES, TIME_SAVED_IDXS, DIL_IDXS = par.TIMES, par.TIME_SAVED_IDXS, par.DIL_IDXS


# Printing of chosen parameters
# -----------------------------
print('P_ACCIDENT: ', par.P_ACCIDENT)
print('MAX_SEN_CYCLE_COUNT: ' + str(par.MAX_SEN_CYCLE_COUNT))
print('PROP_SAT: ', par.PROP_SAT)
print('HTYPE_CHOICE: ', par.HTYPE_CHOICE)
print('PAR_NTA: ', par.PAR_NTA)
print('PAR_SEN: ', par.PAR_SEN)
print('PAR_L_INIT: ', par.PAR_L_INIT)
print('C_EXP: ' + str(C_EXP) + ' cell/mL')
print('PARA_COUNT: ' + str(PARA_COUNT))
if not is_run_in_parallel_from_slurm:
    print('simu_count: ' + str(SIMU_COUNTS))


# Simulation
# ----------
run_idx = 0
for cell_count in C_EXP:

    dir_path = wp.write_simu_pop_subdirectory(cell_count)
    print(dir_path)

    for para_count in PARA_COUNT[run_idx]:
        print('Number of parallelizations: ', para_count)

        # If not already done, creation of a folder for simulations.
        # NB: made only for the first simu to avoid creation of the subdir
        #   inside the if from another job when parallel compputing is run.
        sub_dir_path = wp.write_simu_pop_subdirectory(cell_count, para_count)
        if not os.path.exists(sub_dir_path) and idx == 1:
            os.makedirs(sub_dir_path)

        # If parallel computation run from sbacth command.
        if is_run_in_parallel_from_slurm:
            print('Simulation n°', idx)
            time.sleep(10)  # To avoid non created subdirectory.
            np.random.seed(idx)

            file_path = sub_dir_path + f'output_{idx:02d}.npy'
            if not os.path.exists(file_path):
                outputs = sim.simu_parallel(TIMES,TIME_SAVED_IDXS, DIL_IDXS,
                                            para_count, cell_count)
                # Save dictionary 'output' and postreat.
                np.save(file_path, outputs)
                pps.postreat_from_evo_c(file_path)
            pps.postreat_from_evo_c_if_not_saved(file_path)

        # Otherwise computation in serie.
        else:
            simu_count = SIMU_COUNTS[run_idx]
            for i in range(1, simu_count + 1):
                np.random.seed(i)
                print(f'Simulation n° {i}/{simu_count}')

                file_path = sub_dir_path + f'output_{i:02d}.npy'
                if not(os.path.exists(file_path)):
                    outputs = sim.simu_parallel(TIMES,TIME_SAVED_IDXS,
                                                DIL_IDXS, para_count,
                                                cell_count)
                    # Save dictionary 'output' and postreat.
                    np.save(file_path, outputs)
                    pps.postreat_from_evo_c(file_path)
                pps.postreat_from_evo_c_if_not_saved(file_path)

            # Average on all simulations.
            out_p = pps.postreat_performances(sub_dir_path, simu_count)
            out_s = pps.statistics_simus_if_not_saved(sub_dir_path, simu_count)
            print('Computation time (average):', out_p['computation_time'])

            # Plot average at `c_exp` and `para_count` fixed.
            if IS_PLOT:
                # pl.plot_hist_lmin_at_sen(cell_count, para_count, simu_count,
                #                           FIG_DIRECTORY, day_count=7, width=4)
                # pl.plot_evo_c_n_p_pcfixed_from_stat(cell_count, para_count,
                #                                     simu_count, FIG_DIRECTORY,
                #                                     TMAX_TO_PLOT)
                # pl.plot_evo_l_pcfixed_from_stat(cell_count, para_count,
                #                                 simu_count, FIG_DIRECTORY,
                #                                 TMAX_TO_PLOT)
                # pl.plot_evo_p_anc_pcfixed_from_stat(cell_count, para_count,
                #                                     simu_count,
                #                                     ANC_GROUP_COUNT,
                #                                     FIG_DIRECTORY, TMAX_TO_PLOT)
                # pl.plot_evo_gen_pcfixed_from_stat(cell_count, para_count,
                #                                   simu_count, FIG_DIRECTORY,
                #                                   TMAX_TO_PLOT)
                print()
    run_idx += 1
