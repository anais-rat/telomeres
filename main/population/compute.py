#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 16:52:56 2022

@author: arat


Script to compute population data necessary to plot the figures.

Ideally, this should be run using parallel computing (e.g. on a cluster through
slrum using `slrum_compute.batch`) otherwise, the present script will run the
`SIMU_COUNT` required simulations in serie, which is not recommanded because
very long.
If you use `slrum_compute.batch`, the varible `#SBATCH --array`must be set to
`0-SIMU_COUNT`. Moreover, the script will need to be run twice.
One with `slrum_compute.batch` to simulate `SIMU_COUNT` populations, and once
with `slrum_posttreat.batch` to postreat and average them.

Contrarily to `main.lineage.compute` the present script will not run all the
simulations needed. The user need to adjust manually the parameters `C_EXP`,
and `SIMU_COUNT` (`PARA_COUNT` set to 1 in published data) and `PAR_UPDATES`.
The reason is that population simulations are way longer than lineage
simulations and it is more efficient to run them separately.

"""

from copy import deepcopy
from os.path import join
import numpy as np
import os
import time

import project_path
import telomeres.auxiliary.write_paths as wp
import telomeres.population.simulation as sim
import telomeres.model.parameters as par
import telomeres.population.posttreat as pps

# Recall
# ------

# PAR_DEFAULT_POP = {'htype': HTYPE_CHOICE,
#                    'p_exit': P_EXIT,
#                    'fit': PAR,  # Fitted parameters: p_onset, par_l_init.
#                    'sat': PAR_SAT}
# PAR_DEFAULT_SIM_POP = {'day_count': DAY_COUNT,
#                        't_day_count': TIMES_PER_DAY_COUNT,
#                        'tsaved_day_count': TIMES_SAVED_PER_DAY_COUNT,
#                        'step': STEP}


# Parameters
# ----------

# Experimental concentration initially / at dilution [cell].
C_EXP = np.array([300])  # 250, 300. Cas test: 15

# Number of parallelization.s per simulation.
# NB: s.t. PARA_COUNT[i] 1D array of the numbers of paralelizations to simulate
#     at concentration C_EXP[i].
# WARNING: parallelization might alter the outputs for small `C_EXP` since the
#     resulting subpopulations simulated in parallel are not independent, their
#     saturation times should be coupled. `PARA_COUNT > 1` can be used to study
#     the bias introduced by saturation.
PARA_COUNT = [np.array([1])]

# Number of times the simulation is run.
# NB: Ignored for parallel computing with `slurm_compute.batch`. In this case
#     the number of simulations is set through `#SBATCH --array=0-<to_adust>`
#     with  `<to_ajust>`equal to `SIMU_COUNT - 1`, in `slurm_compute.batch`.
SIMU_COUNT = 30  # 25, 20. Cas test: 3

# Modification of default model parameters given by `par.PAR_DEFAULT_POP`.
# Exemple 1. No updates, default parameters:
#    PAR_UPDATES = None
#
# Exemple 2. To modify p_accident, the rate of accidental death, one should use
#    P_EXIT_NEW = par.P_EXIT.deepcopy()
#    P_EXIT_NEW['accident'] = p_accident_new
#    PAR_UPDATES = {'p_exit': P_EXIT_NEW}
#
# Exemple 3. To modify weither or not type H are accounted:
#    PAR_UPDATES = {'htype': False}
#
# Exemple 4. To modified both previous:
#    PAR_UPDATES = {'htype': False,
#                   'p_exit': P_EXIT_NEW}
#
# ect.. for every key of `PAR_DEFAULT_POP`, making sure that the updated
# parameter conforms to the original format given by `PAR_DEFAULT_POP`.
PAR_UPDATES = None


# True to plot (only possible from local simulation).
IS_PLOT = True
if IS_PLOT:
    import telomeres.population.plot as pl

    IS_SAVED = False
    if IS_SAVED:
        FIG_DIRECTORY = "manuscript"
    else:
        FIG_DIRECTORY = None

    TMAX_TO_PLOT = 8  # In days. math.inf to plot all simulated times.
    ANC_GROUP_COUNT = 10


# Index of the simulation to run (from 1 to SIMU_COUNT)
# -----------------------------------------------------

is_run_in_parallel_from_slurm = "SLURM_ARRAY_TASK_ID" in os.environ.keys()

# If simulation run from slurm .batch file, parallel computation from slurm
# (not python). The current script runs one simulation only, but should be
# launched `SIMU_COUNT` times from parallel jobs.
if is_run_in_parallel_from_slurm:
    idx = int(os.environ["SLURM_ARRAY_TASK_ID"]) + 1  # Index of the job run.

# Otherwise, the script simulates the `SIMU_COUNT` simulations in serie.
else:
    idx = 1  # Initialisation with the first job index.


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
    print('simu_count: ' + str(SIMU_COUNT))


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

            file_path = join(sub_dir_path, f'output_{idx:02d}.npy')
            if not os.path.exists(file_path):
                outputs = sim.simu_parallel(para_count, cell_count)
                # Save dictionary 'output' and postreat.
                np.save(file_path, outputs)
                pps.postreat_from_evo_c(file_path)
            pps.postreat_from_evo_c_if_not_saved(file_path)

        # Otherwise computation in serie.
        else:
            for i in range(1, SIMU_COUNT + 1):
                np.random.seed(i)
                print(f'Simulation n° {i}/{SIMU_COUNT}')

                file_path = join(sub_dir_path, f'output_{i:02d}.npy')
                if not os.path.exists(file_path):
                    outputs = sim.simu_parallel(para_count, cell_count)
                    # Save dictionary 'output' and postreat.
                    np.save(file_path, outputs)
                    pps.postreat_from_evo_c(file_path)
                pps.postreat_from_evo_c_if_not_saved(file_path)

            # Average on all simulations.
            out_p = pps.postreat_performances(sub_dir_path, SIMU_COUNT)
            out_s = pps.statistics_simus_if_not_saved(sub_dir_path, SIMU_COUNT)
            print('Computation time (average):', out_p['computation_time'])

            # Plot average at `c_exp` and `para_count` fixed.
            if IS_PLOT:
                pl.plot_hist_lmin_at_sen(cell_count, para_count, SIMU_COUNT,
                                         FIG_DIRECTORY, day_count=7, width=4)
                pl.plot_evo_c_n_p_pcfixed_from_stat(
                    cell_count, para_count, SIMU_COUNT, FIG_DIRECTORY,
                    TMAX_TO_PLOT, psat=par.PROP_SAT,)
                pl.plot_evo_l_pcfixed_from_stat(
                    cell_count, para_count, SIMU_COUNT, FIG_DIRECTORY,
                    TMAX_TO_PLOT)
                pl.plot_evo_p_anc_pcfixed_from_stat(
                    cell_count, para_count, SIMU_COUNT, ANC_GROUP_COUNT,
                    FIG_DIRECTORY, TMAX_TO_PLOT)
                pl.plot_evo_gen_pcfixed_from_stat(
                    cell_count, para_count, SIMU_COUNT, FIG_DIRECTORY,
                    TMAX_TO_PLOT)
                print()
    run_idx += 1
