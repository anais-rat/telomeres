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
If you use `slrum_compute.batch`, the varible `#SBATCH --array` must be set to
`0-SIMU_COUNT`. Moreover, the script will need to be run twice.
One with `slrum_compute.batch` to simulate `SIMU_COUNT` populations, and once
with `slrum_posttreat.batch` to postreat and average them.

Contrarily to `main.lineage.compute` the present script will not run all the
simulations needed. The user need to adjust manually the parameters `C_EXP`,
`SIMU_COUNT` and `PAR_UPDATES` (and possibly `PARA_COUNT` but it is 1 in our
published data). The reason is that population simulations are way longer than
lineage simulations and it is more efficient to run them separately.

"""

from copy import deepcopy
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

# Plotting options.
IS_PLOT = False  # True to plot (only possible from local simulation).
# .............................................................................
if IS_PLOT:
    STRAIN = 'TetO2-TLC1'
    import telomeres.population.plot as pl
    FIG_DIRECTORY = None  # Figures not saved (since rcParams not updated).
    TMAX_TO_PLOT = 11  # In days. math.inf to plot all simulated times.
    ANC_GROUP_COUNT = 10
    PAR_UPDATE_BIS = None
# .............................................................................

# Posttreat options.
IS_DATA_EXTRACTED_TO_CSV = True


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
#    PAR_UPDATES = {'is_htype_accounted': False}
#
# Exemple 4. To modified both previous:
#    PAR_UPDATES = {'is_htype_accounted': False,
#                   'p_exit': P_EXIT_NEW}
#
# ect.. for every key of `PAR_DEFAULT_POP`, making sure that the updated
# parameter conforms to the original format given by `PAR_DEFAULT_POP`.

# > Default Case.
# .............................................................................
PAR_UPDATE = None
# .............................................................................


# > Default with best-fit r_sat (Uncomment to select these parameters).
# .............................................................................
# R_SAT_NEW = 720  # New r_sat value.

# PAR_SAT_NEW = deepcopy(par.PAR_SAT)
# PAR_SAT_NEW['prop'] = R_SAT_NEW
# PAR_UPDATE = {'sat': PAR_SAT_NEW}
# .............................................................................


# > Case Pol32 (Uncomment for Pol32 parameters).
# .............................................................................
# STRAIN = 'POL32'  # Needed only to plot.
# L_TRANS_NEW = 40  # Translation of 40.
# PAR_NEW = deepcopy(par.PAR)
# PAR_NEW[2][0] = L_TRANS_NEW

# PROP_SAT_NEW = 402  # New r_sat value = 2.01e8 / 5e5, 2.01 mean on telo+.
# PAR_SAT_NEW = deepcopy(par.PAR_SAT)
# PAR_SAT_NEW['prop'] = PROP_SAT_NEW

# PAR_UPDATE = {'fit': PAR_NEW,
#               'sat': PAR_SAT_NEW}
# .............................................................................

# > Case Rad51 (Uncomment for Rad51 parameters).
# .............................................................................
# STRAIN = 'RAD51'  # Needed only to plot.
# PROP_SAT_NEW = 560  # New r_sat value.
# PAR_SAT_NEW = deepcopy(par.PAR_SAT)
# PAR_SAT_NEW['prop'] = PROP_SAT_NEW

# P_EXIT_NEW = deepcopy(par.P_EXIT)
# P_EXIT_NEW['accident'] = 5.4 / 100  # New p_accident, the value of RAD51.

# PAR_UPDATE = {'sat': PAR_SAT_NEW,
#               'p_exit': P_EXIT_NEW
#               }

# PAR_UPDATE_BIS = {'sat': PAR_SAT_NEW}  # Needed only to plot.
# .............................................................................


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


# Printing of default parameters
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

    for para_count in PARA_COUNT[run_idx]:
        print('Number of parallelizations: ', para_count)

        # If parallel computation run from sbacth command.
        if is_run_in_parallel_from_slurm:
            # If not already done, creation of a folder for simulations.
            # NB: made only for the first simu to avoid creation of the subdir
            #   inside the if from another job when parallel compputing is run.
            sub_dir_path = wp.write_simu_pop_subdirectory(
                cell_count, para_count, par_update=PAR_UPDATE)
            if not os.path.exists(sub_dir_path) and idx == 1:
                os.makedirs(sub_dir_path)

            print('Simulation n°', idx)
            time.sleep(10)  # To avoid non created subdirectory.
            np.random.seed(idx)  # To ensure reproducibility.

            sim.simu_parallel(para_count, cell_count, output_index=idx,
                              par_update=PAR_UPDATE)
            pps.postreat_from_evo_c(para_count, cell_count, idx,
                                    par_update=PAR_UPDATE)

        # Otherwise computation in serie.
        else:
            for i in range(1, SIMU_COUNT + 1):
                np.random.seed(i)
                print(f'Simulation n° {i}/{SIMU_COUNT}')
                t_start = time.time()

                sim.simu_parallel(para_count, cell_count, output_index=i,
                                  par_update=PAR_UPDATE)
                pps.postreat_from_evo_c(para_count, cell_count, i,
                                        par_update=PAR_UPDATE)
                t_end = time.time()
                print(t_end - t_start)

            # Average on all simulations.
            out_p = pps.postreat_performances(
                para_count, cell_count, SIMU_COUNT, par_update=PAR_UPDATE)
            out_s = pps.statistics_simus(
                para_count, cell_count, SIMU_COUNT, par_update=PAR_UPDATE)
            print('Computation time (average):', out_p['computation_time'])

            # Plot average at `c_exp` and `para_count` fixed.
            if IS_PLOT:
                pl.plot_hist_lmin_at_sen(
                    cell_count, para_count, SIMU_COUNT, FIG_DIRECTORY,
                    day_count=7, width=4, par_update=PAR_UPDATE)
                pl.plot_evo_c_n_p_pcfixed_from_stat(
                    cell_count, para_count, SIMU_COUNT, FIG_DIRECTORY,
                    TMAX_TO_PLOT, par_update=PAR_UPDATE, strain=STRAIN,
                    par_update_bis=PAR_UPDATE_BIS)
                pl.plot_evo_l_pcfixed_from_stat(
                    cell_count, para_count, SIMU_COUNT, FIG_DIRECTORY,
                    TMAX_TO_PLOT, par_update=PAR_UPDATE)
                pl.plot_evo_p_anc_pcfixed_from_stat(
                    cell_count, para_count, SIMU_COUNT, ANC_GROUP_COUNT,
                    FIG_DIRECTORY, TMAX_TO_PLOT, par_update=PAR_UPDATE)
                pl.plot_evo_gen_pcfixed_from_stat(
                    cell_count, para_count, SIMU_COUNT, FIG_DIRECTORY,
                    TMAX_TO_PLOT, par_update=PAR_UPDATE)
                print()
    run_idx += 1

    if not is_run_in_parallel_from_slurm and IS_DATA_EXTRACTED_TO_CSV:
        pps.statistics_simus_csv(para_count, cell_count, SIMU_COUNT,
                                 par_update=PAR_UPDATE)
