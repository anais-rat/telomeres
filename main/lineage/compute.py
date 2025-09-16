#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 17:51:29 2022

@author: arat

Script to compute all the lineage data necessary to plot the figures.

This is usefull if used to parallelize the `job_count` sets of `SIMU_COUNT`
simulations on a cluster through slrum using `slrum_compute.batch`.
In particular using `slrum_compute.batch`, the varible `#SBATCH --array` must
be set to `0-job_count`.

Otherwise, the present script will run the `job_count` required sets of
simulations in serie, which is not recommanded because very long.

The only parameter that should be adjusted is `PROC_COUNT`. This corresponds
to python's number of processor use to parallelize one set of `SIMU_COUNT`
simulations.
Warning: if slrum is used, the varible `#SBATCH --cpus-per-task` should been
change accordingly to `PROC_COUNT` (set to `= PROC_COUNT + 1` ideally).

"""

if __name__ == "__main__":  # Required on mac to use multiprocessing called in
    # telomeres.lineages.simulation for PROC_COUNT > 1.
    # Need to be removed if run from cluster_compute.batch.

    from copy import deepcopy
    import os
    import numpy as np

    from telomeres.lineage.simulation import (
        simulate_lineages_evolutions,
        compute_postreat_data_vs_exp,
    )
    import telomeres.model.parameters as par
    from telomeres.lineage.posttreat import count_exp_lineages

    # Reminder
    # --------
    # type_of_sort: 'gdeath', 'lmin', 'gnta1', 'gnta2', ..., 'gsen'.
    # gtrig keys: 'nta', 'sen' 'death'.
    # gtrig_to_compare: 'nta1', 'nta2', ..., 'sen' 'death'.
    # characteristics: 'atype', btype', 'htype', 'arrested1', 'arrested2', ...,
    #                  'senescent', 'dead', dead_accidentally', 'dead_naturally'.

    # Definition of parameters (common to all jobs)
    # ---------------------------------------------

    # Adjustable, depending on the machine or cluster from which is run the
    # script.

    PROC_COUNT = 11  # Number of processor used for parallel computing.
    # NB: if run form .batch script `cpu-per-task` should be `PROC_COUNT + 1`.
    #     if run locally, better not exceed your machine capacities.

    # Fixed

    SIMULATION_COUNT = 1000  # Number of simulations used to plot averages.

    CHARAC_S = [
        ["senescent"],
        ["atype", "senescent"],
        ["btype", "senescent"],
        ["btype", "arrested2", "senescent"],
        ["btype"],
    ]
    i1 = 2 * len(CHARAC_S) + 1

    CHARAC_S_2 = [
        ["senescent"],
        ["atype", "senescent"],
        ["btype", "senescent"],
        ["btype"],
    ]

    P_ACC_S = par.P_ACCIDENT * np.array([1.0, 10.0, 20, 30.0, 40.0, 50.0])
    P_ACC_S_TEST = np.array([0.054])  # par.P_ACCIDENT * [25, 34.8, 30, 40]
    P_ACC_S_TEST[0] = 0.054
    L_TRANS_S = np.array([-20, -10, 0, 10, 20, 40])
    L0_S = np.array([-40, -20, -10, 0, 10, 20])
    L1_S = np.array([-168, -84, -42, 0, 42, 84])
    # Not published, done afterwards
    LMODE_S = np.array([-20, -10, 0, 10, 20, 40])

    POSTREAT_DT = par.POSTREAT_DT

    i2 = len(P_ACC_S) * len(CHARAC_S_2)
    i3, i4, i5, i6 = np.array(
        [2 * len(L_TRANS_S), 2 * len(L0_S), len(L1_S), len(LMODE_S)]
    ) * len(CHARAC_S_2)
    i7 = len(P_ACC_S_TEST)

    # Computation
    # -----------

    # > NB Parameters for the job array.
    #  `idx` should ran from 0 to `job_count-1`.
    job_count = i1 + i2 + i3 + i4 + i5 + i6 + i7 + 2
    print("job_count", job_count)
    is_run_in_parallel_from_slurm = "SLURM_ARRAY_TASK_ID" in os.environ.keys()

    # If parallel computation run from sbacth command, only one idx computed.
    if is_run_in_parallel_from_slurm:
        idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
        idxs = [idx]
    # Otherwise computation in serie.
    else:
        idxs = np.arange(job_count)

    # Iteration on all jobs to run.
    for run_idx in idxs:
        print(f"\n Simulation n° {run_idx + 1} / {job_count}")

        # Best-fit parameters.
        if run_idx < len(CHARAC_S):  # Htype seen, everything saved.
            print("1: ", run_idx)
            characteristics = CHARAC_S[run_idx]
            lineage_count = count_exp_lineages(characteristics)
            simulate_lineages_evolutions(
                SIMULATION_COUNT,
                lineage_count,
                characteristics,
                proc_count=PROC_COUNT,
                is_lcycle_count_saved=True,
                is_evo_saved=True,
            )

        elif run_idx < i1 - 1:  # Htype unseen, no evolution array.
            print("2: ", run_idx)
            characteristics = CHARAC_S[run_idx - len(CHARAC_S)]
            lineage_count = count_exp_lineages(characteristics)
            simulate_lineages_evolutions(
                SIMULATION_COUNT,
                lineage_count,
                characteristics,
                proc_count=PROC_COUNT,
                par_update={"is_htype_seen": False},
                is_lcycle_count_saved=True,
            )

        elif run_idx < i1:
            print("2bis: ", run_idx)
            characteristics = ["senescent", "dead"]
            lineage_count = count_exp_lineages(characteristics)
            simulate_lineages_evolutions(
                SIMULATION_COUNT,
                lineage_count,
                characteristics,
                proc_count=PROC_COUNT,
                par_update={"is_htype_seen": False},
                is_lcycle_count_saved=False,
                is_evo_saved=True,
            )

        # Varying parameters.
        # i) Varying pdeath.
        elif run_idx < i1 + i2:
            print("3: ", run_idx)
            idx_par, idx_char = np.divmod(run_idx - i1, len(CHARAC_S_2))
            characteristics = CHARAC_S_2[idx_char]
            lineage_count = count_exp_lineages(characteristics)
            p_exit = deepcopy(par.P_EXIT)
            p_exit["accident"] = P_ACC_S[idx_par]
            simulate_lineages_evolutions(
                SIMULATION_COUNT,
                lineage_count,
                characteristics,
                proc_count=PROC_COUNT,
                par_update={"is_htype_seen": False, "p_exit": p_exit},
            )

        # ii) Varying ltrans.
        elif run_idx < i1 + i2 + i3:
            print("4: ", run_idx)
            p_exit = deepcopy(par.P_EXIT)
            if run_idx < i1 + i2 + i3 / 2:  # Normal types (TetO2-TLC1).
                ridx = run_idx
            else:  # Mutants (RAD51 with pacc x5)
                ridx = run_idx - int(i3 / 2)
                p_exit["accident"] = par.P_ACCIDENT * 5
            idx_par, idx_char = np.divmod(ridx - (i1 + i2), len(CHARAC_S_2))
            characteristics = CHARAC_S_2[idx_char]
            lineage_count = count_exp_lineages(characteristics)
            parameters = deepcopy(par.PAR)
            parameters[2][0] += L_TRANS_S[idx_par]

            simulate_lineages_evolutions(
                SIMULATION_COUNT,
                lineage_count,
                characteristics,
                proc_count=PROC_COUNT,
                par_update={
                    "is_htype_seen": False,
                    "fit": parameters,
                    "p_exit": p_exit,
                },
            )

        # iii) Varying l0.
        elif run_idx < i1 + i2 + i3 + i4:
            print("5: ", run_idx)
            p_exit = deepcopy(par.P_EXIT)
            if run_idx < i1 + i2 + i3 + i4 / 2:  # Normal types (TetO2-TLC1).
                ridx = run_idx
            else:  # Mutants (RAD51 with pacc x5).
                ridx = run_idx - int(i3 / 2)
                p_exit["accident"] = par.P_ACCIDENT * 5
            idx_par, idx_char = np.divmod(ridx - (i1 + i2 + i3), len(CHARAC_S_2))
            characteristics = CHARAC_S_2[idx_char]
            lineage_count = count_exp_lineages(characteristics)
            parameters = deepcopy(par.PAR)
            parameters[2][1] += L0_S[idx_par]
            simulate_lineages_evolutions(
                SIMULATION_COUNT,
                lineage_count,
                characteristics,
                proc_count=PROC_COUNT,
                par_update={
                    "is_htype_seen": False,
                    "fit": parameters,
                    "p_exit": p_exit,
                },
            )

        # iv) Varying l1.
        elif run_idx < i1 + i2 + i3 + i4 + i5:
            print("6: ", run_idx)
            idx_par, idx_char = np.divmod(
                run_idx - (i1 + i2 + i3 + i4), len(CHARAC_S_2)
            )
            characteristics = CHARAC_S_2[idx_char]
            lineage_count = count_exp_lineages(characteristics)
            parameters = deepcopy(par.PAR)
            parameters[2][2] += L1_S[idx_par]
            simulate_lineages_evolutions(
                SIMULATION_COUNT,
                lineage_count,
                characteristics,
                proc_count=PROC_COUNT,
                par_update={"is_htype_seen": False, "fit": parameters},
            )

        # v) Varying lmode.
        elif run_idx < i1 + i2 + i3 + i4 + i5 + i6:
            print("7: ", run_idx)
            idx_par, idx_char = np.divmod(
                run_idx - (i1 + i2 + i3 + i4 + i5), len(CHARAC_S_2)
            )
            characteristics = CHARAC_S_2[idx_char]
            lineage_count = count_exp_lineages(characteristics)
            parameters = deepcopy(par.PAR)
            parameters[2][0] += LMODE_S[idx_par]
            parameters[2][1] -= LMODE_S[idx_par]
            parameters[2][2] -= LMODE_S[idx_par]
            simulate_lineages_evolutions(
                SIMULATION_COUNT,
                lineage_count,
                characteristics,
                proc_count=PROC_COUNT,
                par_update={"is_htype_seen": False, "fit": parameters},
            )

        elif run_idx < i1 + i2 + i3 + i4 + i5 + i6 + i7:
            print("8: ", run_idx)
            idx_par = run_idx - (i1 + i2 + i3 + i4 + i5 + i6)
            characteristics = ["senescent"]
            lineage_count = count_exp_lineages(characteristics, strain="RAD51")
            p_exit = deepcopy(par.P_EXIT)
            p_exit["accident"] = P_ACC_S_TEST[idx_par]
            simulate_lineages_evolutions(
                SIMULATION_COUNT,
                lineage_count,
                characteristics,
                proc_count=PROC_COUNT,
                par_update={"is_htype_seen": False, "p_exit": p_exit},
            )

        else:  # Time vs generation evo with best-fit parameters.
            if run_idx % 2 == 0:
                par_update = {}
                compute_postreat_data_vs_exp(
                    SIMULATION_COUNT, ["senescent"], POSTREAT_DT, proc_count=PROC_COUNT
                )
            else:
                compute_postreat_data_vs_exp(
                    SIMULATION_COUNT, ["senescent"], POSTREAT_DT, proc_count=PROC_COUNT
                )
