#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 23:21:05 2022

@author: arat

Script to plot figures, related to population simulations, of Chapter 3 of the
PhD thesis and paper https://doi.org/10.1101/2023.11.22.568287 (and more).

The data needed to plot these figures can be computed, for every sets of
parameters (i.e. for given `para_count`, `cell_count` and `par_update`), with
the script `main.population.compute.py`. For every fixed set of parameters,
we run `SIMU_COUNT` simulations and average them. Ideally, they should be run
in parallel on a cluster (using `slrum_compute.batch` with the varible
`#SBATCH --array` set to `0-SIMU_COUNT`) otherwise it can be very long.

NB: to have the right `plt.rcParams`: run once up to the end of the Parameters
section, and run again.

"""

if __name__ == "__main__":  # Required on mac to use multiprocessing called in
    # telomeres.lineages.simulation for PROC_COUNT > 1.

    from copy import deepcopy
    import math
    import matplotlib.pyplot as plt
    import matplotlib.font_manager
    import numpy as np
    import seaborn as sns

    import project_path
    import telomeres.dataset.extract_processed_dataset as xtd
    import telomeres.auxiliary.figures_properties as fp
    import telomeres.model.parameters as par
    # import telomeres.auxiliary.parameters_functions as parf
    import telomeres.dataset.plot as pd
    import telomeres.model.plot as parp
    import telomeres.population.plot as pp


# ----------
# Parameters
# ----------

    IS_SAVED = True
    FORMAT = 'manuscript'

    # Global plotting parameters and figure directory.
    # .........................................................................

    # Figures directory if saved.
    FIG_DIR = None
    if IS_SAVED:
        FIG_DIR = FORMAT
    # Global plotting parameters.
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)  # Reset to default.
    if FORMAT == 'manuscript':
        sns.set_style("darkgrid")
        sns.set_context("talk", font_scale=1)
        plt.rcParams.update(fp.PAR_RC_UPDATE_MANUSCRIPT)
    elif FORMAT == 'article':
        sns.set_style("ticks")
        sns.set_context("poster", font_scale=1)
        plt.rcParams.update(fp.PAR_RC_UPDATE_ARTICLE)
    else:
        print("Redefine 'Format' correctly")
    print("Global plotting parameters: \n", sns.plotting_context(), '\n')
    # .........................................................................


# -----------------
# Experimental data
# -----------------

    if FORMAT == 'manuscript':
        bbox_to_anchor1 = (.4, 1)
        bbox_to_anchor2 = None
        lt, l0, l1 = par.PAR_L_INIT
        labels = [r'$f_0$', rf"$f_{{init}} (\cdot \, ; {int(lt)}, {int(l0)}, "
                  rf"{int(l1)})$"]
    else:
        bbox_to_anchor1 = (.22, .94)
        bbox_to_anchor2 = (.33, .68)
        labels = ["(Bourgeron et al., 2015)", r'$f_{init}$']

    # Daily concentrations.
    # Import experimental evolution of cell concentration.
    # > Telomerase negative / doxycycline positive.
    CSTAT_TELO_M = xtd.extract_population_concentration_doxP()
    # > Telomerase positive / doxycycline negative.
    CSTAT_TELO_P = xtd.extract_population_concentration_doxM()
    if FORMAT == 'manuscript':
        pd.plot_data_exp_concentration_curves(
            CSTAT_TELO_M['PD'], CSTAT_TELO_P['PD'], FIG_DIR,
            bbox_to_anchor=bbox_to_anchor1)
        pd.plot_data_exp_concentration_curves(
            CSTAT_TELO_M['c'], CSTAT_TELO_P['c'], None,
            bbox_to_anchor=bbox_to_anchor1)
        pd.plot_data_exp_concentration_curves(
            CSTAT_TELO_M['OD'], CSTAT_TELO_P['OD'], FIG_DIR,
            ylabel=fp.LABELS['ax_OD'], bbox_to_anchor=None,
            fig_name='concentration_OD')
    else:
        pd.plot_data_exp_concentration_curves(
            CSTAT_TELO_M['OD'], CSTAT_TELO_P['OD'], FIG_DIR,
            ylabel=fp.LABELS['ax_OD'], bbox_to_anchor=bbox_to_anchor1)

    # Daily telomere lengths.
    EVO_L_EXP = xtd.extract_population_lmode()
    pd.plot_data_exp_length_curves(
        np.arange(len(EVO_L_EXP[0])), np.mean(EVO_L_EXP, 0),
        np.std(EVO_L_EXP, 0), FIG_DIR)

    # Initial telomere lengths.
    pd.plot_ltelomere_init(FIG_DIR)
    pd.plot_ltelomere_init(FIG_DIR, par_l_init=par.PAR_L_INIT,
                           bbox_to_anchor=bbox_to_anchor2, labels=labels)


# ---------------------
# Sensitivity to N_init
# ---------------------

# Initial distibutions of telomere lengths
# ----------------------------------------

    CELL_COUNTS = np.array([2, 5, 10, 20, 50, 100, 200, 500, 1000, 1e5])

    if FORMAT == 'manuscript':
        parp.plot_distributions_shortest_min(CELL_COUNTS,
                                             fig_subdirectory=FIG_DIR)
        parp.plot_distributions_shortest(CELL_COUNTS, fig_subdirectory=FIG_DIR)


# Comparison
# ----------

    ANC_PROP = 0.5
    TMAX_TO_PLOT = 7

    CELL_COUNTS = np.array([2, 5, 10, 20, 50, 100, 200, 300, 500, 1000])
    XTICKS = [2, 5, 10, 20, 50, 100, 200, 500, 1000]
    SIMU_COUNTS = np.array([25] * len(CELL_COUNTS))
    SIMU_COUNTS[-1] = 15
    DAY_PLOTTED_COUNT = 7

    if FORMAT == 'manuscript':
        bbox_to_anchor = (1, 1)
        bbox_to_anchor2 = (1, 0.96)
        fig_size = (5.8, 4.)  # None  # (7.5, 4.8)
    else:
        bbox_to_anchor = (1, 1.16)
        bbox_to_anchor2 = (1, 1.1)
        fig_size = (7.8, 4.5)

    pp.plot_performances_pfixed(CELL_COUNTS, SIMU_COUNTS, FIG_DIR,
                                xticks=XTICKS, fig_size=fig_size)
    pp.plot_extinct_pfixed(CELL_COUNTS, SIMU_COUNTS, FIG_DIR, xticks=XTICKS,
                           fig_size=fig_size)
    pp.plot_sat_pfixed(CELL_COUNTS, SIMU_COUNTS, FIG_DIR, dsat_count_max=4,
                       xticks=XTICKS, fig_size=fig_size)
    pp.plot_p_pfixed(CELL_COUNTS, SIMU_COUNTS, FIG_DIR,
                     par_sim_update={'day_count': DAY_PLOTTED_COUNT},
                     xticks=XTICKS, fig_size=fig_size,
                     bbox_to_anchor=bbox_to_anchor2)
    pp.plot_evo_pfixed(CELL_COUNTS, SIMU_COUNTS, ANC_PROP, FIG_DIR,
                       TMAX_TO_PLOT, bbox_to_anchor=bbox_to_anchor)


# -------------------------------------
# Validation + accessing new quantities
# -------------------------------------

    PARA_COUNT = 1
    ANC_GROUP_COUNT = 10
    if FORMAT == 'manuscript':
        bbox_to_anchor = None
        fig_size = (5, 11)
        fig_size_gen = (6.8, 4.6)
    else:
        bbox_to_anchor = (1.05, .95)
        fig_size = (6, 11)
        fig_size_gen = (5.8, 3.8)

    # > N_init = 1000, k = 15.
    CELL_COUNT = 1000
    SIMU_COUNT = 15
    TMAX_S = [8, 9]  # Maximum time plotted (day).
    for tmax in TMAX_S:
        pp.plot_evo_c_n_p_pcfixed_from_stat(CELL_COUNT, PARA_COUNT, SIMU_COUNT,
                                            FIG_DIR, tmax)
        pp.plot_evo_l_pcfixed_from_stat(CELL_COUNT, PARA_COUNT, SIMU_COUNT,
                                        FIG_DIR, tmax)

    TMAX = 8
    pp.plot_evo_p_anc_pcfixed_from_stat(CELL_COUNT, PARA_COUNT,
                                        SIMU_COUNT, ANC_GROUP_COUNT,
                                        FIG_DIR, TMAX_TO_PLOT, is_old_sim=True)
    pp.plot_evo_gen_pcfixed_from_stat(CELL_COUNT, PARA_COUNT, SIMU_COUNT,
                                      FIG_DIR, TMAX_TO_PLOT,
                                      fig_size=fig_size_gen,
                                      bbox_to_anchor=bbox_to_anchor)

    # > N_init = 300, k = 30.
    CELL_COUNT = 300
    SIMU_COUNT = 30
    TMAX = 7
    # >> Default r_sat (= 1000).
    pp.plot_evo_p_anc_pcfixed_from_stat(CELL_COUNT, PARA_COUNT,
                                        SIMU_COUNT, ANC_GROUP_COUNT,
                                        FIG_DIR, TMAX_TO_PLOT)
    pp.plot_hist_lmin_at_sen(CELL_COUNT, PARA_COUNT, SIMU_COUNT, FIG_DIR,
                             day_count=TMAX, width=4,
                             bbox_to_anchor=bbox_to_anchor, fig_size=fig_size)

    # >> r_sat = 720.
    if FORMAT == 'manuscript':
        TMAX = 9
        R_SAT_NEW = 720

        PAR_SAT_NEW = deepcopy(par.PAR_SAT)
        PAR_SAT_NEW['prop'] = R_SAT_NEW
        PAR_UPDATE = {'sat': PAR_SAT_NEW}
        pp.plot_evo_c_n_p_pcfixed_from_stat(CELL_COUNT, PARA_COUNT, SIMU_COUNT,
                                            FIG_DIR, TMAX,
                                            par_update=PAR_UPDATE)
        pp.plot_evo_l_pcfixed_from_stat(CELL_COUNT, PARA_COUNT, SIMU_COUNT,
                                        FIG_DIR, TMAX, par_update=PAR_UPDATE)


# --------------------
# Sensitivity analysis
# --------------------

    c = 300
    p = 1
    t_max = 7
    anc_prop = 0.1


# Variable saturation ratio
# -------------------------

    simu_count = 30

    R_SAT_S = np.array([500, 750, 1000, 1500, 2000])
    LINESTYLES = ['-', '-', '--', '-', '-']

    par_sat = deepcopy(par.PAR_SAT)
    par_updates = []
    for r_sat in R_SAT_S:
        par_sat['prop'] = r_sat
        par_updates.append({'sat': deepcopy(par_sat)})

    curve_labels = [str(int(rsat)) for rsat in R_SAT_S]
    pp.plot_evo_w_variable(c, p, simu_count, par_updates, 'prop',
                           curve_labels, anc_prop, FIG_DIR, t_max,
                           linestyles=LINESTYLES, is_interpolated=False)


# Parameters redefinition.
    simu_count = 25


# Variable death rate
# -------------------

    P_DEATH_S = par.P_ACCIDENT * np.array([1., 10., 20, 30., 40., 50.])
    LINESTYLES = ['-', '-', '--', '-', '-', '-']

    p_exit = deepcopy(par.P_EXIT)
    par_updates = []
    curve_labels = []
    for p_death in P_DEATH_S:
        p_exit['accident'] = p_death
        par_updates.append({'p_exit': deepcopy(p_exit)})
        curve_labels.append(r'$\times$' + str(int(p_death / P_DEATH_S[0])))
    pp.plot_evo_w_variable(c, p, simu_count, par_updates, 'accident',
                           curve_labels, anc_prop, FIG_DIR, t_max,
                           linestyles=LINESTYLES)


# Variable maximum number of senescencent cycles
# -----------------------------------------------

    MAX_SEN_COUNT = np.array([2, 6, 10, math.inf])

    p_exit = deepcopy(par.P_EXIT)
    par_updates = []
    curve_labels = []
    for max_sen_count in MAX_SEN_COUNT:
        p_exit['sen_limit'] = max_sen_count
        par_updates.append({'p_exit': deepcopy(p_exit)})
        if max_sen_count == math.inf:
            curve_labels.append(r'$+\infty$')
        else:
            curve_labels.append(rf'{int(max_sen_count)}')
    pp.plot_evo_w_variable(c, p, simu_count, par_updates, 'sen_limit',
                           curve_labels, anc_prop, FIG_DIR, t_max)


# Variable initial distribution
# -----------------------------

    LTRANS, L0, L1 = par.PAR_L_INIT


# > Variable ltrans.
# ..................

    LTRANS_S = np.array([-20, -10, 0, 10, 20, 40])
    LINESTYLES = ['-', '-', '--', '-', '-', '-']

    parameters = deepcopy(par.PAR)
    par_updates = []
    curve_labels = []
    for ltrans_add in LTRANS_S:
        parameters[2][0] = LTRANS + ltrans_add
        par_updates.append({'fit': deepcopy(parameters)})
        if ltrans_add <= 0:
            curve_labels.append(str(ltrans_add))
        else:
            curve_labels.append('+' + str(ltrans_add))
    pp.plot_evo_w_variable(c, p, simu_count, par_updates, 'ltrans',
                           curve_labels, anc_prop, FIG_DIR, t_max,
                           linestyles=LINESTYLES)


# > Variable l0.
# ..............

    L0_S = np.array([-40, -20, -10, 0, 10, 20])
    LINESTYLES = ['-', '-', '-', '--', '-', '-']

    parameters = deepcopy(par.PAR)
    par_updates = []
    curve_labels = []
    for l0_add in L0_S:
        parameters[2][1] = L0 + l0_add
        par_updates.append({'fit': deepcopy(parameters)})
        if l0_add <= 0:
            curve_labels.append(str(l0_add))
        else:
            curve_labels.append('+' + str(l0_add))

    pp.plot_evo_w_variable(c, p, simu_count, par_updates, 'l0', curve_labels,
                           anc_prop, FIG_DIR, t_max, linestyles=LINESTYLES)


# > Variable l1.
# ..............

    L1_S = np.array([-168, -84, -42, 0, 42, 84])
    # l1_s = np.array([-80, -40, -20, -10, 0, 10, 20])
    LINESTYLES = ['-', '-', '-', '--', '-', '-']

    parameters = deepcopy(par.PAR)
    par_updates = []
    curve_labels = []
    for l1_add in L1_S:
        parameters[2][2] = L1 + l1_add
        par_updates.append({'fit': deepcopy(parameters)})
        if l1_add <= 0:
            curve_labels.append(str(l1_add))
        else:
            curve_labels.append('+' + str(l1_add))
    pp.plot_evo_w_variable(c, p, simu_count, par_updates, 'l1', curve_labels,
                           anc_prop, FIG_DIR, t_max, linestyles=LINESTYLES)


# # > Variable mode.
# # ................

#     # NB: Added afterwards, not simulated (and a fortiori neither published).
#     LMODE_S = np.array([-20, -10, 0, 10, 20, 40])
#     LINESTYLES = ['-', '-', '--', '-', '-', '-']
#     parameters = deepcopy(par.PAR)
#     par_updates = []
#     curve_labels = []
#     for lmode_add in LMODE_S:
#         parameters[2][0] = LTRANS + lmode_add
#         parameters[2][1] = L0 - lmode_add
#         parameters[2][2] = L1 - lmode_add
#         par_updates.append({'fit': deepcopy(parameters)})
#         if lmode_add <= 0:
#             curve_labels.append(str(lmode_add))
#         else:
#             curve_labels.append('+' + str(lmode_add))

#     pp.plot_evo_w_variable(c, p, simu_count, par_updates, 'lmode',
#                            curve_labels, anc_prop, FIG_DIR, t_max,
#                            linestyles=LINESTYLES)
