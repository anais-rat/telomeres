#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri Sep  2 15:54:16 2022

@author: arat

Script to plot the figures associated to the finalCut experiment.

"""

if __name__ == "__main__":  # Required on mac to use multiprocessing called in
                            # telomeres.lineages.simulation for PROC_COUNT > 1.

    from copy import deepcopy
    # import math  # Needed if DELAY = math.inf
    import matplotlib.pyplot as plt
    import matplotlib.font_manager
    import numpy as np
    import os
    import seaborn as sns
    import sys

    absolute_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(absolute_path)
    parent_dir = os.path.dirname(current_dir)
    projet_dir = os.path.dirname(parent_dir)
    sys.path.append(projet_dir)

    import telomeres.auxiliary.figures_properties as fp
    from telomeres.dataset.extract_processed_dataset import \
        extract_postreat_lineages
    import telomeres.finalCut.fit_cut_efficiency as fce
    import telomeres.auxiliary.write_paths as wp
    import telomeres.lineage.plot as pl
    from telomeres.model.plot import plot_laws
    import process_dataset as pst


# ----------
# Parameters
# ----------

# "Adjustable"
# ------------

    # True to save figures.
    IS_SAVED = False

    # Plotting style: 'manuscript' or 'article'.
    FORMAT = 'manuscript'
    # FORMAT = 'article'

    # Number of processor used for parallel computing.
    PROC_COUNT = 11  # Add one for cluster.

    # Number of simulations used to plot average quantities.
    SIMU_COUNT = 200

    # New laws (compared to (Rat et al.))!
    PAR = ([0.0247947389, 0.440063202],
           [[0.17, 0.65, 36.0], [2.45423414e-06, 0.122028128, 0.0]],
           [0, 40, 58.0])

    IDX_DOX = 0
    IDX_GAL = 6 * 6  # 6h (= 6x6 [10min]) after Dox addition.
    IDX_RAF = 12 * 6  # 9 * 6  # 12h after Dox addition.
    IDXS_FRAME = [IDX_DOX, IDX_GAL, IDX_RAF]

    # Time [h] during which a cell can still undergo a cut after Galactose
    # removal. math.inf for infinite delay.
    DELAY = 9

    PAR_FINAL_CUT = {'idxs_frame': IDXS_FRAME,
                     'delay': 9}  # [h]

    LENGTHS_CUT = [None, 0, 20, 30, 40, 50, 70]

    KEYS = ['noFc_n2', 'Fc0_n2', 'Fc20_n2', 'Fc30_n2', 'Fc40_n2', 'Fc50_n2',
            'Fc70_n2']


# Fixed
# -----

    # Random seed (for reproducible figures).
    # np.random.seed(1)  # NB: comment to generate new random.

    # Figures directory if saved.
    FIG_DIR = None
    if IS_SAVED:
        FIG_DIR = os.path.join(wp.FOLDER_FC, FORMAT)

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

    CUTS = [key[2:4] for key in KEYS]  # [None, 0, 20, 30, 40, 50, 70]
    for i in range(len(CUTS)):
        if CUTS[i] == '0_':
            CUTS[i] = 0
        elif CUTS[i] == 'Fc':
            CUTS[i] = None
        else:
            CUTS[i] = int(CUTS[i])
    LENGTHS_CUT = {KEYS[i]: CUTS[i] for i in range(len(CUTS))}


# -----------------
# Experimental data
# -----------------

    # 'TetO2-TLC1'.
    DATA_EXP_TET = extract_postreat_lineages(strain='TetO2-TLC1')

    # FinalCut.
    DATA_EXP_FC = pst.DATA_EXP


# ----
# Plot
# ----


# Cycle duration times
# --------------------

    if FORMAT == "manuscript":
        FIG_SIZE = (4.7, 8)
        FONT_SIZE = 24
    else:
        FIG_SIZE = (5.8, 9.5)
        FONT_SIZE = sns.plotting_context()['axes.labelsize']
    IS_EXP = True

    for key in ['noFc_n2']:  # data in DATA_EXP_FC.items():
        data = DATA_EXP_FC[key]
        cycles = data[0]['cycle']
        gmax = np.shape(cycles)[1]  # Maximum lineage length in the dataset.

        pl.plot_lineages_cycles(cycles, IS_EXP, FIG_DIR, FONT_SIZE,
                                lineage_types=data[2], gmax=gmax,
                                fig_size=FIG_SIZE)
        pl.plot_lineages_cycles(cycles, IS_EXP, FIG_DIR,
                                sns.plotting_context()['axes.labelsize'],
                                is_dead=data[1]['death'], gmax=gmax,
                                fig_size=(5.8, 9.5))


# Parameters
# ----------

    # > Fit of the law of cut

    x_s = np.linspace(0, 15, 100)
    plt.figure(dpi=600)
    plt.xlabel('Time (h)')
    plt.ylabel('Proportion of cut')
    plt.plot(x_s, fce.fit_cdf(x_s), label='Simulation', color='darkorange')
    plt.plot(fce.x_exp, 1 - fce.y_exp, 'x', label='Experiment', color='black')
    plt.xticks(np.arange(15))
    plt.legend()
    if not isinstance(FIG_DIR, type(None)):
        plt.savefig(os.path.join(FIG_DIR, 'fit_prop_cut.pdf'),
                    bbox_inches='tight')
    plt.show()

    # > Laws of arrest
    plot_laws(PAR, fig_supdirectory=FIG_DIR, fig_name=f'fc-lminA{PAR[0][-1]}',
              is_par_plot=True)
    plot_laws(PAR, fig_name=f'fc-lminA{PAR[0][-1]}', is_par_plot=False)


# Generations of arrest
# ---------------------

    CHARACTERISTICS_S = [['btype'],
                         ['atype', 'senescent'],
                         ['senescent'],
                         ['btype', 'senescent']]

    LABELS = [r'$nta$', r'$sen_A$', r'$sen$', r'$sen_B$']

    # > Usual conditions (no finalCut, Dox at time 0) w FinalCut parameters.
    # Comparison to 'TetO2-TLC1' data (Rat et al.)

    if FORMAT == "manuscript":
        FIG_SIZE_PARS = (3.9, 2.2)
        FIG_SIZE = (4.5, 11)  # (5.7, 2.9)
        bbox_to_anchor = (0.15, 2)
    else:
        FIG_SIZE = (5.5, 11)
        bbox_to_anchor = (1, 1)

    par_update = {'is_htype_seen': False,
                  'fit': PAR}
    pl.compute_n_plot_gcurves_wrt_charac(DATA_EXP_TET, SIMU_COUNT,
                                         CHARACTERISTICS_S, FIG_DIR,
                                         par_update=par_update,
                                         labels=LABELS,
                                         bbox_to_anchor=bbox_to_anchor,
                                         fig_size=FIG_SIZE,
                                         proc_count=PROC_COUNT)

    # > FinalCut conditions.

    if FORMAT == "manuscript":
        BBOX_TO_ANCHOR = None
    else:
        BBOX_TO_ANCHOR = (1.3, 0)
    TICK_SPACING = 5

    for key in KEYS:
        data = DATA_EXP_FC[key]
        par_final_cut = deepcopy(PAR_FINAL_CUT)
        par_final_cut['lcut'] = LENGTHS_CUT[key]

        par_update = {'finalCut': deepcopy(par_final_cut),
                      'is_htype_seen': False,
                      'fit': PAR}
        print(key, par_update, LENGTHS_CUT[key])

        # Simulation vs experiment.
        pl.compute_n_plot_gcurve(data, SIMU_COUNT, ['senescent'], FIG_DIR,
                                 par_update=par_update, is_exp_plotted=True,
                                 title=key.replace('_', ' '), is_propB=True,
                                 proc_count=PROC_COUNT, tick_spacing=5,
                                 bbox_to_anchor=BBOX_TO_ANCHOR)
        pl.compute_n_plot_gcurve(data, SIMU_COUNT, ['dead'], FIG_DIR,
                                 par_update=par_update, is_exp_plotted=True,
                                 title=key.replace('_', ' '), is_propB=True,
                                 proc_count=PROC_COUNT)

        # Experiment.
        pl.plot_gcurves_exp(data, CHARACTERISTICS_S, FIG_DIR, labels=LABELS,
                            is_gathered=False, fig_size=(5.8, 3.6),
                            title=key.replace('_', ' '), add_to_name=key)
