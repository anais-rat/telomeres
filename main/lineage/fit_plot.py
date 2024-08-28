#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 12:17:16 2024

@author: arat

Script to plot the results of the parameter estimations, some of them in the
annex of Chapter 3 of the PhD thesis and paper
https://doi.org/10.1101/2023.11.22.568287 (and more).

Ideally, the data needed to plot these figures should have been already
computed with the script `main.lineage.fit_compute.py` on a cluster in order to
parallelize all the "sets" of simulations (generally `SIMU_COUNT` simulation
per "set"). If not, the present script will run the required sets of
simulations in serie. which is not recommanded for big `SIMU_COUNT`values
because very long.

"""

if __name__ == "__main__":

    # from copy import deepcopy
    import matplotlib.pyplot as plt
    import matplotlib.font_manager
    import numpy as np
    import os
    import seaborn as sns
    import sys

    import fit_estimate as fct_fit
    import fit_compute as cfit
    import project_path
    from telomeres.model.plot import plot_laws, plot_laws_s
    import telomeres.auxiliary.figures_properties as fp
    from telomeres.dataset.extract_processed_dataset import \
        extract_postreat_lineages
    import telomeres.lineage.plot as pl


# Adjustable parameters
# ---------------------

    # Random seed (for reproducible figures).
    # np.random.seed(1)  # NB: comment to generate new random.

    # True to save figures.
    IS_SAVED = False

    # Plotting style: 'manuscript' or 'article'.
    FORMAT = 'manuscript'
    # FORMAT = 'article'


# Experimental data
# -----------------

    # Extraction and formatting

    # > 'TetO2-TLC1'
    DATA_EXP = extract_postreat_lineages()


# Fixed parameters  (no need to be redefined)
# -------------------------------------------

    # Number of simulations used to plot average quantities.
    SIMU_COUNT = 1000

    # Figures directory if saved.
    FIG_DIR = None
    if IS_SAVED:
        FIG_DIR = FORMAT
        # if (not os.path.exists(FIG_DIR)):
        #     os.makedirs(FIG_DIR)

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


# Results from parameters estimation
# ----------------------------------

    CHARACTERISTICS_S = [['btype'],
                         ['atype', 'senescent'],
                         ['senescent'],
                         ['btype', 'senescent']]
    LABELS = [r'$nta$', r'$sen_A$', r'$sen$', r'$sen_B$']

    if FORMAT == "manuscript":
        FIG_SIZE_PARS = (3.9, 2.2)
        FIG_SIZE = (4, 7.1)  # (5.7, 2.9)
        bbox_to_anchor = (0.15, 2)
        TICK_SPACING = 25
    else:
        FIG_SIZE = (5.5, 11)
        bbox_to_anchor = (1, 1)
        TICK_SPACING = 50

    for key, fits in cfit.FITS.items():
        if key in ['8_4', '8']:
            pass
        else:
            if key == '10':
                par_space = cfit.PAR_SPACES[key]
            else:
                par_space = cfit.PAR_SPACES[key[0]]

            if IS_SAVED:
                fig_sdir = f"{FIG_DIR}/parameters"
                if (not os.path.exists(fig_sdir)):
                    os.makedirs(fig_sdir)
                PATH = f"{fig_sdir}/fit_{key}_gcurves.pdf"
            else:
                PATH = None

            # All results.
            scores = [fct_fit.compute_n_plot_gcurves(
                fit, kwarg=par_space, is_plotted=False,
                simulation_count=SIMU_COUNT, is_printed=False) for fit in fits]
            par_sensitivity = [fct_fit.point_to_cost_fct_parameters(
                point, kwarg=par_space) for point in fits]
            plot_laws_s(par_sensitivity, idx_best=np.argmin(scores),
                        fig_name=key, fig_supdirectory=FIG_DIR, is_zoomed=True)
            print(f'\n {key}: ', scores, min(scores))

            # Best results.
            parameters = par_sensitivity[np.argmin(scores)]
            if FORMAT == 'manuscript':
                if key == '10':
                    FIG_SIZE = (5.5, 11)
                    bbox_to_anchor = None
                    FIG_SIZE_PARS = (5.5, 3)
                plot_laws(parameters, is_par_plot=True, fig_name=key,
                          fig_supdirectory=FIG_DIR, fig_size=FIG_SIZE_PARS,
                          tick_spacing=2 * TICK_SPACING)
            plot_laws(parameters, is_par_plot=False, fig_name=key,
                      fig_supdirectory=FIG_DIR, tick_spacing=TICK_SPACING)
            par_update = {'fit': parameters}
            pl.compute_n_plot_gcurves_wrt_charac(DATA_EXP, SIMU_COUNT,
                                                 CHARACTERISTICS_S, FIG_DIR,
                                                 par_update=par_update,
                                                 labels=LABELS, path=PATH,
                                                 bbox_to_anchor=bbox_to_anchor,
                                                 fig_size=FIG_SIZE)
