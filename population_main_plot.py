#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 23:21:05 2022

@author: arat
"""
import imp
from copy import deepcopy
import math
import matplotlib.pyplot as plt
import matplotlib.font_manager
import numpy as np
import os
import seaborn as sns

import aux_figures_properties as fp
import aux_parameters_functions as parf
import dataset_plot as pd
import parameters as par
import parameters_plot as parl
import population_plot as pp

imp.reload(par)

# NB: to plot at cell_count and para_count fixed run population_main_compute.py


# Parameters
# ----------


IS_SAVED = True
FORMAT = 'manuscript' # 'manuscript' or 'article'.
# WARNING this variable MIGHT need to be changed in 'lineages_plot.py' as well.


# Global plotting parameters and figure directory.
# .............................................................................
fig_dir = None
matplotlib.rcParams.update(matplotlib.rcParamsDefault) # Reset to default.
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{dsfont}'# pmatrix

if FORMAT == 'manuscript':
    if IS_SAVED:
        fig_dir = 'figures_manuscript'
        if (not os.path.exists(fig_dir)):
            os.makedirs(fig_dir)

    sns.set_style("darkgrid")
    sns.set_context("talk", font_scale = 1)

    plt.rcParams.update({'axes.facecolor': ".94",
                         'text.usetex': True,
                         'text.latex.preamble':r'\usepackage{amsfonts,dsfont}',
                         'figure.dpi': fp.DPI,
                         'font.family': "sans-serif", # latex-like: 'serif',
                         'font.sans-serif': "Helvetica", # ... 'cmr10'
                         'legend.frameon': True,
                         'legend.framealpha': 1,
                         'legend.facecolor': 'white',
                         'legend.edgecolor': '#EAEAF2',
                         'legend.fancybox': True,
                         'figure.figsize': (6.4, 4.1),
                         # 'font.size': 14,
                         'legend.title_fontsize': 15.5,
                         'legend.fontsize': 15
                        })
elif FORMAT == 'article':
    if IS_SAVED:
        fig_dir = 'figures_article'
        if (not os.path.exists(fig_dir)):
            os.makedirs(fig_dir)
    sns.set_style("ticks")
    sns.set_context("poster", font_scale = 1)
    plt.rcParams.update({'figure.dpi': 600,
                         'font.family': ['sans-serif'],
                         'font.sans-serif': ['Arial'],
                         'legend.frameon': False,
                         'font.size': 20,
                         'legend.title_fontsize': 18.5,
                         'legend.fontsize': 18})
else:
    print("Redefine 'Format' correctly")
print(sns.plotting_context())
# .............................................................................


# -----------------
# Experimental data
# -----------------

if FORMAT == 'manuscript':
    bbox_to_anchor1 = (.4, 1)
    bbox_to_anchor2 = None
    lt, l0, l1 = par.PAR_L_INIT
    labels = [r'$f_0$',
              rf"$f_{{init}} (\cdot \, ; {int(lt)}, {int(l0)}, {int(l1)})$"]
else:
    bbox_to_anchor1 = (.22, .94)
    bbox_to_anchor2 = (.33, .68)
    labels = ["(Bourgeron et al., 2015)", r'$f_{init}$']

# # Daily concentrations.
if FORMAT == 'manuscript':
    # pd.plot_data_exp_concentration_curves_final(par.C_AVG_M1, par.C_STD_M1,
    #                                             par.C_AVG_P1, par.C_STD_P1,
    #                                             fig_dir,
    #                                             bbox_to_anchor=bbox_to_anchor1)
    # pd.plot_data_exp_concentration_curves_final(par.C_AVG_M2, par.C_STD_M2,
    #                                             par.C_AVG_P2, par.C_STD_P2,
    #                                             None,
    #                                             bbox_to_anchor=bbox_to_anchor1)

    pd.plot_data_exp_concentration_curves_final(par.C_AVG_M3, par.C_STD_M3,
                                                par.C_AVG_P3, par.C_STD_P3,
                                                fig_dir,
                                                ylabel=fp.LABELS['ax_OD'],
                                                bbox_to_anchor=None,
                                                fig_name='concentration_OD')
# else:
#     pd.plot_data_exp_concentration_curves_final(par.C_AVG_M3, par.C_STD_M3,
#                                                 par.C_AVG_P3, par.C_STD_P3,
#                                                 fig_dir,
#                                                 ylabel=fp.LABELS['ax_OD'],
#                                                 bbox_to_anchor=bbox_to_anchor1)
# # Daily telomere lengths.
# pd.plot_data_exp_length_curves(np.arange(len(par.EVO_L_EXP[0])),
#                                 np.mean(par.EVO_L_EXP, 0),
#                                 np.std(par.EVO_L_EXP, 0), fig_dir)

# # Initial telomere lengths.
# pd.plot_l_init(par.l_init_support, par.l_init_prob, fig_dir)

# lnew, dnew = parf.transform_l_init(par.L_INIT_EXP, *par.PAR_L_INIT)
# pd.plot_transform_l_init(par.l_init_support, par.l_init_prob, lnew, dnew,
#                           fig_dir, *par.PAR_L_INIT,
#                           bbox_to_anchor=bbox_to_anchor2, labels=labels)

# par_new = [-20, 20, 20]
# lnew, dnew = parf.transform_l_init(par.L_INIT_EXP, *par_new)
# pd.plot_transform_l_init(par.l_init_support, par.l_init_prob, lnew, dnew,
#                           fig_dir, *par_new,
#                           bbox_to_anchor=bbox_to_anchor2, labels=labels)
# par_new = [40, -40, -40]
# lnew, dnew = parf.transform_l_init(par.L_INIT_EXP, *par_new)
# pd.plot_transform_l_init(par.l_init_support, par.l_init_prob, lnew, dnew,
#                           fig_dir, *par_new,
#                           bbox_to_anchor=bbox_to_anchor2, labels=labels)


# ---------------------
# Sensitivity to N_init
# ---------------------

# Initial distibutions of telomere lengths
# ----------------------------------------
CELL_COUNTS = np.array([2, 5, 10, 20, 50, 100, 200, 500, 1000, 1e5])

# if FORMAT == 'manuscript':
#     parl.plot_distributions_shortest_min(CELL_COUNTS, fig_supdirectory=fig_dir)
#     parl.plot_distributions_shortest(CELL_COUNTS, fig_supdirectory=fig_dir)


# Comparison
# ----------
ANC_PROP = 0.5
TMAX_TO_PLOT = 7

CELL_COUNTS = np.array([2, 5, 10, 20, 50, 100, 200, 300, 500, 1000])
XTICKS = [2, 5, 10, 20, 50, 100, 200, 500, 1000]
SIMU_COUNTS = np.array([25] * len(CELL_COUNTS))
SIMU_COUNTS[-1] = 15
DAY_PLOTTED_COUNT_B = 7
DAY_PLOTTED_COUNT_SEN = 7

if FORMAT == 'manuscript':
    bbox_to_anchor = (1, 1)
    bbox_to_anchor2 = (1, 0.96)
    fig_size = None #(7.5, 4.8)
else:
    bbox_to_anchor = (1, 1.16)
    bbox_to_anchor2 = (1, 1.1)
    fig_size = (7.8, 3.5)

# pp.plot_performances_pfixed(CELL_COUNTS, SIMU_COUNTS, fig_dir, xticks=XTICKS,
#                             fig_size=fig_size)
# pp.plot_extinct_pfixed(CELL_COUNTS, SIMU_COUNTS, fig_dir, xticks=XTICKS,
#                         fig_size=fig_size)
# pp.plot_sat_pfixed(CELL_COUNTS, SIMU_COUNTS, fig_dir, dsat_count_max=4,
#                     xticks=XTICKS, fig_size=fig_size)
# pp.plot_p_pfixed(CELL_COUNTS, SIMU_COUNTS, DAY_PLOTTED_COUNT_B,
#                   DAY_PLOTTED_COUNT_SEN, fig_dir, xticks=XTICKS,
#                   fig_size=fig_size, bbox_to_anchor=bbox_to_anchor2)
# pp.plot_evo_pfixed(CELL_COUNTS, SIMU_COUNTS, ANC_PROP, fig_dir, TMAX_TO_PLOT,
#                     bbox_to_anchor=bbox_to_anchor)
# pp.plot_hsl__pfixed()


# -------------------------------------
# Validation + accessing new quantities
# -------------------------------------
para_count = 1
ANC_GROUP_COUNT = 10
cell_count = 1000
simu_count = 15

if FORMAT == 'manuscript':
    bbox_to_anchor = None
    fig_size = (5, 11)
    fig_size_gen = (6.8, 4.6)
else:
    bbox_to_anchor = (1.05, .95)
    fig_size = (6, 11)
    fig_size_gen = (5.8, 3.8)

# TMAX_TO_PLOT = 9
# pp.plot_evo_c_n_p_pcfixed_from_stat(cell_count, para_count, simu_count,
#                                     fig_dir, TMAX_TO_PLOT)
# pp.plot_evo_l_pcfixed_from_stat(cell_count, para_count, simu_count,
#                                 fig_dir, TMAX_TO_PLOT)

TMAX_TO_PLOT = 8
# pp.plot_evo_c_n_p_pcfixed_from_stat(cell_count, para_count, simu_count,
#                                     fig_dir, TMAX_TO_PLOT)
# pp.plot_evo_l_pcfixed_from_stat(cell_count, para_count, simu_count,
#                                 fig_dir, TMAX_TO_PLOT)
# pp.plot_evo_p_anc_pcfixed_from_stat(cell_count, para_count,
#                                     simu_count, ANC_GROUP_COUNT,
#                                     fig_dir, TMAX_TO_PLOT, is_old_sim=True)
# pp.plot_evo_gen_pcfixed_from_stat(cell_count, para_count, simu_count,
#                                   fig_dir, TMAX_TO_PLOT, fig_size=fig_size_gen)

cell_count = 300
simu_count = 30
# pp.plot_evo_p_anc_pcfixed_from_stat(cell_count, para_count,
#                                     simu_count, ANC_GROUP_COUNT,
#                                     fig_dir, TMAX_TO_PLOT)
# pp.plot_hist_lmin_at_sen(cell_count, para_count, simu_count, fig_dir,
#                           day_count=7, width=4, bbox_to_anchor=bbox_to_anchor,
#                           fig_size=fig_size)


# --------------------
# Sensitivity analysis
# --------------------
c = 300
p = 1
simu_count = 30
t_max = 7
anc_prop = 0.1


# # Variable saturation ratio
# # -------------------------

# R_SAT_S = np.array([500, 750, 1000, 1500, 2000])
# LINESTYLES = ['-', '-', '--', '-', '-']

# par_updates = [{'prop_sat': rsat} for rsat in R_SAT_S]
# curve_labels = [str(int(rsat)) for rsat in R_SAT_S]
# pp.plot_evo_w_variable(c, p, simu_count, par_updates, 'psat',
#                        curve_labels, anc_prop, fig_dir, t_max,
#                        legend_title=fp.LABELS['prop_sat'],
#                        linestyles=LINESTYLES, is_interpolated=False)

# c = 300
# p = 1
# simu_count = 25
# t_max = 7
# anc_prop = 0.1


# # Variable death rate
# # -------------------

# P_DEATH_S = par.P_ACCIDENTAL_DEATH * np.array([1., 10., 20, 30., 40., 50.])
# LINESTYLES = ['-', '-', '--', '-', '-', '-']

# p_exit = deepcopy(par.P_EXIT)
# par_updates = []
# curve_labels = []
# for p_death in P_DEATH_S:
#     p_exit[0] = p_death
#     par_updates.append({'p_exit': deepcopy(p_exit)})
#     curve_labels.append(r'$\times$' + str(int(p_death / P_DEATH_S[0])))
# pp.plot_evo_w_variable(c, p, simu_count, par_updates, 'p_death_acc',
#                         curve_labels, anc_prop, fig_dir, t_max,
#                         legend_title=fp.LABELS['pdeath'],
#                         linestyles=LINESTYLES)


# # Variable maximum number of senescencent cycles
# # -----------------------------------------------

# MAX_SEN_COUNT = np.array([2, 6, 10, math.inf])

# p_exit = deepcopy(par.P_EXIT)
# par_updates = []
# curve_labels = []
# for max_sen_count in MAX_SEN_COUNT:
#     p_exit[3] = max_sen_count
#     par_updates.append({'p_exit': deepcopy(p_exit)})
#     curve_labels.append(rf'{max_sen_count}')
# pp.plot_evo_w_variable(c, p, simu_count, par_updates, 'max_sen_count',
#                         curve_labels, anc_prop, fig_dir, t_max,
#                         legend_title=fp.LABELS['max_sen_count'])

# # Variable initial distribution
# # -----------------------------

# parameters = par.PAR
# ltrans, l0, l1 = par.PAR_L_INIT

# > Variable mode.
# ................


# # > Variable ltrans.
# # ..................

# LTRANS_S = np.array([-20, -10, 0, 10, 20, 40])[::-1]
# LINESTYLES = ['-', '-', '-', '--', '-', '-']
# par_updates = []
# curve_labels = []
# temp = deepcopy(par.PAR_L_INIT)
# for ltrans_add in LTRANS_S:
#     temp[0] = ltrans + ltrans_add
#     par_updates.append({'par_l_init': deepcopy(temp)})
#     if ltrans_add <= 0:
#         curve_labels.append(str(ltrans_add))
#     else:
#         curve_labels.append('+' + str(ltrans_add))
# pp.plot_evo_w_variable(c, p, simu_count, par_updates, 'ltrans', curve_labels,
#                         anc_prop, fig_dir, t_max,
#                         legend_title=fp.LABELS['ltrans'],
#                         linestyles=LINESTYLES)


# # > Variable l0.
# # ..............

# L0_S = np.array([-40, -20, -10, 0, 10, 20])[::-1]
# LINESTYLES = ['-', '-', '--', '-', '-', '-']
# par_updates = []
# curve_labels = []
# temp = deepcopy(par.PAR_L_INIT)
# for l0_add in L0_S:
#     temp[1] = l0 + l0_add
#     par_updates.append({'par_l_init': deepcopy(temp)})
#     if l0_add <= 0:
#         curve_labels.append(str(l0_add))
#     else:
#         curve_labels.append('+' + str(l0_add))
# pp.plot_evo_w_variable(c, p, simu_count, par_updates, 'l0', curve_labels,
#                         anc_prop, fig_dir, t_max, legend_title=fp.LABELS['l0'],
#                         linestyles=LINESTYLES)

# # > Variable l1.
# # ..............

# L1_S = np.array([-168, -84, -42, 0, 42, 84])[::-1]
# LINESTYLES = ['-', '-', '--', '-', '-', '-']
# # l1_s = np.array([-80, -40, -20, -10, 0, 10, 20])
# par_updates = []
# curve_labels = []
# temp = deepcopy(par.PAR_L_INIT)
# for l1_add in L1_S:
#     temp[2] = l1 + l1_add
#     par_updates.append({'par_l_init': deepcopy(temp)})
#     if l1_add <= 0:
#         curve_labels.append(str(l1_add))
#     else:
#         curve_labels.append('+' + str(l1_add))
# pp.plot_evo_w_variable(c, p, simu_count, par_updates, 'l1', curve_labels,
#                         anc_prop, fig_dir, t_max, legend_title=fp.LABELS['l1'],
#                         linestyles=LINESTYLES)
