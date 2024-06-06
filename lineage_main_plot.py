#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 15:54:16 2022

@author: arat

Script to plot figures of Chapter 3 (and more).
Ideally, the data needed to plot these figures should have been already
computed with the script `main_lineage.py`, ideally using parallel computing
through slrum. If not, the present script will run the required simulations in
serie, which is not recommanded because very long.

"""

from copy import deepcopy
import imp
import matplotlib.pyplot as plt
import matplotlib.font_manager
import numpy as np
import os
import seaborn as sns
import scipy.io as sio

import aux_parameters_functions as parf
import aux_figures_properties as fp
import dataset_plot as pd
import lineage_plot as pl
imp.reload(pl)
import lineage_simulation as sim
# imp.reload(sim)
import parameters as par
imp.reload(par)


# Random seed (for reproducible figures).
# NB: Uncomment to generate new random.
np.random.seed(1)

# --------
# Reminder
# --------
# type_of_sort: 'gdeath', 'lmin', 'gnta1', 'gnta2', ..., 'gsen'.
# gtrig keys: 'nta', 'sen' 'death'.
# gtrig_to_compare: 'nta1', 'nta2', ..., 'sen' 'death'.
# characteristics: 'atype', btype', 'htype', 'arrested1', 'arrested2', ...,
#                  'senescent', 'dead', dead_accidentally', 'dead_naturally'.


# ----------
# Parameters
# ----------

IS_SAVED = False
FORMAT = 'manuscript'  # 'manuscript' or 'article'.
# WARNING this variable MIGHT need to be changed in 'lineages_plot.py' as well.

SIMU_COUNT = 1000
PROC_COUNT = 1 # Add one for cluster.

THRESHOLD = 18
GEN_COUNT_BY_LINEAGE_MIN = 1
HIST_LMIN_X_AXIS = np.linspace(0, 250, 251)

POSTREAT_DT = par.CYCLE_MIN


# > Global plotting parameters (no need to be redefined)
# .............................................................................
fig_dir = None

# Global plotting parameters and figure directory.
matplotlib.rcParams.update(matplotlib.rcParamsDefault)  # Reset to default.
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{dsfont}'  # pmatrix

if IS_SAVED:
    fig_dir = 'figures/' + FORMAT
    if (not os.path.exists(fig_dir)):
        os.makedirs(fig_dir)

if FORMAT == 'manuscript':
    sns.set_style("darkgrid")
    sns.set_context("talk", font_scale=1)

    plt.rcParams.update({'axes.facecolor': ".94",
                         'text.usetex': True,
                         'text.latex.preamble':r'\usepackage{amsfonts,dsfont}',
                         'figure.dpi': fp.DPI,
                         'font.family': "sans-serif",  # latex-like: 'serif',
                         'font.sans-serif': "Helvetica",  # ... 'cmr10'
                         'legend.frameon': True,
                         'legend.framealpha': 1,
                         'legend.facecolor': 'white',
                         'legend.edgecolor': '#EAEAF2',
                         'legend.fancybox': True,
                         # 'figure.figsize': (6.4, 4.1),
                         # 'font.size': 14,
                         'legend.title_fontsize': 15.5,
                         'legend.fontsize': 15
                         })
elif FORMAT == 'article':
    sns.set_style("ticks")
    sns.set_context("poster", font_scale=1)
    plt.rcParams.update({'figure.dpi': 600,
                         'font.family': ['sans-serif'],
                         'font.sans-serif': ['Arial'],
                         'legend.frameon': False
                         # 'font.size': 20,
                         # 'legend.fontsize': 18
                         })
else:
    print("Redefine 'Format' correctly")
print(sns.plotting_context())
# .............................................................................


# -----------------
# Experimental data
# -----------------

# > Extraction and formatting
DATA_EXP = sio.loadmat('data/microfluidic/TelomeraseNegative.mat')
DATA_EXP = DATA_EXP['OrdtryT528total160831']
DATA_EXP = sim.postreat_experimental_lineages(DATA_EXP, par.THRESHOLD,
                                              par.GEN_COUNT_BY_LINEAGE_MIN)

DATA_EXP_MUTANT = sio.loadmat('data/microfluidic/rad51/TelomeraseNegMutantRAD51.mat')
DATA_EXP_MUTANT = DATA_EXP_MUTANT['OrdtrRAD51D']
DATA_EXP_MUTANT = sim.postreat_experimental_lineages(DATA_EXP_MUTANT,
                                                     par.THRESHOLD, 2)

DATA_EXP_MUTANT_SEN = sim.select_exp_lineages(DATA_EXP_MUTANT, ['senescent'])
DATA_EXP_MUTANT_SEN = sim.sort_lineages(DATA_EXP_MUTANT_SEN, 'gsen')
GSEN_EXP_MUTANT = DATA_EXP_MUTANT_SEN[1]['sen']


# -------------------
# Plot simulated data
# -------------------

# Cycle duration times
# --------------------


# if FORMAT == "manuscript":
#     FIG_SIZE = (4.7, 8)
#     FONT_SIZE = 24
# else:
#     FIG_SIZE = (5.8, 9.5)
#     FONT_SIZE = sns.plotting_context()['axes.labelsize']

# # > Experimental.
# IS_EXP = True

# cycles_exp = DATA_EXP[0]['cycle']
# GMAX = np.shape(cycles_exp)[1]

# # # Distributions of dycle duration time (cdt) per category.
# # if IS_SAVED and (not os.path.exists(fig_dir + '/' + pd.FDIR_DAT)):
# #     os.makedirs(fig_dir + '/' + pd.FDIR_DAT)
# # pd.plot_cycles_from_dataset(fig_dir, IS_SAVED)

# # Cycle duration times in generation and lineage.
# if FORMAT == 'manuscript': # With legend for types.
#     pl.plot_lineages_cycles(cycles_exp, IS_EXP, fig_dir, FONT_SIZE,
#                             lineage_types=DATA_EXP[2], gmax=GMAX,
#                             fig_size=FIG_SIZE)
#     pl.plot_lineages_cycles(cycles_exp, IS_EXP, fig_dir,
#                             sns.plotting_context()['axes.labelsize'],
#                             is_dead=DATA_EXP[1]['death'], gmax=GMAX,
#                             fig_size=(5.8, 9.5))
# else: # without legend.
#     pl.plot_lineages_cycles(cycles_exp, IS_EXP, fig_dir, FONT_SIZE, gmax=GMAX,
#                             fig_size=FIG_SIZE)
#     cycles_exp_mutant = DATA_EXP_MUTANT[0]['cycle']
#     cycles_exp_mutant_sen = DATA_EXP_MUTANT_SEN[0]['cycle']
#     GMAX_MUTANT = np.shape(cycles_exp_mutant)[1]
#     pl.plot_lineages_cycles(cycles_exp_mutant, IS_EXP, fig_dir, FONT_SIZE,
#                             gmax=None, add_to_name='rad51', fig_size=FIG_SIZE)
#     pl.plot_lineages_cycles(cycles_exp_mutant_sen, IS_EXP, fig_dir, FONT_SIZE,
#                             gmax=None, add_to_name='rad51_sen', fig_size=FIG_SIZE)

# # > Simulated.
# IS_EXP = False

# # >> Type H unseen.
# data = sim.simulate_lineages_evolution(len(cycles_exp), ['senescent'],
#                                         is_htype_seen=False, is_evos=True)
# data = sim.sort_lineages(data, 'gdeath')
# # pl.plot_lineages_cycles(data[0]['cycle'], IS_EXP, fig_dir, FONT_SIZE,
# #                         gmax=GMAX, fig_size=FIG_SIZE)

# if FORMAT == 'manuscript': # With legend for types.
#     pl.plot_lineages_cycles(data[0]['cycle'], IS_EXP, fig_dir, FONT_SIZE,
#                             lineage_types=data[2], gmax=GMAX,
#                             fig_size=FIG_SIZE)
# elif FORMAT == 'article': # 2 other simulations.
#     for seed in [2, 3]:
#         np.random.seed(seed)
#         data = sim.simulate_lineages_evolution(len(cycles_exp), ['senescent'],
#                                                 is_htype_seen=False,
#                                                 is_evos=True)
#         data = sim.sort_lineages(data, 'gdeath')
#         pl.plot_lineages_cycles(data[0]['cycle'], IS_EXP, fig_dir, FONT_SIZE,
#                                 gmax=GMAX, add_to_name=str(seed),
#                                 fig_size=FIG_SIZE)
#     np.random.seed(1)

# # >> Type H seen.
# if FORMAT == 'manuscript':
#     data = sim.simulate_lineages_evolution(len(cycles_exp), ['senescent'],
#                                             is_htype_seen=True, is_evos=True)
#     data = sim.sort_lineages(data, 'gdeath')
#     pl.plot_lineages_cycles(data[0]['cycle'], IS_EXP, fig_dir, FONT_SIZE,
#                             lineage_types=data[2], gmax=GMAX,
#                             fig_size=FIG_SIZE)


# Average of 2D matrices
# ----------------------

# CHARACTERISTICS = ['senescent']
# TYPES_OF_SORT = ['gsen', 'lmin', 'lavg']
# IS_EXP = False
# data = sim.compute_evo_avg_data(DATA_EXP, SIMU_COUNT, CHARACTERISTICS,
#                                 types_of_sort=TYPES_OF_SORT,
#                                 proc_count=PROC_COUNT)

# bbox_to_anchor = (.45, 1.17)
# if FORMAT == 'manuscript':
#     bbox_to_anchor=None

# # > Cycle duration times.
# for key_sort in TYPES_OF_SORT:
#     pl.plot_lineages_cycles(data[key_sort][0]['cycle'], IS_EXP, fig_dir,
#                             FONT_SIZE, fig_size=FIG_SIZE,
#                             curve_to_plot=data[key_sort][1]['sen']['mean'],
#                             evo_avg={'simu_count': SIMU_COUNT,
#                                       'type_of_sort': key_sort},
#                             bbox_to_anchor=bbox_to_anchor)

# TYPE_OF_SORT = 'gdeath'
# data_ = sim.compute_evo_avg_data(DATA_EXP, SIMU_COUNT, ['senescent', 'dead'],
#                                 types_of_sort=[TYPE_OF_SORT],
#                                 proc_count=PROC_COUNT)
# pl.plot_lineages_cycles(data_[TYPE_OF_SORT][0]['cycle'], IS_EXP, fig_dir,
#                         FONT_SIZE, fig_size=FIG_SIZE,
#                         curve_to_plot=data_[TYPE_OF_SORT][1]['sen']['mean'],
#                         evo_avg={'simu_count': SIMU_COUNT,
#                                   'type_of_sort': TYPE_OF_SORT},
#                         bbox_to_anchor=bbox_to_anchor)

# # > Proportion of type-B.
# for key_sort in ['lmin', 'gsen', 'lavg']:
#     pl.plot_lineage_avg_proportions(data[key_sort][0]['prop_btype'], True,
#                             fig_dir, 22,
#                             curve_to_plot=data[key_sort][1]['sen']['mean'],
#                             evo_avg={'simu_count': SIMU_COUNT,
#                                      'type_of_sort': key_sort})

# .............................................................................
if FORMAT == 'article': # Reestablish rcParam modified by plot_lineages_cycles.
    sns.set_style("ticks")
    sns.set_context("poster", font_scale = 1)
    # sns.set_style("darkgrid")
    # sns.set_context("talk", font_scale = 1)
    plt.rcParams.update({'figure.dpi': 600,
                         'font.family': ['sans-serif'],
                         'font.sans-serif': ['Arial'],
                         'legend.frameon': False})
# .............................................................................


# Time/generation evo postreat
# ----------------------------

CHARACTERISTICS = ['senescent']

# pl.compute_n_plot_postreat_time_vs_gen(DATA_EXP, SIMU_COUNT, CHARACTERISTICS,
#                                         POSTREAT_DT, is_htype_seen=False)
# pl.compute_n_plot_postreat_time_vs_gen(DATA_EXP, SIMU_COUNT, CHARACTERISTICS,
#                                         POSTREAT_DT, is_htype_seen=True)


# Generation curves wrt experimental
# ----------------------------------

CHARACTERISTICS_S = [['btype'],
                      ['atype','senescent'],
                      ['senescent'],
                      ['btype','senescent']]

# if FORMAT == 'manuscript':
#     LABELS = [r'$G^{-1}_{{nta}}$', r'$G^{-1}_{{sen_A}}$',
#               r'$G^{-1}_{{sen}}$', r'$G^{-1}_{{sen_B}}$']
#     pl.plot_gcurves_exp(DATA_EXP, CHARACTERISTICS_S, fig_dir, labels=LABELS,
#                         is_gathered=True, fig_size=(5.8, 3.6))
#     pl.plot_gcurves_exp(DATA_EXP, CHARACTERISTICS_S, fig_dir, labels=LABELS,
#                         is_gathered=False, fig_size=(5.8, 3.6))
# else:
#     LABELS = [r'$nta$', r'$sen_A$', r'$sen$', r'$sen_B$']
#     pl.plot_gcurves_exp(DATA_EXP, CHARACTERISTICS_S, fig_dir, labels=LABELS,
#                         is_gathered=False)

# # Print proportion of type B among senescent.
# if FORMAT == 'manuscript':
#     pl.compute_n_plot_gcurve_error(DATA_EXP, None, ['gsen'], ['senescent'],
#                                    is_printed=True, error_types=[1],
#                                    simulation_count=SIMU_COUNT)


# Histograms
# ----------

CHARACTERISTICS_S = [['btype'],
                      ['atype','senescent'],
                      ['senescent'],
                      ['btype','senescent']]


if FORMAT == 'manuscript':
    fig_size = (4, 4.2) # Default: (6.4, 4.8).
else:
    fig_size = (6.4, 4.8)

# # > Number of senescent cycles.
# LCYCLE_TYPES = ['sen']
# for characteristics in CHARACTERISTICS_S:
#     pl.compute_n_plot_lcycle_hist(DATA_EXP, SIMU_COUNT, characteristics,
#                                   LCYCLE_TYPES, fig_dir, fig_size=fig_size)
#     pl.compute_n_plot_lcycle_hist(DATA_EXP, SIMU_COUNT, characteristics,
#                                   LCYCLE_TYPES, fig_dir,
#                                   is_exp_support=True, fig_size=fig_size)

# # > Number of non-terminal arrest per sequence.
# SEQ_COUNT = 4
# LCYCLE_TYPES = ['nta_by_idx'] # ['nta', 'nta_total', 'nta1', 'nta_by_idx']

# pl.compute_n_plot_lcycle_hist(DATA_EXP, SIMU_COUNT, ['btype'], LCYCLE_TYPES,
#                               fig_dir, seq_count=SEQ_COUNT, fig_size=fig_size)

# pl.compute_n_plot_lcycle_hist(DATA_EXP, SIMU_COUNT, ['btype'], LCYCLE_TYPES,
#                               fig_dir, seq_count=SEQ_COUNT,
#                               is_exp_support=True, fig_size=fig_size)


# # > Lmin trigerring
# CHARACTERISTICS_S = [['btype','senescent'], ['atype','senescent']]
# if FORMAT == 'manuscript':
#     WIDTHS = [1, 4]
# else:
#     WIDTHS = [4]
#     pl.compute_n_plot_hist_lmin(DATA_EXP, SIMU_COUNT, [['atype','senescent']],
#                                 HIST_LMIN_X_AXIS, fig_dir, width=1,
#                                 is_htype_seen=True)
# for width in WIDTHS:
#     pl.compute_n_plot_hist_lmin(DATA_EXP, SIMU_COUNT, CHARACTERISTICS_S,
#                                 HIST_LMIN_X_AXIS, fig_dir, width=width)
#     pl.compute_n_plot_hist_lmin(DATA_EXP, SIMU_COUNT, CHARACTERISTICS_S,
#                                 HIST_LMIN_X_AXIS, fig_dir, width=width,
#                                 is_htype_seen=True)
#     pl.compute_n_plot_hist_lmin(DATA_EXP, SIMU_COUNT, [['senescent']],
#                                 HIST_LMIN_X_AXIS, fig_dir, width=width)
#     pl.compute_n_plot_hist_lmin(DATA_EXP, SIMU_COUNT, [['senescent']],
#                                 HIST_LMIN_X_AXIS, fig_dir, width=width,
#                                 is_htype_seen=True)
#     pl.compute_n_plot_hist_lmin(DATA_EXP, SIMU_COUNT,
#                                 [['btype','senescent'], ['atype','senescent'], ['senescent']],
#                                 HIST_LMIN_X_AXIS, fig_dir, width=width,
#                                 is_htype_seen=True)



# Correlations with generation curves
# ------------------------------------


# if FORMAT == 'manuscript':
#     # > Correlation wrt lmin and lavg.
#     CHARACTERISTICS_S = [['atype','senescent'],
#                           ['senescent'],
#                           ['btype','senescent']]
#     TYPES_OF_SORT = ['gsen', 'lmin', 'lavg']
#     cs = len(TYPES_OF_SORT)
#     for characs in CHARACTERISTICS_S:
#         pl.compute_n_plot_gcurves_wrt_sort_n_gen(DATA_EXP, SIMU_COUNT, characs,
#                                                   TYPES_OF_SORT, ['gsen'] * cs,
#                                                   fig_dir, None)
#     TYPES_OF_SORT = ['gnta1', 'lmin', 'lavg']
#     pl.compute_n_plot_gcurves_wrt_sort_n_gen(DATA_EXP, SIMU_COUNT, ['btype'],
#                                               TYPES_OF_SORT, ['gnta1'] * cs,
#                                               fig_dir, None)

#     # > Correlation wrt between arrest.
#     CHARACTERISTICS_S = [['btype', 'senescent'],
#                           ['btype', 'arrested2', 'senescent']]
#     GCURVES = [['gnta1', 'gsen'],
#                 ['gnta1', 'gnta2', 'gsen']]
#     bbox_to_anchor = [None, (.72,.7)]
#     cs = len(GCURVES)
#     for i in range(len(CHARACTERISTICS_S)):
#         characs = CHARACTERISTICS_S[i]
#         cs = len(GCURVES[i])
#         pl.compute_n_plot_gcurves_wrt_sort_n_gen(DATA_EXP, SIMU_COUNT, characs,
#                                                   ['gsen'] * cs, GCURVES[i],
#                                                   fig_dir, bbox_to_anchor[i],
#                                                   is_exp_plotted=True)

#     # import lineage_simulation as sim
#     # import matplotlib.pyplot as plt
#     # SIMU_COUNT = 10000
#     # LTRANS_TO_ADD_S = np.array([-40, 0, 40, 150])
#     # plt.figure()
#     # fig2_data = []
#     # for to_add in LTRANS_TO_ADD_S:
#     #     par_new = deepcopy(par.PAR)
#     #     par_new[2][0] += to_add
#     #     out = sim.compute_lineage_types(DATA_EXP, SIMU_COUNT, ['senescent'],
#     #                                     'gsen', 'gsen', parameters=par_new)
#     #     plt.plot(out[2][0]['btype']['mean'], out[0] / out[0][-1])
#     #     fig2_data.append([out[2][0]['btype']['mean'],
#     #                       out[-1]['mean'] / out[-1]['mean'][-1]])
#     #     print('\n ltrans: ', par_new[2][0])
#     #     print(' lmode:  ', par_new[2][0] + 260)
#     #     print('Proportion of type-B lineages: ', out[2][1]['btype']['mean'], '\n')
#     # plt.show()

#     # plt.figure()
#     # for data in fig2_data:
#     #     plt.plot(data[0], data[1])

#     # SIMU_COUNT = 10000
#     # LTRANS_TO_ADD_S = np.array([-40, 0, 40, 150])
#     # plt.figure()
#     # fig2_data = []
#     # for to_add in LTRANS_TO_ADD_S:
#     #     par_new = deepcopy(par.PAR)
#     #     par_new[2][0] += to_add
#     #     out = sim.compute_lineage_types(DATA_EXP, SIMU_COUNT, ['senescent'],
#     #                                     'lmin', 'gsen', parameters=par_new,
#     #                                     proc_count=47)
#     #     plt.plot(out[2][0]['btype']['mean'], out[0] / out[0][-1])
#     #     fig2_data.append([out[2][0]['btype']['mean'],
#     #                       out[-1]['mean'] / out[-1]['mean'][-1]])
#     #     print('\n ltrans: ', par_new[2][0])
#     #     print(' lmode:  ', par_new[2][0] + 260)
#     #     print('Proportion of type-B lineages: ', out[2][1]['btype']['mean'], '\n')
#     # plt.show()
#     # plt.figure()
#     # for data in fig2_data:
#     #     plt.plot(data[0], data[1])


# Generations of arrest
# ---------------------

# if FORMAT == 'article':
#     SIMU_COUNT = 1000

#     MFACTORS = np.array([25, 34.8, 30, 40])
#     P_ACC_S = par.P_ACCIDENT * MFACTORS

#     p_exit = deepcopy(par.P_EXIT)
#     for i in range(len(P_ACC_S)):
#         p_death_acc = P_ACC_S[i]
#         p_exit[0] = p_death_acc
#         par_update = {'p_exit': deepcopy(p_exit), 'is_htype_seen': False}
#         curve_label = r'$\times$' + str(MFACTORS[i])

#         pl.compute_n_plot_gcurve(DATA_EXP_MUTANT, SIMU_COUNT, ['senescent'],
#                                   fig_dir, par_update=par_update,
#                                   is_exp_plotted=True, bbox_to_anchor=(1.3,0),
#                                   title=curve_label)

# stop

# Sensitivity to initial distribution
# -----------------------------------

parameters = par.PAR
ltrans, l0, l1 = par.PAR_L_INIT

CHARAC_NTA = ['btype']
CHARAC_SEN_S = [['senescent'],
                ['atype','senescent'],
                ['btype','senescent']]
TEXTS = [None, r'\textit{$type~A$}', r'\textit{$type~B$}']

SIMU_COUNT = 1000

if FORMAT == "manuscript":
    FIG_SIZE =  (4.8, 3.8)
    FIG_SIZE2 = (5.5, 11.3)
else:
    FIG_SIZE = None
    FIG_SIZE2 = (6.5, 12)

# # > Accidental death.
# # ...................

# # Detailed simulations on senescence onset.
# if FORMAT == 'article':
#     MFACTORS = np.array([1., 5, 10., 15,  20, 25, 30., 40., 50.])
#     P_ACC_S = par.P_ACCIDENT * MFACTORS
#     LINESTYLES = ['-', '-', '-', '-', '-', '--', '-', '-', '-']
#     median = np.median(GSEN_EXP_MUTANT)
#     # median = GSEN_EXP_MUTANT[int(len(GSEN_EXP_MUTANT) / 2)]
#     p_exit = deepcopy(par.P_EXIT)
#     par_updates = []
#     curve_labels = []
#     for p_death_acc in P_ACC_S:
#         p_exit[0] = p_death_acc
#         par_updates.append({'p_exit': deepcopy(p_exit), 'is_htype_seen': False})
#         curve_labels.append(r'$\times$' + str(int(p_death_acc / P_ACC_S[0])))

#     pl.plot_gcurves_wrt_par(DATA_EXP, SIMU_COUNT, ['senescent'], par_updates,
#                             'p_death_acc', fig_dir, curve_labels=curve_labels,
#                             linestyles=LINESTYLES,
#                             is_exp_plotted=GSEN_EXP_MUTANT,
#                             bbox_to_anchor=(1, 1), add_to_name='rad51')

#     out = pl.plot_medians_wrt_par(DATA_EXP, SIMU_COUNT, ['senescent'], MFACTORS,
#                             par_updates, 'p_death_acc', fig_dir,
#                             x_label="Multiplicative factor for ", y_exp=median,
#                             curve_labels=curve_labels, linestyles=LINESTYLES,
#                             add_to_name='rad51')

# #  Detailed simulations on senescence onset.
# if FORMAT == 'article':
#     MFACTORS = np.array([1., 20.])
#     P_ACC_S = par.P_ACCIDENT * MFACTORS
#     LINESTYLES = ['-', '-']
#     median = np.median(GSEN_EXP_MUTANT)
#     # median = GSEN_EXP_MUTANT[int(len(GSEN_EXP_MUTANT) / 2)]
#     p_exit = deepcopy(par.P_EXIT)
#     par_updates = []
#     curve_labels = []
#     for p_death_acc in P_ACC_S:
#         p_exit[0] = p_death_acc
#         par_updates.append({'p_exit': deepcopy(p_exit), 'is_htype_seen': False})
#         curve_labels.append(r'$\times$' + str(int(p_death_acc / P_ACC_S[0])))

#     pl.plot_gcurves_wrt_par(DATA_EXP_MUTANT, SIMU_COUNT, ['senescent'], par_updates,
#                             'p_death_acc', fig_dir, curve_labels=curve_labels,
#                             linestyles=LINESTYLES,
#                             is_exp_plotted=GSEN_EXP_MUTANT,
#                             bbox_to_anchor=(1, 1), add_to_name='rad51_test')


# P_ACC_S = par.P_ACCIDENT * np.array([1., 10., 20, 30., 40., 50.])
# LINESTYLES = ['-', '-', '--', '-', '-', '-']

# p_exit = deepcopy(par.P_EXIT)
# par_updates = []
# curve_labels = []
# for p_death_acc in P_ACC_S:
#     p_exit[0] = p_death_acc
#     par_updates.append({'p_exit': deepcopy(p_exit), 'is_htype_seen': False})
#     curve_labels.append(r'$\times$' + str(int(p_death_acc / P_ACC_S[0])))
# pl.plot_gcurves_wrt_par(DATA_EXP, SIMU_COUNT, CHARAC_NTA, par_updates,
#                         'p_death_acc', fig_dir, curve_labels=curve_labels,
#                         linestyles=LINESTYLES)
# pl.plot_gcurves_wrt_par(DATA_EXP, SIMU_COUNT, CHARAC_SEN_S[0], par_updates,
#                         'p_death_acc', fig_dir, curve_labels=curve_labels,
#                         linestyles=LINESTYLES)
# pl.plot_gcurves_wrt_par_n_char(DATA_EXP, SIMU_COUNT, CHARAC_SEN_S,
#                                 par_updates, 'p_death_acc', fig_dir,
#                                 texts=TEXTS, curve_labels=curve_labels,
#                                 linestyles=LINESTYLES, fig_size=FIG_SIZE2)

# > Variable lmode.
# .................

LMODE_S = np.array([-20, -10, 0, 10, 20, 40])
LINESTYLES = ['-', '-', '--', '-', '-', '-']

parameters = deepcopy(par.PAR)
par_updates = []
curve_labels = []
for lmode_add in LMODE_S:
    parameters[2][0] = ltrans + lmode_add
    parameters[2][1] = l0 - lmode_add
    parameters[2][2] = l1 - lmode_add
    par_updates.append({'parameters': deepcopy(parameters),
                        'is_htype_seen': False})
    if lmode_add <= 0:
        curve_labels.append(str(lmode_add))
    else:
        curve_labels.append('+' + str(lmode_add))
# pd.plot_linit_wrt_par(ltrans=ltrans+LMODE_S, l0=l0-LMODE_S, l1=l1-LMODE_S,
#                       labels=curve_labels, legend_key='lall',
#                       fig_supdirectory=fig_dir, fig_size=FIG_SIZE)
# pl.plot_gcurves_wrt_par(DATA_EXP, SIMU_COUNT, CHARAC_NTA, par_updates,
#                         'ltrans', fig_dir, curve_labels=curve_labels,
#                         linestyles=LINESTYLES, fig_size=FIG_SIZE)
# pl.plot_gcurves_wrt_par(DATA_EXP, SIMU_COUNT, CHARAC_SEN_S[0], par_updates,
#                         'ltrans', fig_dir, curve_labels=curve_labels,
#                         linestyles=LINESTYLES, fig_size=FIG_SIZE)
# pl.plot_gcurves_wrt_par_n_char(DATA_EXP, SIMU_COUNT, CHARAC_SEN_S,
#                                 par_updates, 'ltrans', fig_dir,
#                                 texts=TEXTS, curve_labels=curve_labels,
#                                 linestyles=LINESTYLES, fig_size=FIG_SIZE2)
# stop

# > Variable ltrans.
# ..................

LTRANS_S = np.array([-20, -10, 0, 10, 20, 40])
LINESTYLES = ['-', '-', '--', '-', '-', '-']

parameters = deepcopy(par.PAR)
par_updates = []
curve_labels = []
for ltrans_add in LTRANS_S:
    parameters[2][0] = ltrans + ltrans_add
    par_updates.append({'parameters': deepcopy(parameters),
                        'is_htype_seen': False})
    if ltrans_add <= 0:
        curve_labels.append(str(ltrans_add))
    else:
        curve_labels.append('+' + str(ltrans_add))
# pd.plot_linit_wrt_par(ltrans=ltrans + LTRANS_S, l0=l0, l1=l1,
#                         labels=curve_labels, legend_key='ltrans',
#                         fig_supdirectory=fig_dir, fig_size=FIG_SIZE)
# pl.plot_gcurves_wrt_par(DATA_EXP, SIMU_COUNT, CHARAC_NTA, par_updates,
#                         'ltrans', fig_dir, curve_labels=curve_labels,
#                         linestyles=LINESTYLES, fig_size=FIG_SIZE)
# pl.plot_gcurves_wrt_par(DATA_EXP, SIMU_COUNT, CHARAC_SEN_S[0], par_updates,
#                         'ltrans', fig_dir, curve_labels=curve_labels,
#                         linestyles=LINESTYLES, fig_size=FIG_SIZE)
# pl.plot_gcurves_wrt_par_n_char(DATA_EXP, SIMU_COUNT, CHARAC_SEN_S,
#                                 par_updates, 'ltrans', fig_dir,
#                                 texts=TEXTS, curve_labels=curve_labels,
#                                 linestyles=LINESTYLES, fig_size=FIG_SIZE2)

# > Variable l0.
# ..............

# L0_S = np.array([-40, -20, -10, 0, 10, 20])
# LINESTYLES = ['-', '-', '-', '--', '-', '-']

# parameters = deepcopy(par.PAR)
# par_updates = []
# curve_labels = []
# for l0_add in L0_S:
#     parameters[2][1] = l0 + l0_add
#     par_updates.append({'parameters': deepcopy(parameters),
#                         'is_htype_seen': False})
#     if l0_add <= 0:
#         curve_labels.append(str(l0_add))
#     else:
#         curve_labels.append('+' + str(l0_add))
# pd.plot_linit_wrt_par(ltrans=ltrans, l0=l0 + L0_S, l1=l1,
#                         labels=curve_labels, legend_key='l0',
#                         fig_supdirectory=fig_dir, fig_size=FIG_SIZE)
# pl.plot_gcurves_wrt_par(DATA_EXP, SIMU_COUNT, CHARAC_NTA, par_updates,
#                         'l0', fig_dir, curve_labels=curve_labels,
#                         linestyles=LINESTYLES, fig_size=FIG_SIZE)
# pl.plot_gcurves_wrt_par(DATA_EXP, SIMU_COUNT, CHARAC_SEN_S[0], par_updates,
#                         'l0', fig_dir, curve_labels=curve_labels,
#                         linestyles=LINESTYLES, fig_size=FIG_SIZE)
# pl.plot_gcurves_wrt_par_n_char(DATA_EXP, SIMU_COUNT, CHARAC_SEN_S,
#                                 par_updates, 'l0', fig_dir, texts=TEXTS,
#                                 curve_labels=curve_labels,
#                                 linestyles=LINESTYLES, fig_size=FIG_SIZE2)

# > Variable l1.
# ..............

# L1_S = np.array([-168, -84, -42, 0, 42, 84])
# LINESTYLES = ['-', '-', '-', '--', '-', '-']
# # l1_s = np.array([-80, -40, -20, -10, 0, 10, 20])


# parameters = deepcopy(par.PAR)
# par_updates = []
# curve_labels = []
# for l1_add in L1_S:
#     parameters[2][2] = l1 + l1_add
#     par_updates.append({'parameters': deepcopy(parameters),
#                         'is_htype_seen': False})
#     if l1_add <= 0:
#         curve_labels.append(str(l1_add))
#     else:
#         curve_labels.append('+' + str(l1_add))
# pd.plot_linit_wrt_par(ltrans=ltrans, l0=l0, l1=l1 + L1_S,
#                       labels=curve_labels, legend_key='l1',
#                       fig_supdirectory=fig_dir, fig_size=FIG_SIZE)
# pl.plot_gcurves_wrt_par(DATA_EXP, SIMU_COUNT, CHARAC_NTA, par_updates,
#                         'l1', fig_dir, curve_labels=curve_labels,
#                         linestyles=LINESTYLES, fig_size=FIG_SIZE)
# pl.plot_gcurves_wrt_par(DATA_EXP, SIMU_COUNT, CHARAC_SEN_S[0], par_updates,
#                         'l1', fig_dir, curve_labels=curve_labels,
#                         linestyles=LINESTYLES, fig_size=FIG_SIZE)
# pl.plot_gcurves_wrt_par_n_char(DATA_EXP, SIMU_COUNT, CHARAC_SEN_S,
#                                 par_updates, 'l1', fig_dir, texts=TEXTS,
#                                 curve_labels=curve_labels,
#                                 linestyles=LINESTYLES, fig_size=FIG_SIZE2)
# stop


# Results from parameters estimation
# ----------------------------------

import lineage_fit_main as lfit
import lineage_fit_results as lfitr

CHARACTERISTICS_S = [['btype'],
                      ['atype','senescent'],
                      ['senescent'],
                      ['btype','senescent']]
LABELS = [r'$nta$', r'$sen_A$', r'$sen$', r'$sen_B$']

if FORMAT == "manuscript":
    FIG_SIZE_PARS = (3.9, 2.2)
    FIG_SIZE = (4, 7.1) #(5.7, 2.9)
    bbox_to_anchor=(0.15, 2)
else:
    FIG_SIZE = (5.5, 11)
    bbox_to_anchor = (1, 1)


# -----------------------------------------------------------------
# Test finalCut parameters.
SIMU_COUNT = 20
PAR = ([0.0249, 0.465], #[0.0247947389, 0.440063202], # [0.024, 0.45], #
       [[0.186824276, 0.725200993, 40.0], [2.45423414e-06, 0.122028128, 0.0]],
       [0, 40, 58.0])
PAR = ([0.028, 0.585], # [0.0249, 0.465] # [0.024, 0.45], #
        [[0.186824276, 0.725200993, 36], [2.45423414e-06, 0.122028128, 0.0]],
        [0, 40, 58.0])
# PAR = ([0.0247947389, 0.440063202],
#         [[0.17, 0.65, 36.0], [2.45423414e-06, 0.122028128, 0.0]],
#         [0, 40, 58.0])
PAR = ([0.0247947389, 0.440063202],
        [[0.186824276, 0.725200993, 27.0], [2.45423414e-06, 0.122028128, 0.0]],
        [0, 40, 58.0])  # Original fit
par_update = {'parameters': PAR}

if FORMAT == 'manuscript':
    FIG_SIZE = (5.5, 11)
    bbox_to_anchor = None
    FIG_SIZE_PARS = (5.5, 3)
# Plot laws.
parf.plot_laws(PAR, is_par_plot=True, fig_name='finalCut',
               fig_supdirectory=None, fig_size=FIG_SIZE_PARS, decimal_count=4)
# Plot gcurves exp/sim.
pl.compute_n_plot_gcurves_wrt_charac(DATA_EXP, SIMU_COUNT,
                                     CHARACTERISTICS_S, fig_dir,
                                     par_update=par_update, proc_count=5,
                                     labels=LABELS, path=None,
                                     bbox_to_anchor=bbox_to_anchor,
                                     fig_size=FIG_SIZE,
                                     xticks=[0, 10, 20, 30, 40, 50, 60])
# Print proportion of type B among senescent.
pl.compute_n_plot_gcurve_error(DATA_EXP, None, ['gsen'], ['senescent'],
                               is_printed=True, error_types=[1],
                               par_update=par_update,
                               simulation_count=SIMU_COUNT)
stop
# -----------------------------------------------------------------


for key, fits in lfitr.FITS.items():
    if key in ['8_4', '8']:
        pass
    else:
        if key == '10':
            par_space = lfitr.PAR_SPACES[key]
        else:
            par_space = lfitr.PAR_SPACES[key[0]]

        if IS_SAVED:
            fig_sdir = f"{fig_dir}/parameters"
            if (not os.path.exists(fig_sdir)):
                os.makedirs(fig_sdir)
            PATH = f"{fig_sdir}/fit_{key}_gcurves.pdf"
        else:
            PATH = None

        # # All results.
        scores = [lfit.compute_n_plot_gcurves(fit, kwarg=par_space,
                      is_plotted=False, simulation_count=SIMU_COUNT,
                      is_printed=False) for fit in fits]
        par_sensitivity = [lfit.point_to_cost_fct_parameters(point,
                                kwarg=par_space) for point in fits]
        # parf.plot_laws_s(par_sensitivity, idx_best=np.argmin(scores),
        #                   fig_name=key, fig_supdirectory=fig_dir,
        #                   is_zoomed=True)
        print(f'\n {key}: ', scores, min(scores))

        # Best results.
        parameters = par_sensitivity[np.argmin(scores)]
        if FORMAT == 'manuscript':
            if key == '10':
                FIG_SIZE = (5.5, 11)
                bbox_to_anchor = None
                FIG_SIZE_PARS = (5.5, 3)
            parf.plot_laws(parameters, is_par_plot=True, fig_name=key,
                            fig_supdirectory=fig_dir, fig_size=FIG_SIZE_PARS)
        # parf.plot_laws(parameters, is_par_plot=False, fig_name=key,
        #                 fig_supdirectory=fig_dir)
        par_update = {'parameters': parameters}
        pl.compute_n_plot_gcurves_wrt_charac(DATA_EXP, SIMU_COUNT,
                                             CHARACTERISTICS_S, fig_dir,
                                             par_update=par_update,
                                             labels=LABELS, path=PATH,
                                             bbox_to_anchor=bbox_to_anchor,
                                             fig_size=FIG_SIZE)
