#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 18:51:29 2023

@author: arat
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import scipy.io as sio

import aux_figures_properties as fp
import lineage_simulation as sim
import parameters as par

EXP_DATA = sio.loadmat('data/microfluidic/TelomeraseNegative.mat')
EXP_DATA = EXP_DATA['OrdtryT528total160831']
EXP_DATA = sim.postreat_experimental_lineages(EXP_DATA, par.THRESHOLD,
                                              par.GEN_COUNT_BY_LINEAGE_MIN)
CHARACTERISTICS = ['senescent']
SIMULATION_COUNT = 100
LINEAGE_COUNT = 80
PROC_COUNT = 3


LENGTH_CUT = 30 # Length of the telomere cut [bp].
AVG_CUT_DELAY = 2 # Average nb of gen between end of galactose and cutting.
PROBA_CUT_ESCAPE = .1 # Proba of/proportion of lineages escaping cutting.

PAR_FINAL_CUT = [LENGTH_CUT, AVG_CUT_DELAY, PROBA_CUT_ESCAPE, par.IDXS_FRAME]

# PAR_FINAL_CUT = par.PAR_FINAL_CUT


LABELS = {'ax_gen': "Generation",
          'ax_lin': 'Lineage'}

fig_dir = 'figures/finalCut/'
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
                     # 'figure.figsize': (6.4, 4.1),
                     # 'font.size': 14,
                     'legend.title_fontsize': 15.5,
                     'legend.fontsize': 15
                     })

# exp_data_selected = sim.select_exp_lineages(EXP_DATA, CHARACTERISTICS)
# lineage_count = len(exp_data_selected[0]['cycle'])


for cut in [0, 20, 30, 40, 50, 70]:
    PAR_FINAL_CUT = [cut, AVG_CUT_DELAY, PROBA_CUT_ESCAPE, par.IDXS_FRAME]
    out = sim.simulate_n_average_lineages(LINEAGE_COUNT, SIMULATION_COUNT,
                                      ['gsen'], CHARACTERISTICS, par_update=
                                      {'par_finalCut':PAR_FINAL_CUT})['gsen']
    lineages = np.arange(LINEAGE_COUNT)
    name = f'Fc{cut}'
    plt.figure()
    plt.plot(out[1]['sen']['mean'], lineages, color='darkorange', label=name)
    plt.fill_betweenx(lineages, out[1]['sen']['perdown'],
                      out[1]['sen']['perup'], alpha=fp.ALPHA,
                      color='darkorange')
    plt.xlabel(LABELS['ax_gen'])
    plt.ylabel(LABELS['ax_lin'])
    plt.legend(loc="lower right")
    plt.savefig(fig_dir + f'{name}.pdf', bbox_inches='tight')
    plt.show()
