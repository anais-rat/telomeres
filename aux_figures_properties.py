#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 15:21:50 2021

@author: arat

Global parameters of the figures, which are common to all plots or should not
be changed locally, are defined below. Other global parameters are also defined
at the beginning of `*_plot.py` scripts; more versatile ones are defined at the
beginning of `main_plot_*.py` scripts.

"""

import matplotlib
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import seaborn as sns


# Classical parameters for plots
# ------------------------------

PERCENT = 95  # Percentage of data to include in the percentile interval.
# ------------------------------
P_DOWN = (100 - PERCENT) / 2  # First percentile to compute.
P_UP = 100 - (100 - PERCENT) / 2  # Second percentile.
# ------------------------------

DPI = 600  # Resolution of plots.
ALPHA = 0.25 # Transparency to fill gaps btw extremum values, std, percentiles.


# Colors management
# -----------------

# Creation of a color map for lineages cycles duration times.
c = ["chartreuse", "limegreen", "red", "r"]
v = [0, 0.2, 0.98, 1.]
l = list(zip(v,c))
CMAP_LINEAGE = LinearSegmentedColormap.from_list('rg', l, N=256)

# Creation of a color maps and palettes for the lines to plot.
# > Viridis color maps.
color_map = matplotlib.cm.get_cmap('viridis')
MY_COLORS = [color_map(.01), color_map(.3), color_map(.7), color_map(.9)]
MY_COLORS_3 = [color_map(.01), color_map(.75), color_map(.9)]
MY_COLORS_5 = [color_map(.01), color_map(.2), color_map(.53), color_map(.7),
               color_map(.9)]
# > Rocket color map.
color_map_rocket = plt.cm.get_cmap('rocket')
MY_COLORS_2_ROCKET = [color_map_rocket(.15), color_map_rocket(.61)]
MY_COLORS_3_ROCKET = [color_map_rocket(.18), color_map_rocket(.48),
                      color_map_rocket(.69)]

# > Associated palettes.
MY_PALETTE_2 = sns.color_palette(MY_COLORS_2_ROCKET)
MY_PALETTE_3 = sns.color_palette(MY_COLORS_3)
MY_PALETTE_5 = sns.color_palette(MY_COLORS_5, desat=0.9)
MY_PALETTE = sns.color_palette(MY_COLORS, desat=0.9)
# Definition and display of the current palette.
sns.set_palette(MY_PALETTE)
if __name__ == "__main__":
    # Here vizualize desired palettes with command sns.palplot(`PALETTE_NAME`).
    sns.palplot(sns.color_palette())  # Default palette.

    sns.palplot(MY_COLORS_2_ROCKET)
    sns.palplot(sns.color_palette('rocket', n_colors=2))
    sns.palplot(MY_COLORS_3_ROCKET)
    sns.palplot(sns.color_palette('rocket', n_colors=3))
    sns.palplot(MY_COLORS_3)
    sns.palplot(sns.color_palette('viridis', n_colors=3))

# > Colors for "simulation" and "experiment" curves.
COLORS_SIM_VS_EXP = MY_COLORS[:2]

# > Colors for cell types.
type_keys = ['b+htype', 'mtype', 'atype']
colors_type = sns.color_palette("rocket", len(type_keys))
COLORS_TYPE = {type_keys[i]: colors_type[i] for i in range(len(type_keys))}
COLORS_TYPE['htype'] = 'grey'
COLORS_TYPE['h+mtype'] = COLORS_TYPE['mtype']
COLORS_TYPE['btype'] = COLORS_TYPE['b+htype']
COLORS_TYPE['all'] = 'grey'
COLORS_TYPE['sen'] = 'black'


# Legend management
# -----------------

# Dictionary of labels common to lineage and population plots.
LABELS = {'ax_time': "Time (day)",
          'ax_cexp': "Cell concentration (cell/mL)",
          'ax_l': "Telomere length (bp)",
          'ax_lavg': "Average telomere length (bp)",
          'ax_lmin_min': "Shortest telomere length (bp)",
          'ax_lmin': "Average shortest telomere length (bp)",
          'ax_lmode': "Mode of the distribution of telomere lengths (bp)",
          'ax_OD': "Optic density ($OD_{600}$)",
          'ax_prop': "Proportion of cells",
          'ax_p_sen': "Proportion of senescent cells",
          'ax_lsen': "Telomere length triggering senscence (bp)",
          'ax_per': "Percent",
          'ax_count': r"Count",
           #
          'cycle': "Cycle duration time (min)",
          'cycle_log': "Cycles duration time (min) in log-scale",
           #
          'max_sen_count': r"$n_{sen}",
          'prop_sat': r'$r_{sat}$',
          'pdeath': r"$p_{accident}$",
          'p_death_acc': r"$p_{accident}$",
          'ltrans': r"$\ell_{trans}$",
          'l0': r"$\ell_{0}$",
          'l1': r"$\ell_{1}$",
          'lall': r"$\ell_{trans},-\ell_{0},-\ell_{1}$",
           #
          'atype_short': "A",
          'btype_short': "B",
          'mtype_short': "M",
          'htype_short': "H",
          'h+mtype_short': "H",
          'b+htype_short': "B",
           #
          'atype': "type A",
          'btype': "type B",
          'mtype': "type M",
          'htype': "type H",
           #
          'atype_sen': "senescent type A",
          'btype_sen': "senescent type B",
          #
          'exp': "Experiment",
          'sim': "Simulation",
           #
          'telo-': 'Telomerase-negative',
          'telo+': 'Telomerase-positive',
           #
          'avg': "Average",
          'ext': "Extremum values",
          'std': "Standard deviation",
          'per': fr"{PERCENT}$\%$ of the values"}
