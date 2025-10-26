#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 15:21:50 2021

@author: arat

Global parameters of the figures (which are common to all plots or should not
be changed locally) are defined below.

Other global parameters are also defined at the beginning of
`telomeres/.../plot.py` scripts ; more versatile ones are defined at the
beginning of `main/.../plot.py` scripts.

Run the script to visualize the color palettes defined here.

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

DPI = 300  # Resolution of plots.
ALPHA = 0.25  # Transparency to fill gaps btw extremum values, std percentiles.


# Global plotting parameters
# --------------------------

PAR_RC_UPDATE_MANUSCRIPT = {
    "axes.facecolor": ".94",
    "text.latex.preamble": r"\usepackage{amsfonts, dsfont}",
    "figure.dpi": DPI,
    # 'text.usetex': True,  # Removed to make `plt.ylabel(wrap=True)` work.
    # Font changed consequently.
    # 'font.family': "sans-serif",  # latex-like: 'serif'
    # 'font.sans-serif': "Helvetica",  # ... 'cmr10'
    "font.family": "sans-serif",
    "font.sans-serif": "cmss10",
    "axes.unicode_minus": False,
    "legend.framealpha": 1,
    "legend.facecolor": "white",
    "legend.edgecolor": "white",  # 'F0F0F0', '#EAEAF2'.
    "legend.fancybox": True,
    "legend.frameon": True,
}

PAR_RC_UPDATE_ARTICLE = {
    "figure.dpi": DPI,
    "text.latex.preamble": r"\usepackage{dsfont}",  # to remove?
    "font.family": "Arial",  # ['sans-serif'],
    "font.sans-serif": "Arial",  # ['Arial'],
    "legend.frameon": False,
}

# NB ['Arial' fonts]. If you are on Linux and you prefer not to install
# proprietary fonts (like Arial), many Linux distributions come with other
# sans-serif fonts which can substitute Arial well in most cases. Use instead:
# plt.rcParams["font.family"] = "Liberation Sans"  # or "DejaVu Sans"


# Colors management
# -----------------

# Creation of a color map for lineages cycles duration times.
c = ["chartreuse", "limegreen", "red", "r"]
v = [0, 0.2, 0.98, 1.0]
CMAP_LINEAGE = LinearSegmentedColormap.from_list("rg", list(zip(v, c)), N=256)

# Creation of a color maps and palettes for the lines to plot.
# > Viridis color maps.
color_map = matplotlib.colormaps["viridis"]
MY_COLORS = [color_map(0.01), color_map(0.3), color_map(0.7), color_map(0.9)]
MY_COLORS_3 = [color_map(0.01), color_map(0.75), color_map(0.9)]
MY_COLORS_5 = [
    color_map(0.01),
    color_map(0.2),
    color_map(0.53),
    color_map(0.7),
    color_map(0.9),
]
# > Rocket color map.
color_map_rocket = matplotlib.colormaps["rocket"]
MY_COLORS_2_ROCKET = [color_map_rocket(0.15), color_map_rocket(0.61)]
MY_COLORS_3_ROCKET = [
    color_map_rocket(0.18),
    color_map_rocket(0.48),
    color_map_rocket(0.69),
]

# > Associated palettes.
MY_PALETTE_2 = sns.color_palette(MY_COLORS_2_ROCKET)
MY_PALETTE_3 = sns.color_palette(MY_COLORS_3)
MY_PALETTE_5 = sns.color_palette(MY_COLORS_5, desat=0.9)
MY_PALETTE = sns.color_palette(MY_COLORS, desat=0.9)

# > Colors for "simulation" and "experiment" curves (previously MY_COLORS[:2]).
COLORS_SIM_VS_EXP = [
    "#942192",  # Simulation. '#ED7D31'
    "#007AAA",  # Exp 1. #4472C4
    "#3A2194",
]  # Exp 2. #A5A5A5

# Definition and display of the current palette.
sns.set_palette(MY_PALETTE)
if __name__ == "__main__":
    # Here visualize desired palettes with command sns.palplot(`PALETTE_NAME`).
    sns.palplot(sns.color_palette())  # Default palette.
    sns.palplot(MY_COLORS_2_ROCKET)
    sns.palplot(sns.color_palette("rocket", n_colors=2))
    sns.palplot(MY_COLORS_3_ROCKET)
    sns.palplot(sns.color_palette("rocket", n_colors=3))
    sns.palplot(MY_COLORS_3)
    sns.palplot(sns.color_palette("viridis", n_colors=3))
    sns.palplot(COLORS_SIM_VS_EXP)

    sns.palplot(MY_COLORS)
    plt.show()

# > Colors for cell types.
type_keys = ["b+htype", "mtype", "atype"]
colors_type = sns.color_palette("rocket", len(type_keys))
COLORS_TYPE = {type_keys[i]: colors_type[i] for i in range(len(type_keys))}
COLORS_TYPE["htype"] = "grey"
COLORS_TYPE["h+mtype"] = COLORS_TYPE["mtype"]
COLORS_TYPE["btype"] = COLORS_TYPE["b+htype"]
COLORS_TYPE["all"] = "grey"
COLORS_TYPE["sen"] = "black"


# Legend management
# -----------------

# Dictionary of labels common to lineage and population plots.
LABELS = {
    "ax_time": "Time (day)",
    "ax_cexp": "Cell concentration (cell/mL)",
    "ax_l": "Telomere length (bp)",
    "ax_lavg": "Average telomere length (bp)",
    "ax_lmin_min": "Shortest telomere length (bp)",
    "ax_lmin": "Average shortest telomere length (bp)   ",
    "ax_lmode": "Mode of the distribution of telomere lengths (bp)",
    "ax_OD": "Optic density ($OD_{600}$)",
    "ax_prop": "Proportion of cells",
    "ax_p_sen": "Proportion of senescent cells",
    "ax_lsen": "Telomere length triggering senescence (bp)",
    "ax_per": "Percent",
    "ax_count": r"Count",
    #
    "leg_cell_count": r"$N_{init}$",
    #
    "cycle": "Cycle duration time (min)",
    "cycle_log": "Cycles duration time (min) in log-scale",
    #
    "sen_limit": r"$n_{sen}$",
    "prop": r"$r_{sat}$",
    "accident": r"$p_{accident}$",
    "ltrans": r"$\ell_{trans}$",
    "l0": r"$\ell_{0}$",
    "l1": r"$\ell_{1}$",
    "lall": r"$\ell_{trans},-\ell_{0},-\ell_{1}$",
    #
    "atype_short": "A",
    "btype_short": "B",
    "mtype_short": "M",
    "htype_short": "H",
    "h+mtype_short": "H",
    "b+htype_short": "B",
    #
    "atype": "type A",
    "btype": "type B",
    "mtype": "type M",
    "htype": "type H",
    #
    "atype_sen": "senescent type A",
    "btype_sen": "senescent type B",
    #
    "exp": "Experiment",
    "sim": "Simulation",
    #
    "telo-": "Telomerase-negative",
    "telo+": "Telomerase-positive",
    #
    "avg": "Average",
    "ext": "Extremum values",
    "std": "Standard deviation",
    "per": rf"{PERCENT}$\%$ of the values",
    #
    "lavg_avg": "Average",
    "lmin_avg": "Average shortest",
    "lmin_max": r"Longest shortest",
    "lmin_min": r"Shortest",
    "lmode": r"Mode",
}
