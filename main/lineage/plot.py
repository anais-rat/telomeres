#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 15:54:16 2022

@author: arat

Script to plot figures, related to lineage simulations, of Chapter 3 of the PhD
thesis and paper https://doi.org/10.1101/2023.11.22.568287 (and more).

Ideally, the data needed to plot these figures should have been already
computed with the script `main.lineage.compute.py` on a cluster in order to
parallelize all the "sets" of simulations (generally `SIMU_COUNT` simulation
per "set"). If not, the present script will run the required sets of
simulations in serie. Although each can be run in parrallel through
`PROC_COUNT > 1`, this is not recommanded for big `SIMU_COUNT`values because
very long.

"""

from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib.font_manager
import numpy as np
import seaborn as sns

import telomeres.auxiliary.figures_properties as fp
import telomeres.dataset.plot as pd
from telomeres.dataset.extract_processed_dataset import extract_postreat_lineages
import telomeres.lineage.plot as pl
import telomeres.lineage.simulation as sim
import telomeres.model.parameters as par


# ----------
# PARAMETERS
# ----------


# Reminder
# --------
# type_of_sort: 'gdeath', 'lmin', 'gnta1', 'gnta2', ..., 'gsen'.
# gtrig keys: 'nta', 'sen' 'death'.
# gtrig_to_compare: 'nta1', 'nta2', ..., 'sen' 'death'.
# characteristics: 'atype', btype', 'htype', 'arrested1', 'arrested2', ...,
#                  'senescent', 'dead', dead_accidentally', 'dead_naturally'.


# Adjustable parameters
# ---------------------

# Random seed (for reproducible figures).
# np.random.seed(1)  # NB: comment to generate new random.

# True to save figures.
IS_SAVED = False

# Plotting style: 'manuscript' or 'article'.
FORMAT = "article"

# Number of processor used for parallel computing.
PROC_COUNT = 1  # Add one for cluster.


# Fixed parameters  (no need to be redefined)
# -------------------------------------------

# Number of simulations used to plot average quantities.
SIMU_COUNT = 1000

# Default parameters of the model.
PAR_DEFAULT = deepcopy(par.PAR_DEFAULT_LIN)

# x-axis of the histogram of the telomere length trigerring sencence.
X_AXIS_HIST_LMIN = par.X_AXIS_HIST_LMIN

# Time step for time evolution along lineages (10 min).
POSTREAT_DT = 5  # !!!! par.CYCLE_MIN

# Figures directory if saved.
FIG_DIR = None
if IS_SAVED:
    FIG_DIR = FORMAT

# Global plotting parameters.
matplotlib.rcParams.update(matplotlib.rcParamsDefault)  # Reset to default.
if FORMAT == "manuscript":
    sns.set_style("darkgrid")
    sns.set_context("talk", font_scale=1)
    plt.rcParams.update(fp.PAR_RC_UPDATE_MANUSCRIPT)
elif FORMAT == "article":
    sns.set_style("ticks")
    sns.set_context("poster", font_scale=1)
    plt.rcParams.update(fp.PAR_RC_UPDATE_ARTICLE)
else:
    print("Redefine 'Format' correctly")
print("Global plotting parameters: \n", sns.plotting_context(), "\n")


# Experimental data
# -----------------

# Extraction and formatting

# > 'TetO2-TLC1'
DATA_EXP = extract_postreat_lineages()

# > 'RAD51'
DATA_EXP_MUTANT = extract_postreat_lineages(strain="RAD51")

DATA_EXP_MUTANT_SEN = sim.select_exp_lineages(DATA_EXP_MUTANT, ["senescent"])
DATA_EXP_MUTANT_SEN = sim.sort_lineages(DATA_EXP_MUTANT_SEN, "gsen")
GSEN_EXP_MUTANT = DATA_EXP_MUTANT_SEN[1]["sen"]


# ----
# PLOT
# ----


# Heatmaps
# --------

if FORMAT == "manuscript":
    FIG_SIZE = (5.8, 9.5)  # (4.7, 8)
else:
    FIG_SIZE = (5.8, 9.5)


# > Cycle duration times
# ----------------------

# Experimental.
IS_EXP = True
CYCLES_EXP = DATA_EXP[0]["cycle"]
GMAX = np.shape(CYCLES_EXP)[1]  # Maximum lineage length in the dataset.


# > Distributions of cycle duration time (cdt) per category.
pd.plot_cycles_from_dataset(FIG_DIR, IS_SAVED)


# > Cycle duration times in generation and lineage.
if FORMAT == "manuscript":  # With legend for types.
    pl.plot_lineages_cycles(
        CYCLES_EXP,
        IS_EXP,
        FIG_DIR,
        lineage_types=DATA_EXP[2],
        gmax=GMAX,
        fig_size=FIG_SIZE,
    )
    pl.plot_lineages_cycles(
        CYCLES_EXP,
        IS_EXP,
        FIG_DIR,
        is_dead=DATA_EXP[1]["death"],
        gmax=GMAX,
        fig_size=FIG_SIZE,
    )

else:  # Without legend.
    pl.plot_lineages_cycles(CYCLES_EXP, IS_EXP, FIG_DIR, gmax=GMAX, fig_size=FIG_SIZE)

    # Same plots for RAD51 data.
    # > Extract data.
    CYCLES_EXP_MUTANT = DATA_EXP_MUTANT[0]["cycle"]
    CYCLES_EXP_MUTANT_SEN = DATA_EXP_MUTANT_SEN[0]["cycle"]

    # > Plot.
    pl.plot_lineages_cycles(
        CYCLES_EXP_MUTANT,
        IS_EXP,
        FIG_DIR,
        gmax=None,
        add_to_name="rad51",
        fig_size=FIG_SIZE,
    )
    pl.plot_lineages_cycles(
        CYCLES_EXP_MUTANT_SEN,
        IS_EXP,
        FIG_DIR,
        gmax=None,
        add_to_name="rad51_sen",
        fig_size=FIG_SIZE,
    )

# Simulated.
IS_EXP = False

# > Type H seen.
if FORMAT == "manuscript":
    # WARNING: we need a unique simulation. It is run below (not already run
    #          and saved contrarily to other figures).
    data = sim.simulate_lineages_evolution(
        len(CYCLES_EXP), ["senescent"], PAR_DEFAULT, is_evo_returned=True
    )
    data = sim.sort_lineages(data, "gdeath")
    pl.plot_lineages_cycles(
        data[0]["cycle"],
        IS_EXP,
        FIG_DIR,
        gmax=GMAX,
        lineage_types=data[2],
        fig_size=FIG_SIZE,
    )

# > Type H unseen.
PAR = deepcopy(PAR_DEFAULT)
PAR["is_htype_seen"] = False

data = sim.simulate_lineages_evolution(
    len(CYCLES_EXP), ["senescent"], PAR, is_evo_returned=True
)
data = sim.sort_lineages(data, "gdeath")
pl.plot_lineages_cycles(
    data[0]["cycle"],
    IS_EXP,
    FIG_DIR,
    # , is_data_saved="Fig2c")
    gmax=GMAX,
    fig_size=FIG_SIZE,
)

if FORMAT == "manuscript":  # With legend for types.
    pl.plot_lineages_cycles(
        data[0]["cycle"],
        IS_EXP,
        FIG_DIR,
        lineage_types=data[2],
        gmax=GMAX,
        fig_size=FIG_SIZE,
    )
elif FORMAT == "article":  # 2 additional simulations.
    for seed in [2, 3]:
        np.random.seed(seed)  # Not working. To repair for reproducible simu...
        data = sim.simulate_lineages_evolution(
            len(CYCLES_EXP), ["senescent"], PAR, is_evo_returned=True
        )
        data = sim.sort_lineages(data, "gdeath")
        pl.plot_lineages_cycles(
            data[0]["cycle"],
            IS_EXP,
            FIG_DIR,
            gmax=GMAX,
            # is_data_saved=["SFig2a", "SFig2b"][seed-2])
            add_to_name=str(seed),
            fig_size=FIG_SIZE,
        )


# > Average of 2D matrices
# ----------------------

bbox_to_anchor = (0.45, 1.17)
if FORMAT == "manuscript":
    bbox_to_anchor = None

IS_EXP = False
CHARACTERISTICS_1 = ["senescent"]
CHARACTERISTICS_2 = ["senescent", "dead"]
TYPES_OF_SORT_1 = ["gsen", "lmin", "lavg"]
TYPE_OF_SORT_2 = "gdeath"

# Simulation.
DATA_SIM1 = sim.compute_evo_avg_data(
    DATA_EXP,
    SIMU_COUNT,
    CHARACTERISTICS_1,
    types_of_sort=TYPES_OF_SORT_1,
    proc_count=PROC_COUNT,
)
DATA_SIM2 = sim.compute_evo_avg_data(
    DATA_EXP,
    SIMU_COUNT,
    CHARACTERISTICS_2,
    types_of_sort=[TYPE_OF_SORT_2],
    proc_count=PROC_COUNT,
)

# Plot.
# > Cycle duration times.
for key_sort in TYPES_OF_SORT_1:  # Config. 1 for various types of sort.
    pl.plot_lineages_cycles(
        DATA_SIM1[key_sort][0]["cycle"],
        IS_EXP,
        FIG_DIR,
        fig_size=FIG_SIZE,
        curve_to_plot=DATA_SIM1[key_sort][1]["sen"]["mean"],
        evo_avg={"simu_count": SIMU_COUNT, "type_of_sort": key_sort},
        bbox_to_anchor=bbox_to_anchor,
    )

pl.plot_lineages_cycles(  # Config. 2, sort by gdeath.
    DATA_SIM2[TYPE_OF_SORT_2][0]["cycle"],
    IS_EXP,
    FIG_DIR,
    fig_size=FIG_SIZE,
    curve_to_plot=DATA_SIM2[TYPE_OF_SORT_2][1]["sen"]["mean"],
    evo_avg={"simu_count": SIMU_COUNT, "type_of_sort": TYPE_OF_SORT_2},
    bbox_to_anchor=bbox_to_anchor,
)

# > Proportion of type-B.
for key_sort in ["lmin", "gsen", "lavg"]:
    pl.plot_lineage_avg_proportions(
        DATA_SIM1[key_sort][0]["prop_btype"],
        True,
        FIG_DIR,
        curve_to_plot=DATA_SIM1[key_sort][1]["sen"]["mean"],
        evo_avg={"simu_count": SIMU_COUNT, "type_of_sort": key_sort},
    )


# Time / generation evolution postreat
# ------------------------------------
# NB: this figures were not published, are are thus not saved.

CHARACTERISTICS = ["senescent"]

pl.compute_n_plot_postreat_time_vs_gen(
    DATA_EXP, SIMU_COUNT, CHARACTERISTICS, POSTREAT_DT, is_htype_seen=False
)
pl.compute_n_plot_postreat_time_vs_gen(
    DATA_EXP, SIMU_COUNT, CHARACTERISTICS, POSTREAT_DT, is_htype_seen=True
)


# Generation curves wrt experimental
# ----------------------------------

CHARACTERISTICS_S = [
    ["btype"],
    ["atype", "senescent"],
    ["senescent"],
    ["btype", "senescent"],
]

if FORMAT == "manuscript":
    LABELS = [
        r"$G^{-1}_{{nta}}$",
        r"$G^{-1}_{{sen_A}}$",
        r"$G^{-1}_{{sen}}$",
        r"$G^{-1}_{{sen_B}}$",
    ]
    # pl.plot_gcurves_exp(DATA_EXP, CHARACTERISTICS_S, FIG_DIR, labels=LABELS,
    #                     is_gathered=True, fig_size=(5.8, 3.6))
    pl.plot_gcurves_exp(
        DATA_EXP,
        CHARACTERISTICS_S,
        FIG_DIR,
        labels=LABELS,
        is_gathered=False,
        fig_size=(5.8, 3.6),
    )
else:
    LABELS = [r"$nta$", r"$sen_A$", r"$sen$", r"$sen_B$"]
    pl.plot_gcurves_exp(
        DATA_EXP,
        CHARACTERISTICS_S,
        FIG_DIR,
        labels=LABELS,
        is_gathered=False,
        is_data_saved=True,
    )

# Print proportion of type B among senescent.
if FORMAT == "manuscript":
    pl.compute_n_plot_gcurve_error(
        DATA_EXP,
        None,
        ["gsen"],
        ["senescent"],
        is_printed=True,
        error_types=[1],
        simulation_count=SIMU_COUNT,
    )


# Histograms
# ----------

CHARACTERISTICS_S = [
    ["btype"],
    ["atype", "senescent"],
    ["senescent"],
    ["btype", "senescent"],
]

if FORMAT == "manuscript":
    FIG_SIZE = (4.4, 4.2)  # Default: (6.4, 4.8).
else:
    FIG_SIZE = (6.4, 4.8)


# > Length of sequences of arrests
# --------------------------------

# Number of senescent cycles.
LCYCLE_TYPES = ["sen"]
for characteristics in CHARACTERISTICS_S:
    pl.compute_n_plot_lcycle_hist(
        DATA_EXP, SIMU_COUNT, characteristics, LCYCLE_TYPES, FIG_DIR, fig_size=FIG_SIZE
    )
    pl.compute_n_plot_lcycle_hist(
        DATA_EXP,
        SIMU_COUNT,
        characteristics,
        LCYCLE_TYPES,
        FIG_DIR,
        is_exp_support=True,
        fig_size=FIG_SIZE,
    )


# Number of non-terminal arrest per sequence.
SEQ_COUNT = 4
LCYCLE_TYPES = ["nta_by_idx"]  # ['nta', 'nta_total', 'nta1', 'nta_by_idx']

pl.compute_n_plot_lcycle_hist(
    DATA_EXP,
    SIMU_COUNT,
    ["btype"],
    LCYCLE_TYPES,
    FIG_DIR,
    seq_count=SEQ_COUNT,
    fig_size=FIG_SIZE,
)
pl.compute_n_plot_lcycle_hist(
    DATA_EXP,
    SIMU_COUNT,
    ["btype"],
    LCYCLE_TYPES,
    FIG_DIR,
    seq_count=SEQ_COUNT,
    is_exp_support=True,
    fig_size=FIG_SIZE,
)


# > Lmin trigerring senescence
# ----------------------------

CHARACTERISTICS_S = [["btype", "senescent"], ["atype", "senescent"]]
if FORMAT == "manuscript":
    WIDTHS = [1, 4]
else:
    WIDTHS = [4]
    pl.compute_n_plot_hist_lmin(
        DATA_EXP,
        SIMU_COUNT,
        [["atype", "senescent"]],
        X_AXIS_HIST_LMIN,
        FIG_DIR,
        width=1,
        is_htype_seen=True,
    )
for width in WIDTHS:
    pl.compute_n_plot_hist_lmin(
        DATA_EXP, SIMU_COUNT, CHARACTERISTICS_S, X_AXIS_HIST_LMIN, FIG_DIR, width=width
    )
    pl.compute_n_plot_hist_lmin(
        DATA_EXP,
        SIMU_COUNT,
        CHARACTERISTICS_S,
        X_AXIS_HIST_LMIN,
        FIG_DIR,
        width=width,
        is_htype_seen=True,
    )
    pl.compute_n_plot_hist_lmin(
        DATA_EXP, SIMU_COUNT, [["senescent"]], X_AXIS_HIST_LMIN, FIG_DIR, width=width
    )
    pl.compute_n_plot_hist_lmin(
        DATA_EXP,
        SIMU_COUNT,
        [["senescent"]],
        X_AXIS_HIST_LMIN,
        FIG_DIR,
        width=width,
        is_htype_seen=True,
    )
    pl.compute_n_plot_hist_lmin(
        DATA_EXP,
        SIMU_COUNT,
        [["btype", "senescent"], ["atype", "senescent"], ["senescent"]],
        X_AXIS_HIST_LMIN,
        FIG_DIR,
        width=width,
        is_htype_seen=True,
    )


# Correlations with generation curves
# ------------------------------------

if FORMAT == "manuscript":
    # Correlation wrt lmin and lavg.
    CHARACTERISTICS_S = [["atype", "senescent"], ["senescent"], ["btype", "senescent"]]
    TYPES_OF_SORT = ["gsen", "lmin", "lavg"]
    cs = len(TYPES_OF_SORT)
    for characs in CHARACTERISTICS_S:
        pl.compute_n_plot_gcurves_wrt_sort_n_gen(
            DATA_EXP, SIMU_COUNT, characs, TYPES_OF_SORT, ["gsen"] * cs, FIG_DIR, None
        )
    TYPES_OF_SORT = ["gnta1", "lmin", "lavg"]
    pl.compute_n_plot_gcurves_wrt_sort_n_gen(
        DATA_EXP, SIMU_COUNT, ["btype"], TYPES_OF_SORT, ["gnta1"] * cs, FIG_DIR, None
    )

    # Correlation between sequences of arrest.
    CHARACTERISTICS_S = [["btype", "senescent"], ["btype", "arrested2", "senescent"]]
    GCURVES = [["gnta1", "gsen"], ["gnta1", "gnta2", "gsen"]]
    bbox_to_anchor = [None, (0.72, 0.7)]
    cs = len(GCURVES)
    for i in range(len(CHARACTERISTICS_S)):
        characs = CHARACTERISTICS_S[i]
        cs = len(GCURVES[i])
        pl.compute_n_plot_gcurves_wrt_sort_n_gen(
            DATA_EXP,
            SIMU_COUNT,
            characs,
            ["gsen"] * cs,
            GCURVES[i],
            FIG_DIR,
            bbox_to_anchor[i],
            is_exp_plotted=True,
        )


# Sensitivity analyses
# --------------------


CHARAC_NTA = ["btype"]
CHARAC_SEN_S = [["senescent"], ["atype", "senescent"], ["btype", "senescent"]]
TEXTS = [None, r"$\mathit{type~A}$", r"$\mathit{type~B}$"]

if FORMAT == "manuscript":
    FIG_SIZE = (4.8, 3.8)
    FIG_SIZE2 = (5.5, 11.3)
else:
    FIG_SIZE = None
    FIG_SIZE2 = (6.5, 12)


# > Accidental death
# ------------------

P_ACC_S = par.P_ACCIDENT * np.array([1.0, 10.0, 20, 30.0, 40.0, 50.0])
LINESTYLES = ["-", "-", "--", "-", "-", "-"]

p_exit = deepcopy(par.P_EXIT)
par_updates = []
curve_labels = []
for p_death_acc in P_ACC_S:
    p_exit["accident"] = p_death_acc
    par_updates.append({"p_exit": deepcopy(p_exit)})
    curve_labels.append(r"$\times$" + str(int(p_death_acc / P_ACC_S[0])))
pl.plot_gcurves_wrt_par(
    DATA_EXP,
    SIMU_COUNT,
    CHARAC_NTA,
    par_updates,
    "accident",
    FIG_DIR,
    curve_labels=curve_labels,
    linestyles=LINESTYLES,
)
pl.plot_gcurves_wrt_par(
    DATA_EXP,
    SIMU_COUNT,
    CHARAC_SEN_S[0],
    par_updates,
    "accident",
    FIG_DIR,
    curve_labels=curve_labels,
    linestyles=LINESTYLES,
)
pl.plot_gcurves_wrt_par_n_char(
    DATA_EXP,
    SIMU_COUNT,
    CHARAC_SEN_S,
    par_updates,
    "accident",
    FIG_DIR,
    texts=TEXTS,
    curve_labels=curve_labels,
    linestyles=LINESTYLES,
    fig_size=FIG_SIZE2,
)

if FORMAT == "article":  # RAD51 data.
    # WARNING: additional plots, not simulated in lineage.main.compute yet.
    # --> Need to simulate, with `PROC_COUNT_TEMP` processor (to adjust).
    print(
        "WARNING: you might want to adjust PROC_COUNT_TEMP before running "
        " the following simulation"
    )
    PROC_COUNT_TEMP = 1  # 11
    MFACTORS = np.array([25, 34.8, 30, 40])
    P_ACC_S = par.P_ACCIDENT * MFACTORS
    p_exit = deepcopy(par.P_EXIT)
    for i in range(len(P_ACC_S)):
        p_death_acc = P_ACC_S[i]
        print("p_accident: ", p_death_acc)
        p_exit["accident"] = p_death_acc
        par_update = {"p_exit": deepcopy(p_exit)}
        curve_label = r"$\times$" + str(MFACTORS[i])

        pl.compute_n_plot_gcurve(
            DATA_EXP_MUTANT,
            SIMU_COUNT,
            ["senescent"],
            FIG_DIR,
            par_update=par_update,
            is_exp_plotted=True,
            bbox_to_anchor=(1.3, 0),
            title=curve_label,
            proc_count=PROC_COUNT_TEMP,
            add_to_name="rad51",
        )
    p_exit = deepcopy(par.P_EXIT)
    p_exit["accident"] = 5.4 / 100
    par_update = {"p_exit": deepcopy(p_exit)}
    curve_label = rf"$p_{{accident}} = ${p_exit['accident'] * 100}%"
    pl.compute_n_plot_gcurve(
        DATA_EXP_MUTANT,
        50,
        ["senescent"],
        FIG_DIR,
        par_update=par_update,
        is_exp_plotted=True,
        bbox_to_anchor=(1.3, 0),
        title=curve_label,
        proc_count=PROC_COUNT_TEMP,
        add_to_name="rad51",
    )


# > Initial distribution of telomere lengths
# ------------------------------------------

LTRANS, L0, L1 = par.PAR_L_INIT


# Variable ltrans.

LTRANS_S = np.array([-20, -10, 0, 10, 20, 40])
LINESTYLES = ["-", "-", "--", "-", "-", "-"]

parameters = deepcopy(par.PAR)
par_updates = []
curve_labels = []
for ltrans_add in LTRANS_S:
    parameters[2][0] = LTRANS + ltrans_add
    par_updates.append({"fit": deepcopy(parameters)})
    if ltrans_add <= 0:
        curve_labels.append(str(ltrans_add))
    else:
        curve_labels.append("+" + str(ltrans_add))
pd.plot_ltelomere_init_wrt_par(
    ltrans=LTRANS + LTRANS_S,
    l0=L0,
    l1=L1,
    labels=curve_labels,
    legend_key="ltrans",
    fig_subdirectory=FIG_DIR,
    fig_size=FIG_SIZE,
)
pl.plot_gcurves_wrt_par(
    DATA_EXP,
    SIMU_COUNT,
    CHARAC_NTA,
    par_updates,
    "ltrans",
    FIG_DIR,
    curve_labels=curve_labels,
    linestyles=LINESTYLES,
    fig_size=FIG_SIZE,
)
pl.plot_gcurves_wrt_par(
    DATA_EXP,
    SIMU_COUNT,
    CHARAC_SEN_S[0],
    par_updates,
    "ltrans",
    FIG_DIR,
    curve_labels=curve_labels,
    linestyles=LINESTYLES,
    fig_size=FIG_SIZE,
)
pl.plot_gcurves_wrt_par_n_char(
    DATA_EXP,
    SIMU_COUNT,
    CHARAC_SEN_S,
    par_updates,
    "ltrans",
    FIG_DIR,
    texts=TEXTS,
    curve_labels=curve_labels,
    linestyles=LINESTYLES,
    fig_size=FIG_SIZE2,
)


# Variable l0.

L0_S = np.array([-40, -20, -10, 0, 10, 20])
LINESTYLES = ["-", "-", "-", "--", "-", "-"]

parameters = deepcopy(par.PAR)
par_updates = []
curve_labels = []
for l0_add in L0_S:
    parameters[2][1] = L0 + l0_add
    par_updates.append({"fit": deepcopy(parameters)})
    if l0_add <= 0:
        curve_labels.append(str(l0_add))
    else:
        curve_labels.append("+" + str(l0_add))
pd.plot_ltelomere_init_wrt_par(
    ltrans=LTRANS,
    l0=L0 + L0_S,
    l1=L1,
    labels=curve_labels,
    legend_key="l0",
    fig_subdirectory=FIG_DIR,
    fig_size=FIG_SIZE,
)
pl.plot_gcurves_wrt_par(
    DATA_EXP,
    SIMU_COUNT,
    CHARAC_NTA,
    par_updates,
    "l0",
    FIG_DIR,
    curve_labels=curve_labels,
    linestyles=LINESTYLES,
    fig_size=FIG_SIZE,
)
pl.plot_gcurves_wrt_par(
    DATA_EXP,
    SIMU_COUNT,
    CHARAC_SEN_S[0],
    par_updates,
    "l0",
    FIG_DIR,
    curve_labels=curve_labels,
    linestyles=LINESTYLES,
    fig_size=FIG_SIZE,
)
pl.plot_gcurves_wrt_par_n_char(
    DATA_EXP,
    SIMU_COUNT,
    CHARAC_SEN_S,
    par_updates,
    "l0",
    FIG_DIR,
    texts=TEXTS,
    curve_labels=curve_labels,
    linestyles=LINESTYLES,
    fig_size=FIG_SIZE2,
)


# Variable l1.

L1_S = np.array([-168, -84, -42, 0, 42, 84])
LINESTYLES = ["-", "-", "-", "--", "-", "-"]
# L1_s = np.array([-80, -40, -20, -10, 0, 10, 20])


parameters = deepcopy(par.PAR)
par_updates = []
curve_labels = []
for l1_add in L1_S:
    parameters[2][2] = L1 + l1_add
    par_updates.append({"fit": deepcopy(parameters)})
    if l1_add <= 0:
        curve_labels.append(str(l1_add))
    else:
        curve_labels.append("+" + str(l1_add))
pd.plot_ltelomere_init_wrt_par(
    ltrans=LTRANS,
    l0=L0,
    l1=L1 + L1_S,
    labels=curve_labels,
    legend_key="l1",
    fig_subdirectory=FIG_DIR,
    fig_size=FIG_SIZE,
)
pl.plot_gcurves_wrt_par(
    DATA_EXP,
    SIMU_COUNT,
    CHARAC_NTA,
    par_updates,
    "l1",
    FIG_DIR,
    curve_labels=curve_labels,
    linestyles=LINESTYLES,
    fig_size=FIG_SIZE,
)
pl.plot_gcurves_wrt_par(
    DATA_EXP,
    SIMU_COUNT,
    CHARAC_SEN_S[0],
    par_updates,
    "l1",
    FIG_DIR,
    curve_labels=curve_labels,
    linestyles=LINESTYLES,
    fig_size=FIG_SIZE,
)
pl.plot_gcurves_wrt_par_n_char(
    DATA_EXP,
    SIMU_COUNT,
    CHARAC_SEN_S,
    par_updates,
    "l1",
    FIG_DIR,
    texts=TEXTS,
    curve_labels=curve_labels,
    linestyles=LINESTYLES,
    fig_size=FIG_SIZE2,
)


# Variable lmode.
# NB: Added afterwards, not published.

LMODE_S = np.array([-20, -10, 0, 10, 20, 40])
LINESTYLES = ["-", "-", "--", "-", "-", "-"]

parameters = deepcopy(par.PAR)
par_updates = []
curve_labels = []
for lmode_add in LMODE_S:
    parameters[2][0] = LTRANS + lmode_add
    parameters[2][1] = L0 - lmode_add
    parameters[2][2] = L1 - lmode_add
    par_updates.append({"fit": deepcopy(parameters)})
    if lmode_add <= 0:
        curve_labels.append(str(lmode_add))
    else:
        curve_labels.append("+" + str(lmode_add))
pd.plot_ltelomere_init_wrt_par(
    ltrans=LTRANS + LMODE_S,
    l0=L0 - LMODE_S,
    l1=L1 - LMODE_S,
    labels=curve_labels,
    legend_key="lmode",
    fig_subdirectory=FIG_DIR,
    fig_size=FIG_SIZE,
)
pl.plot_gcurves_wrt_par(
    DATA_EXP,
    SIMU_COUNT,
    CHARAC_NTA,
    par_updates,
    "lmode",
    FIG_DIR,
    curve_labels=curve_labels,
    linestyles=LINESTYLES,
    fig_size=FIG_SIZE,
)
pl.plot_gcurves_wrt_par(
    DATA_EXP,
    SIMU_COUNT,
    CHARAC_SEN_S[0],
    par_updates,
    "lmode",
    FIG_DIR,
    curve_labels=curve_labels,
    linestyles=LINESTYLES,
    fig_size=FIG_SIZE,
)
pl.plot_gcurves_wrt_par_n_char(
    DATA_EXP,
    SIMU_COUNT,
    CHARAC_SEN_S,
    par_updates,
    "lmode",
    FIG_DIR,
    texts=TEXTS,
    curve_labels=curve_labels,
    linestyles=LINESTYLES,
    fig_size=FIG_SIZE2,
)
