#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 16:22:47 2020

@author: arat

Parameters for:
    1) The model.
    2) The way to process data.
These are default parameters, that have been thoughtfully chosen and are not
intended to be modified.
If 2) modified, `makeFile/processed_dataset.py` must be run again.

We might use the following notations:
    non-terminal-arrest (nta)
    senescence (sen).

"""

import math
import numpy as np


# =============================================================================
# Adjustable parameters
# =============================================================================

# --------------------------
# 1) Parameters of the model
# --------------------------


# Types
# -----

# Modelling assumption for type H cells.
# > 'True': If type H cells taken into account.
# > 'False': Otherwise. In this case it is assumed that type B must exit their
#    sequence of non-terminal arrests to be allowed to enter senescence.
HTYPE_CHOICE = True


# Telomeres
# ---------

# Number of chromosomes per cell.
CHROMOSOME_COUNT = 16

# Overhang / shortening rate (assumed constant) [bp].
OVERHANG = 7

# Post-treatment of the initial distribution of telomere lengths.
# The "original" distribution is slightly transformed as follows
# > Translation by `LTRANS` [bp].
# > Dilatation at the left/right side of the mode of the distribution s.t. the
#   minimum/maximum of the support of the distribution is translated by
#   `L0`/`L1` [bp].
LTRANS, L0, L1 = 0, 40, 58  # Our fit.
# Concatenation ...............................................................
PAR_L_INIT = [LTRANS, L0, L1]
# .............................................................................


# Laws for death and non-terminal-arrest exit
# -------------------------------------------

# Probability to die accidentally (`p_accident`).
P_ACCIDENT = 1 * 0.0043  # Constant = 4.3 * 1e-3 (Coutelier et al. 2018).

# Probability `p_death` to die "naturally" (from senescence).
P_DEATH = 0.58  # Constant = 0.58 for D = 220 (Martin et al. 2021).

# Maximum number of senescent cycles (`math.inf` for no limit).
MAX_SEN_CYCLE_COUNT = math.inf

# Probability `p_repair` of repairing or adapt from a non-terminal arrest.
P_REPAIR = 0.65  # Constant (Martin et al. 2021).


# Laws for non-terminal-arrest and senescence onset
# -------------------------------------------------

# Probability `p_nta` to trigger a non-terminal arrest.
#   Shortest-telomere-length dependent law, with exponential form parametrized
#   by (a, b): p_nta(l) = b exp(-al) (Martin et al. 2021; (0.023, 0.276)).
PAR_NTA = [0.0247947389, 0.440063202]  # Our fit.

# Probability `p_senA` to trigger senescence.
#   Exponential law with deterministic threshold, parametrized by (a, b, lmin):
#   p_sen(l) = 1 if l <= lmin, b exp(-al) otherwise.
#   One law for each cell type: (p_senA, p_senB).
PAR_SEN_A = [0.186824276, 0.725200993, 27.0]  # Our fit.
PAR_SEN_B = [2.45423414e-06, 0.122028128, 0.0]  # Our fit.


# Cycle duration times [min].
# ---------------------------
#   'const': constant cycles, equal to `CYCLES_*_CONST`.
#   'exp': experimental ones, from `CC_durations.mat` (see below).
#   'exp_new': experimental ones, from `TelomeraseNegative.mat` (see below).


# --------------------------------
# 2) Parameters for post-treatment
# --------------------------------


# Experimental conditions
# -----------------------

PAR_FINAL_CUT = None
PAR_FINAL_CUT_P = None  # Temporary


# Population experiment
# ---------------------

# Saturation rules.
# > 'time': saturation reached at a certain, fixed, time.
# > 'prop': ... when cell concentration exceeds `PROP_SAT * c_dilution`.
# One rule per day s.t. `SAT_CHOICE[i]` is the rule on day `i`.
# NB: if `SAT_CHOICE[-1] == 'prop' the option holds from day `len(SAT_CHOICE)`
#     to the end. Otherwise it must be the same length that `DAY_COUNT`.
SAT_CHOICE = ["prop"]

# Exemple. For rule 'time' (times of saturation) on days 1 and 2 and rule
#   'prop' (concentration of saturation) from day 3 onwards, use:
#   SAT_CHOICE = ['time', 'time', 'prop']
#   PROP_SAT = 1e3  # r_sat used for days 3, 4, ...
#   TIMES_SAT = {0: 0.6045 * 24 * 60,  # t_sat_1 saturation time on day 1.
#                1: 1.6267 * 24 * 60}  # t_sat_2 saturation time on day 2.
# ['time', 'time', 'prop']

# Concentration of saturation in terms of proportion of the concentration of
# dilution (r_sat in publication).
PROP_SAT = 1e3  # 720 # 1e3

# Times of saturation [min].
# NB: `i` must be a key of TIMES_SAT every time `SAT_CHOICE[i]` is 'times'
#     meaning that at day `i`, saturation happens at time `TIMES_SAT[i]`.
TIMES_SAT = {}

# Concatenation ...............................................................
PAR_SAT = {"choice": SAT_CHOICE, "prop": PROP_SAT, "times": TIMES_SAT}
# .............................................................................


# Time grid
# ---------

# Criteria to end computation.
# > 'True': simulation ends when the population has extincted.
# > 'False': after `DAY_COUNT` days.
TO_EXTINCTION_CHOICE = False

# Maximal duration [day] of the experiment / number of dilutions + 1.
if not TO_EXTINCTION_CHOICE:
    DAY_COUNT = 20
else:
    DAY_COUNT = math.inf

# Number of times between 2 consecutive dilutions (including time of 1 dil).
# Remark. We add "artificial" times, 'STEP' before each dilution in order to be
#   able to plot time evolution arrays just before dilution (at `day - STEP`)
#   and just after (at `day`). So typically, times (in day) are:
#  `t = [0, dt, ..., 1-dt, 1-STEP, 1, 1+dt, ..., DAY_COUNT-dt, DAY_COUNT-STEP,
#       DAY_COUNT, DAY_COUNT+dt, ..., DAY_COUNT+1-dt, DAY_COUNT+1-STEP,
#       DAY_COUNT+1]`, with `dt = 1 / (TIMES_PER_DAY_COUNT - 1)`.
#   Thus `TIMES_PER_DAY_COUNT + 1` / `TIMES_SAVED_PER_DAY_COUNT + 1` times are
#   actually computed / saved between 2 dilutions: regular discretization of
#   [t_dil1, t_dil2), plus the time `t_dil2 - STEP`, right before dilution.

# > Computed.
TIMES_PER_DAY_COUNT = 200  # At least `24 * 6 / CYCLE_MIN`.
# > Saved.
TIMES_SAVED_PER_DAY_COUNT = 50  # 50  # 150 !!!!

# Time step [day] between dilution time and the time just before.
STEP = 0.001  # Lower than `1 / TIMES_PER_DAY_COUNT`.

PAR_DEFAULT_SIM_POP = {
    "day_count": DAY_COUNT,
    "t_day_count": TIMES_PER_DAY_COUNT,
    "tsaved_day_count": TIMES_SAVED_PER_DAY_COUNT,
    "step": STEP,
}


# Postreat and plot
# -----------------

# x-axis of the histogram of the telomere length triggering senescence.
# NB: shared by lineage and population plotting parameters.
X_AXIS_HIST_LMIN = np.linspace(0, 250, 251)

# Time step for time evolution along lineages (10 min).
POSTREAT_DT = 1

# Simulation parameters.
PAR_DEFAULT_SIM_LIN = {
    "postreat_dt": None,
    "hist_lmins_axis": None,
    "is_lcycle_count_saved": False,
    "is_evo_saved": False,
}

# =============================================================================
# "Fixed" parameters
# =============================================================================


# Experimental data for calibration of the code, importation
# ==========================================================

# .............................................................................
# All fitted parameters.
PAR = [PAR_NTA, [PAR_SEN_A, PAR_SEN_B], PAR_L_INIT]

PAR_SEN = PAR[1]
P_ONSET = PAR[:2]

P_EXIT = {
    "accident": P_ACCIDENT,
    "death": P_DEATH,
    "repair": P_REPAIR,
    "sen_limit": MAX_SEN_CYCLE_COUNT,
}
# .............................................................................

# Model parameters.

PAR_DEFAULT_LIN = {
    "is_htype_accounted": HTYPE_CHOICE,
    "is_htype_seen": True,
    "fit": PAR,  # Fitted parameters: p_onset, par_l_init.
    "p_exit": P_EXIT,
    "finalCut": PAR_FINAL_CUT,
}

PAR_DEFAULT_POP = {
    "is_htype_accounted": HTYPE_CHOICE,
    "fit": PAR,  # Fitted parameters: p_onset, par_l_init.
    "p_exit": P_EXIT,
    "sat": PAR_SAT,
}
