#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 16:22:47 2020

@author: arat
"""
import scipy.io as sio
import math
import numpy as np

import aux_parameters_functions as parf
import dataset_plot as pd


# Adjustable parameters
# =====================

# -----------------------
# Parameters of the model
# -----------------------

# Types.
# ------
#   'True': if hybrib types taken into account.
#   'False': otherwise (i.e. B must finish  nta bf being allowed to enter sen).
HYBRID_CHOICE = True

# Overhang / shortening rate.
# ---------------------------
#   'const': constant overhang, equal to 'OVERHANG'.
#   'uniform': uniform in {0VERHANG_LOW, ...., OVERHANG_UP}.
OVERHANG_CHOICE = 'const'

# ...........................
# > case OVERHANG == 'const'.
OVERHANG = 7
# > case OVERHANG == 'uniform'.
OVERHANG_LOW = 5
OVERHANG_UP = 10
# ...........................



# Rates of accidental events.
# ---------------------------
# Mortality rate / probability to die accidentally (`p_accident`).
P_ACCIDENTAL_DEATH = 1 * .0043 # 4.3 * 1e-3 in (Coutelier et al. 2018).
# Accidenta shortening rate / probability for a telome to shorten accidentally.
# P_L_ABRUPT = 1 * .0043

# Number of long cycles.
# -----------------------
# Geometrical laws (i.e. constant proba) fitted in (Martin et al. 2021).

# > Senescence.
# Probability `p_death` of death 'naturally' from senescence).
P_GEO_SEN = 0.58 # 0.4677 # 0.58 for D=220
# Maximum number of senescent cycles (`math.inf` for no limit).
MAX_SEN_CYCLE_COUNT = math.inf

# > Non-terminal arrest (probability `p_repair` of repairing or adapt).
P_GEO_NTA = 0.65

# ......................................................................
P_EXIT = [P_ACCIDENTAL_DEATH, P_GEO_SEN, P_GEO_NTA, MAX_SEN_CYCLE_COUNT]
# ......................................................................

# Telomere(s) length(s) triggering an arrest.
# -------------------------------------------
#   'shortest': we test only the shortest.
#   'all': all telomeres shortest than a threshold 'L_MIN_MAX'.
TRIG_TELO_CHOICE = 'shortest'

# .................................
# > case TRIG_TELO_CHOICE == 'all'.
L_MIN_MAX = 150
# .................................

# PAR = ([0.025154449389371336, 0.25637137728070253], # 48-42.20 16.35% (n1e4)
#         [[0.5588440524832747, 0.7861336757668571, 0.0],
#         [4.863925108501737e-06, 0.12580565434369623, 0.0]], 0)
# PAR = ([0.01, 0.1], [[0.03, 0.05, 0], [0.003, 0.08, 0]]) # !!!
# PAR = ([0.0229, 0.4546], [[0.6967, 0.9999, 27], [0, 0.1195, 0]], 30) # 8.75, 19.5%
# PAR = ([2.33212133e-02, 4.48589525e-01],
#        [[9.63567774e-01, 9.39539544e-01, 2.66963691e+01],
#          [5.50737210e-05, 1.25412734e-01, 2.78657325e+00]],
#        [28, 0, 0])
PAR = ([0.0247947389, 0.440063202],
       [[0.186824276, 0.725200993, 27.0], [2.45423414e-06, 0.122028128, 0.0]],
       [0, 40, 58.0]) # Final fit.
# PAR = ([0.0247947389, 0.440063202], # [0.0249, 0.465] # [0.024, 0.45], #
#         [[0.186824276, 0.725200993, 35], [2.45423414e-06, 0.122028128, 0.0]],
#         [0, 40, 58.0])
# PAR = ([0.028, 0.58], # [0.0249, 0.465] # [0.024, 0.45], #
#        [[0.186824276, 0.725200993, 38], [2.45423414e-06, 0.122028128, 0.0]],
#        [0, 40, 58.0])
# PAR = ([0.0247947389, 0.440063202],
#         [[0.17, 0.65, 36.0], [2.45423414e-06, 0.122028128, 0.0]],
#         [0, 40, 58.0])
# if __name__ == "__main__":
    # parf.plot_laws(PAR, fig_name='fit_10', is_par_plot=True)
    # parf.plot_laws(PAR, fig_name='fit_10_wo_par', is_par_plot=False)
P_ONSET = PAR[:2]

# Law for the onset of non-terminal arrests.
# ------------------------------------------
TRIG_ARREST_CHOICE = 'exp'
# ....................................
# > case TRIG_ARREST_CHOICE == 'exp' .
#   Telomere-dependent probability to enter a arrest given by the exponential
#   law with parameters (a, b): b exp(-al).
A_EXP_AR = PAR[0][0] # 0.01 # (0.023, 0.276) in (Martin et al. 2021).
B_EXP_AR = PAR[0][1] # 0.1
PAR_NTA = [A_EXP_AR, B_EXP_AR]
# >> Plotting b exp(-al) wrt. l for different values of a and b.
# if __name__ == "__main__":
#     a_to_test = [.005, .01, .05, .1]
#     b = .2
#     parf.plot_laws_nta_various_a(a_to_test, b, is_saved=True, font_scale=1.5)
#     b_to_test = [.005, .01, .05, .1]
#     a = .01
#     parf.plot_laws_nta_various_b(b_to_test, a, is_saved=True, font_scale=1.5)

# > case TRIG_ARREST_CHOICE == 'sigmoid'.
#   Deterministic generation-dependent probability to enter an arrest given by
#   sigmoid law with parameters (a, b) in (Martin thesis).
#   See also function (2) in (Martin et al. 2021).
A_SIGMOID = 61.25
B_SIGMOID = 13.47
# > case TRIG_ARREST_CHOICE == 'lmin_const'.
#   Deterministic (<L_MIN_AR> [bp]) minimal length triggering an arrest.
L_MIN_AR = 61
# > case TRIG_ARREST_CHOICE == 'lmin_gaussian'.
#   Probabilistic: a truncated gaussian law with par from (Martin thesis).
L_MIN_AR_MU = 49
L_MIN_AR_SIGMA = 35.5
# ....................................


# Law for the onset of senescence.
# --------------------------------
TRIG_SEN_CHOICE = 'exp_new'

# ................................
# > case TRIG_SEN_CHOICE == 'exp'.
#   Exponential law: b exp(-al) (Martin et al. 2021).
A_EXP_SEN = 0.0195
B_EXP_SEN = 0.165

# > case TRIG_SEN_CHOICE == 'exp_new'.
#   Exponential law w deterministic threshold possibly differentiating A and B.
#   >> 0: for type-A cells, 1: for type-B cells.
PAR_SEN = PAR[1]
# >> Plotting.
# if __name__ == "__main__":
#     a, b, lmin = (.01, .4, 20)
#     parf.plot_law_sen(a, b, lmin, is_saved=True, font_scale=1.5)

# > case TRIG_SEN_CHOICE == 'lmin_const'.
#   Deterministic threshold.
L_MIN_SEN_ATYPE = 19
L_MIN_SEN_BTYPE = 16
# > case TRIG_SEN_CHOICE == 'lmin_gaussian'.
#   Probabilistic threshold, following a gaussin distribution.
L_MIN_SEN_MU = 33
L_MIN_SEN_SIGMA = 37.5
# ................................


# Cycle duration times [min].
# ---------------------------
#   'const': constant cycles, equal to <CYCLES_*_CONST>.
#   'exp': experimental ones, from `CC_durations.mat` (see below).
#   'exp_new': experimental ones, from `TelomeraseNegative.mat` (see below).
CYCLES_CHOICE = 'exp_new'

# ................................
# > case CYCLES_CHOICE == 'const'.
CYCLES_A_CONST = 90 # Mean division time (mdt.) of A non-senescent.
CYCLES_B_CONST = 155 # Mdt. normal cycles of B-type.
CYCLES_B_LONG_CONST = 440 # Mdt. B cells in 1st seq of long.
CYCLES_B_AVG_CONST = 155 # Mdt. B cells between end of 1st seq. of lc and sen.
CYCLES_SEN_CONST = 610 # Mdt. in senescence (last cycle excluded).
CYCLES_SEN_LAST_CONST = 610 # Mdt. of the last cycle of senescence.
# ................................


# ------------------
# Initial conditions
# ------------------

# Initialization.
# ---------------
#   'new': initial cell(s) generates from scratch according to next parameters.
#   'load': already generated & saved data, to recreate same initial cell(s).
#   'cut': initial cell(s) generates from scratch according to final cut.
DATA_INIT_CHOICE = 'new'

if DATA_INIT_CHOICE == 'load':
    # .... Chose parameters ....
    # > In population.
    DATA_INIT_LOAD_PATH = 'data/data_init/data_init'
    PAR_FINAL_CUT = None
    # ...........................
if DATA_INIT_CHOICE == 'cut':
    # .... Chose parameters ....
    LENGTH_CUT = 70 # Length of the telomere after cut [bp].
    AVG_CUT_DELAY = 1 / 0.577 # 7287 # 0.577 # 1.613 # Average nb of gen between end of galactose and cutting.
    PROBA_CUT_ESCAPE = .1 # Proba of/proportion of lineages escaping cutting.
    IDX_DOX = 0
    IDX_GAL = 36  # 6h after Dox addition.
    IDX_RAF = 9 * 6 # 2 * 36  # 12h after Dox addition.
    IDXS_FRAME = [IDX_DOX, IDX_GAL, IDX_RAF]
    # ...........................
    PAR_FINAL_CUT = [LENGTH_CUT, AVG_CUT_DELAY, PROBA_CUT_ESCAPE, IDXS_FRAME]
else:
    PAR_FINAL_CUT = None
PAR_FINAL_CUT_P = None  # Temporary


# Initial telomere lengths [bp].
# ------------------------------
CHROMOSOME_COUNT = 16
#   'const': constant initial length, equal to <L_INF>.
#   'gaussian': gaussian law of parameter (L_INF, L_SIGMA).
#   'exp': experimental (from 'etat_asymp_val_juillet') translated by `L_INIT`.
#   'two_normal-long': 2 cells, one with long shortest telomere and one with
#       medium (loaded from data obtained from running `af.draw_lengths_one_*`
#       that extracts shortest and medium from `C_FOR_EXTREMUM_LENGHT` cells.
#   'two_normal-short': same with 'short' instead of 'long'.
L_INIT_CHOICE = 'exp'

# ..............................
# > case L_INIT_CHOICE == 'exp'.
PAR_L_INIT = PAR[2]
LTRANS, L0, L1 = PAR_L_INIT
# > case L_INIT_CHOICE == 'gaussian' or 'const'.
L_INF = 342 # Mean telomere length at equilibrium [bp].
L_SIGMA = 100.0
# > case L_INIT_CHOICE == 'two_normal-long' or 'two_normal-short'.
C_FOR_EXTREMUM_LENGHT = 1e6
# ...............................


# Initial clocks (i.e. remaining times before division).
# ------------------------------------------------------
# Cycles drawn accordingly to 'CYCLES_CHOICE'.
# > Desynchronization of initial population.
#    'null': no desynchronization (initially, all cells start their cycle).
#    'uniform': delay for each cell (uniform between 0 and cells' cycle time).
DELAY_CHOICE = 'uniform'


# Population experiment
# ---------------------

# Saturation.
SAT_CHOICE = ['prop']
# ...................
# > case SAT_CHOICE[i] == 'prop'.
#   At day <i>, saturates if the concentration exceeds <PROP_SAT * c_dilution>.
#   NB: if <SAT_CHOICE[-1]> is 'prop' the option holds from day len(SAT_CHOICE)
#       to the end. Otherwise it must be the same length that <DAY_COUNT>.
PROP_SAT = 1e3 #720 # 1e3 # 760
# > case SAT_CHOICE[i] == 'time'.
#   At day <i>, saturation happens at time <TIMES_SAT[i]> [min].
TIMES_SAT = {0: 0.6045 * 24 * 60,
             1: 1.6267 * 24 * 60}
# ...................


# End of computation.
#   'True': simulation ends when the population has extincted.
#   'False': after <DAY_COUNT> days.
TO_EXTINCTION_CHOICE = False
# .........................................
# > case TO_EXTINCTION_CHOICE[i] == 'True'.
DAY_COUNT = 20 # Number of days of the experiment i.e. number of dilutions + 1.
# .........................................
# > Number of times between 2 consecutive dilutions (including time of 1 dil).
TIMES_PER_DAY_COUNT = 200 # Computed.
TIMES_SAVED_PER_DAY_COUNT = 150 # Saved. See below, TIMES_SAVED_PER_DAY_COUNT+1
                                # (the one just below dilution) actually saved.
# > Time step [day] between dilution time and the previous one.
#   NB: lower than 1 / TIMES_PER_DAY_COUNT.
STEP = 0.001

HIST_LMIN_X_AXIS = np.linspace(0, 250, 251)


# "Fixed" parameters
# ===================

# Creation of time related arrays for population simulations.
# -----------------------------------------------------------
# NB: we add "artificial" times at 'STEP' before each dilution in order to be
#  able to plot time evolution arrays just before dilution (at 'day'-'STEP')
#  and just after (at 'day').
#  So typically we chose, for dt = 1 / (TIMES_PER_DAY_COUNT-1):
#  t = [0, dt, ..., 1-dt, 1-STEP, 1, 1+dt, ..., DAY_COUNT-dt, DAY_COUNT-STEP,
#       DAY_COUNT, DAY_COUNT+dt, ..., DAY_COUNT+1-dt, DAY_COUNT+1-STEP,
#       DAY_COUNT+1]
TIMES, TIME_SAVED_IDXS, DIL_IDXS, DIL_SAVED_IDXS = parf.make_time_arrays(
    TIMES_PER_DAY_COUNT, TIMES_SAVED_PER_DAY_COUNT, DAY_COUNT, STEP)

# Imported parameters to post-treat and / or plot.
# ================================================
IS_SAVED = False # Plottings will be saved if true.
# ..............
if IS_SAVED:
    FIG_DIRECTORY = "figures"
# ..............
else:
    FIG_DIRECTORY = None

# -------------
# L_INIT_CHOICE
# -------------

# Case L_INIT_CHOICE == 'exp'.
# ----------------------------
# > Importation postreat of data.
L_INIT_EXP = np.loadtxt('data/etat_asymp_val_juillet')
l_init_support, l_init_prob, l_support_nozero, l_prob_nozero \
    = parf.postreat_l_init_exp(L_INIT_EXP)

# > Visualizations of the translated density (prints & plots).
# if __name__ == "__main__":
#     pd.plot_l_init(l_init_support, l_init_prob, FIG_DIRECTORY)
#     support, proba = parf.transform_l_init(L_INIT_EXP, *PAR_L_INIT)
#     pd.plot_transform_l_init(l_init_support, l_init_prob, support, proba,
#                              FIG_DIRECTORY, *PAR_L_INIT)

# Case L_INIT_CHOICE == 'two_normal-long' or 'two_normal-short'.
# --------------------------------------------------------------
L_LONG_PATH = f'data/data_init/l_init_long_{C_FOR_EXTREMUM_LENGHT:.0e}.txt'
L_MEDIUM_PATH = f'data/data_init/l_init_medium_{C_FOR_EXTREMUM_LENGHT:.0e}.txt'
L_SHORT_PATH = f'data/data_init/l_init_short_{C_FOR_EXTREMUM_LENGHT:.0e}.txt'
C_FOR_EXTREMUM_LENGHT = int(C_FOR_EXTREMUM_LENGHT)
# if __name__ == "__main__":
#     # Load and print the data.
#     l_long = np.loadtxt(L_LONG_PATH)
#     l_medium = np.loadtxt(L_MEDIUM_PATH)
#     l_short = np.loadtxt(L_SHORT_PATH)
#     parf.print_data_on_special_initial_distributions(l_short, l_medium, l_long)


# -------------
# CYCLES_CHOICE
# -------------

# Parameters to postreat.
# -----------------------
THRESHOLD = 18 # [10 min], long cycles are `> THRESHOLD`.
CYCLE_MIN = 5 # [10 min], all cycles are `>= CYCLE_MIN`.
GEN_COUNT_BY_LINEAGE_MIN = 1 # Number of generations per lineage
                             # `>= GEN_COUNT_BY_LINEAGE_MIN`.


# Case CYCLES_CHOICE == 'exp'.
# ----------------------------
cycles_data = sio.loadmat('data/microfluidic/CC_durations.mat') # [10 min]
CDTS_OLD = parf.postreat_cycles_exp(cycles_data, CYCLE_MIN, THRESHOLD)[0] # [min]

# Case CYCLES_CHOICE == 'exp_new'.
# --------------------------------
# WARNING: if parameters updated, need to run `make_cycles_dataset.py` again.
CDTS = parf.extract_cycles_dataset()

# Visualizations of data (prints & plots).
# ----------------------------------------
# if __name__ == "__main__":
#     pd.plot_cycles_from_old_dataset(CDTS_OLD, THRESHOLD, FIG_DIRECTORY)
#     pd.plot_cycles_from_dataset(FIG_DIRECTORY)

# Dataset for finalCul experiment.
# --------------------------------
if DATA_INIT_CHOICE == 'cut':
    CDTS_FINALCUT = parf.extract_cycles_dataset_finalCut()


# Experimental data for calibration of the code, importation
# ==========================================================

# -------------------------------
# Evolution of cell concentration
# -------------------------------
# NB structure of the data:
#   - X[:, 0]: x-axis for telomerase-positive (mean on 3 indep. measuments).
#   - X[:, 2]: corresponding mean standard error (SEM).
#   - X[:, 1]: x-axis for telomerase-negative (mean on 3 indep. measuments).
#   - X[:, 3]: corresponding mean standard error (SEM).
#  and similary for y-axis data with Y.

# > Experiment with no saturation from 'cultureall.mat'.
C_EXP_DATA_NOSAT = sio.loadmat('data/population/cultureall.mat')
Y_NOSAT = C_EXP_DATA_NOSAT['Y']
X_NOSAT = C_EXP_DATA_NOSAT['X']

# > Experiment with saturation from 'senesence_Tet02_tlc1.xlsx'.
#   NB: SEM is mean standard error of population doubling numbers (Y) converted
#     to concentrations, SEM[:, 0] for telomerase-positive, SEM[:, 1] neagtive.
C_EXP_DATA = np.load('data/population/senesence_Tet02_tlc1.npz')
X = C_EXP_DATA['X']
Y = C_EXP_DATA['Y']
SEM = C_EXP_DATA['SEM']

# Data from raw value (corrects the error on errorbars).
(C_AVG_P1, C_STD_P1, C_AVG_M1, C_STD_M1, C_AVG_P2, C_STD_P2, C_AVG_M2,
 C_STD_M2, C_AVG_P3, C_STD_P3, C_AVG_M3, C_STD_M3) = parf.extract_evo_c()

# > Visualizations.
# if __name__ == "__main__":
#     parf.plot_data_exp_concentration_curves(X_NOSAT, Y_NOSAT, X, Y, SEM,
#                                             IS_SAVED)

# -----------------------------
# Evolution of telomere lengths
# -----------------------------
EVO_L_EXP = np.transpose(np.loadtxt('data/population/BioRad_2012-04-26_18hr_36'
                                    'min_analysis.txt'))

# > Visualizations.
# if __name__ == "__main__":
#     parf.plot_data_exp_length_curves(np.arange(len(EVO_L_EXP[0])),
#                                       np.mean(EVO_L_EXP, 0),
#                                       np.std(EVO_L_EXP, 0), FIG_DIRECTORY)

DEFAULT_PARAMETERS_L = {'is_htype_accounted': HYBRID_CHOICE,
                        'is_htype_seen': True,
                        'parameters': PAR,
                        'postreat_dt': None,
                        'hist_lmins_axis': None,
                        'is_lcycle_counts': False,
                        'is_evo_stored': False,
                        'p_exit': P_EXIT,
                        'par_finalCut': PAR_FINAL_CUT
                        }

DEFAULT_PARAMETERS_P = {'hybrid_choice': HYBRID_CHOICE,
                        'p_exit': P_EXIT,
                        'p_onset': P_ONSET,
                        'par_l_init': PAR_L_INIT,
                        'sat_choice': SAT_CHOICE,
                        'times_sat': TIMES_SAT,
                        'prop_sat': PROP_SAT,
                        'par_finalCut': PAR_FINAL_CUT_P
                        }


# if DATA_INIT_CHOICE == 'cut':
#     DEFAULT_PARAMETERS_L['parameters'] = PAR_CUT
#     DEFAULT_PARAMETERS_P['parameters'] = PAR_CUT
