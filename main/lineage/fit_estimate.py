#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 10:21:49 2022

@author: arat

Script used for the parameter estimation. Run from `slurm_fit.batch` file.

"""

import cma  # pip install cma.
from copy import deepcopy
import numpy as np
import multiprocessing as mp
import os
# import pickle

import project_path
from telomeres.lineage.plot import compute_n_plot_gcurve_error
from telomeres.lineage.simulation import type_of_sort_from_characteristics
from telomeres.dataset.extract_processed_dataset import \
    extract_postreat_lineages


# Number of processor used for simulation.
PROC_COUNT = os.cpu_count()

# Proccesed experimental cycle duration times.
DATA_EXP = extract_postreat_lineages()


# Definition of the space of parameters
# -------------------------------------

# Bounds for parameters (a, b) of the law of non-terminal arrests (NTA).
BOUNDS_NTA_A = [0, 1]  # [0, .1] [0, .2]
BOUNDS_NTA_B = [0, 1]  # [.1, .8] [.1, .95]

# Parameters (a, b, lmin) of the law of senescence (SEN).
# > Possibly distinguishing type A (SENA) and B (SENB).
IS_SEN_COMMUN_TO_TYPES = True
# > (a, b) parameter of the SEN (or SENA and SENB) law.
BOUNDS_SEN_A = [0, 1]
BOUNDS_SEN_B = [0, 1]
# > lmin parameter of the SEN (or SENA) law.
IS_LMIN_A_FIXED = False
BOUNDS_SEN_L = [20, 80]
LMIN_A = 27
# > lmin parameter of the SENB law.
BOUNDS_SENB_L = [0, 40]

# Parameter of the transformation of the initial distribution of telomere len.
# > Translation parameter.
IS_LTRANS_FIXED = True
BOUNDS_LTRANS = [0, 40]
LTRANS = 0
# > Dilatation parameters.
IS_L0_FIXED = True
BOUNDS_L0 = [0, 30]
L0 = 0
IS_L1_FIXED = True
BOUNDS_L1 = [0, 60]
L1 = 0
# > All parameters.
PAR_L_INIT = [LTRANS, L0, L1]


# Definition of CMAES parameters.
# -------------------------------

# Initial point.
# NB: if None chosen randomly, uniformly in the parameters domain.
POINT_0 = None

# Initial standard deviation.
# NB: the value expected is the value scaled for the set [0, 1] (the sigma
#     is later rescaled along each variable depending on its bounds)  or s.t.
#     the optimum in [0, 1] is expected to lie in about `POINT_0 +- 3*SIGMA_0`.
SIGMA_0 = 0.5

# Maximum number of iteration per CMAES run.
MAXITER = 50000

# Numbers of evaluations of the cost function per iteration per run of CMAES.
POPSIZES = [32, 52, 72]


# Computation of resulting useful parameters.
# -------------------------------------------

def transform_bounds_of_int_values(bounds):
    """Return the "maximal" (up to 1e-9) interval such that `np.round(bounds)`
    is `bounds`.

    NB: If the set of admissible value for some parameter is {0, 1, 2} then the
        bounded interval should be [-.5 + 1e-9, 2.5] for values 0 and 2 (thus
        rounded from numbers in [-.5 + 1e-9, .5] and (1.5, 2.5]) to have same
        weight than 1 for CMAES.

    """
    return [bounds[0] - .5 + 1e-9, bounds[1] + .5]


PAR_SPACE_CHOICE = {"is_sen_commun_to_types": IS_SEN_COMMUN_TO_TYPES,
                    "is_lmin_a_fixed": IS_LMIN_A_FIXED,
                    "is_ltrans_fixed": IS_LTRANS_FIXED,
                    "is_l0_fixed": IS_L0_FIXED, "is_l1_fixed": IS_L1_FIXED,
                    "lmin_a": LMIN_A,
                    "par_l_init": PAR_L_INIT}


def gather_bounds_with_int_index(kwarg=PAR_SPACE_CHOICE):
    """Return useful information on the parameters to fit depending on the
    space of parameters previously parametred (the number of parameters to fit
    depends on the `IS_*` variables, while their respective bounds are defined
    by the `BOUNDS_*` variables).

    Returns
    -------
    [low_bounds, up_bounds] : list
        List of both lists (of length N each) of the lower bounds and uper
        bounds of all the N parameters that have to be fit.
    int_par_idxs : list
        List of the indexes of the parameters to fit that should be treated by
        CMAES as intergers.

    """
    # NTA bounds.
    low_bounds = [BOUNDS_NTA_A[0], BOUNDS_NTA_B[0]]
    up_bounds = [BOUNDS_NTA_A[1], BOUNDS_NTA_B[1]]
    int_par_idxs = []
    # SEN bounds.
    low_bounds.extend([BOUNDS_SEN_A[0], BOUNDS_SEN_B[0]])
    up_bounds.extend([BOUNDS_SEN_A[1], BOUNDS_SEN_B[1]])
    if not kwarg["is_lmin_a_fixed"]:
        int_par_idxs.append(len(low_bounds))
        bounds_l_new = transform_bounds_of_int_values(BOUNDS_SEN_L)
        low_bounds.append(bounds_l_new[0])
        up_bounds.append(bounds_l_new[1])
    if not kwarg["is_sen_commun_to_types"]:
        bounds_l_new = transform_bounds_of_int_values(BOUNDS_SENB_L)
        low_bounds.extend([BOUNDS_SEN_A[0], BOUNDS_SEN_B[0], bounds_l_new[0]])
        up_bounds.extend([BOUNDS_SEN_A[1], BOUNDS_SEN_B[1], bounds_l_new[1]])
        int_par_idxs.append(len(low_bounds) - 1)
    # PAR_L_INIT bounds if not fixed but to be fit.
    if not kwarg["is_ltrans_fixed"]:
        bounds_l_new = transform_bounds_of_int_values(BOUNDS_LTRANS)
        low_bounds.append(bounds_l_new[0])
        up_bounds.append(bounds_l_new[1])
        int_par_idxs.append(len(low_bounds) - 1)
    if not kwarg["is_l0_fixed"]:
        bounds_l_new = transform_bounds_of_int_values(BOUNDS_L0)
        low_bounds.append(bounds_l_new[0])
        up_bounds.append(bounds_l_new[1])
        int_par_idxs.append(len(low_bounds) - 1)
    if not kwarg["is_l1_fixed"]:
        bounds_l_new = transform_bounds_of_int_values(BOUNDS_L1)
        low_bounds.append(bounds_l_new[0])
        up_bounds.append(bounds_l_new[1])
        int_par_idxs.append(len(low_bounds) - 1)
    return [low_bounds, up_bounds], int_par_idxs


BOUNDS, INT_PAR_IDXS = gather_bounds_with_int_index()

PARAMETER_COUNT = len(BOUNDS[0])
if __name__ == "__main__":
    print(f"Fit on {PARAMETER_COUNT} parameters.")

CMAES_KWARGS = {'maxiter': MAXITER,
                'bounds': BOUNDS,
                'integer_variables': INT_PAR_IDXS
                # 'transformation': [transform_point_flat, None]
                # 'popsize': PARA_COUNT_MAX - 2
                }


# Definition of the cost function.
# --------------------------------

# ...........................
# > Parameters of simulation.

LINEAGE_COUNT_ON_ALL_SIM_MIN = 230
# 150-99.4722, 230-82.6930,  2500-75.9411, 3000-75.3016, 1200-74.0958
if PARAMETER_COUNT == 5:
    COST_MIN_FOR_PRECISION = 34  # 85
    COST_MIN_FOR_PRECISION_MIN = 22  # 55
else:
    COST_MIN_FOR_PRECISION = 25  # 72
    COST_MIN_FOR_PRECISION_MIN = 14  # 45
LINEAGE_COUNT_ON_ALL_SIM = 1200  # 1200 seems good ratio error/t_computation.
LINEAGE_COUNT_ON_ALL_SIM_MAX = 3000


BPROP_ERROR_MAX_1SIMU = .25
BPROP_ERROR_MAX = .27  # bprop exp:  0.5675675675675675 & err = exp - sim.

# .................................
# > Parameters of error computation (should not require modification).

# Types of curves used to fit.
CHARAC_S = [['senescent'], ['btype', 'senescent'], ['atype', 'senescent'],
            ['btype']]

# Respective weights for the error of each curve.
# LINEAGE_COUNTS = np.array([148, 84, 64, 115])
# WEIGHTS = LINEAGE_COUNTS / np.sum(LINEAGE_COUNTS)
WEIGHTS = np.array([.3, 1, 1, 1])  # .3,1,1,1/ 1,1,1,1/ .5,.5,.5,1/ .5,.9,.9,1
# WEIGHTS = np.array([.5,.5,.5,1])
WEIGHTS = WEIGHTS / np.sum(WEIGHTS)

# Type of error between experimental and simulated curves (0 for L2 error).
L2_ERROR_TYPE = 0

# If True, the error take into account the error.
IS_BPROP_ACCOUNTED = False
WEIGHT_BPROP_ERROR = 75

def point_to_cost_fct_parameters(point, kwarg=PAR_SPACE_CHOICE):
    """From a point in the space of parameters to explore return the
    parameters of all the laws adding fixed parameters given in `kwarg`
    and rounding parameters that should be integers.

    """
    point_rounded = point.copy()  # Int variables rounded.
    int_par_idxs = gather_bounds_with_int_index(kwarg)[1]
    for idx in int_par_idxs:
        point_rounded[idx] = np.round(point_rounded[idx])
    if kwarg["is_lmin_a_fixed"]:
        point_rounded = [*point_rounded[:4], deepcopy(kwarg["lmin_a"]),
                         *point_rounded[4:]]
    # NTA parameters.
    par_nta = point_rounded[:2]
    # SEN parameters.
    if kwarg["is_sen_commun_to_types"]:
        par_sen = [list(point_rounded[2:5]), list(point_rounded[2:5])]
    else:
        par_sen = [list(point_rounded[2:5]), list(point_rounded[5:8])]
    # L_INIT parameters.
    par_l_init = deepcopy(kwarg["par_l_init"])
    par_l_init_unfixed_count = 0
    if not kwarg["is_l1_fixed"]:
        par_l_init[2] = point_rounded[-1]
        par_l_init_unfixed_count += 1
    if not kwarg["is_l0_fixed"]:
        par_l_init[1] = point_rounded[-1 - par_l_init_unfixed_count]
        par_l_init_unfixed_count += 1
    if not kwarg["is_ltrans_fixed"]:
        par_l_init[0] = point_rounded[-1 - par_l_init_unfixed_count]
    return par_nta, par_sen, par_l_init

def cost_function(point, is_plotted=False):
    parameters = point_to_cost_fct_parameters(point)
    p_update = {'fit': parameters}
    sum_costs = 0
    btype_prop = 0
    for i in range(len(CHARAC_S)):
        gcurve = type_of_sort_from_characteristics(CHARAC_S[i])
        cost = compute_n_plot_gcurve_error(
            DATA_EXP, LINEAGE_COUNT_ON_ALL_SIM_MIN, [gcurve], CHARAC_S[i],
            par_update=p_update, error_types=[0, 1],
            is_plotted=is_plotted)[gcurve]
        btype_prop = max(btype_prop, cost[1])
        if btype_prop > BPROP_ERROR_MAX_1SIMU or np.isnan(cost[1]):
            return np.NaN
        sum_costs += cost[0] * WEIGHTS[i]
    is_to_recompute = False
    if sum_costs < COST_MIN_FOR_PRECISION_MIN:
        is_to_recompute = True
        lineage_count_on_all_simu = LINEAGE_COUNT_ON_ALL_SIM_MAX
    elif sum_costs < COST_MIN_FOR_PRECISION:
        is_to_recompute = True
        lineage_count_on_all_simu = LINEAGE_COUNT_ON_ALL_SIM
    # If interesting cost, we compute it again but with more precision.
    if is_to_recompute:
        sum_costs = 0
        btype_prop = 0
        for i in range(len(CHARAC_S)):
            gcurve = type_of_sort_from_characteristics(CHARAC_S[i])
            cost = compute_n_plot_gcurve_error(
                DATA_EXP, lineage_count_on_all_simu, [gcurve], CHARAC_S[i],
                par_update=p_update, error_types=[0, 1],
                is_plotted=is_plotted)[gcurve]
            btype_prop = max(btype_prop, cost[1])
            if btype_prop > BPROP_ERROR_MAX or np.isnan(cost[1]):
                return np.NaN
            sum_costs += cost[0] * WEIGHTS[i]
    if IS_BPROP_ACCOUNTED:
        sum_costs += btype_prop * WEIGHT_BPROP_ERROR
    return sum_costs


def parallize_until_no_nan(function, point_count, optimizer, proc_count):
    points = optimizer.ask(point_count)
    # print('optimizer points send to cost_function', points)
    pool = mp.Pool(proc_count)
    outs = pool.map_async(function, points).get()
    # If nan outputs we compute for new points until no nan output.
    isnot_nan = ~np.isnan(outs)
    computed_count = np.sum(isnot_nan)
    while computed_count < point_count:
        # Computation at new points.
        points_temp = optimizer.ask(point_count - computed_count)
        outs_temp = pool.map_async(function, points_temp).get()
        # Update with computed values.
        isnot_nan_idxs = np.arange(point_count)[isnot_nan]
        points = [points[i] for i in isnot_nan_idxs]
        points.extend(points_temp)
        outs = [outs[i] for i in isnot_nan_idxs]
        outs.extend(outs_temp)
        isnot_nan = ~np.isnan(outs)
        computed_count = np.sum(isnot_nan)
    pool.close()
    pool.join()
    return points, outs


def optimize_w_cmaes_multiple(par_bounds, sigma, cmaes_kwargs, popsizes,
                              initial_point=None, proc_count=1):
    """Run CMAES `len(popsizes)` times in a raw (with the functions specified
    by `cmaes_kwargs`), updating each time the 'bestever' parameters. The ith
    run of CMAES evaluate, at each iteration, `popsizes[i]` times the cost
    function.

    Note that the first run will start at the point `initial_point` if not
    None. Otherwise it is, as the initial point of the following runs, drawn
    randomly.

    """
    # if popsizes[-1] > PARA_COUNT_MAX:
    #     raise Exception("Error `optimize_w_cmaes_multiple`, population sizes"
    #                     " too big compared to the maximum of parallelization"
    #                     " allowed")
    bestever = cma.optimization_tools.BestSolution()
    run_count = len(popsizes)
    point = initial_point
    for i in range(run_count):
        cmaes_kwargs['popsize'] = int(popsizes[i])
        cmaes_kwargs['verb_append'] = bestever.evalsall
        cmaes_kwargs['CMA_stds'] = np.array(par_bounds[1]) - \
            np.array(par_bounds[0])
        if isinstance(point, type(None)) or i > 0:
            # For non-indidicated 1st run / following runs initial point drawn.
            point = np.random.uniform(low=par_bounds[0], high=par_bounds[1])
        optimizer = cma.CMAEvolutionStrategy(point, sigma, cmaes_kwargs)
        optimizer.disp()
        # logger = cma.CMADataLogger().register(optimizer,
        #                                       append=bestever.evalsall)
        while not optimizer.stop():
            para_count = int(popsizes[i])
            points = []
            costs = []
            out = parallize_until_no_nan(cost_function, para_count, optimizer,
                                         proc_count=proc_count)
            points.extend(out[0])
            costs.extend(out[1])
            optimizer.tell(points, costs)
            optimizer.disp()
            optimizer.logger.add()
            # text_file = open(join("outcmaes", "history.txt"), "wb")
            # text_file.write(optimizer.pickle_dumps())
            # text_file.close()
        bestever.update(optimizer.best)
        optimizer.result_pretty()
        cma.s.pprint(optimizer.best.__dict__)
    return optimizer


if __name__ == "__main__":
    if os.cpu_count() > POPSIZES[0]:
        res = optimize_w_cmaes_multiple(BOUNDS, SIGMA_0, CMAES_KWARGS,
                                        POPSIZES, initial_point=POINT_0,
                                        proc_count=PROC_COUNT)

# text_file = open(join("outcmaes", "history.txt"), "rb")
# optimizer = pickle.loads(text_file.read())
# text_file.close()


# ---------
# Auxiliary
# ---------

LINEAGE_COUNT_ON_ALL_SIM_TO_PLOT = 1e4


def compute_n_plot_gcurves(point, kwarg, is_plotted=True, proc_count=1,
                           simulation_count=None, is_printed=True):
    """Plot the curves fitted."""
    parameters = point_to_cost_fct_parameters(point, kwarg)
    p_update = {'fit': parameters}
    sum_costs = 0
    btype_prop = 0
    for i in range(len(CHARAC_S)):
        gcurve = type_of_sort_from_characteristics(CHARAC_S[i])
        cost = compute_n_plot_gcurve_error(
            DATA_EXP, LINEAGE_COUNT_ON_ALL_SIM_TO_PLOT, [gcurve], CHARAC_S[i],
            par_update=p_update, error_types=[0, 1], is_printed=is_plotted,
            simulation_count=simulation_count, proc_count=proc_count)[gcurve]
        btype_prop = max(btype_prop, cost[1])
        sum_costs += cost[0] * WEIGHTS[i]
        if is_printed:
            print('\n Errors for individual gcurves:', btype_prop, cost[0])
    if IS_BPROP_ACCOUNTED:
        sum_costs += btype_prop * WEIGHT_BPROP_ERROR
    if is_printed:
        print('Total cost', sum_costs)
    return sum_costs
