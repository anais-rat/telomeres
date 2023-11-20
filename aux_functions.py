#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 16:08:23 2020

@author: arat
"""

# import imp
import math
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rd
# NB: for parallelization issues need to use rd.RandomState() rather than rd.
# rd.RandomState() replaced by rd. but seeds initilize for reproducinility.
# idem with population_simulation
import os
import pandas as pd
import warnings

import aux_parameters_functions as parf
import parameters as par
# imp.reload(par)



# Utilitary functions
# -------------------

def nanstat(arr_stat, p_up, p_down, axis=0):
    # arr_stat = np.asarray(arr_stat)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        stats = {'mean': np.nanmean(arr_stat, axis=axis),
                 'perup': np.nanpercentile(arr_stat, p_up, axis=axis),
                 'perdown': np.nanpercentile(arr_stat, p_down, axis=axis)}
    return stats

def stat(arr_stat, p_up, p_down, axis=0):
    stats = {'mean': np.mean(arr_stat, axis=axis),
             'perup': np.percentile(arr_stat, p_up, axis=axis),
             'perdown': np.percentile(arr_stat, p_down, axis=axis)}
    return stats

def stat_all(arr_stat, p_up, p_down, axis=0):
    stats = {'mean': np.mean(arr_stat, axis=axis),
             'per': [np.percentile(arr_stat, p_down, axis=axis),
                     np.percentile(arr_stat, p_up, axis=axis)],
             'ext': [np.min(arr_stat, axis=axis),
                     np.max(arr_stat, axis=axis)]}
    return stats

#
def nansum(arr, axis=None, keepdims=False):
    """ Similar np.nansum except that if there are only NaN on a sommation axis
    the sum along it is NaN instead of 0.

    """
    arr_sum = np.nansum(arr, axis=axis, keepdims=keepdims)
    arr_isnan = np.isnan(arr).all(axis=axis)
    if arr_isnan.any(): # If at least one axis with only nan.
        # Their sum is Nan instead of 0.
        if isinstance(arr_sum, np.ndarray): # arr_sum is array, array returned.
            arr_sum[arr_isnan] = np.nan
        else: # arr_sum = 0, is int or float not array !
            return np.NaN
    return arr_sum

def nanargsort1D(arr):
    """ Similar np.aargsort except that NaN are not orderred.

    """
    # We compute indexes corresponding to usual the sorting (NaN at the end).
    idxs_sorted = np.argsort(arr)
    # And return only indexes corresponding to non NaN values.
    non_nan_count = sum(~np.isnan(arr))
    return idxs_sorted[:non_nan_count]

def prod(arr):
    """ Return the product of the components of the 1D array `arr` (equivalent
    of `np.sum` function) with convention `prod(arr)` is 1 if `arr` is empty.

    """
    if len(arr) == 0:
        product = 1
    else:
        product = arr[0]
        for i in range(1, len(arr)):
            product = product * arr[i]
    return product

#
def reshape1D_with_nan(arr, len_new):
    """ Return the 1D array `arr` (1, len_current) reshaped under shape
    (1, len_new) either by removing last values or or adding NaN values.

    """
    len_current = len(arr)
    to_add =  len_new - len_current
    if to_add > 0:
        return np.append(arr, np.nan * np.ones(to_add))
    if to_add < 0:
        return np.delete(arr, np.arange(len_new, len_current))
    return arr


def reshape_axis_with_nan(arr, len_new, axis):
    """ Return the array `arr` reshaped turning axis `axis` from lenght
    `len_current` to `len_new` either by removing  values or or adding NaN.""
    NB: for last axis take axis=-1.

    """
    len_current = np.shape(arr)[axis]
    to_add =  len_new - len_current
    if to_add > 0:
        new_dimensions = list(np.shape(arr))
        new_dimensions[axis] = to_add
        new_dimensions = tuple(new_dimensions)
        return np.append(arr, np.nan * np.ones(new_dimensions), axis=axis)
    if to_add < 0:
        return np.delete(arr, np.arange(len_new, len_current), axis=axis)
    return arr

#
def reshape2D_along1_with_0_or_NaN(arr, col_len_new):
    """ Return the 2D array `arr` (row_len, col_len) reshaped under shape
    (row_len, col_len_new) either by removing or adding columns s.t. rows of
    `arr` are extended by NaN if they finish by NaN, 0 otherwise.

    """
    row_len, col_len = np.shape(arr)
    len_to_add =  col_len_new - col_len
    if len_to_add > 0:
        last_col = np.transpose([arr[:, -1]])
        return np.append(arr, last_col*np.zeros((row_len, len_to_add)), axis=1)
    if len_to_add < 0:
        # print(`reshape2D_along1_with_0_or_NaN`)
        return np.delete(arr, np.arange(col_len_new, col_len), axis=1)
    return arr

def reshape2D_along0_w_NaN_along1_w_0_or_NaN(arr, row_len_new, col_len_new):
    """ Return the 2D array  `arr` (row_len, col_len) reshaped under shape
    (row_len_new, col_len_new) by either removing or adding rows of NaN and
    columns of NaN or zeros (s.t. rows of `arr` are extended by NaN if they
    finish by NaN, 0 otherwise).

    """
    row_len = len(arr)
    arr = reshape2D_along1_with_0_or_NaN(arr, col_len_new)
    len_to_add =  row_len_new - row_len
    if len_to_add > 0:
        return np.append(arr, np.nan * np.ones((len_to_add, col_len_new)),
                         axis=0)
    if len_to_add < 0:
        # print(`reshape2D_along0_w_NaN_along1_w_0_or_NaN`)
        return np.delete(arr, np.arange(row_len_new, row_len), axis=0)
    return arr

#
def reshape3D_along2_with_0_or_NaN(arr, col_len_new):
    """ Return the 3D array `arr` (box_len, row_len, col_len) reshaped under
    shape (box_len, row_len, col_len_new) either by removing or adding columns
    s.t. rows  of `arr` are extended by NaN if they finish by NaN, 0 otherwise.

    """
    box_len, row_len, col_len = np.shape(arr)
    len_to_add =  col_len_new - col_len
    if len_to_add > 0:
        last_cols = np.reshape(arr[:, :, -1], (box_len, row_len, 1))
        return np.append(arr, last_cols*np.zeros((row_len, len_to_add)), axis=2)
    if len_to_add < 0:
        # print(`reshape3D_along2_with_0_or_NaN`)
        return np.delete(arr, np.arange(col_len_new, col_len), axis=2)
    return arr

def reshape3D_along0_w_NaN_along1n2_w_0_or_NaN(arr, box_len_new, row_len_new,
                                               col_len_new):
    """ Return the 3D array  `arr` (box_len, row_len, col_len) reshaped under
    shape (box_len_new, row_len_new, col_len_new) either by removing or adding
    boxes of NaN and rows/columns of NaN or zeros (s.t. rows  of `arr` are
    extended by NaN if they finish by NaN, 0 otherwise).

    """
    box_len, row_len = np.shape(arr)[0, 1]
    arr = reshape3D_along2_with_0_or_NaN(arr, col_len_new)
    row_to_add = row_len_new - row_len
    if row_to_add > 0:
        last_rows = np.reshape(arr[:, -1, :], (box_len, 1, col_len_new))
        arr = np.append(arr, last_rows *
                        np.zeros((row_len, row_to_add, col_len_new)), axis=1)
    if row_to_add < 0:
        # print(`reshape3D_along0_w_NaN_along1n2_w_0_or_NaN, row`)
        arr = np.delete(arr, np.arange(row_len_new, row_len), axis=1)
    box_to_add = box_len_new - box_len
    if box_to_add > 0:
        return np.append(arr, np.nan * np.ones((box_to_add, row_len_new,
                                                col_len_new)), axis=0)
    if box_to_add < 0:
        # print(`reshape3D_along0_w_NaN_along1n2_w_0_or_NaN, box`)
        return np.delete(arr, np.arange(box_len_new, box_len), axis=0)
    return arr

#
def pop_to_subpop_format(idxs_pop, c_s):
    """
    Parameters
    ----------
    idxs_pop : ndarray
        1D array (1, cell_count) of indexes of the cells in "population format"
        (i.e. indexes in the whole population).
    c_s : ndarray
        1D array (1, simu_count) indicates how is partitionned the population
        s.t. c_s[i] is the number of cells in the ith subpopulation.

    Returns
    -------
    idxs_subpop : ndarray
        2D array (cell_count, 2) of the indexes of these cells in
        "subpopulation format", i.e. index of the ...:
        > idxs_subpop[i][0] ... subpop to which belongs the ith kept cell.
        > idxs_subpop[i][1] ... cell in this subpopulation.

    """
    # Orderring of indexes.
    idxs_pop = np.sort(idxs_pop)

    # Rigth format for c_s (array of int with NaN replaced by 0).
    c_s = np.nan_to_num(c_s).astype(int)

    # Initialization.
    cell_count = len(idxs_pop)
    idxs_subpop = np.empty((0, 2)) # Indexes in subpop format.
    idx_cell = 0 # Current cell (here 1st cell) index.
    idx_sim = 0 # Index of the subsimu of the current (here the 1st) kept cell.

    # While all indexes have not been converted to subpop format.
    while idx_cell < cell_count:
        # If the index of the current cell (in population format) doesn't exeed
        # the number of cells within the first `idx_sim + 1` subpopulations.
        if idxs_pop[idx_cell] < sum(c_s[:idx_sim + 1]):
            # Then the current cell belongs the `idx_sim`th subpopulation.
            # > We compute its corresponding index in the subsimulation.
            idx_sim_cell = idxs_pop[idx_cell] - sum(c_s[:idx_sim])
            # > Save them and foccus on next cell.
            idxs_subpop = np.append(idxs_subpop, [[idx_sim, idx_sim_cell]],
                                    axis = 0)
            idx_cell += 1
        # Otherwise, it belongs to one of the following subsimulations.
        else:
            idx_sim = idx_sim + 1
    return idxs_subpop.astype(int)

# Histograms
# ----------

def make_hist_from_data(datas, x_axis=None, normalized=True):
    """ Return from the list of values `data`, the coresponding histogram
    corresponding to the axis `x_axis`.

    """
    if x_axis is None: # If no x-axis given.
        x_axis = np.arange(int(np.max(datas) + 1))
    hist = 0. * x_axis
    for data in datas:
        idx = np.where(x_axis == data)[0]
        if len(idx) > 0:
            hist[idx[0]] = hist[idx[0]] + 1
    if normalized:
        if sum(hist) != 0:
            hist = hist / sum(hist)
    return hist

def histogram_w_std(sim_data, x_axis=None):
    """ Compute an 'average histogram' (average and standard deviation) among
    histograms that would correspond to the set of data of several simulations
    given by `sim_data`.

    Parameters
    ----------
    sim_data : ndarray
        2D array (simulation_count, data_per_sim_count).
    x_axis : ndarray, optional
        1D array of the x-axis of the histogram to return.
        The default is None, in which case the axis is made of all possible
        values appearing in the data given in argument.

    Returns
    -------
    x_axis : ndarray
        1D array of histogram's x-axis (depending on the optional argument).
    avg_hist : ndarray
        1D array (len(x_axis), ) s.t. `avg_hist[i]` is the average of the
        number of data with value `x_axis[i]` per simulation (average on
        non-Nan values). NB: is NaN if no suh data.
    std_hist : ndarray
        1D array (len(x_axis), ) s.t. `std_hist[i]` is the standard deviation
        between simulations of the the number of `x_axis[i]` values.
    data_per_sim_count : int

    """
    data_per_sim_count = np.shape(sim_data)[1]
    # Computation of x_axis if not given.
    if x_axis is None: # If no x-axis given.
        x_axis = np.unique(sim_data) # We sort with no repeat all values.
    avg_hist = np.array([])
    std_hist = np.array([])
    # Computation of y_axis and associated error iterating on `x_axis`.
    for x in x_axis:
        where_is_x = sim_data == x
        # NB: we keep track of Nan values to average on non-Nan values.
        where_is_x[~np.isnan(sim_data)] == np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean_per_dataset = 100 * np.nanmean(where_is_x, axis=1)
            avg_hist = np.append(avg_hist, np.nanmean(mean_per_dataset))
            std_hist = np.append(std_hist, np.nanstd(mean_per_dataset))
    return x_axis, avg_hist, std_hist, data_per_sim_count

def histogram_w_stack(sim_data, x_axis=None):
    """

    Parameters
    ----------
    sim_data : ndarray
        3D array (simulation_count, data_per_sim_count).
    x_axis : ndarray, optional
        1D array of the x-axis of the histogram to return.
        The default is None, in which case the axis is made of all the values
        appearing in the data given in argument.

    Returns
    -------
    x_axis : ndarray
        1D array of histogram's x-axis (depending on the optional argument).
    avg_hist : ndarray
        1D array (len(x_axis), ) s.t. `avg_hist[i]` is the average number of
        data with value `x_axis[i]` per simulation (average on non-Nan values).
        NB: is NaN if no suh data.
    std_hist : ndarray
        1D array (len(x_axis), ) s.t. `std_hist[i]` is the standard deviation
        between simulations of the the number of `x_axis[i]` values.

    """
    a, b, c = np.shape(sim_data)
    sim_data_flatten = np.reshape(sim_data, (a, b * c))
    hist = histogram_w_std(sim_data_flatten, x_axis)
    avg_hist = np.empty((c, 0))
    std_hist = np.empty((c, 0))
    is_not_nan_counts = np.sum(~np.isnan(sim_data), axis=(1, 2))
    for x in hist[0]:
        where_is_x = sim_data == x
        x_counts = 100 * np.sum(where_is_x, 1) / is_not_nan_counts[:, None]
        avg_hist = np.append(avg_hist, np.mean(x_counts, axis=0)[:, None], 1)
        std_hist = np.append(std_hist, np.std(x_counts, axis=0)[:, None], 1)
    return hist[0], avg_hist, std_hist

#
def datas_to_hist_s(datas, nta_counts, sen_counts, col_count):
    """ Returns the histograms of the number of cells, B cells, B_SEN cells,
    H cells (if HYBRID_CHOICE), and SEN cells with respect to their `datas`
    (we think of their generations or ancestors) given in argument.

    Parameters
    ----------
    datas : ndarray
        1D array (1, cell_count) of cells' data `data`.
    sen_counts : ndarray
         1D array (1, cell_count) of cells' number of senescent cycle.
    nta_counts : ndarray
    arrest_count : int
        Cells' number of sequence of non-terminal arrest(s) since generation 0
        For non-senescent cell, it is positive if the cell is arrested,
        negative if it is in normal cycle.

    Returns
    -------
    hist : ndarray list
        hist[0]: 1D array (1, col_count), histogram of the total pop.
        hist[1]: 1D array (1, col_count), histogram of B-type cells only.
        hist[2]: 1D array (1, col_count), histogram of B senescent cells.
        hist[-1]: 1D array (1, col_count), histogram of senescent cells.
        if HYBRID_CHOICE:
            hist[3]: 1D array (1, col_count), histogram of H cells.

    """
    datas = datas.astype(int)
    cell_count = len(datas)

    hist = np.zeros(col_count)
    hist_btype = np.zeros(col_count)
    hist_sen_btype = np.zeros(col_count)
    hist_sen = np.zeros(col_count)
    if par.HYBRID_CHOICE:
        hist_htype = np.zeros(col_count)

    # Iteration on cells.
    for i in range(cell_count):
        hist[datas[i]] += 1
        if sen_counts[i] > 0: # Senescent.
            hist_sen[datas[i]] += 1
            if nta_counts[i] > 0: # Type H
                hist_htype[datas[i]] += 1
            elif nta_counts[i] < 0: # Senescent type B.
                hist_btype[datas[i]] += 1
                hist_sen_btype[datas[i]] += 1
        elif nta_counts[i] != 0: # Non-senescent type B.
            hist_btype[datas[i]] += 1

    if par.HYBRID_CHOICE:
        return [hist, hist_btype, hist_sen_btype, hist_sen, hist_htype]
    return [hist, hist_btype, hist_sen_btype, hist_sen]

def change_width_int_hist(x_axis, hist, width):
    x_count = int(len(x_axis) / width)
    x_axis_new = np.linspace(0, x_axis[-1], x_count)
    x_axis_new = x_axis_new.astype('int')
    hist_new = np.array([])
    for i in range(x_count):
        hist_new = np.append(hist_new, np.sum(hist[i * width: (i+1) * width]))
    return x_axis_new, hist_new


# Si vraiment utile faire diff de cummulative evo_c_death !!!
# def death_times_to_hist(times, death_times):
#     """ Returns from an array of cells' time of death `death_times` the
#     histogram of the number of cell dead between each interval of `times`.

#     Parameters
#     ----------
#     times : ndarray
#         1D array (1, time_count) containg times.
#     deaht_times : ndarray
#         1D array (1, dead_cell_count) containing death times.

#     Returns
#     -------
#     hist : ndarray
#         1D array (1, time_count) of the number of dead cell through time.

#     """
#     time_count = len(times)
#     cell_count = len(death_times)
#     hist = np.zeros(time_count)
#     for i in range(1,time_count):
#         for cell in range(cell_count):
#             to_remove_indexes = np.array([])
#             if death_times[cell] < times[i]:
#                 hist[i] += 1
#                 cell_count -= 1
#                 to_remove_indexes = np.append(to_remove_indexes, cell)
#         death_times = np.delete(death_times, to_remove_indexes)
#         cell_count = max(0,cell_count)
#     hist[-1] = len(death_times)
#     return hist


# def death_times_to_hist_by_step(times, death_times, step):
#     """ Returns from an array of cells' time of death <death_times> the
#     histogram of the number of cell dead between interval of <times> of length
#     <step>.

#     Parameters
#     ----------
#     times : ndarray
#         1D array of shape (1,time_count) containg times.
#     deaht_times : ndarray
#         1D array of shape (1,dead_cell_count) containing death times.
#     step : int

#     Returns
#     -------
#     hist : ndarray
#         1D array of shape(1,time_count) containing the number of dead cell on
#         each of the intervals of <times> of lenthg <step>.
#     """
#     time_count = len(times)
#     hist = death_times_to_hist(times, death_times)
#     interval_count = int(time_count / step)
#     for i in range(interval_count - 1):
#         hist[i:i + step] = sum(hist[i:i + step]) # /step
#     hist[interval_count:] = sum(hist[interval_count:]) # /len(hist[interval_count:])
#     return hist

#
def death_times_to_cumulative_death(times, death_times):
    """ Computes from times of death the number of dead cells at each time of
    a given time array <times>.

    Parameters
    ----------
    times : ndarray
        1D array (1, time_count) of times.
    death_times : ndarray
        1D array (1, cell_count) containing cells' time of death.

    Returns
    -------
    evo_c_death : ndarray
        1D array (1, time_count) of time evolution of the number of dead cells.

    Warning
    -------
    The cells whose time of death is superior to the last time of <times> are
    not taken into account.

    """
    evo_c_death = np.zeros(len(times))
    for death_time in death_times:
        evo_c_death += (times - death_time >= 0)
    return evo_c_death

def death_times_to_cumulative_death_by_anc_n_gen(times, death_times, ancs,
                                                 anc_count, gens, gen_count,
                                                 nta_counts):
    """ Computes from times of death the number of dead cells at each time of
    a given time array <times>.

    Parameters
    ----------
    times : ndarray
        1D array (1, time_count) of times.
    death_times : ndarray
        1D array (1, cell_count) containing cells' time of death.

    Returns
    -------
    evo_c_death : ndarray
        1D array (1, time_count) of time evolution of the number of dead cells.

    Warning
    -------
    The cells whose time of death is superior to the last time of <times> are
    not taken into account.

    """
    ancs = ancs.astype(int)
    gens = gens.astype(int)
    time_count = len(times)
    evo_c_death = [np.zeros((time_count, anc_count, gen_count)), # All, B, H.
                   np.zeros((time_count, anc_count, gen_count)),
                   np.zeros((time_count, anc_count, gen_count))]
    cell_count = len(death_times)
    for cell in range(cell_count):
        times_to_update = np.zeros(time_count)
        times_to_update[times - death_times[cell] >= 0] = 1
        evo_c_death[0][:, ancs[cell], gens[cell]] += times_to_update
        if nta_counts[cell] > 0:
            evo_c_death[2][:, ancs[cell], gens[cell]] += times_to_update
        elif nta_counts[cell] < 0:
            evo_c_death[1][:, ancs[cell], gens[cell]] += times_to_update
    return evo_c_death
# def death_times_to_cumulative_death_by_anc_n_gen_new(history_dead):
#     """ 
#     """
#     [anc, gen, t_temp / (60*24), ar_count]
#     ancs, 
#     ancs, gens, death_times, = ancs.astype(int)
#     gens = gens.astype(int)
#     time_count = len(times)
#     evo_c_death = [np.zeros((time_count, anc_count, gen_count)), # All, B, H.
#                    np.zeros((time_count, anc_count, gen_count)),
#                    np.zeros((time_count, anc_count, gen_count))]
#     cell_count = len(death_times)
#     for cell in range(cell_count):
#         times_to_update = np.zeros(time_count)
#         times_to_update[times - death_times[cell] >= 0] = 1
#         evo_c_death[0][:, ancs[cell], gens[cell]] += times_to_update
#         if np.isnan(types[cell]):
#             evo_c_death[2][:, ancs[cell], gens[cell]] += times_to_update
#         elif types[cell] == 1:
#             evo_c_death[1][:, ancs[cell], gens[cell]] += times_to_update
#     return evo_c_death


# ==========================================================
def ancestor_distribution_to_cumulative_distributions(evo_anc_dist):
    """
    Parameters
    ----------
    evo_anc_dist : ndarray
        2D-array of shape (time_count, ancestor_count) of the time evolution of
        the distribution of ancestors in the population.

    Returns
    -------
    cumul_dists : ndarray
        2D-array of shape (ancestor_count, time_count) containing the times
        evolutions of the proportion (cumulated on ancestors) of every ancestor.

    """
    anc_count = np.shape(evo_anc_dist)[1]
    cumul_dists = np.copy(evo_anc_dist)
    cumul_dists = np.transpose(cumul_dists)
    for j in range(2, anc_count+1):
        for i in range(1,j):
            cumul_dists[-i] += cumul_dists[-j]
    return cumul_dists
# ==========================================================

#
def cell_idxs_sorted_by_shortest_telomere(lengths):
    """ Return the indexes of cells sorted by increasing shortest telomere.

    """
    shortest_lengths = np.min(lengths, axis=(1, 2))
    return np.argsort(shortest_lengths)



# Model related functions (generate distributions, test certain laws)
# -------------------------------------------------------------------

#
def inv_cdf(u, x_values, probas):
    """ Return cdf^{-1}_X(u): the inverse of the cumulative distribution
    function, of a discrete random variable X whose law is described by
    `x_values` and `probas`, evaluated at u.

    Parameters
    ----------
    u : float
        in [0, 1], argument of F^{-1}.
    x_values : nd array
        ORDERED 1D array (1, ) of the values taken by the (discrete) random
        variable of interest.
    probas : ndarray
        Probabilities associated to the values of X `x_values`, s.t.
        `proba[i] == P(X = N[i])`.
    """
    inv_idx = 0
    cdf = probas[0]
    while cdf < u and inv_idx < len(probas) - 1:
        inv_idx = inv_idx + 1
        cdf = cdf + probas[inv_idx]
    return x_values[inv_idx]

if __name__ == "__main__":
    # Visualization of the inverse of the cumulative distrib. function of the
    # experimental initial telomere length (draw line step in figure options).
    plt.clf()
    U = np.linspace(0, 1, 1000)
    # Original cdt.
    support, probas = parf.transform_l_init(par.L_INIT_EXP)
    x_invs = np.round(np.array([inv_cdf(u, support, probas) for u in U]))
    plt.plot(U, x_invs, label='Original')
    # Cdt of the transformed distribution.
    support, probas = parf.transform_l_init(par.L_INIT_EXP, *par.PAR_L_INIT)
    x_invs = np.round(np.array([inv_cdf(u, support, probas) for u in U]))
    plt.plot(U, x_invs, label='Transformed')
    # Cdt of the distribution transformed with old (wrong) method.
    support, probas = parf.transform_l_init_old(par.L_INIT_EXP,*par.PAR_L_INIT)
    x_invs = np.round(np.array([inv_cdf(u, support, probas) for u in U]))
    plt.plot(U, x_invs, label='Transformed old')
    plt.title('Inverse of the cdf of the transformed inital distribution of '
              'telomere lengths')
    plt.legend()

#
def draw_lengths(chromosome_count, par_l_init=par.PAR_L_INIT):
    """ Returns a distribution of `2 * chromosome_count` initial telomeres
     lengths drawn accordingly to `par.L_INIT_CHOICE`:
        > 'const' for constant initial length (=L_INF).
        > 'gaussian' for gaussian distributrion (avg=L_INF, sd=L_SIGMA).
        > 'exp' for experimental one (from etat_asymp_val_juillet).

    Parameters
    ----------
    chromosome_count : int
        Number of chromosomes to create.

    Returns
    -------
    lengths : ndarray
        2D array (2, chromosome_count) of the lengths of the two telomeric
        extremities of `chromosome_count` chromosomes.

    """
    cell_shape = (2, chromosome_count)
    if par.L_INIT_CHOICE == 'const':
        return par.L_INF * np.ones(cell_shape)

    if par.L_INIT_CHOICE == 'gaussian':
        return rd.normal(par.L_INF, par.L_SIGMA, cell_shape)

    if par.L_INIT_CHOICE == 'exp':
        u_rands = rd.uniform(0, 1, 2 * chromosome_count)
        lengths = np.array([])
        for u in u_rands:
            support, proba = parf.transform_l_init(par.L_INIT_EXP, *par_l_init)
            lengths = np.append(lengths, np.round(inv_cdf(u, support, proba)))
         # Return the lengths obtained with translated initial distribution.
        return np.reshape(lengths, cell_shape)
    print("Error: `L_INIT_CHOICE` from `parameters.py` has unexpected value")
    return None

#
def draw_cell_lengths(cell_count, par_l_init=par.PAR_L_INIT):
    """ Returns initial telomere length distributions of `cell_count` cells
     drawn accordingly to `par.L_INIT_CHOICE`.

    Parameters
    ----------
    cell_count : int
        Number of cell  disribution to create.

    Returns
    -------
    lengths : ndarray
        3D array (cell_count, 2, chromosome_count) of the lengths of the two
        telomeric extremities of the 16 chromosomes of the `cell_count` cells.

    """
    if par.L_INIT_CHOICE == 'two_normal-long':
        if cell_count != 2:
            raise Exception(f"Creating {cell_count} cells is not compatible "
                            f"with 'par.L_INIT_CHOICE' that loads 2 cells.")
        l_long = np.loadtxt(par.L_LONG_PATH)
        l_medium = np.loadtxt(par.L_MEDIUM_PATH)
        lengths = np.array([l_long, l_medium])
        return lengths

    if par.L_INIT_CHOICE == 'two_normal-short':
        if cell_count != 2:
            raise Exception(f"Creating {cell_count} cells is not compatible "
                            f"with `par.L_INIT_CHOICE` that loads 2 cells.")
        l_short = np.loadtxt(par.L_SHORT_PATH)
        l_medium = np.loadtxt(par.L_MEDIUM_PATH)
        lengths = np.array([l_short, l_medium])
        return lengths

    lengths = np.array([draw_lengths(16, par_l_init)
                        for i in range(cell_count)])
    return lengths

def cdf_to_distribution(cdfs, distrib_x=None):
    if isinstance(distrib_x, type(None)):
        distrib_x = np.unique(cdfs)
    x_count = len(distrib_x)
    distrib_y = np.zeros(x_count)
    for i in range(x_count-1):
        distrib_y[i] = sum(np.logical_and(distrib_x[i] <= cdfs,
                                          cdfs < distrib_x[i+1]))
    return distrib_x, distrib_y / x_count

def distibutions_cell_shortest_n_average(cell_count, simu_count,
                                         is_plotted=False):
    file_shortest = "data/data_init/distribution_shortest_" + \
                    f"c{cell_count}_s{simu_count}.csv"
    file_average = "data/data_init/distribution_average_" + \
                   f"c{cell_count}_s{simu_count}.csv"
    x_axis = parf.transform_l_init(par.L_INIT_EXP, *par.PAR_L_INIT)[0]
    if not os.path.exists(file_shortest):
        tmp_distrib_shortest = np.empty((0, len(x_axis)))
        tmp_distrib_average = np.empty((0, len(x_axis)))
        for s in range(simu_count):
            print(s)
            lengths = draw_cell_lengths(cell_count)
            tmp_shortest = np.min(lengths, axis=(1, 2))
            tmp_shortest = cdf_to_distribution(tmp_shortest, x_axis)[1]
            tmp_distrib_shortest = np.append(tmp_distrib_shortest,
                                             [tmp_shortest], 0)
            tmp_average = np.mean(lengths, axis=(1, 2))
            tmp_average = cdf_to_distribution(tmp_average, x_axis)[1]
            tmp_distrib_average = np.append(tmp_distrib_average, [tmp_average],
                                            0)
        distrib_shortest = np.mean(tmp_distrib_shortest, 0)
        distrib_average = np.mean(tmp_distrib_average, 0)
        std_shortest = np.std(tmp_distrib_shortest, 0)
        std_average = np.std(tmp_distrib_average, 0)
        pd.DataFrame(distrib_shortest).to_csv(file_shortest, header=None,
                                              index=None)
        pd.DataFrame(std_shortest).to_csv(file_shortest.replace('.csv',
                                                                '_std.csv'),
                                          header=None, index=None)
        pd.DataFrame(distrib_average).to_csv(file_average, header=None,
                                             index=None)
        pd.DataFrame(std_average).to_csv(file_average.replace('.csv',
                                         '_std.csv'), header=None, index=None)
    else:
        distrib_shortest = np.loadtxt(file_shortest)
        distrib_average = np.loadtxt(file_average)
    if is_plotted:
        plt.plot(x_axis, distrib_shortest, distrib_average)
    return # distrib_shortest, distrib_average

#
def draw_lengths_one_long(cell_count, par_l_init=par.PAR_L_INIT):
    """ Generates telomere lengths of `cell_count` cells and return the
    telomere distribution of the cell with the biggest shortest telomere.

    Warning
    ------
    Argument `cell_count` is an integer >= 2.

    """
    lengths_init = draw_cell_lengths(cell_count, par_l_init)
    lengths_min = np.min(lengths_init, axis=(1, 2))
    longuest_cell_idx = np.argmax(lengths_min)
    return lengths_init[longuest_cell_idx]

#
def draw_lengths_one_short(cell_count, par_l_init=par.PAR_L_INIT):
    """ Generates telomere lengths of `cell_count` cells and return the
    telomere distribution of the cell with the smallest shortest telomere.

    Warning
    ------
    Argument `cell_count` is an integer >= 2.

    """
    lengths_init = draw_cell_lengths(cell_count, par_l_init)
    print('Generation of the population finished')
    lengths_min = np.min(lengths_init, axis=(1, 2))
    shortest_cell_idx = np.argmin(lengths_min)
    return lengths_init[shortest_cell_idx]

#
def find_nearest(arr, value):
    """ Return the index of the value of the array `arr` that is the (first)
    closest to `value`.

    """
    return (np.abs(arr - value)).argmin()

#
def draw_lengths_one_medium(cell_count, par_l_init=par.PAR_L_INIT):
    """ Generates telomere lengths of `cell_count` cells and return the
    telomere distribution of the cell with shortest telomere the closest to the
    average shortest telemore length in the generated initial population.

    Warning
    ------
    Argument `cell_count` is an integer >= 2.

    """
    lengths_init = draw_cell_lengths(cell_count, par_l_init)
    lmins = np.min(lengths_init, axis=(1, 2))
    lmin_avg = np.mean(lmins)
    average_cell_idx = find_nearest(lmins, lmin_avg)
    return lengths_init[average_cell_idx]


def draw_lengths_one_veryshort(cell_count, par_l_init=par.PAR_L_INIT):
    """ Generates telomere lengths of `cell_count` cells and return the
    telomere distribution of the cell composed of the (16) chromosomes of the
    whole population with the shortest telomeres.

    Warning
    -------
    Argument `cell_count` is an integer >= 2.

    """
    chromosome_count = 16 * cell_count
    lengths_init = draw_lengths(chromosome_count, par_l_init)
    lengths_new = np.array([[], []])
    for chromosome in range(16):
        lmin_idx = np.mod(np.argmin(lengths_init), chromosome_count)
        chromosome_new = np.transpose([lengths_init[:, lmin_idx]])
        lengths_new = np.append(lengths_new, chromosome_new, axis=1)
        lengths_init = np.delete(lengths_init, lmin_idx, 1)
        chromosome_count -= 1
    return rd.permutation(lengths_new.T).T


def lmin_cdf_inv(u, lmin_max, alpha):
    """ Return the inverse of the cumulative distribution function of the
    probability for a  telomere lenght to triggering senescence defined by
    B(x)exp(-int_0^x B**2) (details in parameters.py) evaluated at u.

    """
    return lmin_max - np.sqrt(- 2 * np.log(u) / alpha)


#
def sigmoid(gen):
    """ Return the sigmoid function (4.2.) [Martin Thesis] evaluated at `gen`,
    i.e. the probability to have a long cycle at generation `gen` when there
    where only normal cycles so far.

    """
    if gen > 0:
        return 1 / (1 + np.exp(- (gen - par.A_SIGMOID) / par.B_SIGMOID))
    return 0


#
def is_nta_trig(gen, length_min, lengths, parameters=par.PAR_NTA):
    """ Return True if a non-terminal arrest is triggered for a type A cell
    of telomere distribution `lengths`, minimal telomere length `length_min`
    and generation `gen`, False otherwise, following the rule given by
    `par.TRIG_TELO_CHOICE` and `TRIG_ARREST_CHOICE`.

    Optionnal: no matter the parameters fixed in `parameters`, one can specify
    new parameters `[a, b]` in argument.

    """
    if par.TRIG_ARREST_CHOICE == 'sigmoid':
        proba_trig = min(1, sigmoid(gen)) # Proba to trigger a non-term arrest.
        return bool(rd.binomial(1, proba_trig))

    if par.TRIG_ARREST_CHOICE == 'lmin_const':
        ltrig = par.L_MIN_AR # Minimal len triggering a non-terminal arrest.
        return length_min < ltrig

    # Computation of the lengths of the telomere to test.
    if par.TRIG_TELO_CHOICE == 'shortest': # Only the shortest telo is tested.
        lengths_min = [length_min]
    elif par.TRIG_TELO_CHOICE == 'all': # All telo shorter than L_MIN_MAX are.
        lengths_min = lengths.flatten()
        lengths_min = lengths_min[lengths_min <= par.L_MIN_MAX]

    if par.TRIG_ARREST_CHOICE == 'exp':
        for lmin in lengths_min: # For all lengths to test.
            # Computation of the probability to trigger an arrest.
            proba = min(1, parameters[1] * math.exp(- parameters[0] * lmin))
            # Testing (enters an arrest if at least one telo triggers).
            if rd.binomial(1, proba):
                return True

    if par.TRIG_ARREST_CHOICE == 'lmin_gaussian':
        for lmin in lengths_min:
            ltrig = max(0, rd.normal(par.L_MIN_AR_MU,
                                                   par.L_MIN_AR_SIGMA))
            if lmin < ltrig:
                return True

    # If still nothing returned, no telomere has triggerd an arrest.
    return False


#
def is_sen_atype_trig(length_min, lengths, parameters=par.PAR_SEN[0]):
    """ Return True if senescence of a type A cell of telomere length
    distribution `lengths` and minimal telomere length `length_min` is
    triggered, False otherwise, according to `par.TRIG_TELO_CHOICE` and
    `TRIG_SEN_CHOICE`.

    """
    if par.TRIG_SEN_CHOICE == 'lmin_const':
        lmin_trig = par.L_MIN_SEN_ATYPE # Minimal len triggering senescence.
        return length_min < lmin_trig

    # Computation of the lengths of the telomere to test.
    if par.TRIG_TELO_CHOICE == 'shortest': # Only the shortest telo is tested.
        lengths_min = [length_min]
    elif par.TRIG_TELO_CHOICE == 'all': # All telo shorter than L_MIN_MAX are.
        lengths_min = lengths.flatten()
        lengths_min = lengths_min[lengths_min <= par.L_MIN_MAX]

    if par.TRIG_SEN_CHOICE == 'exp':
        for lmin in lengths_min: # For all lengths to test.
            # Computation of the probability to trigger senescence.
            proba = min(1, par.B_EXP_SEN * math.exp(- par.A_EXP_SEN * lmin))
            # Testing (enters senescence as soon as one telo triggers).
            if rd.binomial(1, proba):
                return True

    if par.TRIG_SEN_CHOICE == 'exp_new':
        for lmin in lengths_min: # For all lengths to test.
            # Computation of the probability to trigger senescence.
            if lmin <= parameters[2]:
                proba = 1
            else:
                proba = min(1, parameters[1] * math.exp(-parameters[0] * lmin))
            # Testing (enters senescence as soon as one telo triggers).
            if rd.binomial(1, proba):
                return True

    if par.TRIG_SEN_CHOICE == 'lmin_gaussian':
        for lmin in lengths_min:
            ltrig = max(0, rd.normal(par.L_MIN_SEN_MU,
                                                   par.L_MIN_SEN_SIGMA))
            if lmin < ltrig:
                return True

    if par.TRIG_SEN_CHOICE == 'lmin_rand':
        for lmin in lengths_min:
            u = rd.uniform(0, 1)
            ltrig = lmin_cdf_inv(u, par.L_MIN_MAX_ATYPE, par.L_ALPHA_ATYPE)
            if lmin < ltrig:
                return True

    # If still nothing returned, no telomere has triggerd an arrest.
    return False

#
def is_sen_btype_trig(length_min, lengths, parameters=par.PAR_SEN[1]):
    """ Return True if senescence of a type B cell of telomere length
    distribution `lengths` and minimal telomere length `length_min` is
    triggered, False otherwise, according to `par.TRIG_TELO_CHOICE` and
    `TRIG_SEN_CHOICE`.

    """
    if par.TRIG_SEN_CHOICE == 'lmin_const':
        lmin_trig = par.L_MIN_SEN_BTYPE # Minimal len triggering senescence.
        return length_min < lmin_trig

    # Computation of the lengths of the telomere to test.
    if par.TRIG_TELO_CHOICE == 'shortest': # Only the shortest telo is tested.
        lengths_min = [length_min]
    elif par.TRIG_TELO_CHOICE == 'all': # All telo shorter than L_MIN_MAX are.
        lengths_min = lengths.flatten()
        lengths_min = lengths_min[lengths_min <= par.L_MIN_MAX]

    if par.TRIG_SEN_CHOICE == 'exp':
        for lmin in lengths_min: # For all lengths to test.
            # Computation of the probability to trigger senescence.
            proba = min(1, par.B_EXP_SEN * math.exp(- par.A_EXP_SEN * lmin))
            # Testing (enters senescence as soon as one telo triggers).
            if rd.binomial(1, proba):
                return True

    if par.TRIG_SEN_CHOICE == 'exp_new':
        for lmin in lengths_min: # For all lengths to test.
            # Computation of the probability to trigger senescence.
            if lmin <= parameters[2]:
                proba = 1
            else:
                proba = min(1, parameters[1] * math.exp(-parameters[0] * lmin))
            # Testing (enters senescence as soon as one telo triggers).
            if rd.binomial(1, proba):
                return True

    if par.TRIG_SEN_CHOICE == 'lmin_gaussian':
        for lmin in lengths_min:
            lmin_trig = max(0, rd.normal(par.L_MIN_SEN_MU,
                                                       par.L_MIN_SEN_SIGMA))
            if lmin < lmin_trig:
                return True

    if par.TRIG_SEN_CHOICE == 'lmin_rand':
        for lmin in lengths_min:
            u = rd.uniform(0, 1)
            lmin_trig = lmin_cdf_inv(u, par.L_MIN_MAX_BTYPE, par.L_ALPHA_BTYPE)
            if lmin < lmin_trig:
                return True

    # If still nothing returned, no telomere has triggered non-terminal arrest.
    return False

# if __name__ == "__main__":
#     def lmin_A():
#         """ Return a minimal telomere length triggering senescence of A-type
#         cells accordingly to `par.TRIG_SEN_CHOICE`.

#         """
#         if par.TRIG_SEN_CHOICE == 'const':
#             return par.L_MIN_SEN_ATYPE

#         u = rd.uniform(0, 1)
#         l = lmin_cdf_inv(u, par.L_MIN_MAX_ATYPE, par.L_ALPHA_ATYPE)
#         return max(l, 0)

#     # Visualizations.
#     plt.figure('lmin_A')
#     x = np.linspace(0, 1, 100)
#     plt.plot(x,lmin_cdf_inv(x, par.L_MIN_MAX_ATYPE, par.L_ALPHA_ATYPE))
#     plt.plot(x,np.array([lmin_A() for i in x]))

#
def draw_overhang():
    """Return a overhang value drawn accordingly to the law `OVERHANG_CHOICE`.

    """
    if par.OVERHANG_CHOICE == 'const':
        return par.OVERHANG
    if par.OVERHANG_CHOICE == 'uniform':
        return np.random.randint(par.OVERHANG_LOW, par.OVERHANG_UP)
    print("Error: `OVERHANG_CHOICE` from `parameters.py` has unexpected value")
    return None


#
def draw_cycles_A(cell_count):
    """ Return a distribution of cycle duration times of a population of
    `cell_count` non-senescent A-type cells accordingly to `par.CYCLES_CHOICE`.

    """
    if par.CYCLES_CHOICE == 'const':
        return par.CYCLES_A_CONST * np.ones(cell_count)
    if par.CYCLES_CHOICE == 'exp':
        return rd.choice(par.CDTS_OLD['nor-telo+'], cell_count)
    if par.CYCLES_CHOICE == 'exp_new':
        return rd.choice(par.CDTS['norA'], cell_count)
    print("Error: `CYCLES_CHOICE` from `parameters.py` has unexpected value")
    return None

#
# def draw_cycle_B_avg():
#     """ Return a cycle duration time of non-senescent B-type cell after its
#     1st seq of long cycles accordingly to `par.CYCLES_CHOICE`.

#     """
#     if par.CYCLES_CHOICE == 'const':
#         return par.CYCLES_B_AVG_CONST
#     if par.CYCLES_CHOICE == 'exp':
#         return rd.choice(par.CDTS_OLD['btype'])
#     print("Error: `CYCLES_CHOICE` from `parameters.py` has unexpected value")
#     return None

#
def draw_cycle_B_lc():
    """ Return a cycle duration time for a B-type cell in its 1st seq of long
    cycles accordingly to `par.CYCLES_CHOICE`.

    """
    if par.CYCLES_CHOICE == 'const':
        return par.CYCLES_B_LONG_CONST
    if par.CYCLES_CHOICE == 'exp':
        return rd.choice(par.CDTS_OLD['nta1'])
    if par.CYCLES_CHOICE == 'exp_new':
        return rd.choice(par.CDTS['nta'])
    print("Error: `CYCLES_CHOICE` from `parameters.py` has unexpected value")
    return None

#
def draw_cycle_B_lc_af_1ST():
    """ Return a cycle duration time for a B-type cell in its 2nd, or 3rd, ...
    seq of long cycles accordingly to `par.CYCLES_CHOICE`.

    """
    if par.CYCLES_CHOICE == 'const':
        return par.CYCLES_B_LONG_CONST
    if par.CYCLES_CHOICE == 'exp':
        return rd.choice(par.CDTS_OLD['nta2+'])
    if par.CYCLES_CHOICE == 'exp_new':
        return rd.choice(par.CDTS['nta'])
    print("Error: `CYCLES_CHOICE` from `parameters.py` has unexpected value")
    return None

#
def draw_cycle_B_nc():
    """ Return a cycle duration time for a B-type cell in normal cycle
    accordingly to `par.CYCLES_CHOICE`.

    """
    if par.CYCLES_CHOICE == 'const':
        return par.CYCLES_B_CONST
    if par.CYCLES_CHOICE == 'exp':
        return rd.choice(par.CDTS_OLD['norB'])
    if par.CYCLES_CHOICE == 'exp_new':
        return rd.choice(par.CDTS['norB'])
    print("Error: `CYCLES_CHOICE` from `parameters.py` has unexpected value")
    return None

#
def draw_cycle_sen():
    """ Return a cycle duration time for a senescent cell (last cycle excluded)
    accordingly to `par.CYCLES_CHOICE`.

    """
    if par.CYCLES_CHOICE == 'const':
        return par.CYCLES_SEN_CONST
    if par.CYCLES_CHOICE == 'exp':
        return rd.choice(par.CDTS_OLD['sen'])
    if par.CYCLES_CHOICE == 'exp_new':
        return rd.choice(par.CDTS['sen'])
    print("Error: `CYCLES_CHOICE` from `parameters.py` has unexpected value")
    return None

#
def draw_cycle_sen_last():
    """ Return a cycle duration time for the last cycle of senescence
    accordingly to `par.CYCLES_CHOICE`.

    """
    if par.CYCLES_CHOICE == 'const':
        return par.CYCLES_SEN_LAST_CONST
    if par.CYCLES_CHOICE == 'exp':
        # Because of lack of data same as other cycles of the senescence.
        return rd.choice(par.CDTS_OLD['sen'])
    if par.CYCLES_CHOICE == 'exp_new':
        return rd.choice(par.CDTS['sen_last'])
    print("Error: `CYCLES_CHOICE` from `parameters.py` has unexpected value")
    return None

#
def draw_cycle(arrest_count, is_senescent):
    """ Return a cycle duration time for a new born cell in the state entered
    as argument.

    Parameters
    ----------
    arrest_count : int
        Number of sequence of non-terminal arrests the lineage of the cell went
        through so far. For non-senescent cell, it is positive if the cell is
        arrested, negative if it is in normal cycle.
    is_senescent : bool
        Is False if the cell is not senescent, True if senescent.

    """
    if is_senescent: # The cell is senescent.
        return draw_cycle_sen()
    if arrest_count == 0: # Non-senescent type A.
        return draw_cycles_A(1)[0]
    if arrest_count == 1: # Non-senescent type B in a 1st sequence of arrests.
        return draw_cycle_B_lc()
    if arrest_count < 0: # Non-senescent type B in normal cycle.
        return draw_cycle_B_nc()
    # Non-senescent type B in a 2nd, 3rd... sequence of arrests.
    return draw_cycle_B_lc_af_1ST()


#
def draw_delays(cycles):
    """ Return a delay for cells (with cycle duration times `cycles`) to start
    with accordingly to `DELAY_CHOICE`:
        > 'null' for no delay.
        > 'uniform' for uniform delay between 0 and their division time.

    Parameters
    ----------
    cycles : ndarray
        1D array (1, cell_count) of cells' cycle duration time.

    Returns
    -------
    delays : ndarray
        1D array (1, cell_count) of associated cells' delay.

    """
    cell_count = len(cycles)
    if par.DELAY_CHOICE == 'null':
        return np.zeros(cell_count)
    if par.DELAY_CHOICE == 'uniform':
        return rd.uniform(0, cycles, cell_count)
    print("Error: `DELAY_CHOICE` from `parameters.py` has unexpected value")
    return None

#
def desynchronize(cycles):
    """ Delays cell's cycle duration time `cycles` accordingly `delay_choice`.

    """
    if par.DELAY_CHOICE == 'null':
        return cycles
    if par.DELAY_CHOICE == 'uniform':
        delays = rd.uniform(0, cycles, len(cycles))
        return cycles - delays
    print("Error: `DELAY_CHOICE` from `parameters.py` has unexpected value")
    return None

#
def draw_len_nta():
    """ Return a number of long cycles for non-terminal arrests of type B.

    """
    return rd.geometric(par.P_GEO_NTA)

#
def draw_len_sen():
    """ Return a number of long cycles for the senescence.

    """
    return rd.geometric(par.P_GEO_SEN)

def is_repaired():
    """  Returns if a type-B cell in non-terminal arrest exits the sequence of
    long cycles (True) or continue the sequence (False).

    """
    return rd.binomial(1, par.P_GEO_NTA)

#
def is_dead(sen_count=0, max_sen_count=par.MAX_SEN_CYCLE_COUNT):
    """ Return if a senescent cell dies (True) or continue to divide (False).

    """
    return (rd.binomial(1, par.P_GEO_SEN) or
            sen_count > max_sen_count)

def is_accidentally_dead(p_death_acc=par.P_ACCIDENTAL_DEATH):
    """ Return if a cell accidentally dies (True) or continue to divide
    (False).

    """
    return rd.binomial(1, p_death_acc)

