#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 13:42:05 2024

@author: anais

    Copyright (C) 2024  Anaïs Rat

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""

import numpy as np
import warnings


# Computation on arrays
# ---------------------


# Statistical quantities.


def stat(arr_stat, p_up, p_down, axis=0):
    """Calculate statistics (mean, percentiles) of an array.

    Parameters
    ----------
    arr_stat : ndarray
        Input array containing numerical data.
    p_up : float
        Percentile value for upper bound calculation.
    p_down : float
        Percentile value for lower bound calculation.
    axis : int, optional
        Axis along which the statistics are computed. The default is 0.

    Returns
    -------
    stats : dict
        Dictionary containing the calculated statistics.
        - 'mean': Mean of the array along the specified axis.
        - 'perup': Upper percentile value along the specified axis.
        - 'perdown': Lower percentile value along the specified axis.
    """
    stats = {
        "mean": np.mean(arr_stat, axis=axis),
        "perup": np.percentile(arr_stat, p_up, axis=axis),
        "perdown": np.percentile(arr_stat, p_down, axis=axis),
    }
    return stats


def nanstat(arr_stat, p_up, p_down, axis=0):
    """Same function as `stat`, except that `nanstat` handles NaN values."""
    with warnings.catch_warnings():  # Prevent printing warnings comming from
        # applying np.nanmean, etc., to arrays full of NaN values.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        stats = {
            "mean": np.nanmean(arr_stat, axis=axis),
            "perup": np.nanpercentile(arr_stat, p_up, axis=axis),
            "perdown": np.nanpercentile(arr_stat, p_down, axis=axis),
        }
    return stats


def stat_all(arr_stat, p_up, p_down, axis=0):
    """Similar to function `stat` with a slightly different output format and
    the additional statistical quantity: extremum values.

     Returns
        -------
    stats : dict
        Dictionary containing the calculated statistics.
        - 'mean': Mean of the array along the specified axis.
        - 'per': List containing the lower and upper percentiles along the
                 specified axis.
        - 'ext': List containing the minimum and maximum values along...

    """
    stats = {
        "mean": np.mean(arr_stat, axis=axis),
        "per": [
            np.percentile(arr_stat, p_down, axis=axis),
            np.percentile(arr_stat, p_up, axis=axis),
        ],
        "ext": [np.min(arr_stat, axis=axis), np.max(arr_stat, axis=axis)],
    }
    return stats


# Sum, product...


def nansum(arr, axis=None, keepdims=False):
    """Similar to `np.nansum`, except that if there are only NaN values along
    a summation axis, the sum along that axis is NaN instead of 0.

    """
    arr_sum = np.nansum(arr, axis=axis, keepdims=keepdims)
    arr_isnan = np.isnan(arr).all(axis=axis)
    if arr_isnan.any():  # If at least one axis with NaN only.
        # The sum along NaN? axis is Nan instead of 0.
        if isinstance(arr_sum, np.ndarray):  # If arr_sum is an array.
            arr_sum[arr_isnan] = np.nan  # Array returned.
        else:  # Otherwise arr_sum = 0, is int or float, not array!
            return np.nan
    return arr_sum


# Modification of arrays
# ----------------------


# Sorting.


def nanargsort1D(arr):
    """Similar to `np.argsort` for 1D array, except that NaN are not orderred."""
    # Compute indexes corresponding to the usual sorting (NaN at the end).
    idxs_sorted = np.argsort(arr)
    # And return only indexes corresponding to non NaN values.
    non_nan_count = sum(~np.isnan(arr))
    return idxs_sorted[:non_nan_count]


# Reshaping.


def reshape_with_nan(arr, len_new, axis=-1):
    """Return the array `arr` reshaped along the axis `axis`, turning it from
    length `len_current` to `len_new` either by removing values or adding NaN.
    NB: for the last axis, take `axis=-1` (default).

    """
    len_current = np.shape(arr)[axis]
    to_add = len_new - len_current
    if to_add > 0:
        new_dimensions = list(np.shape(arr))
        new_dimensions[axis] = to_add
        new_dimensions = tuple(np.array(new_dimensions).astype("int"))
        return np.append(arr, np.nan * np.ones(new_dimensions), axis=axis)
    if to_add < 0:
        return np.delete(arr, np.arange(len_new, len_current), axis=axis)
    return arr


def reshape_list_with_nan(lst, len_new):
    """Return the list `lst` reshaped from length `len_current` to `len_new`
    either by removing values or adding nan.

    """
    len_current = len(lst)
    if len_new > len_current:
        lst.extend([np.nan] * (len_new - len_current))
    elif len_new < len_current:
        lst = lst[:len_new]
    return lst


def reshape2D_along1_with_0_or_NaN(arr, col_len_new):
    """Return the 2D array `arr` (row_len, col_len) reshaped to shape
    (row_len, col_len_new), either by removing or adding columns so that rows
    of `arr` are extended by NaN if they end with NaN, and by 0 otherwise.

    """
    row_len, col_len = np.shape(arr)
    len_to_add = col_len_new - col_len
    if len_to_add > 0:
        last_col = np.transpose([arr[:, -1]])
        return np.append(arr, last_col * np.zeros((row_len, len_to_add)), axis=1)
    if len_to_add < 0:
        return np.delete(arr, np.arange(col_len_new, col_len), axis=1)
    return arr


def reshape2D_along0_w_NaN_along1_w_0_or_NaN(arr, row_len_new, col_len_new):
    """Return the 2D array `arr` (row_len, col_len) reshaped to shape
    (row_len_new, col_len_new) by either removing or adding rows of NaN and
    columns of NaN or zeros (such that rows of `arr` are extended by NaN if
    they end with NaN, and by 0 otherwise).

    """
    row_len = len(arr)
    arr = reshape2D_along1_with_0_or_NaN(arr, col_len_new)
    len_to_add = row_len_new - row_len
    if len_to_add > 0:
        return np.append(arr, np.nan * np.ones((len_to_add, col_len_new)), axis=0)
    if len_to_add < 0:
        return np.delete(arr, np.arange(row_len_new, row_len), axis=0)
    return arr


def convert_idxs_pop_to_subpop(idxs_pop, c_s):
    """Convert indexes from population format to subpopulation format.

    Parameters
    ----------
    idxs_pop : ndarray
        1D array (1, cell_count) of indexes in "population format" i.e. indexes
        (integer in [0, cell_count-1]) in the whole population.
    c_s : ndarray
        1D array (1, simu_count) indicates how the population is partitioned
        such that c_s[i] is the number of cells in the ith subpopulation.

    Returns
    -------
    idxs_subpop : ndarray
        2D array (cell_count, 2) of the indexes of these cells in
        "subpopulation format", i.e. index of the ...:
        > idxs_subpop[i][0] ... subpop to which belongs the ith cell.
        > idxs_subpop[i][1] ... cell in this subpopulation.

    """
    # Orderring of the indexes (in population format).
    idxs_pop = np.sort(idxs_pop)
    # Rigth format for c_s (array of int with NaN replaced by 0).
    c_s = np.nan_to_num(c_s).astype(int)
    # Cumulative subpopulation sizes.
    # (Warning: starts with 0, ends with repetition for later convinience).
    cc_s = [sum(c_s[:i]) for i in range(len(c_s) + 2)]

    # Initialization.
    cell_count = len(idxs_pop)
    idxs_subpop = np.zeros((cell_count, 2))  # Indexes in subpop format.
    idx_cell = 0  # Current cell index (here 1st cell).
    idx_subpop = 0  # Index of the subpopulation of the current cell.
    # (Here 1st cell put in the 1st subpop).

    # While all indexes have not been converted to subpopulation format.
    while idx_cell < cell_count:
        # If the index of the current cell (in population format) doesnt exceed
        # the number of cells within the first `idx_subpop + 1` subpopulations.
        if idxs_pop[idx_cell] < cc_s[idx_subpop + 1]:
            # Then the current cell belongs the `idx_subpop`th subpopulation.
            # > We compute its corresponding index in the subpopulation.
            idx_subpop_cell = idxs_pop[idx_cell] - cc_s[idx_subpop]
            # > Save them and foccus on next cell.
            idxs_subpop[idx_cell] = [idx_subpop, idx_subpop_cell]
            idx_cell += 1
        # Otherwise, it belongs to one of the following subsimulations.
        else:
            idx_subpop += 1
    return idxs_subpop.astype(int)


# Histogram-related functions
# ---------------------------


# Computing histograms out of general data.


def make_histogram(data, x_axis=None, normalized=True):
    """Generate a histogram from the given data.

    Parameters
    ----------
    data : ndarray
        Array of data values.
    x_axis : ndarray, optional
        Values to use for the histogram bins along the x-axis. If None, bins
        are generated from 0 to the maximum value in `data`.
    normalized : bool, optional
        If True, normalize the histogram to have a total area of 1.

    Returns
    -------
    hist : ndarray
        Histogram of the data.

    """
    if x_axis is None:  # If no x-axis given.
        x_axis = np.arange(int(np.max(data) + 1))
    hist = 0.0 * x_axis
    for datum in data:
        idx = np.where(x_axis == datum)[0]
        if len(idx) > 0:
            hist[idx[0]] = hist[idx[0]] + 1
    if normalized:
        if sum(hist) != 0:
            hist = hist / sum(hist)
    return hist


def make_histogram_wo_nan(data, x_axis=None, normalized=True):
    """Generate a histogram from the given data excluding NaN values.

    Parameters
    ----------
    data : ndarray
        Input array of data values, possibly containing NaN values.
    x_axis : ndarray, optional
        Values to use for the histogram bins along the x-axis. If None, bins
        are generated from 0 to the maximum value in `data`.
    normalized : bool, optional
        If True, normalize the histogram to have a total area of 1.

    Returns
    -------
    hist : array_like
        Histogram of the non-NaN data

    """
    data_tmp = data[~np.isnan(data)]
    return make_histogram(data_tmp, x_axis=x_axis, normalized=normalized)


def make_average_histogram(data_s, x_axis=None):
    """Compute an "average histogram" (average and standard deviation) among
    histograms corresponding to the sets of data given by `data_s`.

    Parameters
    ----------
    data_s : ndarray
        2D array (dataset_count, data_per_set_count).
    x_axis : ndarray, optional
        1D array of the histogram's x-axis. If None (default value), the axis
        is made of all possible values appearing in the data given as argument.

    Returns
    -------
    x_axis : ndarray
        1D array of the histogram's x-axis (depending on the optional argument)
    avg_hist : ndarray
        1D array (len(x_axis), ) where `avg_hist[i]` is the average number (in
        in percentage of the dataset) of data with value `x_axis[i]` per
        dataset (average on non-NaN values). NB: NaN if no such data.
    std_hist : ndarray
        1D array (len(x_axis), ) where `std_hist[i]` is the standard deviation
        between datasets of the number of `x_axis[i]` values per dataset.
    data_per_set_count : int
        (Common) size of the datasets.

    """
    data_per_set_count = np.shape(data_s)[1]
    # Computation of x_axis if not given.
    if x_axis is None:  # If no x-axis given.
        x_axis = np.unique(data_s)  # We sort all values without repetition.
        if isinstance(x_axis[-1], type(np.nan)):
            x_axis = x_axis[:-1]
    bin_count = len(x_axis)
    avg_hist = np.zeros(bin_count)
    std_hist = np.zeros(bin_count)
    # Computation of y-axes and associated error iterating on `x_axis`.
    for i in range(bin_count):
        where_is_x = (data_s == x_axis[i]).astype("float")
        # NB: we keep track of NaN values to average on non-NaN values.
        where_is_x[np.isnan(data_s)] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean_per_dataset = 100 * np.nanmean(where_is_x, axis=1)
            avg_hist[i] = np.nanmean(mean_per_dataset)
            std_hist[i] = np.nanstd(mean_per_dataset)
    return x_axis, avg_hist, std_hist, data_per_set_count


def make_stacked_average_histogram(data_s, x_axis=None):
    """Compute a stacked "average histogram" (average and standard deviation)
    among histograms corresponding to the sets of data given by `data_s`.

    Parameters
    ----------
    data_s : ndarray
        3D array (dataset_count, data_per_set_n_stack_count, stack_count).
    x_axis : ndarray, optional
        1D array of the histogram's x-axis. If None (default value), the axis
        is made of all possible values appearing in the data given as argument.

    Returns
    -------
    x_axis : ndarray
        1D array of the histogram's x-axis (depending on the optional arg).
    avg_hist : ndarray
        2D array (stack_count, len(x_axis)) where `avg_hist[i, j]` is the
        average number of data with value `x_axis[j]` in the i-th stack per
        dataset (average on non-NaN values). NB: NaN if no such data.
    std_hist : ndarray
        2D array (len(x_axis), stack_count) where `std_hist[i, j]` is the
        standard deviation between datasets of the number of `x_axis[j]` values
        in the i-th stack per dataset.

    """
    a, b, stack_count = np.shape(data_s)
    if x_axis is None:  # If no x-axis given.
        x_axis = np.unique(np.reshape(data_s, (a, b * stack_count)))
    bin_count = len(x_axis)
    avg_hist = np.empty((stack_count, bin_count))
    std_hist = np.empty((stack_count, bin_count))
    is_not_nan_counts = np.sum(~np.isnan(data_s), axis=(1, 2))
    for j in range(bin_count):
        where_is_x = data_s == x_axis[j]
        x_counts = 100 * np.sum(where_is_x, 1) / is_not_nan_counts[:, None]
        avg_hist[:, j] = np.mean(x_counts, axis=0)
        std_hist[:, j] = np.std(x_counts, axis=0)
    return x_axis, avg_hist, std_hist


# Computing specific histograms.


def make_cell_count_histograms(traits, nta_counts, sen_counts, bin_count, htype_choice):
    """Assume the population is structured w.r.t. a certain trait (typically
    generation or ancestor index) as described by `traits`. Compute the
    histograms of the number of: cells, type B cells, senescent type B, type H
    (if HTYPE_CHOICE), and senescent cells with respect to the individual
    trait, whose repartition in the population is described by `traits`.

    Parameters
    ----------
    traits : ndarray
        1D array (cell_count, ) containing cells' trait (generation or ancestor
        index), s.t. trait[i] is the trait of the i-th cells in the population.
    nta_counts : ndarray
         1D array (cell_count, ) with cells' number of non-terminal arrest.
    sen_counts : ndarray
         1D array (cell_count, ) with cells' number of senescent cycle.
    bin_count : int
        Number of bins for the output histograms.

    Returns
    -------
    hist : ndarray list
        List of the histograms of cell counts with respect to the trait among
        different subpopulation:
        hist[0]: 1D array (bin_count, ), the total population.
        hist[1]: 1D array (bin_count, ), type B cells only.
        hist[2]: 1D array (bin_count, ), type B senescent cells.
        hist[-1]: 1D array (bin_count, ), senescent cells.
        if HTYPE_CHOICE:
            hist[3]: 1D array (bin_count, ), type H cells.

    """
    traits = traits.astype(int)
    cell_count = len(traits)

    hist = np.zeros(bin_count)
    hist_btype = np.zeros(bin_count)
    hist_sen_btype = np.zeros(bin_count)
    hist_sen = np.zeros(bin_count)
    if htype_choice:
        hist_htype = np.zeros(bin_count)

    # Iteration on cells.
    for i in range(cell_count):
        hist[traits[i]] += 1
        if sen_counts[i] > 0:  # Senescent.
            hist_sen[traits[i]] += 1
            if nta_counts[i] > 0:  # Type H
                hist_htype[traits[i]] += 1
            elif nta_counts[i] < 0:  # Senescent type B.
                hist_btype[traits[i]] += 1
                hist_sen_btype[traits[i]] += 1
        elif nta_counts[i] != 0:  # Non-senescent type B.
            hist_btype[traits[i]] += 1

    if htype_choice:
        return [hist, hist_btype, hist_sen_btype, hist_sen, hist_htype]
    return [hist, hist_btype, hist_sen_btype, hist_sen]


# Postreating histograms.


def rescale_histogram_bin(x_axis, hist, width_rescale):
    """Resample the input histogram to change its bin width by combining
    adjacent bins.
    WARNING: `x_axis` is asumed to be regular, and `width_rescale` a positive
             integer.

    Parameters
    ----------
    x_axis, hist : ndarray
        1D arrays representing the original x-axis and y-axis values of the
        histogram.
    width_rescale : int
        Width of the new bins are `width_rescale` times the original bins.

    Returns
    -------
    x_axis_new, hist_new: ndarray
        1D arrays representing the new x-axis and y-axis values of the
        resampled histogram.

    """
    x_count, dx = len(x_axis), x_axis[1] - x_axis[0]
    x_count_new, dx_new = x_count // width_rescale, dx * width_rescale
    x_axis_new = np.linspace(
        x_axis[0], x_axis[0] + dx_new * (x_count_new - 1), x_count_new
    )
    if np.all(x_axis_new == x_axis_new.astype("int")):
        x_axis_new = x_axis_new.astype("int")
    hist_new = np.zeros(x_count_new)
    for i in range(x_count_new):
        hist_new[i] = np.sum(hist[i * width_rescale : (i + 1) * width_rescale])
    return x_axis_new, hist_new
