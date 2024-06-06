#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 16:08:23 2020

@author: arat

Defined below are auxiliary functions useful throughout the whole project.

"""

import imp
import math
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rd
# NB: for parallelization issues need to use rd.RandomState() rather than rd.
# rd.RandomState() replaced by rd. but seeds initilize for reproducinility.
# idem with population_simulation
import warnings

import aux_parameters_functions as parf
import parameters as par
imp.reload(par)


# -------------------
# Utilitary functions
# -------------------

# Computation on arrays
# ---------------------

# > Statistical quantities.

def stat(arr_stat, p_up, p_down, axis=0):
    """ Calculate statistics (mean, percentiles) of an array.

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
    stats = {'mean': np.mean(arr_stat, axis=axis),
             'perup': np.percentile(arr_stat, p_up, axis=axis),
             'perdown': np.percentile(arr_stat, p_down, axis=axis)}
    return stats

def nanstat(arr_stat, p_up, p_down, axis=0):
    """ Same function as `stat`, except that `nanstat` handles NaN values.

    """
    with warnings.catch_warnings():  # Prevent printing warnings comming from
    # applying np.nanmean, etc., to arrays full of NaN values.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        stats = {'mean': np.nanmean(arr_stat, axis=axis),
                 'perup': np.nanpercentile(arr_stat, p_up, axis=axis),
                 'perdown': np.nanpercentile(arr_stat, p_down, axis=axis)}
    return stats

def stat_all(arr_stat, p_up, p_down, axis=0):
    """ Similar to function `stat` with a slightly different output format and
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
    stats = {'mean': np.mean(arr_stat, axis=axis),
             'per': [np.percentile(arr_stat, p_down, axis=axis),
                     np.percentile(arr_stat, p_up, axis=axis)],
             'ext': [np.min(arr_stat, axis=axis),
                     np.max(arr_stat, axis=axis)]}
    return stats

# > Sum, product...

def nansum(arr, axis=None, keepdims=False):
    """ Similar to `np.nansum`, except that if there are only NaN values along
    a summation axis, the sum along that axis is NaN instead of 0.

    """
    arr_sum = np.nansum(arr, axis=axis, keepdims=keepdims)
    arr_isnan = np.isnan(arr).all(axis=axis)
    if arr_isnan.any():  # If at least one axis with NaN only.
        # The sum along NaN? axis is Nan instead of 0.
        if isinstance(arr_sum, np.ndarray):  # If arr_sum is an array.
            arr_sum[arr_isnan] = np.nan  # Array returned.
        else:  # Otherwise arr_sum = 0, is int or float, not array!
            return np.NaN
    return arr_sum

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

# Modification of arrays
# ----------------------

# > Sorting.

def nanargsort1D(arr):
    """ Similar to `np.argsort` for 1D array, except that NaN are not orderred.

    """
    # Compute indexes corresponding to the usual sorting (NaN at the end).
    idxs_sorted = np.argsort(arr)
    # And return only indexes corresponding to non NaN values.
    non_nan_count = sum(~np.isnan(arr))
    return idxs_sorted[:non_nan_count]

# > Reshaping.

def reshape_with_nan(arr, len_new, axis=-1):
    """ Return the array `arr` reshaped along the axis `axis`, turning it from
    length `len_current` to `len_new` either by removing values or adding NaN.
    NB: for the last axis, take `axis=-1` (default).

    """
    len_current = np.shape(arr)[axis]
    to_add =  len_new - len_current
    if to_add > 0:
        new_dimensions = list(np.shape(arr))
        new_dimensions[axis] = to_add
        new_dimensions = tuple(np.array(new_dimensions).astype('int'))
        return np.append(arr, np.nan * np.ones(new_dimensions), axis=axis)
    if to_add < 0:
        return np.delete(arr, np.arange(len_new, len_current), axis=axis)
    return arr

def reshape2D_along1_with_0_or_NaN(arr, col_len_new):
    """ Return the 2D array `arr` (row_len, col_len) reshaped to shape
    (row_len, col_len_new), either by removing or adding columns so that rows
    of `arr` are extended by NaN if they end with NaN, and by 0 otherwise.

    """
    row_len, col_len = np.shape(arr)
    len_to_add =  col_len_new - col_len
    if len_to_add > 0:
        last_col = np.transpose([arr[:, -1]])
        return np.append(arr, last_col*np.zeros((row_len, len_to_add)), axis=1)
    if len_to_add < 0:
        return np.delete(arr, np.arange(col_len_new, col_len), axis=1)
    return arr

def reshape2D_along0_w_NaN_along1_w_0_or_NaN(arr, row_len_new, col_len_new):
    """ Return the 2D array `arr` (row_len, col_len) reshaped to shape
    (row_len_new, col_len_new) by either removing or adding rows of NaN and
    columns of NaN or zeros (such that rows of `arr` are extended by NaN if
    they end with NaN, and by 0 otherwise).

    """
    row_len = len(arr)
    arr = reshape2D_along1_with_0_or_NaN(arr, col_len_new)
    len_to_add =  row_len_new - row_len
    if len_to_add > 0:
        return np.append(arr, np.nan * np.ones((len_to_add, col_len_new)),
                         axis=0)
    if len_to_add < 0:
        return np.delete(arr, np.arange(row_len_new, row_len), axis=0)
    return arr

def convert_idxs_pop_to_subpop(idxs_pop, c_s):
    """ Convert indexes from population format to subpopulation format.

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

# > Computing histograms out of general data.

def make_histogram(data, x_axis=None, normalized=True):
    """ Generate a histogram from the given data.

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
    hist = 0. * x_axis
    for datum in data:
        idx = np.where(x_axis == datum)[0]
        if len(idx) > 0:
            hist[idx[0]] = hist[idx[0]] + 1
    if normalized:
        if sum(hist) != 0:
            hist = hist / sum(hist)
    return hist

def make_histogram_wo_nan(data, x_axis=None, normalized=True):
    """ Generate a histogram from the given data excluding NaN values.

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
    """ Compute an "average histogram" (average and standard deviation) among
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
        1D array of the histogram's x-axis (depending on the optional argument).
    avg_hist : ndarray
        1D array (len(x_axis), ) where `avg_hist[i]` is the average number of
        data with value `x_axis[i]` per dataset (average on non-NaN values).
        NB: NaN if no such data.
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
    bin_count = len(x_axis)
    avg_hist = np.zeros(bin_count)
    std_hist = np.zeros(bin_count)
    # Computation of y-axes and associated error iterating on `x_axis`.
    for i in range(bin_count):
        where_is_x = data_s == x_axis[i]
        # NB: we keep track of NaN values to average on non-NaN values.
        where_is_x[~np.isnan(data_s)] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean_per_dataset = 100 * np.nanmean(where_is_x, axis=1)
            avg_hist[i] = np.nanmean(mean_per_dataset)
            std_hist[i] = np.nanstd(mean_per_dataset)
    return x_axis, avg_hist, std_hist, data_per_set_count

def make_stacked_average_histogram(data_s, x_axis=None):
    """ Compute a stacked "average histogram" (average and standard deviation)
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

# > Computing specific histograms.

def make_cell_count_histograms(traits, nta_counts, sen_counts, bin_count):
    """ Assume the population is structured w.r.t. a certain trait (typically
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
    if par.HTYPE_CHOICE:
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

    if par.HTYPE_CHOICE:
        return [hist, hist_btype, hist_sen_btype, hist_sen, hist_htype]
    return [hist, hist_btype, hist_sen_btype, hist_sen]

# > Postreating histograms.

def rescale_histogram_bin(x_axis, hist, width_rescale):
    """ Resample the input histogram to change its bin width by combining
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
    x_axis_new = np.linspace(x_axis[0], x_axis[0] + dx_new * (x_count_new - 1),
                             x_count_new)
    if np.all(x_axis_new == x_axis_new.astype('int')):
        x_axis_new = x_axis_new.astype('int')
    hist_new = np.zeros(x_count_new)
    for i in range(x_count_new):
        hist_new[i] = np.sum(hist[i * width_rescale:(i+1) * width_rescale])
    return x_axis_new, hist_new

# Using Cumulative distribution function (cdt)
# --------------------------------------------

def cdf_to_distribution(cdfs, distrib_x=None):
    """ Convert a cumulative distribution function (CDF) to a probability
    distribution.

    Parameters
    ----------
    cdfs : ndarray
        1D array (x_count,) representing the cumulative distribution function.
    distrib_x : ndarray, optional
        1D array (x_count,) representing the x-axis values of the distribution.
        If None (default value), unique values from the CDF are used.

    Returns
    -------
    distrib_x : ndarray
        1D array (x_count,) representing the x-axis values of the probability 
        distribution.
    distrib_y : ndarray
        1D array (x_count,) representing the y-axis values (probabilities) of
        the probability distribution.

    """
    if isinstance(distrib_x, type(None)):
        distrib_x = np.unique(cdfs)
    x_count = len(distrib_x)
    distrib_y = np.zeros(x_count)
    for i in range(x_count - 1):
        distrib_y[i] = sum(np.logical_and(distrib_x[i] <= cdfs,
                                          cdfs < distrib_x[i + 1]))
    return distrib_x, distrib_y / x_count

def inverse_cdf(u, x_values, probas):
    """ Return cdf^{-1}_X(u): the inverse of the cumulative distribution
    function, of a discrete random variable X, whose law is described by
    `x_values` and `probas`, evaluated at u.

    Parameters
    ----------
    u : float
        Float in [0, 1], argument of F^{-1}.
    x_values : nd array
        ORDERED 1D array (1, ) of the values taken by the (discrete) random
        variable of interest.
    probas : ndarray
        Probabilities associated to the values of X `x_values`, s.t.
        `proba[i] == P(X = x_values[i])`.
    """
    inv_idx = 0
    cdf = probas[0]
    while cdf < u and inv_idx < len(probas) - 1:
        inv_idx = inv_idx + 1
        cdf = cdf + probas[inv_idx]
    return x_values[inv_idx]


# -------------------------------------------------------------------
# Model-related functions (generate distributions, test certain laws)
# -------------------------------------------------------------------

# Telomere lengths
# ----------------

# > Overhang distribution.

def draw_overhang():
    """Return a overhang value drawn accordingly to the law `OVERHANG_CHOICE`:
    - 'const': constant overhang, equal to `par.OVERHANG`.
    - 'uniform': random overhang, drawn from a uniform distribution on
                `[par.OVERHANG_LOW, par.OVERHANG_UP]`.

    """
    if par.OVERHANG_CHOICE == 'const':
        return par.OVERHANG
    if par.OVERHANG_CHOICE == 'uniform':
        return np.random.randint(par.OVERHANG_LOW, par.OVERHANG_UP)
    raise Exception("Error: `OVERHANG_CHOICE` from `parameters.py` has "
                    "unexpected value")

# > Initial distribution of telomere length

if __name__ == "__main__":
    # Visualization of the inverse of the cumulative distrib. function of the
    # experimental initial telomere length (draw line step in figure options).
    plt.clf()
    U = np.linspace(0, 1, 1000)
    # Original cdt.
    SUPPORT, PROBAS = parf.transform_l_init(par.L_INIT_EXP)
    x_invs = np.round(np.array([inverse_cdf(u, SUPPORT, PROBAS) for u in U]))
    plt.plot(U, x_invs, label='Original')
    # Cdt of the transformed distribution.
    SUPPORT, PROBAS = parf.transform_l_init(par.L_INIT_EXP, *par.PAR_L_INIT)
    x_invs = np.round(np.array([inverse_cdf(u, SUPPORT, PROBAS) for u in U]))
    plt.plot(U, x_invs, label='Transformed')
    # Cdt of the distribution transformed with old (wrong) method.
    SUPPORT, PROBAS = parf.transform_l_init_old(par.L_INIT_EXP,*par.PAR_L_INIT)
    x_invs = np.round(np.array([inverse_cdf(u, SUPPORT, PROBAS) for u in U]))
    plt.plot(U, x_invs, label='Transformed (before correction)')
    plt.title('Inverse of the cdf of the\ninital distribution of '
              'telomere lengths')
    plt.legend()

def draw_cell_lengths(chromosome_count, par_l_init=par.PAR_L_INIT):
    """ Return a distribution of `2 * chromosome_count` initial telomere
    lengths drawn independently according to `par.L_INIT_CHOICE`:
    - 'const' for constant initial length (= `L_INF`).
    - 'gaussian' for Gaussian distribution (avg = `L_INF`, sd = `L_SIGMA`).
    - 'exp' for experimental distribution (from `data/etat_asymp_val_juillet`).

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
        telomere_count = 2 * chromosome_count
        u_rands = rd.uniform(0, 1, telomere_count)
        lengths = np.zeros(telomere_count)
        support, proba = parf.transform_l_init(par.L_INIT_EXP, *par_l_init)
        for i in range(telomere_count):
            lengths[i] = np.round(inverse_cdf(u_rands[i], support, proba))
         # Return the lengths obtained with translated initial distribution.
        return np.reshape(lengths, cell_shape)

    raise Exception("Error: `L_INIT_CHOICE` from `parameters.py` has "
                    "unexpected value")

def draw_cells_lengths(cell_count, par_l_init=par.PAR_L_INIT):
    """ Returns initial telomere length distributions of `cell_count` cells
     drawn independently accordingly to `par.L_INIT_CHOICE`:
    - 'const': constant initial length (= `L_INF`).
    - 'gaussian': Gaussian distribution (avg = `L_INF`, sd = `L_SIGMA`).
    - 'exp': experimental distribution (from `data/etat_asymp_val_juillet`).
    - 'two_normal-long': 2 cells only: one with "average" shortest telomere
        length, and one with "long" shortest telomere.
    - 'two_normal-short': one cell with "average" shortest telomere, and one
        cell with "short" shortest telomere.
    WARNING: for the last 2 options, if the data needed to create `par.L_*` has
             not been generated yet, one should create it using
             `aux_make_special_telomere_distribution.py`.

    Parameters
    ----------
    cell_count : int
        Number of cell disribution to create.

    Returns
    -------
    lengths : ndarray
        3D array (cell_count, 2, chromosome_count) of the lengths of the two
        telomeric extremities of the `par.CHROMOSOME_COUNT` chromosomes of the
        `cell_count` cells.

    """
    if par.L_INIT_CHOICE[:3] == 'two':
        if cell_count != 2:
            raise Exception(f"Creating {cell_count} cells is not compatible "
                            "with `par.L_INIT_CHOICE` that loads 2 cells.")
        if par.L_INIT_CHOICE == 'two_normal-long':
            l_ext = np.loadtxt(par.L_LONG_PATH)
            l_medium = np.loadtxt(par.L_MEDIUM_PATH)
        if par.L_INIT_CHOICE == 'two_normal-short':
            l_ext = np.loadtxt(par.L_SHORT_PATH)
            l_medium = np.loadtxt(par.L_MEDIUM_PATH)
        else:
            raise Exception("Error: `L_INIT_CHOICE` from `parameters.py` has "
                    "unexpected value")
        lengths = np.array([l_ext, l_medium])
        return lengths

    return np.array([draw_cell_lengths(par.CHROMOSOME_COUNT, par_l_init) for i
                     in range(cell_count)])

# > Telome cutting in final cut experiments.

if not isinstance(par.PAR_FINAL_CUT, type(None)):
    import finalCut_fit_cut_efficiency as fce

    def is_cut_exponential(cdt_under_gal, dt_w_gal=36):
        """ Test if a cell, which has spent `cdt_under_gal` * 10 minutes of its
        cell cycle experiencing galactose, had one of its telomeres cut during
        the cycle, given that galactose was added `dt_w_gal` * 10 minutes ago
        (at the end of the cell cycle).

        """
        ten_min_to_h = 1 / 6
        a = (dt_w_gal - cdt_under_gal) * ten_min_to_h  # Birth time [h].
        b = dt_w_gal * ten_min_to_h  # Minimun between division time [h] and
                                     # time at which Galactose was removed [h].
        # We compute P(T<b | T>a) where T is the r.v. of the time of cut [h].
        proba = (fce.fit_cdf(b) - fce.fit_cdf(a)) / (1 - fce.fit_cdf(a))
        return rd.binomial(1, proba)

# Laws of arrest in the cell cycle
# --------------------------------

# > Onset of on-terminal arrest (nta).

def law_sigmoid(gen, a, b):
    """ Return the sigmoid function (2) (Martin et al., 2021) evaluated at
    `gen`, which corresponds to the probability of having a long cycle at
    generation `gen` when there have only been normal cycles so far.

    """
    # Compute generation-dependent probability to trigger an arrest.
    proba = 0
    if gen > 0:
        proba = 1 / (1 + np.exp(- (gen - a) / b))
    # Test if an arrest is triggered according to this probiblity.
    return bool(rd.binomial(1, proba))

def law_exponential(length, a, b):
    # Compute telomere-length dependent probability to trigger an arrest.
    proba = min(1, b * math.exp(- a * length))
    # Test if an arrest is triggered according to this probiblity.
    return bool(rd.binomial(1, proba))

def law_exponential_w_threshold(length, a, b, lmin):
    if length <= lmin:  # If deterministic threshold reached.
        return True  # Senescence is triggered.
    # Otherwise, test with exponential law.
    return law_exponential(length, a, b)

def law_gaussian_threshold(length, mu, sigma):
    ltrig = max(0, rd.normal(mu, sigma))  # Threshold length.
    return length <= ltrig

def is_nta_trig(gen, length_min, lengths, parameters=par.PAR_NTA):
    """ Test if a non-terminal arrest (nta) is triggered (True) for a type A
    cell with telomere distribution `lengths`, minimal telomere length
    `length_min`, and generation `gen`, following the rule given by
    `par.TRIG_TELO_CHOICE` and `P_NTA_CHOICE`.

    Optional: regardless of the parameters fixed in `parameters.py`, one can
    specify new parameters [a, b] as arguments.

    """
    if par.P_NTA_CHOICE == 'sigmoid':  # See (2) (Martin et al., 2021).
        return law_sigmoid(gen, *parameters)

    elif par.P_NTA_CHOICE == 'deterministic':  # Determinitic threshold.
        return length_min <= parameters

    # Computation of the lengths of the telomere to test.
    if par.TRIG_TELO_CHOICE == 'shortest':  # Only the shortest telo is tested.
        lengths_min = [length_min]
    elif par.TRIG_TELO_CHOICE == 'all':  # All telo shorter than L_MIN_MAX are.
        lengths_min = lengths.flatten()
        lengths_min = lengths_min[lengths_min <= par.L_MIN_MAX]

    if par.P_NTA_CHOICE == 'exponential':
        for lmin in lengths_min:  # For every lengths to test, test if it
            # triggers nta (nta triggered if at least 1 telo triggers).
            if law_exponential(lmin, *parameters):
                return True

    elif par.P_NTA_CHOICE == 'gaussian':  # Probabilistic threshold.
        for lmin in lengths_min:
            if law_gaussian_threshold(lmin, *parameters):
                return True
    # If still nothing returned, no telomere has triggerd an arrest.
    return False

# > Onset of terminal/senescent arrest.

def is_sen_trig(length_min, lengths, parameters):
    """ Test if senescence of a cell with telomere length distribution
    `lengths` and minimal telomere length `length_min` is triggered (True), or
    not (False), according `par.TRIG_TELO_CHOICE` and `par.P_SEN_CHOICE` in
    the case of a stochastic threshold.

    """
    if par.P_SEN_CHOICE == 'deterministic':  # Determinitic threshold.
        return length_min <= parameters

    # Computation of the lengths of the telomere to test.
    if par.TRIG_TELO_CHOICE == 'shortest':  # Only the shortest telo is tested.
        lengths_min = [length_min]
    elif par.TRIG_TELO_CHOICE == 'all':  # All telo shorter than L_MIN_MAX are.
        lengths_min = lengths.flatten()
        lengths_min = lengths_min[lengths_min <= par.L_MIN_MAX]
    # Test.
    # > Exponential law with deterministic thershold.
    if par.P_SEN_CHOICE == 'exponential-threshold':
        for lmin in lengths_min:  # For all lengths to test,
            # Test if it triggers senescence.
            if law_exponential_w_threshold(lmin, *parameters):
                return True
    # > Exponential law.
    elif par.P_SEN_CHOICE == 'exponential':
        for lmin in lengths_min:
            if law_exponential(lmin, *parameters):
                return True
    # > Gaussian law.
    elif par.P_SEN_CHOICE == 'gaussian':
        for lmin in lengths_min:
            if law_gaussian_threshold(lmin, *parameters):
                return True
    # If still nothing returned, no telomere has triggered a nta.
    return False

def is_sen_atype_trig(length_min, lengths, parameters=par.PAR_SEN_A):
    """ Test if senescence of a type A cell with telomere length distribution
    `lengths` and minimal telomere length `length_min` is triggered (True), or
    not (False), according to `par.TRIG_TELO_CHOICE` and `par.P_SEN_CHOICE`.

    """
    return is_sen_trig(length_min, lengths, parameters)

def is_sen_btype_trig(length_min, lengths, parameters=par.PAR_SEN_B):
    """ Test if senescence of a type B cell of telomere length distribution
    `lengths` and minimal telomere length `length_min` is triggered (True) or
    not (False), according to `par.TRIG_TELO_CHOICE` and `par.P_SEN_CHOICE`.

    """
    return is_sen_trig(length_min, lengths, parameters)

# > Exit of non-terminal arrest.

def is_repaired():
    """  Test if a non-terminally arrested cell exits its sequence of
    non-terminal long cycles (True) or continues the sequence (False).

    """
    return rd.binomial(1, par.P_REPAIR)

# > Onset of death.

def is_dead(sen_count=0, max_sen_count=par.MAX_SEN_CYCLE_COUNT):
    """ Test if a senescent cell dies (True) or continues to divide (False).

    """
    return (rd.binomial(1, par.P_DEATH) or
            sen_count > max_sen_count)

def is_accidentally_dead(p_death_acc=par.P_ACCIDENT):
    """ Test if a cell accidentally dies (True) or continues to divide (False).

    """
    return rd.binomial(1, p_death_acc)

# Distribution of cycle duration time (cdt)
# -----------------------------------------

def draw_cycles_atype(cell_count):
    """ Return a distribution of cycle duration times [min] of a population of
    `cell_count` non-senescent type A cells according to `par.CYCLES_CHOICE`.

    """
    if par.CYCLES_CHOICE == 'const':
        return par.CYCLES_A_CONST * np.ones(cell_count)
    if par.CYCLES_CHOICE == 'exp':
        return rd.choice(par.CDTS_OLD['nor-telo+'], cell_count)
    if par.CYCLES_CHOICE == 'exp_new':
        return rd.choice(par.CDTS['norA'], cell_count)
    print("Error: `CYCLES_CHOICE` from `parameters.py` has unexpected value")
    return None

def draw_cycle_btype_long():
    """ Return a cycle duration time [min] for a type B cell in its 1st
    sequence of long cycles accordingly to `par.CYCLES_CHOICE`.

    """
    if par.CYCLES_CHOICE == 'const':
        return par.CYCLES_B_LONG_CONST
    if par.CYCLES_CHOICE == 'exp':
        return rd.choice(par.CDTS_OLD['nta1'])
    if par.CYCLES_CHOICE == 'exp_new':
        return rd.choice(par.CDTS['nta'])
    print("Error: `CYCLES_CHOICE` from `parameters.py` has unexpected value")
    return None

def draw_cycle_btype_long_af_1ST():
    """ Return a cycle duration time for a type B cell in its 2nd, or 3rd, ...
    sequence of long cycles accordingly to `par.CYCLES_CHOICE`.

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
def draw_cycle_btype_normal():
    """ Return a cycle duration time for a type B cell in normal cycle
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

def draw_cycle(arrest_count, is_senescent):
    """ Return a cycle duration time [min] for a new born cell in the state
    entered as argument.

    Parameters
    ----------
    arrest_count : int
        Number of sequence of non-terminal arrests the lineage of the cell went
        through so far. For non-senescent cell, it is positive if the cell is
        arrested, negative if it is in normal cycle.
    is_senescent : bool
        Is False if the cell is not senescent, True if senescent.

    """
    if is_senescent:  # The cell is senescent.
        return draw_cycle_sen()
    if arrest_count == 0:  # Non-senescent type A.
        return draw_cycles_atype(1)[0]
    if arrest_count == 1:  # Non-senescent type B in a 1st sequence of arrests.
        return draw_cycle_btype_long()
    if arrest_count < 0:  # Non-senescent type B in a normal cycle.
        return draw_cycle_btype_normal()
    # Non-senescent type B in a 2nd, 3rd... sequence of arrests.
    return draw_cycle_btype_long_af_1ST()

def draw_cycle_finalCut(arrest_count, is_senescent, is_galactose):
    """ Return a cycle duration time [min] for a new born cell in the state
    entered as argument (see `draw_cycle` docstring) in galactose or raffinose
    conditions (if `is_galactose` is True or False respectively).

    """
    # If the cell is arrested.
    if is_senescent or arrest_count > 0:
        if is_galactose:
            return rd.choice(par.CDTS_FINALCUT['gal']['arr'])
        return rd.choice(par.CDTS_FINALCUT['raf']['arr'])
    # Otherwise it experiences a normal cycle.
    if is_galactose:
        return rd.choice(par.CDTS_FINALCUT['gal']['nor'])
    return rd.choice(par.CDTS_FINALCUT['raf']['nor'])

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
