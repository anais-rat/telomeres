#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 11:37:41 2024

@author: anais
"""

import scipy.io as sio
from copy import deepcopy
import numpy as np

from telomeres.auxiliary.functions import reshape_with_nan, nanargsort1D
from telomeres.dataset.extract_processed_dataset import \
    extract_postreat_lineages


# Auxiliary
# ---------

def type_of_sort_from_characteristics(characteristics):
    """Return the most natural criteria to sort lineages of given
    characteristics.

    """
    if 'dead' in characteristics:
        return 'gdeath'
    if 'senescent' in characteristics:
        return 'gsen'
    return 'gnta1'


# Tools for characterization of lineage(s)
# ----------------------------------------

def is_as_expected_lineage(gtrigs, lineage_type, is_accidental_death,
                           characteristics):
    """Test if the lineage described by `gtrigs`, `lineage_type`,
    `is_accidental_death` has the characteristics given by `characteristics`.

    Parameters
    ----------
    gtrigs : dict
        Dictionnary with generations of event (nta, senescence and death) of
        the format returned by `simulate_lineage_evolution`.
    lineage_type : int
        0 for type A, 1 for type B and Nan for H (in case of simulated
        lineage) or unidentified (for experimental).
    is_accidental_death : bool
        True if the lineage died accidentally, False otherwise.
    characteristic : list of string
        The list of characteristics to test among:
        - 'arrestedi' : for lineage experiencing at least `i` nta(s).
        - 'senescent' : that has enter senescence (in the strict sense).
        - 'atype' : for type-A lineage
        - 'dead' : for dead lineage (NB: always True for simulated lineages).
        - 'dead_naturally' : for lineage dead (apparently) not by accident.
        - 'dead_accidentally' : for lineage dead by accident.

    Returns
    -------
    is_as_expected : bool
        True if the lineage correponds to the characteristics given, False
        otherwise.

    """
    is_as_expected = True  # No characteristic checked yet necessarily True.
    if 'atype' in characteristics:
        is_as_expected = lineage_type == 0
    if 'btype' in characteristics:
        is_as_expected = lineage_type == 1
    if 'htype' in characteristics:
        is_as_expected = is_as_expected and np.isnan(lineage_type)
    arrested_idx = np.array(['arrested' in c for c in characteristics])
    if any(arrested_idx):
        nta_idxs = []
        for i in range(len(arrested_idx)):
            if arrested_idx[i]:
                nta_idxs.append(int(characteristics[i][-1]))
        nta_idx = max(nta_idxs)
        nta_count = len(gtrigs['nta'])
        is_as_expected = is_as_expected and (nta_idx < nta_count)
    is_dead = not np.isnan(gtrigs['death'])
    if 'dead' in characteristics:
        is_as_expected = is_as_expected and is_dead
    if 'dead_accidentally' in characteristics:
        is_as_expected = is_as_expected and is_accidental_death
    if 'dead_naturally' in characteristics:
        is_dead_naturally = is_dead and (not is_accidental_death)
        is_as_expected = is_as_expected and is_dead_naturally
    if 'senescent' in characteristics:
        is_senescent = not np.isnan(gtrigs['sen'])
        is_senescent = np.logical_and(is_dead, is_senescent)
        is_as_expected = is_as_expected and is_senescent
    return is_as_expected


def is_as_expected_lineages(gtrigs_s, lineage_types, is_accidental_deaths,
                            characteristics):
    """Test if the lineages described by `gtrigs_s`, `lineage_types`,
    `is_accidental_deaths` have the characteristics given by `characteristics`.

    The output is a bool 1D array of length `lineage_count`. See also
    `is_as_expected_lineage`.
    NB: since characteristics are common to all lineages we repeat the code of
        `is_as_expected_lineage` instead of referring to it.

    """
    lineage_count = len(lineage_types)
    is_as_expected = np.ones(lineage_count).astype(bool)
    if 'atype' in characteristics:
        is_as_expected = lineage_types == 0
    if 'btype' in characteristics:
        is_as_expected = lineage_types == 1
    if 'htype' in characteristics:
        is_as_expected = np.logical_and(is_as_expected,
                                        np.isnan(lineage_types))
    is_w_arrested = np.array(['arrested' in c for c in characteristics])
    if any(is_w_arrested):
        arrested_charac = [characteristics[i] for i in
                           np.arange(len(is_w_arrested))[is_w_arrested]]
        nta_idxs = [int(c[-1]) - 1 for c in arrested_charac]
        nta_idx = max(nta_idxs)
        if nta_idx > len(gtrigs_s['nta'][0]):
            return np.zeros(lineage_count).astype(bool)
        is_arrested = ~np.isnan(gtrigs_s['nta'][:, nta_idx])
        is_as_expected = np.logical_and(is_as_expected, is_arrested)
    is_dead = ~np.isnan(gtrigs_s['death'])
    if 'dead' in characteristics:
        is_as_expected = np.logical_and(is_as_expected, is_dead)
    if 'dead_accidentally' in characteristics:
        is_as_expected = np.logical_and(is_as_expected, is_accidental_deaths)
    if 'dead_naturally' in characteristics:
        is_dead_naturally = np.logical_and(is_dead, ~is_accidental_deaths)
        is_as_expected = np.logical_and(is_as_expected, is_dead_naturally)
    if 'senescent' in characteristics:
        is_senescent = ~np.isnan(gtrigs_s['sen'])
        # Warning for experimental lineage: `gtrigs_s['sen']` can be an integer
        #  for a non-senescent lineage if the lineage has not died.
        #  Thus, we need in addition to require that the lineage is dead.
        is_senescent = np.logical_and(is_dead, is_senescent)
        is_as_expected = np.logical_and(is_as_expected, is_senescent)
    return is_as_expected


# Extract information
# -------------------

def select_exp_lineages(exp_data, characteristics):
    """Extract from `exp_data` the data of the lineages having the
    characteristics given as argument, returning it under same format (i.e. an
    output of `lineage.posttreat.postreat_experimental_lineages`).

    """
    evo, gtrigs, lineage_types, is_unseen_htypes, is_accidental_deaths, \
        lcycle_per_seq_counts = deepcopy(exp_data)
    selected_lineage_idxs = is_as_expected_lineages(gtrigs, lineage_types,
                                                    is_accidental_deaths,
                                                    characteristics)
    evo['cycle'] = evo['cycle'][selected_lineage_idxs]
    for key in gtrigs:
        gtrigs[key] = gtrigs[key][selected_lineage_idxs]
    lineage_types = lineage_types[selected_lineage_idxs]
    if not isinstance(is_unseen_htypes, type(None)):
        is_unseen_htypes = is_unseen_htypes[selected_lineage_idxs]
    is_accidental_deaths = is_accidental_deaths[selected_lineage_idxs]
    for key in lcycle_per_seq_counts:
        lcycle_per_seq_counts[key] \
            = lcycle_per_seq_counts[key][selected_lineage_idxs]
    selected_data = (evo, gtrigs, lineage_types, is_unseen_htypes,
                     is_accidental_deaths, lcycle_per_seq_counts)
    return selected_data


def count_exp_lineages(characteristics, strain='TetO2-TLC1'):
    """Return the number of lineages having certain characteristics in a
    certain experimental dataset.

    Parameters
    ----------
    characteristic : list of string
        The list of characteristics of the experimental lineages to count:
        - 'arrestedi' : lineages experiencing at least `i` nta(s).
        - 'senescent' : that has enter senescence (in the strict sense).
        - 'atype', 'btype', 'htype': specific lineage types.
        - 'dead' : dead lineages (NB: always True for simulated lineages).
        - 'dead_naturally' : lineages dead (apparently) not by accident.
        - 'dead_accidentally' : lineages dead by accident.
    strain : string, optional
        Dataset in which to count: 'TetO2-TLC1' (default) or 'RAD51'.

    """
    data_exp = extract_postreat_lineages(strain=strain)
    data_exp_selected = select_exp_lineages(data_exp, characteristics)
    return len(data_exp_selected[0]['cycle'])


# Postreat
# --------


def sort_lineages(data, type_of_sort):
    """Sort the information on lineages (contained in `data` formatted as any
    ouput of `simulate_lineages_evolution` or
    `lineage.posttreat.postreat_experimental_lineages`)
    accordingly to `type_of_sort`.
    WARNING : also forget the lineages than cannot be sorted (for exemple if we
        ask to sort by generation of the onset of the 1st nta, the data of
        lineages that did not experienced a 1st arrest is not returned).

    Parameters
    ----------
    data : list
        List of data of sets of lineages formatted as the ouputs of
        `simulate_lineages_evolution` or
        `lineage.posttreat.postreat_experimental_lineages`
        (see corresponding docstrings).
    type_of_sort : string
        Indicates how to sort data:
        - 'lmin' : sort by initial length of shortest telomere.
        - 'lavg' : sort by initial average telomere length.
          WARNING: does not work for experimental data (ie `data` being output
                   of `lineage.posttreat.postreat_experimental_lineages`).
        - 'gntai' : sort by generation of the ith non-terminal arrest.
        - 'gsen' : sort by generation of senescence.
        - 'gdeath' : sort by generation of death.
        - 'len' : sort by length.

    Returns
    -------
    data_sorted : list
        Exactly `data` except that all the arrays of `data` have been sorted
        along the lineage axis (0 axis) and that the lineages that could not be
        sorted have been forgotten.

    """
    evo_s, gtrigs_s, lineage_types, is_unseen_htypes, is_accidental_deaths, \
        lcycle_per_seq_counts = deepcopy(data)

    if type_of_sort == 'gdeath':
        idxs_sorted = nanargsort1D(gtrigs_s['death'])
    elif type_of_sort in ['lmin', 'lavg']:
        if isinstance(evo_s, type(None)):
            raise ValueError(f"ERROR: to sort by '{type_of_sort}' you need to"
                             " set `is_evo_returned` to True.")
        idxs_sorted = np.argsort(evo_s[type_of_sort][:, 0])
    elif 'gnta' in type_of_sort:
        nta_idx = int(type_of_sort[-1]) - 1
        idxs_sorted = nanargsort1D(gtrigs_s['nta'][:, nta_idx].flatten())
    elif type_of_sort == 'gsen':
        # NB: we need to be carefull for experimental lineage: need to remove
        #     lineages ended by long cycle (thus classified as senescent) but
        #     that have not died at the end of measurement.
        is_dead = ~np.isnan(gtrigs_s['death'])
        dead_lineages = np.arange(len(lineage_types))[is_dead]
        # Indexes of senescent lineages ordered by increasing gen of sen onset.
        idxs_sorted = nanargsort1D(gtrigs_s['sen'])
        # Computation of bool array its it h component indicates if the lineage
        # index `idxs_sorted[i]` is in `dead_lineages`.
        is_idx_by_gsen_in_dead = np.in1d(idxs_sorted, dead_lineages)
        # Indexes of senescent and dead cells by increasing gen of sen onset.
        idxs_sorted = idxs_sorted[is_idx_by_gsen_in_dead]
    elif type_of_sort == 'len':
        lin_length = np.sum(~np.isnan(evo_s['cycle']), axis=1)
        idxs_sorted = nanargsort1D(lin_length)
    elif type_of_sort == 'none':
        return data
    else:
        raise ValueError("ERROR: wrong `type_of_sort` argument for "
                         "`sort_lineage` function")
    # Ordering of data.
    # > `evo_s`.
    if not isinstance(evo_s, type(None)):
        for key, evos in evo_s.items():
            evo_s[key] = evos[idxs_sorted]
    # > `gtrigs_s`.
    for key in gtrigs_s:
        gtrigs_s[key] = gtrigs_s[key][idxs_sorted]
    # > `lcycle_per_seq_counts` if computed
    if not isinstance(lcycle_per_seq_counts['nta'], type(None)):
        lcycle_per_seq_counts = \
            {'nta': lcycle_per_seq_counts['nta'][idxs_sorted],
             'sen': lcycle_per_seq_counts['sen'][idxs_sorted]}
    # > Gathering of all sorted outputs.
    if not isinstance(is_unseen_htypes, type(None)):
        is_unseen_htypes = is_unseen_htypes[idxs_sorted]
    return (evo_s, gtrigs_s, lineage_types[idxs_sorted], is_unseen_htypes,
            is_accidental_deaths[idxs_sorted], lcycle_per_seq_counts)


def compute_exp_lcycle_counts(cycles, gtrigs, is_lcycle):
    """Compute the number of successive long cycles of every sequence of arrest
    of every lineage, orderring them similarly to `gtrigs`.

    Parameters
    ----------
    cycles : ndarray
        2D array (lineage_count, gen_count) s.t. `cycles[i, j]` is the cycle
        duration time of the jth cell of the ith lineage (in minutes).
        NB: is Nan if the the ith lineage is already dead at generation j.
    gtrigs : dict
        Dictionnary with generations of event (nta, senescence and death) of
        the format returned by `simulate_lineages_evolution`.
    is_lcycle : ndarray
        2D array (lineage_count, gen_count) s.t. `is_lcycle[i, j]` is True if
        the cycle duration time of the jth cell of the ith lineage is long,
        False otherwise.
        NB: is Nan if the the ith lineage is already dead at generation j.

    Returns
    -------
    lcycle_per_seq_counts : dict
        Dictionnary of the number of long cycles per sequence of long cycles
        gathered by entries st.:
        nta : ndarray
            2D array with same shape as `gtrigs['nta']` s.t.
            `lcycle_per_seq_counts['nta'][i, j]` is the number of successive
            long cycles of the jth nta of the ith lineage.
        sen : ndarray
            1D array (lineage_count, ) (ie with shape of `gtrigs['sen']`) s.t.
            `lcycle_per_seq_counts['sen'][i]` is the number of successive
            senescent (long)  cycles of the ith lineage.
        NB: Nan value whenever there is if no such sequence.

    """
    lineage_count = len(gtrigs['sen'])

    lcycle_per_seq_counts = {'nta': np.nan * gtrigs['nta'],
                             'sen': np.nan * np.zeros(lineage_count)}
    for lin in range(lineage_count):
        # > nta.
        nta_max = sum(~np.isnan(gtrigs['nta'][lin]))
        idx_gmax = sum(~np.isnan(cycles[lin])) - 1
        idx_nta_max_end = int(np.nanmin([idx_gmax, gtrigs['sen'][lin]]))
        # If the lineage is not senescent and ends with long cycle.
        if np.isnan(gtrigs['sen'][lin]) and is_lcycle[lin, idx_gmax]:
            # The last sequence of nta is not counted.
            nta_max -= 1
        temp = []
        for nta_idx in range(nta_max - 1):
            idxs_min = int(gtrigs['nta'][lin, nta_idx])
            idxs_max = int(gtrigs['nta'][lin, nta_idx + 1])
            temp.append(np.nansum(is_lcycle[lin, idxs_min:idxs_max]))
        if nta_max > 0:
            temp.append(np.nansum(is_lcycle[lin,
                        int(gtrigs['nta'][lin, nta_max-1]):idx_nta_max_end]))
        lcycle_per_seq_counts['nta'][lin, :nta_max] = temp
        # sen.
        if (not np.isnan(gtrigs['sen'][lin])
                or np.isnan(gtrigs['death'][lin])):
            lcycle_per_seq_counts['sen'][lin] = gtrigs['death'][lin] - \
                gtrigs['sen'][lin] + 1
    return lcycle_per_seq_counts


def postreat_experimental_lineages(data, threshold, gen_count_min,
                                   par_multiple_thresholds=None):
    """Extract and return from `data` the cell cycles, from strictly after
    DOX addition (or new conditions if par_multiple_thresholds not None)
    to the end of measurements, of all the lineages longer than
    `gen_count_min`. Besides, compute and return information on each
    lineage (generations at which event triggered and type).

    Parameters
    ----------
    data : ndarray
        Data file having the structure of
        `TelomeraseNegative.mat['OrdtryT528total160831']`.
    threshold : int
        In 10 minutes, threeshold between long cycles (> threeshold) and normal
        cycles (<= threeshold).
    gen_count_min : int
        Lineages counting striclty less than `gen_count_min`
        generations (from after DOX addition) are forgotten from the data.
    par_multiple_thresholds : list
        par_multiple_thresholds[0]: threshold during the new conditions (C2).
        par_multiple_thresholds[1]: time (in 10 min) of change of environment
        s.t. conditions C2 between `init_times` and par_multiple_thresholds[1].

    Returns
    -------
    cycles : dict
        Dictionnary with only one entry 'cycle' that is a 2D array
        (lineage_count, gen_count) with all cycle duration times (in min).
        NB: any lineage shorter than the longest one extended with NaN values.
    gtrigs : dict
        Dictionnary of generations at which an event is triggered, entries st.:
        nta : ndarray
            2D array (lineage_count, nta_count) of generations at which
            non-terminal arrests (nta) are triggered.
            NB: `gtrigs['nta'][l, i]` for the ith arrest of the lth lineage is
                NaN if no such arrest.
        sen : ndarray
            1D array (lineage_count,) of the generations at which senescence
            (in the broad sense of serie of long cycles terminated by either
             death or the end of measurements for lineages that have not died)
            is trigerred, nan if never trigerred.
        death : ndarray
            1D array (lineage_count,) of the generations at which death is
            triggered (i.e. the cell having this generation has died at the end
             of its cycle). Is NaN if the lineage has not died.
    lineage_types : ndarray
        1D array (lineage_count,) st. lineage_types[i] is
        > 0 if the lineage is type A (normal cycles, senescence and death),
        > 1 if it is "type B" (ie. at least 1 long cycle followed by 1 normal),
        > NaN when we cannot say: any lineage (accidentally) dead but without
          arresting (i.e. acc. died after only normal cycles) or alive but that
          cannot be said type B (i.e. with a long cycle followed by normal).
    is_unseen_htypes : NoneType
        None (here only to have same data structure than simulated lineages).
    is_accidental_death : ndarray
        1D array (lineage_count,) st. `is_accidental_death[i]` True if the ith
        lineage has died but ends with a normal cycle, False otherwise.
    NB: all arrays are returned ordered along the lineage axis, by increasing
        lineages length.

    """
    lineage_count = len(data['endlineage'][0])
    lineages = np.arange(lineage_count)

    # Array, for each lineage, of the time of the fisrt generation of interest.
    # Under usual conditions, it is when DOX was added.
    init_times = data['DOXaddition'][0].astype(int)
    # NB: for finalCut data 'DOXaddition' corresponds to Gal addition.
    if not isinstance(par_multiple_thresholds, type(None)):
        threshold_new, time_change = par_multiple_thresholds
    # Extraction of cycles duration times after DOX addition saving some info.
    # > Initialization of lists:
    cycles = []  # `cycles[i]`: array of cycle durations of the ith lineage.
    is_long = []  # `is_long[i]`: bool array indicating if they are long.
    # ... and arrays indicating:
    is_last_long = np.array([]).astype(bool)  # if lineages' last cycle is long
    gen_counts = np.array([])  # lineages lengths (in generation).

    # > Iteration on all lineages.
    for i in lineages:
        # Extraction of times of division/death (except first gen) in ith lin.
        div_times = data['track'][0, i].astype(int)[:, 1]
        birth_times = data['track'][0, i].astype(int)[:, 0]  # Same for birth.
        # We keep only generations born after or at Dox (/Gal) addition.
        is_kept = birth_times >= init_times[i]
        # cycles.append(data['track'][0, i].astype(int)[div_times >
        #                                               init_times[i], :2])
        # And turn "times of division" to cycle duration times.
        cycles.append(div_times[is_kept] - birth_times[is_kept])
        # Update other data concerning the ith lineage.
        if isinstance(par_multiple_thresholds, type(None)):  # Only 1 threshold
            is_long.append(cycles[i] > threshold)
        else:  # Change of environment -> multiple thresholds.
            div_times_kept = div_times[div_times > init_times[i]]
            idx_change = np.argmin(div_times_kept < time_change)
            is_long.append(np.append(cycles[i][:idx_change] > threshold_new,
                                     cycles[i][idx_change:] > threshold))
        if len(cycles[i]) == 0:  # If no cycle after Dox addition.
            is_last_long = np.append(is_last_long, False)
        else:
            is_last_long = np.append(is_last_long, is_long[-1][-1])
        gen_counts = np.append(gen_counts, len(cycles[i]))

    # Removal of lineages too short and ordering of lineages.
    # > Indexes of lineages to keep.
    lineages_too_short = lineages[gen_counts < gen_count_min]
    lineages_kept = np.delete(lineages, lineages_too_short)
    lineage_kept_count = len(lineages_kept)
    # > Ordering of kept lineages by increasing length (i.e. number of gen).
    lineages_kept = lineages_kept[np.argsort(gen_counts[lineages_kept])]
    # > Update of data, also turning lists to array extending by NaN.
    gen_counts = gen_counts[lineages_kept]
    gen_count = int(max(gen_counts))  # Maximal lineage length.
    is_last_long = is_last_long[lineages_kept]
    cycles = np.array([reshape_with_nan(cycles[i], gen_count) for i in
                       lineages_kept])
    is_long = np.array([reshape_with_nan(is_long[i], gen_count) for i in
                        lineages_kept])

    # Computation of generations at which nta or senescence triggered.
    # NB: we add a normal cycle before DOX addition so that long cycles that
    #     are starting a lineage are considered as the 1st long of a sequence.
    is_long_with_1st_normal = np.append(np.zeros((lineage_kept_count, 1)),
                                        is_long, axis=1)
    gtrigs_temp = [np.where(is_long_with_1st_normal[i, 1:] -
                            is_long_with_1st_normal[i, :-1] == 1)[0] for i in
                   range(lineage_kept_count)]
    # Of number of nta(s) per lineage.
    # NB: by default of arrests including senescence and death.
    nta_counts = np.array([len(gtrigs_temp[i]) for i in
                           range(lineage_kept_count)])
    nta_count = max(nta_counts)

    # Extraction of bool array indicating if (orderred kept) lineages died.
    is_dead = data['endlineage'][0][lineages_kept].astype(bool)
    # Computation of lineages' generations at nta, sen or death was triggered.
    # > Initialization with NaN values by default.
    gtrigs = {'nta': np.nan * np.zeros((nta_count, lineage_kept_count)),
              'sen': np.nan * np.zeros(lineage_kept_count),
              'death': np.nan * np.zeros(lineage_kept_count)}
    # > Update for all kept lineages.
    for lin_idx in range(lineage_kept_count):
        # If the lineage has died, update of the gen of death.
        if is_dead[lin_idx]:
            gtrigs['death'][lin_idx] = gen_counts[lin_idx] - 1
        # If the lineage experienced at least 1 arrest.
        if nta_counts[lin_idx] > 0:
            # If its last cycle is long (no matter if the lineage has died)
            if is_last_long[lin_idx]:  # last arrest counted as sen entry.
                gtrigs['sen'][lin_idx] = gtrigs_temp[lin_idx][-1]
                nta_counts[lin_idx] -= 1  # Update of number of nta.
                # If other arrest(s), counted as nta(s).
                if nta_counts[lin_idx] > 0:
                    gtrigs['nta'][:nta_counts[lin_idx], lin_idx] = \
                        gtrigs_temp[lin_idx][:-1]
            # Otherwise the last arrest (and previous) for sure non-terminal.
            else:
                gtrigs['nta'][:nta_counts[lin_idx], lin_idx] =\
                    gtrigs_temp[lin_idx]
    # We reshape `gtrigs['nta']` under rigth form.
    gtrigs['nta'] = np.transpose(gtrigs['nta'])
    # and remove posible columns full of nan.
    is_col_of_nan = np.all(np.isnan(gtrigs['nta']), axis=0)
    gtrigs['nta'] = np.delete(gtrigs['nta'], is_col_of_nan, axis=1)

    # Computation of lineages' type.
    lineage_types = np.nan * np.zeros(lineage_kept_count)
    is_type_a = np.logical_and(is_dead, np.logical_and(nta_counts == 0,
                               ~np.isnan(gtrigs['sen'])))
    is_type_b = nta_counts > 0
    lineage_types[is_type_a] = 0
    lineage_types[is_type_b] = 1
    # Computation of (accidental) death before senescence (dead wo senescence).
    is_not_senescent = np.isnan(gtrigs['sen'])
    is_accidental_deaths = np.logical_and(is_dead, is_not_senescent)
    # Conversion from [10 min] to [min].
    cycles = 10 * cycles
    # And remaining data.
    is_unseen_htypes = None
    lcycle_per_seq_counts = compute_exp_lcycle_counts(cycles, gtrigs, is_long)
    return ({'cycle': cycles}, gtrigs, lineage_types, is_unseen_htypes,
            is_accidental_deaths, lcycle_per_seq_counts)
