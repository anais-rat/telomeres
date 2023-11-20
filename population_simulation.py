#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 11:15:36 2022

@author: arat
"""

# import imp
from copy import deepcopy
import multiprocessing as mp
import numpy as np
import numpy.random as rd
from scipy import stats
import os
import psutil
import time
import warnings

import aux_functions as fct
import aux_keys as ks
import parameters as par
# imp.reload(par)

# printer les rand pour voir si pb kernel pendant parallélisation


def population_init(c_init_s, is_saved):
    """ Generate 'simu_count' initial subpopulations contentrated according to
    'c_init_s' following 'par.DATA_INIT_CHOICE'. Save them if 'is_saved' True.

    Parameters
    ----------
    c_init_s : ndarray
        1D array (simu_count,) of the initial concentrations per subpopulation.
    is_saved : bool
        True to save the output, False otherwise.

    Returns
    -------
    dic_s : dictionary list
        List of dictionaries (with same keys) containing subpopulations initial
        data s.t. dic_s[i]['entry'] is the data 'entry' of the ith population.

    """
    simu_count = len(c_init_s)
    simus = np.arange(simu_count)

    # Computation of subpopulations' cumulative concentration.
    ccum_s = np.array([sum(c_init_s[:i]) for i in range(simu_count + 1)])
    ccum_s = ccum_s.astype(int)

    # Creation of the initial subpopupulations.
    dic_s = [{} for subsimu in simus]
    # > From scratch, denpending on parameters defined in `parameters.py`.
    if par.DATA_INIT_CHOICE == 'new':
        for i in simus:
            # Initial number of cells.
            c_init = c_init_s[i]
            # Lengths generated (depending on `PAR_L_INIT`and `L_INIT_CHOICE`).
            dic_s[i]['lengths'] = fct.draw_cell_lengths(c_init)
            # Cycle duration times (cdt) (depending on `CYCLES_CHOICE`).
            cycles = fct.draw_cycles_A(c_init)
            # Remaining time before death (depending on `DELAY_CHOICE`).
            dic_s[i]['clocks'] = fct.desynchronize(cycles)
            # Other data account for non-sencescent type A generation 0 cells.
            dic_s[i]['ancestors'] = ccum_s[i] + np.arange(c_init)
            zeros = np.zeros(c_init)
            dic_s[i]['nta_counts'] = zeros
            dic_s[i]['generations'] = zeros.astype(int)
            dic_s[i]['sen_counts'] = zeros.astype(int)
        # Saving.
        if is_saved:
            for i in simus:
                np.save(par.DATA_INIT_LOAD_PATH + f'_c{c_init}_{i}.npy',
                        dic_s[i])

    # > From already generated & saved data.
    if par.DATA_INIT_CHOICE == 'load':
        for i in simus:
            dic_s[i] = np.load(par.DATA_INIT_LOAD_PATH + f'_c{c_init}_{i}.npy',
                               allow_pickle='TRUE').item()
            if len(dic_s[i]['types']) != c_init_s[i]:
                raise Exception(f"The loaded data of the {i}th subpopulation "
                                f"does not match the desired concentration "
                                f"{c_init_s[i]} in this subpopulation.")
    return dic_s


def population_evolution(times, time_saved_idxs, ancestor_count, cells_data):
    """ Simulate the evolution of a given population over a given set of times
    (supposed to be 1 day). Includes saturation.

    Parameters
    ----------
    times : ndarray
        1D array (time_count,) containing initial and simulation times (min).
    time_saved_idxs : ndarray
        1D array (time_saved_count,) of integers: the index of `times` at which
        data should be saved.
    ancestor_count : int
        Number of original ancestors in the "original population" (at the
        origin of the population passed in argument).
        NB: when `times[0]` is 0 this is exactly `cell_count` (the "initial
            population" passed in argument is the original one).
    cells_data : dict
        Dictionary gathering the data of each of the `cell_count` of the
        initial population data. Entries ('key') are detailed below.

        > 1D arrays (cell_count,).
        'ancestors' : ndarray
            Indexes of cells' ancestor in the original population.
        'nta_count' : int
            Cells' number of sequence of non-terminal arrest(s) since
            generation 0. For non-senescent cell, it is positive if the cell is
            arrested, negative if it is in normal cycle.
            Ex: 0 for non-senescent type A, 1 for non-senescent type B in a 1st
                seq. of arrests, <0 for non-senescent type B in normal cycle...
        'clocks' : ndarray
            Cells' remaining time until division.
        'generations' : ndarray
            Cells' generation.
        'sen_counts' : ndarray
            Cells' numbers of senescent cycles.

        > 3D array (cell_count, 2, 16).
        'lengths' : ndarray
            Cells' telomere lengths at both extremities of their 16 chromosomes
            lengths[i, 0] one telomeric extremity in the i-th cell.
            lengths[i, 1] and the other one.

    Returns
    -------
    output : dict
        Only one output: a big dictionnary whose entries are detailed below.

        > Time evolution arrays.
        Information on the population computed at every time of 'times'.
        np.nan values returned at times where population is extinct.

        >> 1D arrays (time_saved_count,).
        'evo_lavg_sum' : ndarray
            Temporal evolution of the sum of all cells' average telo lengths.
        'evo_lmin_sum' : ndarray
            Temporal evolution of the sum of all cells' shortest telo length.
        'evo_lmin_max' : ndarray
            Temporal evolution of the longest shortest telomere length.
        'evo_lmin_min' : ndarray
            Temporal evolution of the minimal telomere length.

        >> 2D array (time_saved_count, *).
        'evo_lmode' : ndarray
            Time evolution of the mode in the distribution of telo lengths. At
            time `times[i]`: [i, 0] mode, [i, 1] count of values = to the mode.
        'evo_c_ancs': ndarray
            (* = ancestor_count): evolution of the concentration (= number) of
            cells with respect to their ancestor in the original population.
            Idem with only the number of type B, H or senescent cells...
            See `ks.evo_c_anc_keys`.
        'evo_c_gens' : ndarray
            (* = gen_max + 1): time evolution of the distrib of generations.
            `gen_max` the biggest generation reached during the simulation.
            See `ks.evo_c_gen_keys`.

        > Final population data.
        The entrie of `cells_data` ('ancestors', 'nta_counts', 'clocks',
        'generations', 'sen_counts') updated with population data at final.

        > Population history.
        'sat_time' : float
            Time at which saturation occurs, NaN if saturation is not reached.
        'history_dead' : ndarray
            (dead_cell_count, 4). history_dead[i] is the array (4,) of the
            ancestor, the generation, the time of death in minutes and the type
            of the i-th dead cell.
        'extinction_time': float
            Population extinction time (min) if extinction, math.inf otherwise.
        'evo_lmin_gsen' : dict
            Dictionnary with entries `ks.type_keys` (i.e. 'atype', 'btype', and
            possibly 'mtype') each made of the `time_saved_count` lists, one
            for each time, of the lengths of the telomere that has triggering
            senescence per type.

    Raises
    ------
    Exception
        If the initial population is empty.

    """
    # Initialization of the dictionary to return, with all data.
    d = deepcopy(cells_data)

    # Number of cell(s) in the initial population.
    init_cell_count = len(d['clocks'])
    if init_cell_count == 0:
        raise Exception("Function `population_evolution` does not accept empty"
                        " populations.")

    # Computation of saturation parameters.
    # NB: if SAT_CHOICE[-1] is 'prop', i.e. population saturates when reaching
    #     a proportion of `init_cell_count` at day `len(SAT_CHOICE)`, then
    #     saturation on the following days follows the same saturation rule.
    day = int(times[0] / (24 * 60)) # Current day converting min to day.
    day = min(len(par.SAT_CHOICE) - 1, day) # Day defining the saturation rule.
    c_sat = par.PROP_SAT * init_cell_count

    # Creation of additional population data arrays (init_cell_count,) to speed
    # up the computation of evolution arrays.
    lengths_avg = np.mean(d['lengths'], axis=(1, 2)) # Cells' average telomere.
    lengths_min = np.min(d['lengths'], axis=(1, 2)) # Cells' shortest telomere.

    # Creation of needed lists of keys and time evolution arrays.
    time_count = len(times)
    time_count_saved = len(time_saved_idxs)
    gen_count = int(np.max(d['generations'])) + 1
    for key in ks.evo_c_anc_keys:
        d[key] = np.zeros((time_count, ancestor_count))
    for key in ks.evo_c_gen_keys:
        d[key] = np.zeros((time_count, gen_count))
    for key in ks.evo_l_keys_0:
        d[key] = np.zeros(time_count)
    d['evo_lmode'] = np.zeros((time_count_saved, 2))

    # Initialization of time evolution arrays at the first time.
    c_current = init_cell_count
    temp_anc = fct.datas_to_hist_s(d['ancestors'], d['nta_counts'],
                                   d['sen_counts'], ancestor_count)
    temp_gen = fct.datas_to_hist_s(d['generations'], d['nta_counts'],
                                   d['sen_counts'], gen_count)
    for i in range(len(ks.evo_c_anc_keys)):
        d[ks.evo_c_anc_keys[i]][0] = temp_anc[i]
    for i in range(len(ks.evo_c_gen_keys)):
        d[ks.evo_c_gen_keys[i]][0] = temp_gen[i]
    d['evo_lavg_sum'][0] = np.sum(lengths_avg)
    d['evo_lmin_sum'][0] = np.sum(lengths_min)
    d['evo_lmin_max'][0] = np.max(lengths_min)
    d['evo_lmin_min'][0] = np.min(lengths_min)
    if time_saved_idxs[0] == 0:
        evo_lmin_gsen = {key: [[]] for key in ks.type_keys}
        d['evo_lmode'][0] = np.transpose(stats.mode(d['lengths'].flatten(
                                         ).astype('float')))
        time_saved_idx = 1
    else:
        evo_lmin_gsen = {key: [[]] for key in ks.type_keys}
        time_saved_idx = 0
    # Lists of lmin between two saving times
    lmins_gsen = {key: [] for key in ks.type_keys} 

    # Initialization of population history data.
    sat_time = np.NaN
    history_dead = np.empty((0, 4))

    # Iteration on times.
    for i in range(1, time_count):
        t_temp = times[i-1]
        c_previous = c_current

        if i == time_saved_idxs[time_saved_idx]:
            for key in ks.type_keys:
                evo_lmin_gsen[key].append(list(lmins_gsen[key]))
                lmins_gsen[key] = []

        # We let cells of time `times[i-1]` evolve up to `times[i]`.
        # > Initialization of evolution arrays at time `times[i]`.
        for key in ks.evo_keys_0: # By default identical to previous time.
            d[key][i] = d[key][i-1]
        # > Initialization of indexes of cells died between times `i-1` & `i`.
        dead_idxs = np.array([]).astype(int)
        # > Update of cells' time left to division (from time `i-1` to `i`).
        d['clocks'] = d['clocks'] - (times[i] - times[i-1])
        # > Indexes of cells that have divided/died between times `i-1` & `i`.
        div_idxs = d['clocks'] <= 0
        div_cells = np.arange(c_previous)[div_idxs]
        c_surplus = sum(div_idxs) + c_current - c_sat
        # If saturation was reached in the interval `[times[i-1], times[i])`.
        if ((par.SAT_CHOICE[day] == 'prop' and c_surplus > 0) or
          (par.SAT_CHOICE[day] == 'time') and (times[i] > par.TIMES_SAT[day])):
            # We order divided/died cell indexes by increasing division time.
            div_cells = div_cells[np.argsort(d['clocks'][div_idxs])]

        # > Iteration on divided/died cells between times[i-1] & [i] or t_sat.
        for cell in div_cells:
            # If saturation reached.
            if ((par.SAT_CHOICE[day] == 'prop' and c_current >= c_sat) or
                (par.SAT_CHOICE[day] == 'time') and (t_temp >=
                                                     par.TIMES_SAT[day])):
                sat_time = t_temp
                # Population stops evolving up to the end of `times` and we
                # remove times that do not need to be saved.
                for key in ks.evo_keys_0:
                    d[key][i+1:] = d[key][i]
                    d[key] = d[key][time_saved_idxs]
                d['evo_lmode'][time_saved_idx:] = \
                    d['evo_lmode'][time_saved_idx - 1]
                # Or add empty list for remaining saved times.
                time_left_count = time_count_saved- len(evo_lmin_gsen['atype'])
                for key in ks.type_keys:
                    evo_lmin_gsen[key].extend(time_left_count * [[]])
                d.update({'evo_lmin_gsen': evo_lmin_gsen, 'sat_time': sat_time,
                          'history_dead': history_dead,
                          'extinction_time': np.NaN})
                # Memory used running of 'population_evolution'.
                d['memory'] = psutil.Process(os.getpid()).memory_info().rss
                return d

            # Otherwise, we let the cell divide or die.
            # Strorage of the (positive) time elapsed since division/death.
            delay = - d['clocks'][cell]
            t_temp = times[i] - delay # Update of time.

            # We determine if the cell dies or divides.
            is_dead_accidentally = False
            # > Accidental death (authorized only from generation 1).
            if fct.is_accidentally_dead():
                if d['generations'][cell] > 0:
                    is_dead_accidentally = True
            is_dead = is_dead_accidentally
            # > Natural death (for senescent cells not died accidentally).
            if (not is_dead) and d['sen_counts'][cell] > 0:
                is_dead = fct.is_dead(sen_count=d['sen_counts'][cell])

            # If the cell has died.
            if is_dead:
                # It disappears from the population.
                dead_idxs = np.append(dead_idxs, cell)
                c_current -= 1
                anc, gen = d['ancestors'][cell], d['generations'][cell]
                ar_count = d['nta_counts'][cell]
                d['evo_c_ancs'][i, anc] -= 1
                d['evo_c_gens'][i, gen] -= 1
                if d['sen_counts'][cell] > 0: # Senescent.
                    d['evo_c_sen_ancs'][i, anc] -= 1
                    if ar_count < 0: # Senescent B-type.
                        d['evo_c_B_ancs'][i, anc] -= 1
                        d['evo_c_B_sen_ancs'][i, anc] -= 1
                    elif ar_count > 0: # (Senescent) H-type.
                        d['evo_c_H_ancs'][i, anc] -= 1
                elif ar_count != 0 and is_dead_accidentally: # B-type non-sen
                    d['evo_c_B_ancs'][i, anc] -= 1           #  dead acc.

                d['evo_lavg_sum'][i] -= lengths_avg[cell]
                d['evo_lmin_sum'][i] -= lengths_min[cell]
                cell_data = np.array([anc, gen, t_temp / (60*24), ar_count])
                history_dead = np.append(history_dead, [cell_data], axis=0)

            # Otherwise it divides, the cell is updated as one of its
            # daughters and we create its other daughter.
            else:
                # Update of the generation and generation arrays if new gen.
                d['generations'][cell] += 1
                if d['generations'][cell] >= gen_count:
                    gen_count += 1
                    zero_column = np.zeros((time_count, 1))
                    for key in ks.evo_c_gen_keys:
                        d[key] = np.append(d[key], zero_column, axis=1)

                # Update of `evo_c` arrays that are already known.
                # NB: when not exact (ie nta/sen_counts of daughters not
                #     known yet) we retrieve mothers from the counts, daughters
                #     added latter, when their nta/sen_counts are known.
                c_current += 1 
                d['evo_c_ancs'][i, d['ancestors'][cell]] += 1
                d['evo_c_gens'][i, d['generations'][cell]] += 2
                d['evo_c_gens'][i, d['generations'][cell]-1] -= 1

                # Creation of a new cell (one of the two daughters).
                # NB: default values except for 'ancestors', 'generations'.
                for key in ks.data_keys:
                    d[key] = np.append(d[key], [d[key][cell]], axis=0)

                # Update of daughters' telomere lengths.
                # NB: random repartition of mother cell's breads into
                #     daughters' taking into account coupling.
                r = rd.binomial(1, .5, 16)
                daughter1 = (d['lengths'][-1] - fct.draw_overhang() *
                             np.array([r, 1-r]))
                daughter2 = (d['lengths'][-1] - fct.draw_overhang() *
                             np.array([1-r, r]))
                daughters_lengths = {cell: daughter1, -1: daughter2}
                d['lengths'][cell] = daughter1
                d['lengths'][-1] = daughter2

                # Update of `evo_l*` and `lengths_*` data (exact values)
                #  at time `i`, except `evo_lmin_*[i]` later computed.
                d['evo_lavg_sum'][i] -= lengths_avg[cell]
                d['evo_lmin_sum'][i] -= lengths_min[cell]
                lengths_avg[cell] = np.mean(daughter1)
                lengths_avg = np.append(lengths_avg, np.mean(daughter2))
                lengths_min[cell] = np.min(daughter1)
                lengths_min = np.append(lengths_min, np.min(daughter2))
                d['evo_lavg_sum'][i] += np.sum(lengths_avg[[cell, -1]])
                d['evo_lmin_sum'][i] += np.sum(lengths_min[[cell, -1]])
                lmin_daughters = np.min(lengths_min[[cell, -1]])
                if lmin_daughters < d['evo_lmin_min'][i]:
                    d['evo_lmin_min'][i] = lmin_daughters

                # Update of `evo_c` arrays and `nta_counts` & `sen_counts`
                # daughters' data depending on their mother's state (ie current
                # except for lengths generation) and daughter's lmin (current).
                # If sen mother daugthers' data exact except evo_c, sen_counts.
                if d['sen_counts'][cell] > 0:
                    # Necessarily senescent daughters as well.
                    d['sen_counts'][cell] += 1
                    d['sen_counts'][-1] += 1
                    d['evo_c_sen_ancs'][i, d['ancestors'][cell]] += 1
                    if d['nta_counts'][cell] < 0: # Mother is sen-B.
                        # It necessarily give birth to sen-B daughters.
                        d['evo_c_B_ancs'][i, d['ancestors'][cell]] += 1
                        d['evo_c_B_sen_ancs'][i, d['ancestors'][cell]] += 1
                    elif d['nta_counts'][cell] > 0: # H-type mother.
                        # Type-H (senescent) daughters as well.
                        d['evo_c_H_ancs'][i, d['ancestors'][cell]] += 1

                # Otherwise, the mother was non-senescent, need also to update
                #  `nta_counts` and `sen_counts` daughters' data.
                else:
                    # > If the mother was non-senescent type A.
                    if d['nta_counts'][cell] == 0:
                        # Daughters' state depends on their state (generation
                        #  and/or length). Iteration on daughters.
                        for dau in [cell, -1]:
                            anc = d['ancestors'][dau]
                            gen = d['generations'][dau]
                            lmin = lengths_min[dau]
                            lengths = daughters_lengths[dau]
                            # If senescence is triggered.
                            if fct.is_sen_atype_trig(lmin, lengths):
                                # The daugher enters senescence.
                                d['sen_counts'][dau] = 1
                                d['evo_c_sen_ancs'][i, anc] += 1
                                lmins_gsen['atype'].append(lmin)
                            # Otherwise, if a first arrest is triggered.
                            elif fct.is_nta_trig(gen, lmin, lengths):
                                # It enters a 1st arrest and becomes type B.
                                d['nta_counts'][dau] = 1
                                d['evo_c_B_ancs'][i, anc] += 1
                    # > Otherwise mother was non-sen type B.
                    else:
                        # If mother not arrested type B.
                        if d['nta_counts'][cell] < 0:
                            # Daugthers stay type-B.
                            d['evo_c_B_ancs'][i, d['ancestors'][cell]] += 1
                            for dau in [cell, -1]: # For each daughter test...
                                anc = d['ancestors'][dau]
                                gen = d['generations'][dau]
                                lmin = lengths_min[dau]
                                lengths = daughters_lengths[dau]
                                # ... If senescence is triggered.
                                if fct.is_sen_btype_trig(lmin, lengths):
                                    # The daugher enters sen and stays B.
                                    d['sen_counts'][dau] = 1
                                    d['evo_c_sen_ancs'][i, anc] += 1
                                    d['evo_c_B_sen_ancs'][i, anc] += 1
                                    lmins_gsen['btype'].append(lmin)
                                # ... Elif a new seq of arrest is triggered.
                                elif fct.is_nta_trig(gen, lmin, lengths):
                                    # I stays type-B but enters a new arrest.
                                    ar =  int(1 - d['nta_counts'][dau])
                                    d['nta_counts'][dau] = ar
                        # Otherwise the mother was non-senescent arrested (B).
                        elif par.HYBRID_CHOICE: # If H type taken into account.
                            # Mother retrieve from non-sen B counts.
                            d['evo_c_B_ancs'][i, d['ancestors'][cell]] -= 1
                            # For each daughter we test if sen is triggered.
                            for dau in [cell, -1]:
                                anc = d['ancestors'][dau]
                                gen = d['generations'][dau]
                                lmin = lengths_min[dau]
                                # If triggered, the daugther becomes sen H.
                                if fct.is_sen_btype_trig(lmin,
                                                       daughters_lengths[dau]):
                                    d['sen_counts'][dau] = 1
                                    d['evo_c_sen_ancs'][i, anc] += 1
                                    d['evo_c_H_ancs'][i, anc] += 1
                                    if d['nta_counts'][dau] == 1:
                                        lmins_gsen['mtype'].append(lmin)
                                    else: # NB: different def of htype w lmin!
                                        lmins_gsen['htype'].append(lmin)
                                else: # Otherwise it stays type B.
                                    d['evo_c_B_ancs'][i, anc] += 1
                                    # If it adapts/repairs, it exits arrest.
                                    if fct.is_repaired():
                                        d['nta_counts'][dau] *= - 1
                        else: # Daugthers cannot turn sen bf having exited arr.
                            # They stay type B.
                            d['evo_c_B_ancs'][i, d['ancestors'][cell]] += 1
                            for dau in [cell, -1]:
                                if fct.is_repaired():
                                    d['nta_counts'][dau] *= - 1

                # Update daughters' clock depending on their updated state.
                for dau in [cell, -1]:
                    d['clocks'][dau] = fct.draw_cycle(d['nta_counts'][dau],
                                              d['sen_counts'][dau] > 0) - delay

        # > If at least one cell has died, update of the population data.
        if len(dead_idxs) > 0:
            # If all cells are dead, we return data up to the end of `times`.
            if c_current == 0:
                # Computation of extinction time.
                extinction_time = times[i] + np.max(d['clocks'][dead_idxs])
                # Evolution arrays are set to nan at times `times[i:]` and we
                # remove times that do not need to be saved.
                for key in ks.evo_keys_0:
                    d[key][i:] = np.nan
                    d[key] = d[key][time_saved_idxs]
                d['evo_lmode'][time_saved_idx:] = [np.nan, np.nan]
                # Population data returned empty.
                empty = np.array([]).astype(int)
                for key in ks.data_keys:
                    d[key] = empty 
                d['lengths'] = np.empty((0, 2, 16))
                # We add empty list for remaining times.
                time_left_count = time_count_saved- len(evo_lmin_gsen['atype'])
                for key in ks.type_keys:
                    evo_lmin_gsen[key].extend(time_left_count * [[]])
                d.update({'evo_lmin_gsen': evo_lmin_gsen, 'sat_time': sat_time,
                          'history_dead': history_dead,
                          'extinction_time': extinction_time})
                # Memory used running of 'population_evolution'.
                d['memory'] = psutil.Process(os.getpid()).memory_info().rss
                return d
            # Otherwise, we remove dead cells' data of population data.
            for key in ks.data_keys:
                d[key] = np.delete(d[key], dead_idxs, axis=0)
            lengths_avg = np.delete(lengths_avg, dead_idxs)
            lengths_min = np.delete(lengths_min, dead_idxs)
            # And update the `evo_*min_length` that may have changed.
            d['evo_lmin_max'][i] = np.max(lengths_min)
            d['evo_lmin_min'][i] = np.min(lengths_min)

        # > If at least one division and no death.
        elif c_previous < c_current:
            # Update of `d['evo_lmin_max']` that may have changed.
            d['evo_lmin_max'][i] = np.max(lengths_min)

        if i == time_saved_idxs[time_saved_idx]:
            d['evo_lmode'][time_saved_idx] = np.transpose(
                stats.mode(d['lengths'].flatten().astype('float')))
            time_saved_idx += 1

    #     # ---------------------------------------------------------------------
    #     # Uncomment only to test counts are exact.
    #     # ----------------------------------------
    #     A_count = len(d['nta_counts'][d['nta_counts'] == 0])
    #     neg_count = len(d['nta_counts'][d['nta_counts'] < 0])
    #     pos_count = len(d['nta_counts'][d['nta_counts'] > 0])
    #     sen_A = np.sum(d['sen_counts'][d['nta_counts'] == 0] > 0)
    #     sen_B = np.sum(d['sen_counts'][d['nta_counts'] < 0] > 0)
    #     sen_H = np.sum(d['sen_counts'][d['nta_counts'] > 0] > 0)

    #     if par.HYBRID_CHOICE:
    #         if sen_H != np.sum(d['evo_c_H_ancs'][i]):
    #             print('Error: H count failed. Time index: ', i,
    #                   '\n sen_H', sen_H, "np.sum(d['evo_c_H_ancs'][i])",
    #                   np.sum(d['evo_c_H_ancs'][i]))
    #             return
    #         if pos_count + neg_count - sen_H != np.sum(d['evo_c_B_ancs'][i]):
    #             print('Error: B type count failed. Time index: ', i,
    #                   '\n pos_count + neg_count', pos_count + neg_count,
    #                   "np.sum(d['evo_c_B_ancs'][i])",
    #                   np.sum(d['evo_c_B_ancs'][i]),
    #                   "sen_H", sen_H)
    #             return
    #     else:
    #         if sen_H != 0:
    #             print('Error: there should not be senescent cells with '
    #                   'positive arrest count. Time index: ', i,
    #                   '\n sen_H', sen_H)
    #             return
    #         if pos_count + neg_count != np.sum(d['evo_c_B_ancs'][i]):
    #             print('Error: B type count failed. Time index: ', i,
    #                   '\n pos_count + neg_count', pos_count + neg_count,
    #                   "np.sum(d['evo_c_B_ancs'][i])",
    #                   np.sum(d['evo_c_B_ancs'][i]))
    #             return
    
    #     if (A_count + pos_count + neg_count) != np.sum(d['evo_c_ancs'][i]):
    #         print('Error: count on all cells failed. Time index: ', i,
    #               "\n len(d['nta_counts'])", len(d['nta_counts']),
    #               "np.sum(d['evo_c_ancs'][i])", np.sum(d['evo_c_ancs'][i]))
    #         return
    #     if (sen_A + sen_B + sen_H) != np.sum(d['evo_c_sen_ancs'][i]):
    #         print('Error: senescent count failed. Time index: ', i,
    #               "\n np.sum(d['sen_counts'] > 0)",
    #                   np.sum(d['sen_counts'] > 0),
    #               "np.sum(d['evo_c_sen_ancs'][i])",
    #               np.sum(d['evo_c_sen_ancs'][i]))
    #         return
    #     if sen_B != np.sum(d['evo_c_B_sen_ancs'][i]):
    #         print('Error: sen type B count failed. Time index: ', i,
    #               '\n sen_B', sen_B, "np.sum(d['evo_c_B_sen_ancs'][i]",
    #               np.sum(d['evo_c_B_sen_ancs'][i]))
    #         return

    # evo_c_from_anc = fct.nansum(d['evo_c_ancs'], axis=1)
    # evo_c_from_gen = fct.nansum(d['evo_c_gens'], axis=1)
    
    # print("Are 'evo_c_from_anc' and 'evo_c_from_gen' equal?",
    #       np.argmin(evo_c_from_anc == evo_c_from_gen), 
    #       (evo_c_from_anc[~np.isnan(evo_c_from_anc)] == 
    #        evo_c_from_gen[~np.isnan(evo_c_from_gen)]).all())
    # # ------------------------------------------------------------------------

    # End of `times` reached without saturation or extinction.
    # Outputs returned.
    d.update({'evo_lmin_gsen': evo_lmin_gsen, 'sat_time': sat_time,
              'history_dead': history_dead, 'extinction_time': np.NaN})
    # Memory used running of 'population_evolution'.
    d['memory'] = psutil.Process(os.getpid()).memory_info().rss
    # We remove times that do not need to be saved.
    for key in ks.evo_keys_0:
        d[key] = d[key][time_saved_idxs]
    return d


def gather_evo_and_dilute(output_s, c_dilution, para_count,
                          gen_count_previous):
    """ The evolutions of 'subpop_count' subpopulations that have been run on
    the same time array 'times' through "population_evolution". This function:
    > Gathers time evolution arrays of subpopulations and return the evolution
       of the overall population on 'times'.
    > Randomly dilutes the whole population at last time down to 'c_dilution'
       and return the data of the diluted population in an appropriate
       subpopulation format (with as much subpopulations of concentration
       'c_dilution // para_count' as possible in the limit of 'para_count'
       subpopulations).

    Parameters
    ----------
    output_s : tuple
        Tuple of 'subpop_count' output_s (of type 'dict') returned by
        "population_evolution" (run with times argument 'times').
        NB: 'tuple' objects are not modificable but 'list' are.
    times : ndarray
        1D array (time_count,) of simulated times in minute.
    ancestor_count : int
        Number of original ancestors / cells composing the population at time
        0 (not necessarily 'times[0]') (generally identical to 'c_dilution').
    c_dilution : int
        Concentration at which dilute the whole population.
    gen_max_previous : int
        The biggest generation reached since time 0, will be taken as minimal
        number of rows for the returned 'gen_distributions'.

    Returns
    -------
    out : dict
        See below and "population_evolution" description for entries detail.

        > Subpopulations after division.
    c_init_s : ndarray
        1D array (subpop_after_dil_count,) concentrations in the
        new subpopulations (created after dilution).

        > Time evolution arrays of the overall population.
        >> From summing on subpopulations evo arrays.
            'evo_c_ancs', 'evo_c_B_ancs', 'evo_c_B_ancs', 'evo_c', 'evo_c_B',
            'evo_c_H', 'evo_c_gens', 'evo_lavg_sum', 'evo_lmin_sum'.
        >> Taking the minimum / maximum on all subsimulations.
            'evo_lmin_min', 'evo_lmin_max'.
        >> Or avering (with good weight) on subpopulations. 'evo_lmode'.

        > Population data after dilution in subpopulation format, i.e.:
          '*[i]' for data '*' in the ith subpopulation after dilution.
              'lengths_s', 'ancestors_s', 'clocks_s', 'generations_s',
              l'ong_cycle_counts_s', 'sen_counts_s', 'types_s'.

        > Population history.
    sat_time : float
        Average (on subpop that have saturated) of the time of saturation.
        NaN if no subpop has reached saturation.
    sat_prop : float
        Proportion of subpopulations that have saturated.
    history_dead : ndarray
        Concatenation of subpopulations' histories (same format).
    evo_lmin_gsen : dict
        Concatenation of subpopulations' `evo_lmin_gen` by entry ('atype',
        'btype' possibly 'htype'), each being a list of "shape"
        `(day_count, time_saved_per_day_count, *)`.

    """
    # Creation of arrays of subpopulations' initial and final concentrations.
    subpop_count = len(output_s)
    subpops = np.arange(subpop_count)
    c_init_s = [fct.nansum(output_s[i]['evo_c_ancs'][0]) for i in subpops]
    c_final_s = [fct.nansum(output_s[i]['evo_c_ancs'][-1]) for i  in subpops]
    
    # Computation of `evo_c_gens` length (subpopulations' generation
    #  distributions will be reshaped under same shape).
    gen_counts = [len(output_s[i]['evo_c_gens'][0]) for i in subpops]
    gen_count = max(gen_count_previous, max(gen_counts))

    # Computation of population's evolution arrays with those of subpopulations
    out = {}
    # > Sum on subpopulations (with extention by zeros or nan).
    for key in ks.evo_ancs_n_sum_keys:
        out[key] = fct.nansum([output_s[i][key] for i in subpops], 0)
    for key in ks.evo_c_gen_keys:
        out[key] = fct.nansum([fct.reshape2D_along1_with_0_or_NaN(
                             output_s[i][key], gen_count) for i in subpops], 0)
    # > Minimum / maximum on all subsimulations: `evo_lmin_min`/`evo_lmin_max`.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        out['evo_lmin_max'] = np.nanmax([output_s[i]['evo_lmin_max'] for i in
                                         subpops], axis=0)
        out['evo_lmin_min'] = np.nanmin([output_s[i]['evo_lmin_min'] for i in
                                         subpops], axis=0)
    # > Weighted average on all subsimulations: `evo_lmode`.
        counts = [output_s[i]['evo_lmode'][:, 1] for i in subpops]
        weights = counts / fct.nansum(counts, 0)
        out['evo_lmode'] = fct.nansum([output_s[i]['evo_lmode'][:, 0] for i in
                                       subpops] * weights, axis=0)

    # Computation of population's history data:
    # > Concatenation of subpopulations' data.
    out['history_dead'] = np.concatenate([output_s[i]['history_dead'] for i in
                                          subpops], axis=0)
    out['evo_lmin_gsen'] = {key: [output_s[i]['evo_lmin_gsen'][key] for i in
                                  subpops] for key in ks.type_keys}
    # > Average on saturated subpopulations.
    # Computation of the number of subpop that have saturated.
    sat_count = np.sum(~np.isnan([output_s[i]['sat_time'] for i in subpops]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        out['sat_time'] = np.nanmean([output_s[i]['sat_time'] for i in
                                      subpops], axis=0)
    # if sat_count == 0: # If no saturated subsim, the saturation time is NaN.
    #     out['sat_time'] = np.nan
    # else: # Otherwise it is the average saturation time (on saturated subsims).
    #     out['sat_time'] = np.nansum([output_s[i]['sat_time'] for i in
    #                                   subpops]) / sat_count
    out['sat_prop'] = sat_count / subpop_count

    # Concentration before dilution.
    c_final = int(np.nansum(c_final_s)) 
    
    # If the whole population is dead.
    if c_final == 0:
        # The population data returned is empty.
        if len(output_s) == 1: # If only one subsimulation.
            # We compute the memory used running of 'gather_evo_and_dilute'.
            current_memory = psutil.Process(os.getpid()).memory_info().rss
            # And return the maximum with the memory used to run the subsimu.
            out['memory'] = max(current_memory, output_s[0]['memory'])
        return out, [], []

    # Otherwise, we dilute if necessary and keep data in subpop format of only
    # remaining cells only.
    # > Definition of the subpop format after dilution and selection of kept
    #   cells (in pop format).
    c_subpop = c_dilution // para_count # Expected concentration for subpops.
    # > If the population at last time counts more than `c_dilution` cells.
    if c_final >= c_dilution:
        # Classical subpopulation format.
        subpop_count = para_count
        c_init_s = c_subpop * np.ones(para_count)
        c_init_s[-1] += c_dilution % para_count
        # Dilution donw to `c_dilution` choosing cells to keep index uniformly.
        kept_cell_idxs = rd.choice(c_final, c_dilution,
                                                 replace=False)
    # > Otherwise, no dilution but subpop format possibly not classical.
    else:
        # We redefine the repartion in subsimulations for these cells.
        # NB: at most `para_count` subsimulations with a maximum of them
        #     at concentration `c_subpop`.
        subpop_count = min(c_final // c_subpop, para_count)
        c_left = c_final % c_subpop
        c_init_s = c_subpop * np.ones(subpop_count)
        if c_left != 0:
            if subpop_count == para_count:
                c_init_s[-1] += c_left
            else:
                subpop_count += 1
                c_init_s = np.append(c_init_s, c_left)
        kept_cell_idxs = np.arange(c_final)
    # Conversion of indexes of kept cells from pop to subpop format.
    idxs = fct.pop_to_subpop_format(kept_cell_idxs, c_final_s)

    # Weither the (non-empty) population has been diluted or not we compute its
    # data into subpopulation format.
    ccum_s = np.array([sum(c_init_s[:i]) for i in range(subpop_count + 1)])
    ccum_s = ccum_s.astype(int) # Cumulative concent. in the subpops after dil.
    # Initialization.
    data_s = [{} for subsimu in range(subpop_count)]
    # Iteration on the number of subpopulation to create.
    for s in range(subpop_count):
        rg = np.arange(ccum_s[s], ccum_s[s + 1])
        for key in ks.data_keys:
            data_s[s][key] = np.array([output_s[idxs[i, 0]][key][idxs[i, 1]]
                                       for i in rg])

    if len(output_s) == 1: # If only one subsimulation, we add memory data.
        current_memory = psutil.Process(os.getpid()).memory_info().rss
        out['memory'] = max(current_memory, output_s[0]['memory'])
    return  out, c_init_s, data_s


def simu_parallel(times, time_saved_idxs, dil_idxs, para_count, c_init):
    """ Simulates the evolution of a population initially concentrated at
    'c_init' through the simulation of 'para_count' subpopulations run in
    parallel between each dilution and return the "population history", namely:
        > "Time evolution arrays" (time_count,): the temporal evolution of ...
        - c_tot: the total concentration of cell (sum of the subsimulations)
        - Gen_tot: (n_t, nb_gen_max) the generation distribution
        - evo_lavg: the average telomere length in the population
            (ponderated average of subsimu)
        - evo_lmin_avg: the average shortest telome length among cells (idem)
        - evo_lmin_max:
        - evo_lmin_min: the shortest telomere length among all telomeres
            (the shortest of all subsimu)
        - propB_tot: proportion of B-type cell in the whole population
        - evo_lmin_gsen:
        > "saturation data" [array(1,nb_d)]: 
        >> sat_idxs: sat_idxs[i] index of the time times[sat_idxs[i]] at which
            saturation is reached on day `i` (average among saturated subsimus)
            math.inf when no subsimulation has reached saturation at day 'i'.
        >> sat_time: sat_time[i] estimation (assuming exponential growth) of
            the time at which... (idem)
        >> sat_prop: sat_prop[i] proportion of submiluations that have
            reached saturation at day `i`.
        > "history of dead cells" 'history_dead' [array(nb_dead_cell,4)]
        (see population_evolution) times in days
    Plus, we return the initial population data: 'Data_init'
    [tuple of arrays [(2,16*c_init), (1,c_init)]]

    Parameters
    ----------
    times : ndarray
        1D array (1,time_count) of the computation times (in min) of the whole
        experiment.
    dil_idxs : ndarray
        1D array of ORDERED integers: indexes i such that times[i] is a
        time of dilution.
    para_count : int
        Maximal number of paralelizations.
    c_init : int
        Initial concentration of the whole population.

    Returns
    -------

    Notations
    ---------
    In the following, two ways to store the data of the whole population
    > 'population format': usual format, with arrays.
    > 'subpopulation format': list of arrays, each element of the list 
        corresponding to the data (in usual array format) of a subsimulation.
        We denote _s subpopulation format.

    Warning
    -------
    The population entered in argument must be non-empty.

    """
    start = time.time()
            
    # Computation of subpopulations' concentration in c_init_s (para_count,).
    # The `c_init` initial cells are reparted as follows:
    #  > `para_count-1` subpopulations with same concentration `c_init_subpop`.
    #  > One of `c_init_subpop+c_left` cells, c_left >= 0 as small as possible.
    simu_count = para_count
    c_init_subpop = c_init // simu_count
    c_init_s = c_init_subpop * np.ones(simu_count)
    c_init_s[-1] += c_init % simu_count
    c_init_s = c_init_s.astype(int)

    # Initialization of usefull quantities.
    d = {} # Dictionary of all data to return.
    c_current = c_init # Total cell number at the begining of current day.
    gen_count = 1 # Number of current or past generations in the whole pop.
    day = 0
    day_count = len(dil_idxs) + 1
    if para_count == 1: # If no para, we initialize max of memory used so far.
        memory_max = psutil.Process(os.getpid()).memory_info().rss
    tsaved_idxs = time_saved_idxs.copy()
    evo_keys = ks.evo_keys.copy()

    # Creation of the initial popupulation in subpopulation format.
    data_s = population_init(c_init_s, False)

    # Storage of a part of initial population data.
    d['day_init_data'] = {}
    for key in ks.data_stored_keys:
        d['day_init_data'][key] = [np.concatenate([data_s[s][key] for s in
                                                   range(simu_count)])]

    # At every time `times[dil_idxs[i]]` of dilution, as long as the population
    # is not extinct if TO_EXTINCTION_CHOICE is True.
    while ((par.TO_EXTINCTION_CHOICE and not(np.isnan(c_current))) or
           (not(par.TO_EXTINCTION_CHOICE) and day < day_count)):

        print('Day n°', day + 1, 'c_last', c_current)

        # We let subpopulations evolve on parallel on `times_temp`, i.e. up to:
        # > If it remains at least one dilution: the next time of dilution.
        if day == 0:
            times_temp = times[:dil_idxs[0] + 1]
        elif day + 1 < day_count:
            times_temp = times[dil_idxs[day-1]:dil_idxs[day] + 1]
        # Otherwise, up to the end of 'times'.
        else:
            times_temp = times[-par.TIMES_PER_DAY_COUNT-1:]
        tsaved_idxs_temp = tsaved_idxs[day] - tsaved_idxs[day][0]
        tsaved_idxs_temp = np.append(tsaved_idxs_temp, len(times_temp) - 1)

        # > If para_count is 1, no parallelization.
        if para_count == 1:
            output_s = [population_evolution(times_temp, tsaved_idxs_temp,
                                             c_init, data_s[0])]
        # > Otherwise, initialization of the parallelization.
        else:
            pool = mp.Pool(simu_count)
            pool_s = [pool.apply_async(population_evolution, args=(times_temp,
                      tsaved_idxs_temp, c_init, data_s[i])) for i
                      in range(simu_count)]
            # > Results retrieval from pool_s (list of pool.ApplyResult obj).
            output_s = [r.get() for r in pool_s]
            # > We prevent the current process to put more data on the queue.
            pool.close()
            # > Execution of next line postponed until processes in queue done.
            pool.join()
        # Computation of population's evolution and history data on `t_temp`
        # and dilution in subpop format of the population of time `t_temp[-1]`.
        temp, c_init_s, data_s = gather_evo_and_dilute(output_s, c_init,
                                                       para_count, gen_count)
        # Update of concentration just after dilution
        c_current = fct.nansum(c_init_s)

        # If no parallelization, we update the max of the memory used so far.
        if para_count == 1:
            memory_max = max(memory_max, temp['memory'])

        # Upate of population's evolution and history data from beginning.
        if day == 0:
            gen_count  = len(temp['evo_c_gens'][0])
            d.update(temp)
            for key in ks.sat_keys:
                d[key] = np.array([d[key]])
        else:
            # > Reshape 'evo_gen' arrays if new generations have appeared.
            gen_count_new = len(temp['evo_c_gens'][0])
            if gen_count < gen_count_new:
                for key in ks.evo_c_gen_keys:
                    d[key] = fct.reshape2D_along1_with_0_or_NaN(d[key],
                                                                gen_count_new)
                gen_count = gen_count_new
            # > Update `evo_*` arrays.
            for key in evo_keys:
                # NB: time before dil (last of evo_*) replaced by the one just
                #     after dilution (1st of *_temp).
                d[key] = np.append(d[key][:-1], temp[key], axis=0)
            for key in ks.type_keys:
                d['evo_lmin_gsen'][key].extend(temp['evo_lmin_gsen'][key])
            #  > Update history data.
            for key in ks.sat_keys:
                d[key] = np.append(d[key], temp[key])
            d['history_dead'] = np.append(d['history_dead'],
                                          temp['history_dead'], axis=0)

        # If the whole population has died on `t_temp`.
        if np.isnan(c_current):
            # Computation of `extinction_time`.
            d['extinction_time'] = np.nanmax([output_s[sim]['extinction_time']
                                       for sim in range(simu_count)]) / (60*24)
            # Convertion from minute to day.
            d['sat_time'] = d['sat_time'] / (60 * 24)

            time_idxs = np.concatenate([tsaved_idxs[d] for d in
                                        range(day_count)])
            time_idxs = np.append(time_idxs, tsaved_idxs[day_count-1][-1] + 1)

            # If computation ends at the end of the day after extinction.
            if par.TO_EXTINCTION_CHOICE is True:
                time_idxs = np.concatenate([tsaved_idxs[d] for d in 
                                            range(day + 1)])
                time_idxs = np.append(time_idxs, tsaved_idxs[day][-1] + 1)
                # `evo_*` arrays will be returned on `times[:dil_idxs[day]+1]`.
            # Otherwise it ends at the end of `times`.
            else:
                # `evo_*` arrays are set to nan for all remaining saved times.
                time_left_count = len(time_idxs) - len(d['evo_c_ancs'])
                nan_arr = np.nan * np.zeros(time_left_count)
                for key in ks.evo_l_keys: # 1D evo arrays.
                    d[key] = np.append(d[key], nan_arr)
                nan_arr = np.nan * np.zeros((time_left_count, c_init))
                for key in ks.evo_c_anc_keys:
                    d[key] = np.append(d[key], nan_arr, axis=0)
                nan_arr = np.nan * np.zeros((time_left_count, gen_count))
                for key in ks.evo_c_gen_keys:
                    d[key] = np.append(d[key], nan_arr, axis=0)
                day_left_count = day_count - (day + 1)
                for key in ks.type_keys:
                    d['evo_lmin_gsen'][key].extend(day_left_count * [[]])
            # If no parallelization, we compute the max of the memory used
            # during computation.
            if para_count == 1:
                memory = psutil.Process(os.getpid()).memory_info().rss
                d['memory'] = max(memory_max, memory) / 8388608
            d['times'] = times[time_idxs] / (60 * 24)
            # Add computation times to data.
            end = time.time()
            time_elapsed =  (end - start) / 60 
            d['computation_time'] = time_elapsed
            return d

        # Update of the number of subsimulations to run next.
        simu_count = len(c_init_s)
        
        # Storage of a partial population data after dilution.
        for key in ks.data_stored_keys:
            d['day_init_data'][key].append(np.concatenate([data_s[s][key]
                                           for s in range(simu_count)]))
        # If 'TO_EXTINCTION_CHOICE' True and we've reached the end of `times`.
        if (par.TO_EXTINCTION_CHOICE and (day + 1 >= day_count)):
            # We add one day, `times` and `tsaved_idxs` extended.
            first_idx = tsaved_idxs[day][-1] + 1
            tsaved_idxs[day + 1] = np.arange(first_idx, first_idx +
                                             par.TIMES_PER_DAY_COUNT + 1,
                      par.TIMES_PER_DAY_COUNT // par.TIMES_SAVED_PER_DAY_COUNT)
            tsaved_idxs[day + 1][-1] -= 1

            times = np.append(times[:-1], times[:dil_idxs[0]+1] + day *
                              times[dil_idxs[0]])
        day += 1

    d['sat_time'] = d['sat_time'] / (60 * 24)

    time_idxs = np.concatenate([tsaved_idxs[d] for d in range(day)])
    time_idxs = np.append(time_idxs, tsaved_idxs[day - 1][-1] + 1)
    d['times'] = times[time_idxs] / (60 * 24)

    # Computation of the extinction time.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        d['extinction_time'] = np.nanmax([output_s[sim]['extinction_time'] for
                                          sim in range(simu_count)]) / (60*24)

    # If no parallelization, we compute the max of the memory used.
    if para_count == 1:
        memory = psutil.Process(os.getpid()).memory_info().rss
        d['memory'] = max(memory_max, memory) / 8388608

    # Add computation times to data.
    end = time.time()
    time_elapsed =  (end - start) / 60
    d['computation_time'] = time_elapsed
    return d
