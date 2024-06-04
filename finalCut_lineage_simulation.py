#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 11:39:21 2024

@author: arat
"""

from copy import deepcopy
# import imp
import glob
import math
import multiprocessing as mp
import numpy as np
import numpy.random as rd
import os
import scipy.io as sio
import warnings

import aux_figures_properties as fp
# imp.reload(fp)
import aux_functions as fct
import aux_write_paths as wp
import parameters as par
import lineage_simulation as sim
# imp.reload(par)


def simulate_lineage_evolution_w_cut(lineage_idx,
                                     is_htype_seen=True, parameters=par.PAR,
                                     p_exit=par.P_EXIT, par_finalCut=None):
    """ Simulate the evolution of one lineage from after addition of the DOX to
    death (the cell before DOX addition is type A non-senescent with telomere
    lengths drawn from telomerase-positive distribution at equilibrium).

    Parameters
    ----------
    lineage_idx : int
        If data of the first cell need to be loaded, indicates the index of the
        cell to load among all loaded cell.
    is_htype_seen : bool, optional
        If set to False, the generation of the last nta of type H lineages is
        considered as the generation of the onset of the senescence.
        Plus, H-type lineages will not be classified as H but as either A-type
        or B-type lineages (depending on weither or not they experienced more
        than one nta). The fact that it was H is stored in `is_unseen_htype`.
    parameters : list, optional
        0: The parameters [a, b] of the law of the onset of the 1st arrest.
        1: The parameters [a, b, l_min] of the law of the onset of sen for A&B
        2: The parameters [ltrans, l0, l1] of the law of transfo of `l_init`.
        If not indicated, the parameters of the script `parameters` are taken.
    p_exit : list, optional
        The parameters of death and exit of a sequence of arrest if different
        from par.P_EXIT:
        0: rate of accidental death (p_accident / par.P_ACCIDENTAL_DEATH)
        1: terminal arrest (p_death / par.P_GEO_SEN)
        2: non-terminal arrest (p_repair / par.P_GEO_NTA)
    par_finalCut : ndarray, optional
        Default is None which corresponds to usual initial telomere lengths.
        Otherwise, the initial telomere lengths are drawn s.t. to mimick final
        cut experiment: one telomere (uniformly chosen) is set to
        `par_finalCut[0]` bp after a random number of generations (geometric
        law of parameter `1 / par_finalCut[1]`) with proba `1-par_finalCut[2]`.

    Returns
    -------
    evo : dict
        Dictionnary of evolution arrays, with entry:
        cycle : ndarray
            1D array (gen_count,) of cells' cycle times (min) over generations.
        lengths : ndarray
            3D array (gen_count, 2, 16) of cell's telomere lengths over gens.
        lavg : ndarray
            1D array (gen_count,) of cells' average telo length over gens.
        lmin : ndarray
            1D array (gen_count,) of cells' shortest telo length over gens.
    gtrigs : dict
        Dictionnary of generations at which an event is triggered, entries st.:
        nta : ndarray
            1D array (nta_count,) of generations at which non-terminal arrests
            (nta) are triggered, NaN if no such arrest.
        senescence : int or NaN
            Generation at which senescence is triggered, nan if accidently
            dead before the onset of senescence.
        death : int
            Generation at which death is triggered.
    lineage_type : str
        Type of the lineage (0 for A type, 1 for B or np.NaN for H).
    is_unseen_htype : bool or NoneType
        If no H-type (i.e. `par.HYBRID_CHOICE` is False) or `is_htype_seen` is
        True, always None, otherwise True if the lineage was H but classified
        as A or B and False if the lineage was A or B type.
        (NB: NaN in the case of experimental lineages for unknown).
    is_accidental_death : bool
        True if the lineage died accidentally, False otherwise.

    """
    par_nta, par_sen, par_l_init = parameters
    is_unseen_htype = False
    if is_htype_seen or (not par.HYBRID_CHOICE):
        is_unseen_htype = None

    # Initialization of <evo_*> arrays (t = 0) with the data of the first cell:
    # non-sencescent type A with generation -1 (s.t the 1st cell born under DOX
    # has generation 0). At t=0: Dox addition under raffinose.
    t_current = 0
    is_galactose = False
    is_accidental_death = False
    is_senescent = False
    nta_count = 0
    generation = -1
    evo_cycle = fct.draw_cycles_atype(1)
    evo_lengths = fct.draw_cells_lengths(1, par_l_init)

    # Cut conditions.
    # Telomere `[t1, t2]` will be cut `gen_cut` generations after Gal addition
    gen_cut = math.inf  # with probability `fct.is_cut_escaped`.
    is_cut_escaped = fct.is_cut_escaped(1, proba=par_finalCut[2])[0]
    if not is_cut_escaped:
        gen_delay_cut = fct.draw_delays_cut(1, avg_delay=par_finalCut[1])[0]
        t2 = rd.randint(par.CHROMOSOME_COUNT) # Index of the chromosome cut.
        t1 = rd.randint(2)  # Index of the chrosome extremity cut.

    idx_dox, idxf_gal, idxf_raf = par_finalCut[3]
    t_gal = (idxf_gal - idx_dox) * 10  # [min] s.t. 1 frame every 10 min.
    t_raf = (idxf_raf - idx_dox) * 10

    evo_lavg = [np.mean(evo_lengths)]
    evo_lmin = [np.min(evo_lengths)]
    gtrigs = {'nta': np.array([]), 'sen': np.NaN}
    lcycle_per_seq_count = {'nta': np.array([]), 'sen': np.nan}

    # While the lineage is not extinct.
    lineage_is_alive = True
    while lineage_is_alive:
        t_current += evo_cycle[-1] # Time just before division of current cell.
        if t_current > t_gal:
            if is_galactose == False:  # If the cell is 1st to experience gal.
                # `is_galactose` still False s.t. its cdt drawn as if no gal.
                gen_cut = generation + gen_delay_cut
            else:
                is_galactose = True
        elif t_current > t_raf:
            is_galactose == False

        # If the current cell dies accidentally, we store it with its state
        #    (senescent or not) and generation of death, and the lineage dies.
        # Plus we update the lineage data depending on `is_unseen_acc`.
        if (generation >= 0 and fct.is_accidentally_dead(p_exit[0])):
            is_accidental_death = True
            gtrigs['death'] = generation
            lineage_is_alive = False
            # If H-type accounted but not recognized and the lineage is not H.
            if is_unseen_htype is False:
                # If the lineage was arrested non-senescent at its death.
                if nta_count > 0:
                    is_senescent = True
                    # Then its last arrest is recognized as a terminal arrest.
                    gtrigs['sen'] = gtrigs['nta'][-1]
                    gtrigs['nta'] = gtrigs['nta'][:-1]
                    nta_count = - nta_count + 1
                    lcycle_per_seq_count['sen'] = \
                        lcycle_per_seq_count['nta'][-1]
                    lcycle_per_seq_count['nta'] = \
                            lcycle_per_seq_count['nta'][:-1]

        # If it is senescent we test if it dies.
        elif is_senescent and fct.is_dead():
            lineage_is_alive = False # If so the lineage extincts.
            gtrigs['death'] =  generation # We strore the gen of death.

        # Otherwise it divides, we add one generation and create the next cell.
        else:
            generation += 1
            # Extend evolution arrays at the new generation w default values.
            evo_lengths = np.append(evo_lengths, [evo_lengths[-1]], axis=0)
            evo_cycle = np.append(evo_cycle, evo_cycle[-1])

            # Computation of telomere lengths following the shortening model.
            loss = rd.RandomState().binomial(1, .5 ,16)
            evo_lengths[-1] -= fct.draw_overhang() * np.array([loss, 1 - loss])
            if generation == gen_cut:  # With possible cut.
                evo_lengths[-1][t1, t2]  = par_finalCut[0]
            # Update of other length-related evolution arrays.
            evo_lavg = np.append(evo_lavg, np.mean(evo_lengths[-1]))
            evo_lmin = np.append(evo_lmin, np.min(evo_lengths[-1]))

            # Update of other new-born cell's data depending its mother's data
            # (current or previous data) and its telomere lengths.
            # > If non-senescent mother.
            if not is_senescent:
                # If the mother is (non-senescent) type A.
                if nta_count == 0:
                    # If senescence is triggered, the cell enters senescence.
                    if fct.is_sen_atype_trig(evo_lmin[-1], evo_lengths[-1],
                                             par_sen[0]):
                        is_senescent = True
                        gtrigs['sen'] = generation
                        lcycle_per_seq_count['sen'] = 1
                    # Otherwise, if a 1st arrest triggered, it becomes type B.
                    elif fct.is_nta_trig(generation, evo_lmin[-1],
                                         evo_lengths[-1], par_nta):
                        nta_count = 1
                        gtrigs['nta'] = np.array([generation])
                        # 1st sequence of nta.
                        lcycle_per_seq_count['nta'] = np.append(
                            lcycle_per_seq_count['nta'], 1)
                # Otherwise mother was (non-senescent) type B.
                elif nta_count < 0: # > If not arrested type B.
                    # If senescence is triggered, the cell enters sen.
                    if fct.is_sen_btype_trig(evo_lmin[-1], evo_lengths[-1],
                                             par_sen[1]):
                        is_senescent = True
                        gtrigs['sen'] = generation
                        lcycle_per_seq_count['sen'] = 1
                    # Elif new arrest triggered, enters a new arrest.
                    elif fct.is_nta_trig(generation, evo_lmin[-1],
                                         evo_lengths[-1], par_nta):
                        nta_count = 1 - nta_count
                        gtrigs['nta'] = np.append(gtrigs['nta'],
                                                  generation)
                        # New sequence of nta.
                        lcycle_per_seq_count['nta'] = np.append(
                            lcycle_per_seq_count['nta'], 1)
                else: # > Otherwise mother was (non-senescent) arrested B.
                    # If H type taken into account, cell can turn sen (H).
                    if par.HYBRID_CHOICE and fct.is_sen_btype_trig(
                            evo_lmin[-1], evo_lengths[-1], par_sen[1]):
                        is_senescent = True
                        if is_htype_seen:
                            gtrigs['sen'] = generation
                            lcycle_per_seq_count['sen'] = 1
                        else:
                            is_unseen_htype = True
                            gtrigs['sen'] = gtrigs['nta'][-1]
                            gtrigs['nta'] = gtrigs['nta'][:-1]
                            nta_count = - nta_count + 1
                            # The sequence of nta is considered as sen.
                            lcycle_per_seq_count['sen'] = \
                                lcycle_per_seq_count['nta'][-1] + 1
                        # And the last seq of nta is forgotten.
                        lcycle_per_seq_count['nta'] = \
                            lcycle_per_seq_count['nta'][:-1]
                    # Else, if it adapts/repairs it exits arrest.
                    elif fct.is_repaired():
                        nta_count *= - 1
                    # Otherwise it stays arrested.
                    else:
                        # Update of the length of current seq of lcycles.
                        lcycle_per_seq_count['nta'][-1] += 1
            # Otherwise the cell is senescent, keeps same data than its mother.
            else:
                # New sen cycle, we update the count.
                lcycle_per_seq_count['sen'] += 1

            # Update of the cell cycle duration time and array of cyle times.
            if is_galactose:
                evo_cycle[-1] = fct.draw_cycle_galactose(nta_count,
                                                         is_senescent)
            else:
                evo_cycle[-1] = fct.draw_cycle(nta_count, is_senescent)
    # Computation of the type of the lineage (ie of the last cell).
    if nta_count == 0:
        lineage_type = 0
    elif nta_count < 0:
        lineage_type = 1
    else:
        lineage_type = np.NaN

    # Return data removing data of the 1st cell (born before DOX addition).
    return ({'cycle': evo_cycle[1:], 'lavg': evo_lavg[1:],
             'lmin': evo_lmin[1:]}, gtrigs, lineage_type, is_unseen_htype,
            is_accidental_death, lcycle_per_seq_count)