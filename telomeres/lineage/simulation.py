#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 11:42:48 2021

@author: arat

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

from copy import deepcopy
import glob
import multiprocessing as mp
import numpy as np
import os
import warnings

import telomeres.auxiliary.figures_properties as fp
import telomeres.auxiliary.functions as afct
import telomeres.model.functions as mfct
import telomeres.finalCut.functions as fc_fct
import telomeres.auxiliary.write_paths as wp
from telomeres.model.parameters import (
    PAR_DEFAULT_LIN,
    PAR_DEFAULT_SIM_LIN,
    OVERHANG,
    CHROMOSOME_COUNT,
)
from telomeres.lineage.posttreat import (
    select_exp_lineages,
    count_exp_lineages,
    type_of_sort_from_characteristics,
    is_as_expected_lineage,
    sort_lineages,
)

SIMU_COUNT_MIN_TO_SAVE = 2  # Strict minimum.
IS_LOAD_PRINTED = False
IS_TO_SIM_PRINTED = False


def simulate_lineage_evolution(parameters, rng):
    """Simulate the evolution of one lineage from after addition of the DOX to
    death (the cell before DOX addition is type A non-senescent with telomere
    lengths drawn from telomerase-positive distribution at equilibrium).

    Parameters
    ----------
    parameters : dict
        Dictionary with same entries as `par.PAR_DEFAULT_LIN`, namely:
        is_htype_accounted : bool
            True if type H allowed in the model (otherwise non-terminally
             arrested cells must return to normal cycle to be able to enter
             senescence).
        is_htype_seen : bool
            If set to False, the generation of the onset of the last sequence
            of nta of type H lineages is considered as the generation of the
            onset of the senescence; as it would have been classified
            experimentally. Plus, type H lineages will not be classified as H
            but as either type A or B (depending on weither or not they
            experienced more than one nta). The fact that it was H is saved in
            `is_unseen_htype`.
        'fit' : list
            0: The parameters [a, b] of the law of the onset of the 1st arrest.
            1: The parameters [a, b, l_min] of the law of the onset of sen for
               type A and B.
            2: The parameters [ltrans, l0, l1] of transfo applied to `l_init`.
        p_exit : dict
            The parameters of death and exit of a sequence of arrest:
            'accident': rate of accidental death (p_accident).
            'death': terminal arrest (p_death).
            'repair': non-terminal arrest (p_repair).
            'sen_limit': maximum nb of senescent cycles (math.inf: no limit).
        'finaCut' : dict
            None corresponds to usual experimental setting with from Dox
            addition to lineage death. Otherwise, mimick the experiment where
            - Dox is added at frame `par_finalCut['idxs_frame'][0]`
            - Gal at frame `par_finalCut['idxs_frame'][1]` which will initiate
              the stochastic cut of 1 telomere (uniformly chosen) to length
              `par_finalCut['lcut']` bp with constant probability in time
              (exponential law). Galactose also affects cell growth. It is
              removed at frame `par_finalCut['idxs_frame'][2]`.

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
        Dictionnary of generations at which an event is triggered, with entires
        nta : ndarray
            1D array (nta_count,) of generations at which a sequence of
            non-terminal arrests (nta) is triggered, NaN if no such sequence.
        senescence : int or NaN
            Generation at which senescence is triggered, nan if accidently
            dead before the onset of senescence.
        death : int
            Generation at which death is triggered.
    lineage_type : str
        Type of the lineage (0 for A type, 1 for B or np.NaN for H).
    is_unseen_htype : bool or NoneType
        Indicate if the lineage is a type H classified A or B.
        If no type H (i.e. `parameters['is_htype_accounted']` is False) or
        `is_htype_seen` is True, always None, otherwise True if the lineage was
        H but classified as A or B, and False if the lineage was a "true"A or B
        type (NB: NaN in the case of experimental lineages for unknown).
    is_accidental_death : bool
        True if the lineage has died accidentally, False otherwise.
    lcycle_per_seq_count : dict
        Length / number of long cycles per sequence of arrest. Keys are:
        'sen': integer
            Number of senescent cycles / length of senescence.
        'nta': ndarray
            lcycle_per_seq_count['nta'][i] length of the ith sequence of nta.

    """
    par_nta, par_sen, par_l_init = parameters["fit"]
    p_exit = parameters["p_exit"]

    # Initialization of <evo_*> arrays (t = 0) with the data of the first cell:
    # non-sencescent type A with generation -1 (s.t the 1st cell born under DOX
    # has generation 0). At t=0: Dox addition, no Galactose (and thus no cut).
    is_unseen_htype = False
    if parameters["is_htype_seen"] or (not parameters["is_htype_accounted"]):
        is_unseen_htype = None
    is_galactose = False
    is_accidental_death = False
    is_senescent = False
    nta_count = 0
    generation = -1
    evo_lengths = mfct.draw_cells_lengths(1, par_l_init, rng)
    evo_lavg = [np.mean(evo_lengths)]
    evo_lmin = [np.min(evo_lengths)]
    gtrigs = {"nta": np.array([]), "sen": np.NaN}
    lcycle_per_seq_count = {"nta": np.array([]), "sen": np.nan}
    # > Final cut exp conditions.
    # gen_cut = math.inf  # Generation at wich a cut happens.
    if not isinstance(parameters["finalCut"], type(None)):
        # Cell cycles [min] under rafinose conditions.
        evo_cycle = np.array(
            [
                fc_fct.draw_cycle_finalCut(nta_count, is_senescent, is_galactose, rng)
            ]
        )
        # Time at the end of the 1st cycle.
        t_current = evo_cycle[0] * rng.uniform()  # Time since Dox addition.
        # Times [min] of change in experimental conditions.
        idx_dox, idxf_gal, idxf_raf = parameters["finalCut"]["idxs_frame"]
        delay = parameters["finalCut"]["delay"] * 60
        t_gal = (idxf_gal - idx_dox) * 10  # (1 frame every 10 min).
        t_raf = (idxf_raf - idx_dox) * 10
    else:  # Cell cycle [min] under usual conditions.
        evo_cycle = mfct.draw_cycles_atype(1, rng)
    is_telo_cut = False  # No cut initially.

    # While the lineage is not extinct.
    lineage_is_alive = True
    while lineage_is_alive:
        # If the current cell dies accidentally, we store it with its state
        #    (senescent or not) and generation of death, and the lineage dies.
        # Plus we update the lineage data depending on `is_unseen_acc`.
        if generation >= 0 and mfct.is_accidentally_dead(p_exit["accident"], rng):
            is_accidental_death = True
            gtrigs["death"] = generation
            lineage_is_alive = False
            # If H-type accounted but not recognized and the lineage is not H.
            if is_unseen_htype is False:
                # If the lineage was arrested non-senescent at its death.
                if nta_count > 0:
                    is_senescent = True
                    # Then its last arrest is recognized as a terminal arrest.
                    gtrigs["sen"] = gtrigs["nta"][-1]
                    gtrigs["nta"] = gtrigs["nta"][:-1]
                    nta_count = -nta_count + 1
                    lcycle_per_seq_count["sen"] = lcycle_per_seq_count["nta"][-1]
                    lcycle_per_seq_count["nta"] = lcycle_per_seq_count["nta"][:-1]

        # If it is senescent we test if it dies.
        elif is_senescent and mfct.is_dead(lcycle_per_seq_count["sen"], p_exit, rng):
            lineage_is_alive = False  # If so the lineage extincts.
            gtrigs["death"] = generation  # We strore the gen of death.

        # Otherwise it divides, we add one generation and create the next cell.
        else:
            generation += 1
            # Extend evolution arrays at the new generation w default values.
            evo_lengths = np.append(evo_lengths, [evo_lengths[-1]], axis=0)
            evo_cycle = np.append(evo_cycle, evo_cycle[-1])

            # Computation of telomere lengths following the shortening model.
            loss = rng.binomial(1, 0.5, 16)
            evo_lengths[-1] -= OVERHANG * np.array([loss, 1 - loss])
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
                    if mfct.is_sen_trig(evo_lmin[-1], par_sen[0], rng):
                        is_senescent = True
                        gtrigs["sen"] = generation
                        lcycle_per_seq_count["sen"] = 1
                    # Otherwise, if a 1st arrest triggered, it becomes type B.
                    elif mfct.is_nta_trig(evo_lmin[-1], par_nta, rng):
                        nta_count = 1
                        gtrigs["nta"] = np.array([generation])
                        # 1st sequence of nta.
                        lcycle_per_seq_count["nta"] = np.append(
                            lcycle_per_seq_count["nta"], 1
                        )
                # Otherwise mother was (non-senescent) type B.
                elif nta_count < 0:  # > If not arrested type B.
                    # If senescence is triggered, the cell enters sen.
                    if mfct.is_sen_trig(evo_lmin[-1], par_sen[1], rng):
                        is_senescent = True
                        gtrigs["sen"] = generation
                        lcycle_per_seq_count["sen"] = 1
                    # Elif new arrest triggered, enters a new arrest.
                    elif mfct.is_nta_trig(evo_lmin[-1], par_nta, rng):
                        nta_count = 1 - nta_count
                        gtrigs["nta"] = np.append(gtrigs["nta"], generation)
                        # New sequence of nta.
                        lcycle_per_seq_count["nta"] = np.append(
                            lcycle_per_seq_count["nta"], 1
                        )
                else:  # > Otherwise mother was (non-senescent) arrested B.
                    # If H type taken into account, cell can turn sen (H).
                    if parameters["is_htype_accounted"] and mfct.is_sen_trig(
                        evo_lmin[-1], par_sen[1], rng
                    ):
                        is_senescent = True
                        if parameters["is_htype_seen"]:
                            gtrigs["sen"] = generation
                            lcycle_per_seq_count["sen"] = 1
                        else:
                            is_unseen_htype = True
                            gtrigs["sen"] = gtrigs["nta"][-1]
                            gtrigs["nta"] = gtrigs["nta"][:-1]
                            nta_count = -nta_count + 1
                            # The sequence of nta is considered as sen.
                            lcycle_per_seq_count["sen"] = (
                                lcycle_per_seq_count["nta"][-1] + 1
                            )
                        # And the last seq of nta is forgotten.
                        lcycle_per_seq_count["nta"] = lcycle_per_seq_count["nta"][:-1]
                    # Else, if it adapts/repairs it exits arrest.
                    elif mfct.is_repaired(p_exit["repair"], rng):
                        nta_count *= -1
                    # Otherwise it stays arrested.
                    else:
                        # Update of the length of current seq of lcycles.
                        lcycle_per_seq_count["nta"][-1] += 1
            # Otherwise the cell is senescent, keeps same data than its mother.
            else:
                # New sen cycle, we update the count.
                lcycle_per_seq_count["sen"] += 1

            # Update of the cell cycle duration time and array of cyle times.
            if isinstance(parameters["finalCut"], type(None)):
                evo_cycle[-1] = mfct.draw_cycle(nta_count, is_senescent, rng)
            else:  # Experimental conditions of the fincalCut experiment.
                evo_cycle[-1] = fc_fct.draw_cycle_finalCut(
                    nta_count, is_senescent, is_galactose, rng
                )
            # If finalcut experiment, possible change of condition and cut.
            if not isinstance(parameters["finalCut"], type(None)):
                t_current += evo_cycle[-1]  # Time just bf div of current cell.
                cdt_after_gal = evo_cycle[-1]
                # cdt_w_gal = evo_cycle[-1]  # When Gal is active: the time
                # # spent under Gal.
                if t_current >= t_raf:  # If Galactose has been removed.
                    if is_galactose:  # If during the current cycle.
                        is_galactose = False
                        # cdt_w_gal -= t_current - t_raf
                elif t_current >= t_gal:  # Otherwise if Galactose is active.
                    if not is_galactose:  # If cell is 1st to experience it
                        # If the cell (gen =- 1, that experiences Gal) is
                        # arrested the lin is excluded from data: new lin simu.
                        if nta_count > 0 or is_senescent:
                            return simulate_lineage_evolution(parameters, rng)
                        is_galactose = True
                        # cdt_w_gal = t_current - t_gal
                        cdt_after_gal = t_current - t_gal
                        # We reset the generation and forget ancestors.
                        evo_cycle = evo_cycle[-1:]
                        evo_lengths = evo_lengths[-1:]
                        evo_lavg = evo_lavg[-1:]
                        evo_lmin = evo_lmin[-1:]
                        for key, gtrig in gtrigs.items():
                            gtrigs[key] = gtrig - np.abs(generation)
                        generation = -1  # = generation - dgen_rescale
                # Possible cut if the length after cut is not None (ie there
                # is a Cas9 cut), ...
                # NB: We assume that the cut happens at the end of the cycle,
                #     and will thus influence only the next generation.
                if not isinstance(parameters["finalCut"]["lcut"], type(None)):
                    # and that cut not already done  # & gal still present.
                    # if (not is_telo_cut) and is_galactose:
                    #     if fc_fct.is_cut_exponential(
                    #             cdt_w_gal, dt_w_gal=t_current - t_gal):
                    if (not is_telo_cut) and np.logical_and(
                        t_gal <= t_current, t_current <= t_raf + delay
                    ):
                        if fc_fct.is_cut_exponential(
                            cdt_after_gal=cdt_after_gal,
                            dt_since_gal=t_current - t_gal,
                            rng=rng,
                        ):
                            is_telo_cut = True
                            # Index of the chromosome cut...
                            t2 = rd.randint(CHROMOSOME_COUNT)
                            t1 = rng.randint(2)  # ... and extremity cut.
                            evo_lengths[-1][t1, t2] = parameters["finalCut"]["lcut"]

    # Computation of the type of the lineage (ie of the last cell).
    if nta_count == 0:
        lineage_type = 0
    elif nta_count < 0:
        lineage_type = 1
    else:
        lineage_type = np.NaN

    # Return data removing data of the 1st cell (born before DOX addition).
    return (
        {"cycle": evo_cycle[1:], "lavg": evo_lavg[1:], "lmin": evo_lmin[1:]},
        gtrigs,
        lineage_type,
        is_unseen_htype,
        is_accidental_death,
        lcycle_per_seq_count,
    )


def simulate_lineages_evolution(
    lineage_count,
    characteristics,
    parameters,
    is_lcycle_count_returned=False,
    is_evo_returned=False,
    rng=None,
):
    """Simulate the independent evolution of `lineage_count` lineages having
    characteristics `characteristics`.

    Parameters
    ----------
    lineage_count : int
        Number of lineages returned.
    characteristics : list of string
        Characteristics of the lineages returned.
        See `is_as_expected_lineage` docstring.
    parameters : dict
        Dictionnary with the parameters of the model in the same format than
        `PAR_DEFAULT_LIN`.
    is_lcycle_count_returned : bool
        If True, the number of consecutive long cycles per sequence of arrest
        is computed and returned, otherwise `is_lcycle_count_returned` is None.
    is_evo_returned : bool
        If True, evolution arrays computed of every lineages are concatenated
        and returned, otherwise `is_evo_returned` is set to None.

    Returns
    -------
    evo_s : dict (or Nonetype)
        Dictionnary of evolution arrays over generations gathering all
        lineages. Same entries as the output `evo` of
        `simulate_lineage_evolution`. Each entry, say `key`, is an nD array
        (lineage_count, *) consisting of the concatenation along 0 axis of the
        arrays evo[key] (dimension *), extended by Nan values if needed, of all
        the lineages simulated and kept.
    gtrigs_s : dict
        Dictionnary of the generation for each kept lineage at which an event
        is triggered. Same description as `evo_s` replacing `evo` by `gtrigs`.
    lineage_types : ndarray
        1D array (lineage_count,) of lineages types (0, 1 or NaN for A B or H).
    is_unseen_htypes : ndarray or Nonetype
        If no H-type (i.e. `parameters['is_htype_accounted']` is False) or
        `is_htype_seen` is True, always None, otherwise 1D array
        (lineage_count,) of the `is_unseen_htype` values of lineages.
    is_accidental_deaths : ndarray
        1D array (lineage_count,) of bool st. `is_accidental_deaths[i]` is True
        if the ith lineage has died accidentally, False otherwise.
    lcycle_per_seq_counts : dict
        Dictionnary of the number of arrests (/long cycles) per sequence of
        arrests gathered by entries st.:
        nta : ndarray
            2D array with same shape as `gtrigs_s['nta']` s.t.
            `lcycle_per_seq_counts['nta'][i, j]` is the number of successive
            long cycles of the jth nta of the ith lineage.
        sen : ndarray
            1D array (lineage_count, ) (ie with shape of `gtrigs_s['sen']`) st.
            `lcycle_per_seq_counts['sen'][i]` is the number of successive
            senescent cycles of the ith lineage.
        NB: Nan value whenever there is if no such sequence.

    """
    # Initialization.
    print("New subsimulation.")
    rng = np.random.default_rng(rng)
    lineage_types = np.array([])
    is_unseen_htypes = np.array([])
    is_accidental_deaths = np.array([])
    if is_lcycle_count_returned:
        lcycle_per_seq_count_s = {"nta": [], "sen": []}
    nta_counts = np.array([])
    evo_s = None
    if is_evo_returned:
        evo_s = {}
    gtrigs_s = {"nta": [], "sen": np.array([]), "death": np.array([])}

    # While all lineages have not been simulated.
    count = 0
    while count < lineage_count:
        # We simulate another lineage.
        (
            evos,
            gtrigs,
            lineage_type,
            is_unseen_htype,
            is_accidental_death,
            lcycle_per_seq_count,
        ) = simulate_lineage_evolution(parameters, rng)

        # And keep it only if it has the expected characteristic.
        if not is_as_expected_lineage(
            gtrigs, lineage_type, is_accidental_death, characteristics
        ):
            pass
        else:
            count += 1

            # > Update of `is_accidental_deaths` `lineage_types` `nta_counts`.
            lineage_types = np.append(lineage_types, lineage_type)
            is_unseen_htypes = np.append(is_unseen_htypes, is_unseen_htype)
            is_accidental_deaths = np.append(is_accidental_deaths, is_accidental_death)
            nta_counts = np.append(nta_counts, len(gtrigs["nta"]))
            # and the number of generations in the lineage (gen of death + 1).
            gen_count_temp = gtrigs["death"] + 1

            # > Update of `gtrigs_s`.
            gtrigs_s["nta"].append(gtrigs["nta"])
            gtrigs_s["sen"] = np.append(gtrigs_s["sen"], gtrigs["sen"])
            gtrigs_s["death"] = np.append(gtrigs_s["death"], gtrigs["death"])

            # Update of `lcycle_per_seq_counts` if asked to be computed.
            if is_lcycle_count_returned:
                for key in ["nta", "sen"]:
                    lcycle_per_seq_count_s[key].append(lcycle_per_seq_count[key])

            # > Update of `evo_s`, iterating on all keys, if asked.
            if is_evo_returned:
                if count == 1:
                    gen_count = gen_count_temp
                    for key, evo in evos.items():
                        evo_s[key] = [evo]
                else:
                    # > We reshaphe either `evos` either `evo_s` if necessary.
                    if gen_count > gen_count_temp:
                        for key, evo in evos.items():
                            evos[key] = afct.reshape_with_nan(evo, gen_count)
                            # > And add the current lineage to previous ones.
                            evo_s[key] = np.append(evo_s[key], [evos[key]], 0)
                    # If `evo_s` reshape, update of current max number of gen.
                    elif gen_count < gen_count_temp:
                        gen_count = gen_count_temp
                        for key, evo in evos.items():
                            evo_s[key] = afct.reshape_with_nan(evo_s[key], gen_count, 1)
                            evo_s[key] = np.append(evo_s[key], [evo], 0)
                    # Otherwise we add directly, no need to reshape first.
                    else:
                        for key, evo in evos.items():
                            evo_s[key] = np.append(evo_s[key], [evo], 0)
    # `gtrigs_s['nta']` and `lcycle_per_seq_count_s` converted from list to
    #  array extending with NaN.
    nta_count = int(max(nta_counts))
    gtrigs_s["nta"] = np.array(
        [afct.reshape_with_nan(gtrigs, nta_count) for gtrigs in gtrigs_s["nta"]]
    )
    lcycle_per_seq_counts = {"nta": None, "sen": None}
    if is_lcycle_count_returned:
        seq_count = max([len(lc) for lc in lcycle_per_seq_count_s["nta"]])
        lcycle_per_seq_counts = {
            "nta": np.array(
                [
                    afct.reshape_with_nan(count, seq_count)
                    for count in lcycle_per_seq_count_s["nta"]
                ]
            ),
            "sen": np.array(lcycle_per_seq_count_s["sen"]),
        }
    # If no type H to keep track of `is_unseen_htypes` simply set to nan.
    if parameters["is_htype_seen"] or (not parameters["is_htype_accounted"]):
        is_unseen_htypes = None
    return (
        evo_s,
        gtrigs_s,
        lineage_types,
        is_unseen_htypes,
        is_accidental_deaths,
        lcycle_per_seq_counts,
    )


def simulate_lineages_evolutions(
    simulation_count,
    lineage_count,
    characteristics,
    par_update=None,
    proc_count=1,
    is_lcycle_count_saved=False,
    is_evo_saved=False,
    rng=None,
):
    """Simulate (possibly in parallel) `simu_count` times
    `simulate_lineages_evolutions` (with parameters entered as argument).

    Parameters
    ----------
    simulation_count : int
        Number of sets of `lineage_count` lineages simulated.
    proc_count : int, optional
        Number of processors used for parallel computation (if 1 no parallel
        computation).

    Returns
    -------
    output_s : list
        List of `simulation_count` outputs of `simulate_lineages_evolutions`.

    """
    rng = np.random.default_rng(rng)
    # Updatatable parameters.
    p = deepcopy(PAR_DEFAULT_LIN)
    if isinstance(par_update, dict):
        p.update(par_update)

    is_saved = simulation_count > SIMU_COUNT_MIN_TO_SAVE
    simus = np.arange(simulation_count)
    paths = wp.write_lineages_paths(
        simulation_count,
        lineage_count,
        characteristics,
        is_lcycle_count_saved,
        is_evo_saved,
        par_update=par_update,
        make_dir=is_saved,
    )
    for path in paths:
        if os.path.exists(path):
            print("\n Lineages loaded from \n", path)
            output_s = np.load(path, allow_pickle=True).item()
            return output_s

    # > If proc_count is 1, no parallelization.
    print("proc_count: ", proc_count)
    print("cpu_count: ", os.cpu_count())
    print(paths[0])
    # stop
    kwargs = {
        "is_lcycle_count_returned": is_lcycle_count_saved,
        "is_evo_returned": is_evo_saved,
    }
    if proc_count == 1:
        output_s = [
            simulate_lineages_evolution(
                lineage_count, characteristics, p, **kwargs, rng=rng
            )
            for s in simus
        ]
    # > Otherwise, initialization of the parallelization.
    else:
        if proc_count > os.cpu_count() - 1:
            raise Exception(
                "`proc_count` is too big for your computing "
                f"capacity of {os.cpu_count()} processors."
            )
        pool = mp.Pool(processes=proc_count)
        pool_s = [
            pool.apply_async(
                simulate_lineages_evolution,
                args=(lineage_count, characteristics, p),
                kwds=kwargs | {"rng": rng.integers(2**32) + i},
            )
            for i in simus
        ]
        # > Results retrieval from pool_s (list of pool.ApplyResult obj).
        output_s = [r.get() for r in pool_s]
        # > We prevent the current process to put more data on the queue.
        pool.close()
        # > Execution of next line postponed until processes in queue done.
        pool.join()
    if is_saved:
        np.save(paths[0], {s: output_s[s] for s in simus})
    return output_s


# --------
# Postreat
# --------


def compute_lineage_type_stats(
    lineage_types_s, is_unseen_htypes_s, is_accidental_deaths_s, is_htype_accounted
):
    """Compute statistics on the types of lineages: average and percentile of
    simulations having ith lineage type-A, -B and -H, and accidental death for
    all lineage i (dict `lineage_types_stat_on_sim` with entries arrays of len
    `(lineage_count,)`) and the statisitics on the pourcentage of each lineage
    type per simulation (avg, percent and extremum) of the prop on all
    simulations: dict `type_proportion_stat` with entries float).

    """
    is_atype = lineage_types_s == 0
    is_btype = lineage_types_s == 1
    type_lineage_stat_on_sim = {
        "atype": afct.stat(is_atype.astype(int), fp.P_UP, fp.P_DOWN, 0),
        "btype": afct.stat(is_btype.astype(int), fp.P_UP, fp.P_DOWN, 0),
        "accidental_death": np.mean(is_accidental_deaths_s.astype(int), 0),
    }
    if is_htype_accounted:
        is_htype_seen = isinstance(is_unseen_htypes_s, type(None))
        if not is_htype_seen:
            is_htype = np.isnan(lineage_types_s)
        else:
            is_htype = is_unseen_htypes_s
        type_lineage_stat_on_sim["htype"] = afct.stat(
            is_htype.astype(int), fp.P_UP, fp.P_DOWN, 0
        )
    type_per_sim = {}
    type_per_sim["atype"] = np.mean(is_atype, 1)  # Shape (simu_count, ).
    type_per_sim["btype"] = np.mean(is_btype, 1)
    type_per_sim["accidental_death"] = np.mean(is_accidental_deaths_s, 1)
    type_proportion_stat = {}
    for key, stats in type_per_sim.items():
        type_proportion_stat[key] = afct.stat_all(type_per_sim[key], fp.P_UP, fp.P_DOWN)
    return type_lineage_stat_on_sim, type_proportion_stat


def is_alive_at_time(time, div_times):
    """
    Parameters
    ----------
    time : float
        Non- negative time (min) at which to test which cells are alive.
    div_times : ndarray
        2D array (lineage_count, gmax) s.t. `div_times[i, j]` is the time (>0
        min) at which the cell of generation j in the ith lineage had divided.

    Returns
    -------
    is_alive : ndarray
        2D array (lineage_count, gmax) s.t. `is_alive[i, j]` is True if the
        cell of generation `j` in the ith lineage was alive at time `time`,
        False otherwise.

    """
    lineage_count = len(div_times)
    # We add a column of zeros in front of `div_times` (assuming generation 0
    #  is born at time zero, not in div_times).
    div_times_extended = np.append(np.zeros((lineage_count, 1)), div_times, 1)
    # To be alive at time `time` a cell must have a division time greater to
    #  `time` and its mother a division time  lower or equal to `time`.
    return np.logical_and(div_times_extended[:, :-1] <= time, div_times > time)


def is_alive_n_times_from_cycles(div_times, times=None):
    """Return for each time t of `times` (the times given as argument or each
    time of division if not indicated) a bool array of the shape of `cycles`
    indicating True in any `(i, j)` component s.t. the cell of the jth
    generation of the ith lineage is alive at time t.


    Parameters
    ----------
    div_times : ndarray
        2D array (lineage_count, gen_count) s.t. `cycles[i, j]` is the time
        (min > 0) at which the jth cell of the ith lineage had divided.
        NB: is Nan if the the ith lineage is already dead at generation j.
    times : ndarray, optional
        1D array (time_count,) of times (min) at which compute 'is_alive'.
        NB: 0 and all times at which a division had occured if not given.

    Returns
    -------
    times : ndarray
        1D array (time_count,) given as argument or computed if no times given.
    is_alive : ndarray
        3D array (time_count, lineage_count, gen_count) s.t. `is_alive[i,j,k]`
        is True if the kth cell of the jth lineage is alive at time `times[i]`,
        False otherwise.

    """
    if isinstance(times, type(None)):
        times = np.unique(div_times)
        times = np.append(0, times[~np.isnan(times)])
    is_alive = np.concatenate(
        [[is_alive_at_time(time, div_times)] for time in times], 0
    )
    return times, is_alive


def cell_types_from_gtrigs_n_lin_types(
    cycles, gtrigs, lineage_types, data_type, gmax=None
):
    """Compute the type of all cells in a set of lineages with generations of
    event given by `gtrigs` and llineage types by `lineage_types`.

    Parameters
    ----------
    gtrigs : dict
        Dictionnary with generations of event (nta, senescence and death) of
        the format returned by `simulate_lineage_evolution`.
    lineage_types : ndarray
        1D array (lineage_count,) of lineages types: 0, 1 or NaN respectively
        for A, B and H (if simulated lineages with is_unseen_htypes False),
        or not known (if experimental lineages).
    data_type : string
        'exp' if experimental data, 'sim' if simulated data.

    Returns
    -------
    cell_types : ndarray
        2D array (lineage_count, gmax) s.t. `cell_types[i, j]` is 0, 1 or Nan
        depending on the type of the cell of generation `j` in the ith lineage.
        NB: is -1 if no cell.

    """
    lineage_count, gen_count = np.shape(cycles)
    gmax = gmax or gen_count
    lineages = np.arange(lineage_count)
    cell_types = np.zeros((lineage_count, gmax))

    # > Type A lineages are made only of Type A cells -> zero default value OK.

    # > Type B lineages are made of type A cells and B from the 1st nta.
    for b_lin_idx in lineages[lineage_types == 1]:
        nta1_idx = int(gtrigs["nta"][b_lin_idx, 0])
        cell_types[b_lin_idx, nta1_idx:] = 1

    nan_lin_idxs = lineages[np.isnan(lineage_types)]
    for nan_lin_idx in nan_lin_idxs:
        # > Unknown lineages in the case of experimental data.
        if data_type == "exp":
            # Cells of unknow lin are unknown type after 1st arrest, A before.
            # Computation of idx of arrest if arrested lineage (nan if not).
            arrest_idx = gtrigs["sen"][nan_lin_idx]
            # If arrest at generation `arrest_idx`, cells after are unknown.
            if not np.isnan(arrest_idx):
                cell_types[nan_lin_idx][int(arrest_idx) :] = np.nan
        # > Type H lineages in the case of simulated data.
        elif data_type == "sim":
            # Type A before 1st arest, H after senescence, B in between.
            nta1_idx = int(gtrigs["nta"][nan_lin_idx, 0])
            sen_idx = int(gtrigs["sen"][nan_lin_idx])
            cell_types[nan_lin_idx, nta1_idx:] = 1
            cell_types[nan_lin_idx, sen_idx:] = np.nan
        else:
            raise ValueError(
                "ERROR: wrong `data_exp` argument for "
                "`cell_types_from_gtrigs_n_lin_types` function"
            )

    # > -1  values at every generation where no cell (lineage already extinct).
    cycles_extended = afct.reshape_with_nan(cycles, gmax, 1)
    cell_types[np.isnan(cycles_extended)] = -1
    return cell_types


def evo_in_gen_n_time(data, div_times, gen_count=None, times=None):
    """Compute evolution of average (and possibly percentile) population data
    along generation and time.

    Parameters
    ----------
    data : list
        List formatted as (evo_s, gtrigs_s, lineage_types, is_unseen_htypes,
        is_accidental_deaths) an output of either `simulate_lineages_evolution`
        or `lineage.posttreat.postreat_experimental_lineages`, for `data_type`
        being 'exp or 'sim', respectively.
    div_times : ndarray
        2D array (lineage_count, gen_count) s.t. `cycles[i, j]` is the time
        (min) at which the jth cell of the ith lineage had divided.
        NB: is Nan if the the ith lineage is already dead at generation j.
    gen_count : int, optional
        Generation up to which compute the postreated generation-data.
        NB: The maximal number of generation `gmax` in the lineages in
            argmument if not given.
    times : ndarray, optional
        1D array (time_count,) of times at which compute postreated time-data.
        NB: all times (in min) at which a division had occured if not given.

    Returns
    -------
    postreated_data : dict
        Dictionnary with key 1) 'gen' and 'time', each entry giving a list of
        data characterizing the evolution of specific quantities along resp.
        generation or time as follows: the ith element of each list is
        - i=0: 1D array: the evolution axis (`generations` or `times`)
        - i=1: a dict with same key as `data[0]`, each entry being:
          > if `is_stat_computed` is True: a dict of statistics (with entries
            'mean', 'perup', 'perdown') and additionnal entries
            - 'lmin_min' (dict of stats on the miminum of lim on all lineages),
            - 'prop_type' (2D array (3, gen_count) or(3, time_count)) s.t. the
              1st/2nd/3rd row is the evo of the proportion of type A/B/H cells.
          > otherwise: only the entry 'mean' of the dict (i.e. a 1D array)
            returned when `is_stat_computed` is True.
        NB: arrays are nan when all lineages are dead
           (might happen in the case `gen_count > gmax`).

    """
    # Determination of the type of data.
    # > If the dictionnary of evolution arrays has only one entry.
    if len(data[0]) == 0:  # Experimental data.
        data_type = "exp"
    else:  # Otherwise, simulated.
        data_type = "sim"

    # Definition of `gen_count` if not given as argument.
    lineage_count, gmax = np.shape(data[0]["cycle"])
    if isinstance(gen_count, type(None)):
        gen_count = gmax
    generations = np.arange(gen_count)

    times, is_alive = is_alive_n_times_from_cycles(div_times, times)
    evo_g = {}
    evo_t = {}
    # "Evo" data, we iterate on all evo data, adding the key 'lmin_min'.
    for key, evo_data in data[0].items():
        evo_data_reshaped = afct.reshape_with_nan(evo_data, gen_count, 1)
        # Avoid error nanmean/min/... when mean/... computed over axis of nan.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            # > In generation: average on all lineages at each gen.
            evo_g[key] = np.nanmean(evo_data_reshaped, 0)
            # > In time: average on all lineages at each time.
            evo_t[key] = np.array([])
            for a in is_alive:
                if len(evo_data[a]) == 0:
                    evo_t[key] = np.append(evo_t[key], np.nan)
                else:
                    evo_t[key] = np.append(evo_t[key], np.nanmean(evo_data[a]))
            # > New key 'lmin_min'.
            if key == "lmin":
                evo_g["lmin_min"] = np.nanmin(evo_data_reshaped, 0)
                evo_t["lmin_min"] = np.array([])
                for a in is_alive:
                    if len(evo_data[a]) == 0:
                        evo_temp = np.nan
                    else:
                        evo_temp = np.nanmin(evo_data[a], axis=0)
                    evo_t["lmin_min"] = np.append(evo_t["lmin_min"], evo_temp)

    # Proportion of each type, senescent, ...
    if "cell_types" in data[0].keys():
        cell_types = data[0]["cell_types"]
    else:
        cell_types = cell_types_from_gtrigs_n_lin_types(
            data[0]["cycle"], *data[1:3], data_type
        )
    is_senescent = np.empty((0, gmax))
    for i in range(lineage_count):
        temp = np.arange(gmax) >= data[1]["sen"][i]
        is_senescent = np.append(is_senescent, [temp], 0)
    # Add nan to the bool matrix at every generation after death.
    is_senescent_alive = is_senescent + 0 * data[0]["cycle"]

    # > Evolution in generation, iteration on generations.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        evo_g["prop_sen"] = np.nanmean(is_senescent_alive, 0)
    evo_g["prop_type"] = np.empty((3, 0))
    evo_g["prop_type_sen"] = np.empty((3, 0))
    for gen in range(gmax):
        # Average on alive cell at `gen`, i.e. those with type not equal to -1.
        is_galive = cell_types[:, gen] != -1
        is_gsen_alive = np.logical_and(is_senescent[:, gen], is_galive)
        types_alive = cell_types[is_galive, gen]
        types_alive_sen = cell_types[is_gsen_alive, gen]
        with warnings.catch_warnings():
            # For mean to return nan for empty array w.o. printing a warning.
            warnings.simplefilter("ignore", category=RuntimeWarning)
            temp = np.concatenate(
                [
                    [[np.mean(types_alive == 0, 0)]],
                    [[np.mean(types_alive == 1, 0)]],
                    [[np.mean(np.isnan(types_alive), 0)]],
                ],
                0,
            )
            temp_sen = np.concatenate(
                [
                    [[np.mean(types_alive_sen == 0, 0)]],
                    [[np.mean(types_alive_sen == 1, 0)]],
                    [[np.mean(np.isnan(types_alive_sen), 0)]],
                ],
                axis=0,
            )
        evo_g["prop_type"] = np.append(evo_g["prop_type"], temp, 1)
        evo_g["prop_type_sen"] = np.append(evo_g["prop_type_sen"], temp_sen, 1)
    # Reshape.
    for key in ["prop_sen", "prop_type", "prop_type_sen"]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            evo_g[key] = afct.reshape_with_nan(evo_g[key], gen_count, -1)

    # > Iteration on all times, averaging only omong alive cells.
    evo_t["prop_sen"] = np.array([])
    evo_t["prop_type"] = np.empty((3, 0))
    evo_t["prop_type_sen"] = np.empty((3, 0))
    for a in is_alive:
        a_sen = np.logical_and(a, is_senescent)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            temp = np.mean(is_senescent[a])
            temp_t = np.concatenate(
                [
                    [[np.mean(cell_types[a] == 0)]],
                    [[np.mean(cell_types[a] == 1)]],
                    [[np.mean(np.isnan(cell_types[a]))]],
                ],
                0,
            )
            temp_tsen = np.concatenate(
                [
                    [[np.mean(cell_types[a_sen] == 0, 0)]],
                    [[np.mean(cell_types[a_sen] == 1, 0)]],
                    [[np.mean(np.isnan(cell_types[a_sen]))]],
                ],
                axis=0,
            )
        evo_t["prop_sen"] = np.append(evo_t["prop_sen"], temp)
        evo_t["prop_type"] = np.append(evo_t["prop_type"], temp_t, 1)
        evo_t["prop_type_sen"] = np.append(evo_t["prop_type_sen"], temp_tsen, 1)
    return {
        "gen": [generations, evo_g],
        "time": [times, evo_t],
        "cell_types": cell_types,
    }


# ----------------------------------------------
# Statistics on evolutions of simulated lineages
# ----------------------------------------------


def statistics_on_sorted_lineages(data_s, is_htype_accounted, parameters_sim):
    """Return statistics on a set of simulations with common number of
    lineages and characteristics.

    Parameters
    ----------
    data_s : list
        A list of `simulation_count` sets of (simulated) data, each sorted
        similarly and formatted as an output of `sort_lineages` (or an output
        of `simulate_lineages_evolution`).
    parameters_sim : dict
        See PAR_DEFAULT_SIM_LIN. Keys:
        postreat_dt : int
            If not None (default value), data postreated and statistics on all
            sets of lineages computed from time 0 to `tmax` (maximal time of
            extinction among all lineages) with constant time step
            `postreat_dt` (in 10 min).
        hist_lmins_axis
        is_evo_stored

    Returns
    -------
    evo_avg : dict
        Dictionnary with the structure of the output `evo_s` of
        `simulate_lineages_evolution` (excep that the key 'cell_types' has been
        added), consisting of the average on all simulations.
    gtrigs_stat : dict
        Dictionnary with same entries as the output `gtrigs_s` of
        `simulate_lineages_evolution`. Each entry is itself a dictionnary,
        gathering statistics on all simulations, with entries:
        - 'mean' : for the average on simulation.
        - 'perup' : for the `fp.P_UP` percentile.
        - 'perup' : for the `fp.P_DOWN` percentile.

    """
    simus = np.arange(len(data_s))
    nta_counts = np.array([np.shape(data_s[s][1]["nta"])[1] for s in simus])
    nta_count = int(max(nta_counts))

    # Time/gen evolution postreatment of data if asked.
    evo_avg = None
    postreat_stat = {"gen": None, "time": None}
    postreat_dt = parameters_sim["postreat_dt"]
    if parameters_sim["is_evo_saved"]:
        evo_avg = {}
        gen_counts = [np.shape(data_s[s][0]["lavg"])[1] for s in simus]
        gen_count = int(max(gen_counts))

        # Average (only on simulations) all evolution arrays.
        for evo_key in data_s[0][0]:  # 'cycle', 'lavg', 'lmin.
            # We prevent RuntimeWarnings print when nanmean of empty arr.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                evo_avg[evo_key] = np.nanmean(
                    np.concatenate(
                        [
                            [afct.reshape_with_nan(data_s[s][0][evo_key], gen_count, 1)]
                            for s in simus
                        ],
                        0,
                    ),
                    axis=0,
                )
        # Add proportions by type to evolution arrays (dead:-1 H:nan A:0 B:1).
        cell_types_s = np.concatenate(
            [
                [
                    cell_types_from_gtrigs_n_lin_types(
                        data_s[s][0]["cycle"], *data_s[s][1:3], "sim", gmax=gen_count
                    )
                ]
                for s in simus
            ],
            0,
        )
        cell_counts = np.sum(cell_types_s != -1, 0).astype(float)
        # Not to divide by zero and have nan values where no cells.
        cell_counts[cell_counts == 0] = np.nan
        evo_avg["prop_atype"] = np.sum(cell_types_s == 0, 0) / cell_counts
        evo_avg["prop_btype"] = np.sum(cell_types_s == 1, 0) / cell_counts
        evo_avg["prop_htype"] = np.sum(np.isnan(cell_types_s), 0) / cell_counts
        if not isinstance(postreat_dt, type(None)):  # Postreat.
            # Computation of division times of all sets of lineages.
            div_times_s = [
                np.concatenate(
                    [
                        np.sum(data_s[s][0]["cycle"][:, : i + 1], 1)[:, None]
                        for i in range(gen_counts[s])
                    ],
                    1,
                )
                for s in simus
            ]
            tmax = np.max([np.nanmax(div_times_s[s]) for s in simus])
            times = np.arange(0, tmax + postreat_dt * 10, postreat_dt * 10)
            postreat_data_s = [
                evo_in_gen_n_time(data_s[s], div_times_s[s], gen_count, times)
                for s in simus
            ]
            # 1st argument of `postreat_stat`'s entries: gen & time arrays.
            postreat_stat = {"gen": [np.arange(gen_count)], "time": [times]}
            for key in ["gen", "time"]:
                # 2nd argument: dictionary of evo arrays.
                postreat_stat[key].append({})
                for evo_key in postreat_data_s[0][key][1].keys():
                    if evo_key == "lmin_min":
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=RuntimeWarning)
                            postreat_stat[key][1]["lmin_min"] = np.nanmin(
                                [postreat_data_s[s][key][1][evo_key] for s in simus], 0
                            )
                    else:
                        postreat_stat[key][1][evo_key] = afct.nanstat(
                            [postreat_data_s[s][key][1][evo_key] for s in simus],
                            fp.P_UP,
                            fp.P_DOWN,
                            0,
                        )
    # Concatenation of `lcycle_counts` if computed for each/the 1st simulation.
    lcycle_counts_s = {"nta": None, "sen": None}
    if not isinstance(data_s[0][5]["nta"], type(None)):
        seq_count = max([np.shape(data_s[s][5]["nta"])[1] for s in simus])
        lcycle_counts_s = {
            "nta": np.array(
                [
                    afct.reshape_with_nan(data_s[s][5]["nta"], seq_count, 1)
                    for s in simus
                ]
            ),
            "sen": np.array([data_s[s][5]["sen"] for s in simus]),
        }
    gtrigs_stat = {}
    for trig_key in data_s[0][1]:
        if trig_key == "nta":
            gtrigs = np.concatenate(
                [
                    [afct.reshape_with_nan(data_s[s][1]["nta"], nta_count, axis=1)]
                    for s in simus
                ],
                0,
            )
        else:
            gtrigs = [data_s[s][1][trig_key] for s in simus]
        gtrigs_stat[trig_key] = afct.nanstat(gtrigs, fp.P_UP, fp.P_DOWN, 0)

    lineage_types_s = np.concatenate([[data_s[s][2]] for s in simus], axis=0)
    is_unseen_htypes_s = np.concatenate([[data_s[s][3]] for s in simus], 0)
    is_accidental_deaths_s = np.concatenate([[data_s[s][4]] for s in simus], 0)
    lineage_type_stat = compute_lineage_type_stats(
        lineage_types_s, is_unseen_htypes_s, is_accidental_deaths_s, is_htype_accounted
    )
    # Statistics on histograms of lmin triggering senescence.
    hist_lmins = None
    if not isinstance(parameters_sim["hist_lmins_axis"], type(None)):
        lmin_max = 0
        lmins_s = []
        for s in simus:
            gsens = data_s[s][1]["sen"]
            sen_lin_idxs = np.arange(len(gsens))[~np.isnan(gsens)]
            lmins = [
                data_s[s][0]["lmin"][lmin, int(gsens[lmin])] for lmin in sen_lin_idxs
            ]  # int(gsens[l] - 1) !!!
            lmins_s.append(lmins)
            lmin_max = max(lmin_max, max(lmins))
        hist_axis = np.arange(lmin_max + 2)
        hist_s = [afct.make_histogram(lmins, hist_axis) for lmins in lmins_s]
        hist = afct.stat_all(hist_s, fp.P_UP, fp.P_DOWN)
        hist_lmins = [hist_axis, hist]
    return (
        evo_avg,
        gtrigs_stat,
        lineage_type_stat,
        postreat_stat,
        lcycle_counts_s,
        hist_lmins,
    )


def run_with_limited_time(func, time_limit, args, kwargs={}):
    """Run a function with time limit.

    From Ariel Cabib's answer on https://stackoverflow.com/questions/366682.

    Parameters
    ----------
    func
        The function to run.
    args : tuple
        The function arguments.
    kwargs : dict
        The function keyword arguments.
    time : float
        The time limit in seconds.

    Returns
    -------
    bool
        False if the function ended successfully in the time limit given.
        True otherwise, when the function took too long to be run.

    """
    process = mp.Process(target=func, args=args, kwargs=kwargs)
    process.start()
    process.join(time_limit)
    if process.is_alive():
        process.terminate()
        return True
    return False


def file_names_w_expected_data(path, parameters_sim):
    """

    Parameters
    ----------
    path : string
    parameters_sim : dict
        See PAR_DEFAULT_SIM_LIN. Keys:
        postreat_dt : int
            If not None (default value), data postreated and statistics on all
            sets of lineages computed from time 0 to `tmax` (maximal time of
            extinction among all lineages) with constant time step
            `postreat_dt` (in 10 min).
        hist_lmins_axis
        is_evo_stored
        is_lcycle_counts : bool
            If True, the number of consecutive long cycles per sequence of
            arrest is computed and saved, otherwise set to None.


    """
    is_tpostreat = not isinstance(parameters_sim["postreat_dt"], type(None))
    is_hist_lmins = not isinstance(parameters_sim["hist_lmins_axis"], type(None))
    names = []
    if parameters_sim["is_evo_saved"] and not is_tpostreat:
        names = glob.glob(path.replace(".npy", "-dt*.npy"))
    elif not (
        is_tpostreat
        or is_hist_lmins
        or parameters_sim["is_lcycle_count_saved"]
        or parameters_sim["is_evo_saved"]
    ):
        names = glob.glob(path.replace(".npy", "*.npy"))
    return names


def simulate_n_average_lineages(
    lineage_count,
    simulation_count,
    types_of_sort,
    characteristics,
    par_update=None,
    par_sim_update=None,
    parameters_comput=None,
    proc_count=1,
    is_saved=None,
    rng=None,
):
    """Simulates `simulation_count` times the evolution, through
    `simulate_lineages_evolution`, of `lineage_count` lineages with given
    characteristics (corresponding to `characteristics`) and return statistics
    (average and `par.PERCENT` percentiles) on those sets of lineages for every
    way of orderring lineages given by `types_of_sort`.

    Parameters
    ----------
    lineage_count : int
        Number (positive integer) of lineages per simulations.
    simulation_count : int
        Number of simulations to run and average.
    types_of_sort : list
        List of strings: all the way(s) to order the simulated lineages (see
        `sort_lineage` docstring).
    characteristics : list of string
        Charateristics of lineages to keep (cf `is_expected_lineage` docstring)
    par_update : dict or None, optional
        Dictionnary with the updates of the default paramerters (namely
        `par.PAR_DEFAULT_LIN`) to make to simulate. In particular the only
        relevant keys of `par_update` are those of `par.PAR_DEFAULT_LIN`.
        If None (default value) the parameters taken are exactly default ones.
        Some details on a few keys:
        is_htype_seen : bool, optional
            See `simulate_lineage_evolution` docstring.
        fit
            0: The parameters [a, b] of the law of the onset of the 1st arrest.
            1: The parameters [a,b,l_min] of the law of the onset of sen A &B.
            2: The parameters [ltrans,l0, 1] of the law of transfo of `l_init`.
            If not indicated, the parameters of `parameters.py` are taken.
    par_sim_update : dict or None, optional
        Updates to apply on `par.PAR_DEFAULT_SIM_LIN`. Details on a few keys:
        postreat_dt : int, optional
            If not None (default value), data postreated and statistics on all
            sets of lineages computed from time 0 to `tmax` (maximal time of
            extinction among all lineages) with constant time step`postreat_dt`
            (10 min). If None evolution arr and corresponding data not computed
        is_lcycle_counts : bool
            If True, the number of consecutive long cycles per sequence of
            arrest is computed and saved, otherwise set to None.
    parameters_comput : list or Nonetpype, optional
        If None no specific time limit for computing, otherwise if the time
        to compute `simulate_lineages_evolution` with `parameters_comput[0]`
        lineages is greater than `parameters_comput[1]` (sec) the whole
        simulation (with a priori more) is not run (None result instead).

    Returns
    -------
    stats : dict
        Dictionnary with entries the element(s) of `types_of_sort`. For each
        entry `key`, `stats[key]` is the statistics (format of the outputs of
        `statistics_on_sorted_lineages`) on all simulations where lineages have
        been orderred according to `key`.

    """
    p = deepcopy(PAR_DEFAULT_LIN)  # Current dictionnary of model parameters.
    if isinstance(par_update, dict):  # Updated as required.
        p.update(par_update)
    psim = deepcopy(PAR_DEFAULT_SIM_LIN)  # ... parameters of simulation.
    if isinstance(par_sim_update, dict):
        psim.update(par_sim_update)
    is_saved = is_saved or simulation_count > SIMU_COUNT_MIN_TO_SAVE
    if not isinstance(psim["postreat_dt"], type(None)):
        psim["is_evo_saved"] = True
    stats_path = wp.write_stat_path(
        simulation_count,
        lineage_count,
        types_of_sort,
        characteristics,
        par_update=par_update,
        par_sim_update=par_sim_update,
        make_dir=is_saved,
    )
    if IS_TO_SIM_PRINTED:
        print(stats_path)
    # Creation of directories where to save if not existing.
    wp.write_path_directory_from_file(stats_path, make_dir=is_saved)
    files_w_data = file_names_w_expected_data(stats_path, psim)
    if len(files_w_data) != 0:
        stats = np.load(files_w_data[0], allow_pickle=True).item()
        np.save(stats_path, stats)
        if IS_LOAD_PRINTED:
            print("1) Loaded from ", files_w_data[0])
    # If already simulated & saved.
    if os.path.exists(stats_path):
        # We load it.
        stats = np.load(stats_path, allow_pickle=True).item()
    else:  # Otherwise we set stats entries with only NaN.
        stats = {}
        for type_of_sort in types_of_sort:
            stats[type_of_sort] = None
    # We (re)simulate it for all keys not simulated or corresponding to a None
    #  value and save it.
    # Parameters for simulation.
    is_simulated = False
    is_too_long = False
    is_time_tested = isinstance(parameters_comput, type(None))  # True if not to test.
    simus = np.arange(simulation_count)
    is_evo_temp = psim["is_evo_saved"] or not isinstance(
        psim["hist_lmins_axis"], type(None)
    )
    # Orderring and statistics.
    for type_of_sort in types_of_sort:
        if isinstance(stats[type_of_sort], type(None)):
            path = wp.write_stat_path(
                simulation_count,
                lineage_count,
                [type_of_sort],
                characteristics,
                par_update=par_update,
                par_sim_update=par_sim_update,
                make_dir=is_saved,
            )
            files_w_data = file_names_w_expected_data(path, psim)
            if len(files_w_data) != 0:
                if IS_LOAD_PRINTED:
                    print("3) Loaded from ", files_w_data[0])
                stats[type_of_sort] = np.load(
                    files_w_data[0], allow_pickle=True
                ).item()[type_of_sort]
                np.save(path, {type_of_sort: stats[type_of_sort]})
            if os.path.exists(path):
                stats[type_of_sort] = np.load(path, allow_pickle=True).item()[
                    type_of_sort
                ]
            if isinstance(stats[type_of_sort], type(None)):
                if not is_simulated:
                    if not is_time_tested:
                        args = [parameters_comput[0], characteristics, p]
                        if run_with_limited_time(
                            simulate_lineages_evolution,
                            parameters_comput[1],
                            args,
                            kwargs={"rng": rng},
                        ):
                            print("Too long")
                            is_too_long = True
                        is_time_tested = True
                    if not is_too_long:
                        print("Simulation running...", is_saved)
                        if is_saved:
                            # If lineages have to be sorted by lmin or lavg,
                            # need to store data in a first time.
                            is_evo_saved = is_evo_temp or type_of_sort[0] == "l"
                            print(is_evo_temp, is_evo_saved)
                            data_s_0 = simulate_lineages_evolutions(
                                simulation_count,
                                lineage_count,
                                characteristics,
                                par_update=par_update,
                                proc_count=proc_count,
                                is_lcycle_count_saved=psim["is_lcycle_count_saved"],
                                is_evo_saved=is_evo_saved,
                                rng=rng,
                            )
                        else:
                            data_s_0 = [
                                simulate_lineages_evolution(
                                    lineage_count,
                                    characteristics,
                                    p,
                                    is_lcycle_count_returned=psim[
                                        "is_lcycle_count_saved"
                                    ],
                                    is_evo_returned=is_evo_temp,
                                    rng=rng,
                                )
                                for s in simus
                            ]
                    is_simulated = True
                if not is_too_long:
                    data_s = [sort_lineages(data_s_0[s], type_of_sort) for s in simus]
                    stats[type_of_sort] = statistics_on_sorted_lineages(
                        data_s, p["is_htype_accounted"], psim
                    )
                else:
                    stats[type_of_sort] = None
                if is_saved:
                    np.save(path, {type_of_sort: stats[type_of_sort]})
    if len(types_of_sort) > 1:
        np.save(stats_path, stats)
    return stats


# # Compute the right simulation to fit experimental data
# # -----------------------------------------------------

# def simulate_n_average_lineages_as_exp(exp_data, simulation_count,
#                                        types_of_sort, characteristics,
#                                        par_update=None, par_sim_update=None,
#                                        parameters_comput=None, proc_count=1):
#     # Experimental count of lineages having desired charateristics.
#     exp_data_selected = select_exp_lineages(exp_data, characteristics)
#     lineage_count = len(exp_data_selected[0]['cycle'])
#     # Corresponding simulation.
#     out = simulate_n_average_lineages(
#         lineage_count, simulation_count, types_of_sort, characteristics,
#         par_update=par_update, par_sim_update=par_sim_update,
#         parameters_comput=parameters_comput, proc_count=proc_count)
#     return out


# -----------------------------------
# Compute specific quantities to plot
# -----------------------------------

# Gcurves.
# --------


def compute_gtrigs(
    exp_data,
    simu_count,
    characteristics,
    gcurve,
    type_of_sort,
    par_update=None,
    proc_count=1,
    is_propB=False,
    rng=None,
):
    p_update = {"is_htype_seen": False}  # Default to compare with exp.
    p_update.update(par_update or {})

    # Experimental data.
    gtrigs_exp = select_exp_lineages(exp_data, characteristics)
    lineage_count = len(gtrigs_exp[0]["cycle"])
    if is_propB:
        exp_bprop = np.mean(exp_data[2][~np.isnan(exp_data[2])] == 1)
    if type_of_sort[0] != "l":
        gtrigs_exp = sort_lineages(gtrigs_exp, type_of_sort)[1]
    else:
        gtrigs_exp = gtrigs_exp[1]
    lineages = np.arange(lineage_count)

    # Simulated data.
    gtrigs_sim = simulate_n_average_lineages(
        lineage_count,
        simu_count,
        [type_of_sort],
        characteristics,
        par_update=p_update,
        proc_count=proc_count,
        rng=rng,
    )[type_of_sort]
    if is_propB:
        sim_bprop = gtrigs_sim[2][1]["btype"]["mean"]
    gtrigs_sim = gtrigs_sim[1]
    if "nta" in gcurve:
        gtrigs_exp = gtrigs_exp["nta"][:, int(gcurve[-1]) - 1]
        for stat_key, gtrigs in gtrigs_sim["nta"].items():
            gtrigs_sim[stat_key] = gtrigs[:, int(gcurve[-1]) - 1]
    else:
        gtrigs_exp = gtrigs_exp[gcurve[1:]]
        gtrigs_sim = gtrigs_sim[gcurve[1:]]
    if is_propB:
        return lineages, gtrigs_exp, gtrigs_sim, [exp_bprop, sim_bprop]
    else:
        return lineages, gtrigs_exp, gtrigs_sim


# Lineage types.
# --------------


def compute_lineage_types(
    exp_data,
    simu_count,
    characteristics,
    type_of_sort,
    gcurve,
    par_update=None,
    proc_count=1,
    rng=None,
):
    p_update = {"is_htype_seen": False}  # Default to compare with exp.
    p_update.update(par_update or {})

    # Experimental data.
    lin_types_exp = select_exp_lineages(exp_data, characteristics)
    lineage_count = len(lin_types_exp[0]["cycle"])
    if type_of_sort[0] != "l":
        lin_types_exp = sort_lineages(lin_types_exp, type_of_sort)[2]
    else:
        lin_types_exp = lin_types_exp[2]
    lineages = np.arange(lineage_count)

    # Simulated data.
    data = simulate_n_average_lineages(
        lineage_count,
        simu_count,
        [type_of_sort],
        characteristics,
        par_update=p_update,
        proc_count=proc_count,
        rng=rng,
    )[type_of_sort]
    lin_types_sim = data[2]

    if "nta" in gcurve:
        gtrigs_exp = data[1]["nta"][:, int(gcurve[-1]) - 1]
        gtrigs_sim = {}
        for stat_key, gtrigs in data[1]["nta"].items():
            gtrigs_sim[stat_key] = gtrigs[:, int(gcurve[-1]) - 1]
    else:
        gtrigs_exp = data[1][gcurve[1:]]
        gtrigs_sim = data[1][gcurve[1:]]
    return lineages, lin_types_exp, lin_types_sim, gtrigs_exp, gtrigs_sim


# Histograms.
# -----------


def compute_lmin_histogram_data(
    exp_data,
    simulation_count,
    characteristics_s,
    hist_lmins_axis,
    lineage_count_on_all_simu=None,
    par_update=None,
    proc_count=1,
    rng=None,
):
    """Warning: the number of experimental lineages having the given
    characteristics must be identical to all the types of sort given.

    """
    p_update = {"is_htype_seen": False}  # Default to compare with exp.
    p_update.update(par_update or {})
    psim_update = {"hist_lmins_axis": hist_lmins_axis}

    data = []
    lineage_counts = []
    for characs in characteristics_s:
        # We keep only exp lineages having required characteristics.
        exp_data_selected = select_exp_lineages(exp_data, characs)
        lineage_count = len(exp_data_selected[0]["cycle"])
        lineage_counts.append(lineage_count)
        if not isinstance(lineage_count_on_all_simu, type(None)):
            simulation_count = int(lineage_count_on_all_simu / lineage_count)
        # Simulation.
        # NB: we fit on experimental data where H type, if existing, are
        #     unseen, thus htype unseen for simulations as well.
        type_of_sort = type_of_sort_from_characteristics(characs)
        axis, hist = simulate_n_average_lineages(
            lineage_count,
            simulation_count,
            [type_of_sort],
            characs,
            par_update=p_update,
            par_sim_update=psim_update,
            proc_count=proc_count,
            rng=rng,
        )[type_of_sort][5]
        data.append([axis, hist])
    return lineage_counts, data


def compute_lcycle_histogram_data(
    exp_data,
    simulation_count,
    characteristics,
    par_update=None,
    proc_count=1,
    rng=None,
):
    p_update = deepcopy(par_update) or {}
    if "is_htype_accounted" in list(p_update.keys()):
        is_htype_accounted = p_update["is_htype_accounted"]
    else:
        is_htype_accounted = PAR_DEFAULT_LIN["is_htype_accounted"]
    psim_update = {"is_lcycle_count_saved": True}

    # Definition of some parameters
    exp_data_selected = select_exp_lineages(exp_data, characteristics)
    lineage_count = len(exp_data_selected[0]["cycle"])
    type_of_sort = type_of_sort_from_characteristics(characteristics)

    # Simulation.
    p_update["is_htype_seen"] = False
    data_sim = simulate_n_average_lineages(
        lineage_count,
        simulation_count,
        [type_of_sort],
        characteristics,
        par_update=p_update,
        par_sim_update=psim_update,
        proc_count=proc_count,
        rng=rng,
    )[type_of_sort]
    # Extraction of data on the number of long cycless.
    lcycle_counts_exp = exp_data_selected[5]
    lcycle_counts_sim = data_sim[4]

    # Same if type H accounted but simulating their "detection."
    data_sim_h = None
    if is_htype_accounted:
        p_update["is_htype_seen"] = True
        data_sim_h = simulate_n_average_lineages(
            lineage_count,
            simulation_count,
            [type_of_sort],
            characteristics,
            par_update=p_update,
            par_sim_update=psim_update,
            proc_count=proc_count,
            rng=rng,
        )[type_of_sort]
        lcycle_counts_sim_h = data_sim_h[4]
    return (lineage_count, lcycle_counts_exp, lcycle_counts_sim, lcycle_counts_sim_h)


# Evolution 2D arrays
# --------------------


def compute_evo_avg_data(
    exp_data,
    simulation_count,
    characteristics,
    types_of_sort=["lmin"],
    par_update=None,
    proc_count=1,
    rng=None,
):
    p_update = {"is_htype_seen": False}  # Default to compare with exp.
    p_update.update(par_update or {})
    psim_update = {"is_evo_saved": True}
    # Definition of some parameters
    exp_data_selected = select_exp_lineages(exp_data, characteristics)
    lineage_count = len(exp_data_selected[0]["cycle"])

    # Simulation.
    data_s = simulate_n_average_lineages(
        lineage_count,
        simulation_count,
        types_of_sort,
        characteristics,
        par_update=p_update,
        par_sim_update=psim_update,
        proc_count=proc_count,
        rng=rng,
    )
    return {key: data[:2] for key, data in data_s.items()}


# Time / gen postreat
# -------------------


def compute_postreat_data(
    exp_data,
    simulation_count,
    characteristics,
    postreat_dt,
    par_update=None,
    proc_count=1,
    rng=None,
):
    p_update = {"is_htype_seen": False}  # Default to compare with exp.
    p_update.update(par_update or {})
    psim_update = {"postreat_dt": postreat_dt, "is_evo_saved": True}
    # Definition of some parameters
    exp_data_selected = select_exp_lineages(exp_data, characteristics)
    lineage_count = len(exp_data_selected[0]["cycle"])
    type_of_sort = type_of_sort_from_characteristics(characteristics)

    # Simulation.
    data_s = simulate_n_average_lineages(
        lineage_count,
        simulation_count,
        [type_of_sort],
        characteristics,
        par_update=p_update,
        par_sim_update=psim_update,
        proc_count=proc_count,
        rng=rng,
    )[type_of_sort]
    return data_s[3]


def compute_postreat_data_vs_exp(
    simulation_count,
    characteristics,
    postreat_dt,
    par_update=None,
    proc_count=1,
    strain="TetO2-TLC1",
    rng=None,
):
    """Simulate lineages and return average time evolutions.

    Each simulation gather the evolution of `lineage_count` lineages having
    characteristics `characteristics`, where `lineage_count` is the number of
    lineages having these characteristics in the experimental dataset
    associated to `strain`.

    Parameters
    ----------
    simulation_count:
        Number of simulations to run and average.
    characteristic : list of string
        Characteristics of the lineages to simulated among 'arrestedi',
        'senescent', 'atype', 'btype', 'htype', 'dead', 'dead_naturally',
        'dead_accidentally'.
    postreat_dt : float
        Time step in 10 min.
    strain : string, optional
        Experimental dataset: 'TetO2-TLC1' (default) or 'RAD51'.

    """
    p_update = {"is_htype_seen": False}  # Default to compare with exp.
    p_update.update(par_update or {})
    psim_update = {"postreat_dt": postreat_dt, "is_evo_saved": True}
    # Definition of some parameters
    lineage_count = count_exp_lineages(characteristics)
    type_of_sort = type_of_sort_from_characteristics(characteristics)

    # Simulation.
    data_s = simulate_n_average_lineages(
        lineage_count,
        simulation_count,
        [type_of_sort],
        characteristics,
        par_update=p_update,
        par_sim_update=psim_update,
        proc_count=proc_count,
        rng=rng,
    )[type_of_sort]
    return data_s[3]
