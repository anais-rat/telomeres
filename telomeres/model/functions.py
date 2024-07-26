#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 16:08:23 2020

@author: arat

Defined below are auxiliary functions useful throughout the whole project.

"""

import math
import numpy as np
import numpy.random as rd
# NB: for parallelization issues need to use rd.RandomState() rather than rd.
# rd.RandomState() replaced by rd. but seeds initilize for reproducinility.
# idem with population_simulation

from telomeres.auxiliary.functions import inverse_cdf
from telomeres.dataset.extract_processed_dataset import \
    extract_distribution_cycles, extract_distribution_telomeres_init
from telomeres.model.posttreat import transform_l_init
from telomeres.model.parameters import CHROMOSOME_COUNT, PAR_L_INIT


# -------------
# PRE-LOAD DATA
# -------------

# Distributions of cycle duration times [min].
# WARNING. If `auxiliary.extract_processed_dataset.PAR_CYCLES_POSTREAT` updated
#    need to update the processed data saved in `data/processed/cycles*` by
#    running `makeFile/processed_dataset.py`.
CDTS = extract_distribution_cycles()

# Distribution of initial telomere lengths.
# WARNING. If `PAR_L_INIT` is updated, need to create a .csv file containing
#    the distribution transformed according to these new parameters by running
#    `makeFile/processed_dataset.py`.
DISTRIBUTION_PAR = extract_distribution_telomeres_init(par_l_init=PAR_L_INIT)


# -----------------------
# Model-related functions
# -----------------------


# Telomere lengths
# ----------------

# > Initial distribution of telomere length

def draw_cell_lengths(chromosome_count, par_l_init):
    """Draw `2 x chromosome_count` initial telomere lengths.

    Lengths drawn independently from the distribution
    `data/processed/telomeres_initial_distribution/original.csv`, possibly
    modified through `par_l_init`.

    Parameters
    ----------
    chromosome_count : int
        Number of chromosomes to create.
    par_l_init : list
        Parameters (l_trans, l_0, l_1) defined in (Rat et al.).
        See also `transform_l_init` docstring.

    Returns
    -------
    lengths : ndarray
        2D array (2, chromosome_count) of the lengths of the two telomeric
        extremities of `chromosome_count` chromosomes.

    """
    telomere_count = 2 * chromosome_count
    lengths = np.zeros(telomere_count)
    # Distribution transformed according to `par_l_init`.
    if np.all(np.array(par_l_init) == np.array(PAR_L_INIT)):  # If default
        # `par_l_init` parameters, use the already computed distribution.
        support, proba = DISTRIBUTION_PAR
    else:  # Otherwise, compute the transformed distribution.
        support, proba = transform_l_init(par_l_init=par_l_init)
    # Draw from the transformed distribution.
    u_rands = rd.uniform(0, 1, telomere_count)
    for i in range(telomere_count):
        lengths[i] = np.round(inverse_cdf(u_rands[i], support, proba))
    # Return the lengths obtained with translated initial distribution.
    return np.reshape(lengths, (2, chromosome_count))


def draw_cells_lengths(cell_count, par_l_init):
    """Draw the initial telomere lengths of `cell_count` cells.

    Telomere lengths drawn independently from the distribution
    `data/processed/telomeres_initial_distribution/original.csv`, possibly
    modified through `par_l_init`.

    Parameters
    ----------
    cell_count : int
        Number of cell disributions to create.
    par_l_init : list
        Parameters (l_trans, l_0, l_1) defined in (Rat et al.).
        See also `transform_l_init` docstring.

    Returns
    -------
    lengths : ndarray
        3D array (cell_count, 2, chromosome_count) of the lengths of the two
        telomeric extremities of the `CHROMOSOME_COUNT` chromosomes of the
        `cell_count` cells.

    """
    return np.array([draw_cell_lengths(CHROMOSOME_COUNT, par_l_init) for i in
                     range(cell_count)])


# Laws of arrest in the cell cycle
# --------------------------------


def law_exponential(length, a, b):
    """Return True with probability `p(length) = b exp(-a length)`."""
    # Compute telomere-length dependent probability to trigger an arrest.
    proba = min(1, b * math.exp(- a * length))
    # Test if an arrest is triggered according to this probiblity.
    return bool(rd.binomial(1, proba))


# > Onset of on-terminal arrest (nta).

def is_nta_trig(length, parameters):
    """Test if a telomere with length `length` triggers a non-terminal arrest.

    Test with probabibility `p_nta(length) = b exp(-a length)`.

    Parameters
    ----------
    length : integer
        Length of the telomere tested (assumed to be the cell's shortest).
    parameters : list
        Parameters (a, b) for `p_nta`.

    """
    return law_exponential(length, *parameters)


# > Onset of terminal/senescent arrest.

def is_sen_trig(length, parameters):
    """Test if a telomere with length `length` triggers senescence.

    If `length <= lmin` senescence is trigerred, otherwise it is trigerred with
    probability `p(length) = b exp(-a length)`.

    """
    a, b, lmin = parameters
    if length <= lmin:  # If deterministic threshold reached.
        return True  # Senescence is triggered.
    # Otherwise, test with exponential law.
    return law_exponential(length, a, b)


# > Exit of non-terminal arrest.

def is_repaired(p_repair):
    """Test if a non-terminally arrested cell exits its sequence of
    non-terminal long cycles (True) or continues the sequence (False).

    """
    return rd.binomial(1, p_repair)


# > Onset of death.

def is_dead(sen_count, p_exit):
    """Test if a senescent cell dies (True) or continues to divide (False).

    Parameters
    ----------
    sen_count : int
        Number of sencescent ancestors in the lineage of the cell.
    p_exit : dict
        p_exit['sen_limit'] : Maximum number of senescent cycles.
        p_exit['death'] : Probability die "naturally", from senescence.

    """
    return (rd.binomial(1, p_exit['death']) or sen_count > p_exit['sen_limit'])


def is_accidentally_dead(p_death_acc):
    """Test if a cell accidentally dies (True) or continues to divide."""
    return rd.binomial(1, p_death_acc)


# Distribution of cycle duration time (cdt)
# -----------------------------------------


def draw_cycles_atype(cell_count):
    """Draw `cell_count`cell cycle durations [min] of non-senescent A cells."""
    return rd.choice(CDTS['norA'], cell_count)


def draw_cycle(arrest_count, is_senescent):
    """Return a cycle duration time [min] for a newborn cell in the state
    entered as argument.

    Parameters
    ----------
    arrest_count : int
        Number of sequences of non-terminal arrests the lineage of the cell
        went through so far. For non-senescent cell, it is positive if the cell
        is arrested, negative if it is in normal cycle.
    is_senescent : bool
        Is False if the cell is not senescent, True if senescent.

    """
    if is_senescent:  # The cell is senescent.
        return rd.choice(CDTS['sen'])
    if arrest_count == 0:  # Non-senescent type A.
        return draw_cycles_atype(1)[0]
    if arrest_count == 1:  # Non-senescent type B in a 1st sequence of arrests.
        return rd.choice(CDTS['nta'])
    if arrest_count < 0:  # Non-senescent type B in a normal cycle.
        return rd.choice(CDTS['norB'])
    # Non-senescent type B in a 2nd, 3rd... sequence of arrests.
    return rd.choice(CDTS['nta'])


def desynchronize(cycles):
    """Delays cell's cycle duration time `cycles`."""
    delays = rd.uniform(0, cycles, len(cycles))
    return cycles - delays
