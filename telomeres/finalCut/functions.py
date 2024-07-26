#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 15:29:43 2024

@author: anais

"""

import numpy.random as rd
import telomeres.finalCut.fit_cut_efficiency as fce
from telomeres.finalCut.parameters import CDTS_FINALCUT

# > Telome cutting in final cut experiments.

def is_cut_exponential(cdt_after_gal, dt_since_gal):
    """Test if a cell has one of its telomeres cut during.

    cdt_after_gal : float
        Time that the cell spent after Galactose addition (this is extaclty its
        cycle duration time if born after Galactose addition).
    dt_since_gal : float
        Time spent between galactose addition and the end of the cell cycle.

    """
    min_to_h = 1 / 60
    a = (dt_since_gal - cdt_after_gal) * min_to_h
    b = dt_since_gal * min_to_h
    # We compute P(T<b | T>a) where, for time 0 being the time where galactose
    # was added, T is the r.v. of the time of cut [h], a is the time of birth
    # (if the cell was born after Gal addition) or 0 (otherwize).
    proba = (fce.fit_cdf(b) - fce.fit_cdf(a)) / (1 - fce.fit_cdf(a))
    return rd.binomial(1, proba)


def draw_cycle_finalCut(arrest_count, is_senescent, is_galactose):
    """Return a cycle duration time [min] for a new born cell in the state
    entered as argument (see `draw_cycle` docstring) in galactose or raffinose
    conditions (if `is_galactose` is True or False respectively).

    """
    # If the cell is arrested.
    if is_senescent or arrest_count > 0:
        if is_galactose:
            return rd.choice(CDTS_FINALCUT['gal']['arr']) * 10
        return rd.choice(CDTS_FINALCUT['raf']['arr']) * 10
    # Otherwise it experiences a normal cycle.
    if is_galactose:
        return rd.choice(CDTS_FINALCUT['gal']['nor']) * 10
    return rd.choice(CDTS_FINALCUT['raf']['nor']) * 10
