#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 15:29:43 2024

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
