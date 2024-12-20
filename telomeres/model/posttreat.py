#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 13:11:13 2024

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
from scipy import interpolate

from telomeres.dataset.extract_processed_dataset import \
    extract_distribution_telomeres_init

from sys import path
import os
absolute_path = os.path.abspath(__file__)  # Path to extract_processed_dataset.
current_dir = os.path.dirname(absolute_path)  # Path to auxiliary directory.
parent_dir = os.path.dirname(current_dir)  # Path to telomeres directory.
projet_dir = os.path.dirname(parent_dir)
path.append(os.path.join(projet_dir))

DISTRIBUTION_RAW = extract_distribution_telomeres_init()


def transform_l_init(distribution=None, par_l_init=None):
    """Transform the distribution according to `par_l_init = (ltrans, l0, l1)`.

    The transformation consists in (see (Rat PhD thesis) and (Rat et al.)):
      > Translation by `ltrans` [bp].
      > Dilatations at both sides of the mode of the distribution s.t. the
        mimimum / maximum of the support of the distribution is translated by
        `l0` / `l1`: for `f`the distribution and `f_new` its transformation,
            min supp(f_new) = min supp(f) + l0
            max supp(f_new) = max supp(f) + l1
            mode(f_new) = mode(f)

    Parameters
    ----------
    distribution : list of ndarray
        Distribution function in the form `(lengths, densities)` such that
        `densities[i] = P(L = lengths[i])`. Default is None, in which case the
        distribution of `data/processed/` is used.
    par_l_init : None or list
        If None (default) no transformation applied.

    """
    if isinstance(distribution, type(None)):
        distribution = DISTRIBUTION_RAW
    if isinstance(par_l_init, type(None)):
        return np.copy(distribution)
    else:
        ltrans, l0, l1 = par_l_init
    lengths, densities = distribution

    # Indexes of interest.
    # > Mode.
    lmod_idx = np.where(lengths == np.argmax(densities))[0][0]
    lmod = lengths[lmod_idx]
    # > Minimum of the support.
    linf_idx = np.where(densities > 0)[0][0]
    linf = int(lengths[linf_idx])
    # > Maximum of the support.
    lsup_idx = np.where(densities > 0)[0][-1]
    lsup = int(lengths[lsup_idx])

    # Transformation of the distribution.
    lengths_tf = np.copy(lengths)
    lengths_tf[linf_idx:lmod_idx] = (lengths_tf[linf_idx:lmod_idx] - linf) \
        * (lmod - linf - l0) / (lmod - linf) + linf + l0
    lengths_tf[lmod_idx:] = (lengths_tf[lmod_idx:] - lmod) * \
        (lsup + l1 - lmod) / (lsup - lmod) + lmod
    lengths_tf[:linf_idx] = np.linspace(1, lengths_tf[linf_idx],
                                        linf_idx+1)[:-1]

    lengths_new = np.arange(1, np.round(lengths_tf[-1]) + 1)
    densities_new = interpolate.interp1d(lengths_tf, densities)(lengths_new)
    mass = np.sum(densities_new * np.diff(np.append(0, lengths_new)))
    densities_new = densities_new / mass
    lengths_new = lengths_new + ltrans
    return lengths_new, densities_new


def transform_l_init_old(distribution=None, par_l_init=None):
    """Function `transform_l_init` before corrections."""
    if isinstance(distribution, type(None)):
        distribution = DISTRIBUTION_RAW
    if isinstance(par_l_init, type(None)):
        return np.copy(distribution)
    else:
        ltrans, l0, l1 = par_l_init
    lengths, densities = distribution

    # Indexes of interest.
    lmod_idx = np.where(lengths == np.argmax(densities))[0][0]
    lmod = lengths[lmod_idx]
    # print(np.sum(np.diff(np.append(lengths, lengths[-1]) + 1) * densities))
    # print('std1', lmod, np.std((lengths - lmod) * densities))

    linf_idx = np.where(densities > 0)[0][0]
    linf = int(lengths[linf_idx])

    lsup_idx = np.where(densities > 0)[0][-1]
    lsup = int(lengths[lsup_idx])

    # Transformation of the distribution.
    lengths_new = np.append(lengths, lengths[-1] + 1)
    lengths_new[linf_idx:lmod_idx] = (lengths_new[linf_idx:lmod_idx] - linf) \
        * (lmod - linf - l0) / (lmod - linf) + linf + l0
    lengths_new[lmod_idx:] = (lengths_new[lmod_idx:] - lmod) * \
        (lsup + l1 - lmod) / (lsup - lmod) + lmod
    lengths_new[:linf_idx] = np.linspace(lengths_new[0], lengths_new[linf_idx],
                                         linf_idx + 1)[:-1]
    densities_new = densities / np.sum(np.diff(lengths_new) * densities)
    lengths_new = lengths_new[:-1] + ltrans
    return lengths_new, densities_new
