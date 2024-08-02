#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 11:45:46 2022

@author: arat

List the keys of commonly-used dictionaries of the `telomeres` package to avoid
rewriting them elsewhere and risking spelling mistakes.

Importing the lists of keys of this script, rather that writing them again, is
thus recommanded.

"""

from telomeres.model.parameters import HTYPE_CHOICE


# Single cell data
# ----------------

data_saved_keys = ['nta_counts', 'lengths', 'ancestors', 'generations',
                   'sen_counts']
data_keys = data_saved_keys.copy()
data_keys.append('clocks')

if HTYPE_CHOICE:
    type_keys = ['atype', 'btype', 'mtype', 'htype']
    # Type M for misclassified cells (type B that would be recognize as A).
else:
    type_keys = ['atype', 'btype']


# History data
# ------------

history_keys = ['sat_time', 'sat_prop', 'history_dead']
sat_keys = ['sat_time', 'sat_prop']


# Time evolution data
# -------------------

# Telomere lengths.
evo_l_keys_0 = ['evo_lavg_sum', 'evo_lmin_sum', 'evo_lmin_max', 'evo_lmin_min']
evo_l_keys = [*evo_l_keys_0, 'evo_lmode']

# Number of cells.
# > With respect to the ancestor index.
evo_c_anc_keys = ['evo_c_ancs', 'evo_c_B_ancs', 'evo_c_B_sen_ancs',
                  'evo_c_sen_ancs']
# > With respect to the generation.
evo_c_gen_keys = ['evo_c_gens']
if HTYPE_CHOICE:
    evo_c_anc_keys.append('evo_c_H_ancs')
# > All.
evo_c_keys = evo_c_anc_keys.copy()
evo_c_keys.extend(evo_c_gen_keys)


# Keys for postreatment
# ---------------------

# Keys of data to sum.
to_sum_keys = evo_c_keys.copy()
evo_l_to_sum_keys = ['evo_lavg_sum', 'evo_lmin_sum']
to_sum_keys.extend(evo_l_to_sum_keys)

evo_ancs_n_sum_keys = evo_c_anc_keys.copy()
evo_ancs_n_sum_keys.extend(evo_l_to_sum_keys)

evo_keys = evo_c_keys.copy()
evo_keys_0 = evo_keys.copy()
evo_keys_0.extend(evo_l_keys_0)
evo_keys.extend(evo_l_keys)

all_keys = evo_keys.copy()
all_keys.extend(history_keys)

# New keys.
evo_p_anc_keys = evo_c_anc_keys.copy()
evo_p_anc_keys = [evo_p_anc_keys[i].replace('_c', '_p') for i in
                  range(len(evo_c_anc_keys))]
if HTYPE_CHOICE:
    evo_p_anc_keys.append('evo_p_H_sen_ancs')

evo_c_keys_new = [evo_c_anc_keys[i].replace('_ancs', '') for i in
                  range(len(evo_c_anc_keys))]
evo_1Dkeys_new = evo_c_keys_new.copy()
for key in evo_c_keys_new:
    if '_c_' in key:
        evo_1Dkeys_new.append(key.replace('_c_', '_p_'))
    if HTYPE_CHOICE:
        evo_1Dkeys_new.append('evo_p_H_sen')

evo_l_keys_af_postreat = [key.replace('_sum', '_avg') for key in evo_l_keys]

evo_c_keys_to_postreat = evo_c_anc_keys.copy()
evo_c_keys_to_postreat.extend(evo_p_anc_keys)
evo_c_keys_to_postreat.extend(evo_c_gen_keys)
evo_c_keys_to_postreat.extend(evo_1Dkeys_new)
