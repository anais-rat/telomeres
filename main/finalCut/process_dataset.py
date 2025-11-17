#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 08:20:32 2024

@author: arat

Run this script to process the experimental data contained in
`data_finalCut/raw`. This will create the folder `data_finalCut/processed`,
and fill it with processed data.

"""

import os
import scipy.io as sio

import telomeres.finalCut.make_cycles_dataset as mk


IS_SAVED = True  # If True data is saved.
if __name__ == "__main__":
    IS_PRINTED = True
else:
    IS_PRINTED = False


# Threshold between normal/long cycles and associated datasets
# ------------------------------------------------------------

# Two experimental conditions:
# - Raffinose
# - Galactose

# For each condition we compute:
# 1) A threshold D between normal (<= D) and long (> D) cycles
# 2) A dataset of normal cycles and a dataset of long cycles
#    based on the data of experiment 'noFc_noCas9_noDox'
#    ('noFc_1' and 'noFc_2' for comparison only).
KEYS = ["noFc_1", "noFc_2", "noFc_noCas9_noDox"]

# Extraction of CDTs in each dataset separately.
CYCLES_EXP, thresholds = {}, {}
for key in KEYS:
    mk.print_if_true("\n" + key, IS_PRINTED)
    path = os.path.join(mk.FOLDERS[key], "results_analysis.mat")
    out = sio.loadmat(path)["results_analysis"]
    idx_dox, idx_gal, idx_nogal = mk.IDXS_CDT[key]
    if IS_SAVED:
        SKEY = key
    else:
        SKEY = None
    cycles = mk.extract_cycles_dataset(
        out, gal_delay=idx_nogal - idx_gal, dox_frame_idx=idx_dox, saved_key=SKEY
    )
    CYCLES_EXP[key] = cycles
    # Print thresholds in each dataset for information.
    thresholds[key] = mk.compute_cycle_thresholds(cycles, IS_PRINTED)
    if IS_PRINTED:
        print("Number of cycles in the dataset 'gal'", len(cycles["gal"]))
        print("Number of cycles in the dataset 'raf_dox'", len(cycles["raf_dox"]))


# Gather both datasets to make the dataset of 'noFC_n2'.
mk.print_if_true("\n" + "noFc_n2", IS_PRINTED)
out = mk.gather_cycle_dataset(["noFc_1", "noFc_2"], "noFc_n2", is_printed=IS_PRINTED)

# # 1) Computation of the thresholds based on the dataset of 'noFC_n2'.
# THRESHOLDS = compute_cycle_thresholds(cycles, IS_PRINTED)

# 1) Computation of the thresholds based on the dataset of 'noFc_noCas9_noDox'.
THRESHOLDS = thresholds["noFc_noCas9_noDox"]

# 2) Creation of the datasets of normal/long cycles based on the thresholds.
NCYCLES, LCYCLES = mk.make_cycles_dataset(
    "noFc_n2", thresholds=THRESHOLDS, is_printed=IS_PRINTED
)


# Post-treatment: generations of arrests (based on THRESHOLDS) and cycles
# ---------------------------------------------------------------------

DATA_EXP = {}

# Iteration on all experiments with Dox.
for key in list(mk.IDXS_CDT.keys()):
    if key != "noFc_noCas9_noDox":
        # Postreat.
        if IS_PRINTED:
            print(key)
        path = os.path.join(mk.FOLDERS[key], "results_analysis.mat")
        DATA_EXP[key] = mk.postreat_experimental_lineages_from_path(
            path,
            "results_analysis",
            threshold=THRESHOLDS["raf"],
            gen_count_min=1,
            par_multiple_thresholds=[THRESHOLDS["gal"], mk.IDXS_CDT[key][2]],
        )
        # # Save.
        # np.save(os.path.join(FOLDERS[key], "postreat.npy"), out)

# Gather post-treated data of replicated experiments.
for key in ["noFc_n2", "Fc0_n2", "Fc20_n2", "Fc30_n2", "Fc40_n2", "Fc50_n2", "Fc70_n2"]:
    key_1 = key.replace("_n2", "_1")
    key_2 = key.replace("_n2", "_2")
    DATA_EXP[key] = mk.gather_postreated_output(DATA_EXP[key_1], DATA_EXP[key_2])
    # np.save(os.path.join(FOLDERS[key], "postreat.npy"), DATA_EXP[key])

# Compute proportion of long and normal cycles.
data_key = "noFc_noCas9_noDox"
# > noFc_noCas9_noDox dataset
out = mk.compute_prop_arrest(data_key, THRESHOLDS)
if IS_PRINTED:
    print("\n", data_key)
    print(f"Proportion of long cycles in {data_key} under gal: {out['gal']} %")
    print(f"Proportion of long cycle in {data_key} under raf: {out['raf_dox']} %")
