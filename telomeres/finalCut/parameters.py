#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 11:17:15 2024

@author: anais
"""

import os.path as op
import numpy as np

ABSOLUTE_PATH = op.abspath(__file__)  # Path to extract_processed_dataset.
CURRENT_DIR = op.dirname(ABSOLUTE_PATH)  # Path to auxiliary directory.
PARENT_DIR = op.dirname(CURRENT_DIR)  # Path to telomeres directory.
PROJECT_DIR = op.dirname(PARENT_DIR)

FOLDER = op.join(PROJECT_DIR, "data_finalCut", "processed")


def extract_cycles_dataset_finalCut(folder=FOLDER):
    """Extract the data postreated by `finalCut.make_dataset_cycles.py`.

    If no such data return None for CDTS_FINALCUT to be defined even in the
    abscence of FinalCut data (since called from telomeres.lineage.simulation).
    However if you are in a finalCut config, `finalCut.make_dataset_cycles.py`
    absolutly need to be run first.

    """
    cdts = {"raf": {}, "gal": {}}
    sdir = op.join(folder, "noFc_n2")
    if not op.exists(op.join(sdir, "cycles_normal_raf_dox.csv")):
        return None
    cdts["raf"]["nor"] = np.loadtxt(op.join(sdir, "cycles_normal_raf_dox.csv"))
    cdts["raf"]["arr"] = np.loadtxt(op.join(sdir, "cycles_arrest_raf_dox.csv"))
    cdts["gal"]["nor"] = np.loadtxt(op.join(sdir, "cycles_normal_gal.csv"))
    cdts["gal"]["arr"] = np.loadtxt(op.join(sdir, "cycles_arrest_gal.csv"))
    return cdts


# AVG_CUT_DELAY = 1 / 0.577 # 7287 # 0.577 # 1.613 # Average nb of gen between
#  end of galactose and cutting.
# PROBA_CUT_ESCAPE = .1 # Proba of/proportion of lineages escaping cutting.
IDX_DOX = 0
IDX_GAL = 36  # 6h after Dox addition.
IDX_RAF = 9 * 6  # 2 * 36  # 12h after Dox addition.
IDXS_FRAME = [IDX_DOX, IDX_GAL, IDX_RAF]

# CDTS Dataset for finalCul experiment [10 min].
CDTS_FINALCUT = extract_cycles_dataset_finalCut()

PAR_FINAL_CUT = {"idxs_frame": IDXS_FRAME, "edlyay": 9}

PAR_FINAL_CUT_POP = None  # Temporary
