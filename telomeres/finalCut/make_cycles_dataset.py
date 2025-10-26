#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 18:31:14 2023

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

from os.path import join
import os
import numpy as np
import pandas as pd
import scipy.io as sio

import telomeres.auxiliary.functions as af
import telomeres.lineage.posttreat as pst


# Data information
# ----------------

# Folders containing the data.
ABSOLUTE_PATH = os.path.abspath(__file__)  # Path to the current .py
CURRENT_DIR = os.path.dirname(ABSOLUTE_PATH)  # Path to finalCut directory.
PARENT_DIR = os.path.dirname(CURRENT_DIR)  # Path to telomeres directory.
PROJECT_DIR = os.path.dirname(PARENT_DIR)

SUBF = os.path.join(PROJECT_DIR, "data_finalCut", "raw", "uFluidicData_maths_23-11-30")

DIR = os.path.join(PROJECT_DIR, "data_finalCut", "processed")

FOLDERS = {
    "noFc_noCas9_noDox": join(SUBF, "noFc", "noCas9", "noDox", "20220901_Gal82"),
    "noFc_1": join(SUBF, "noFc", "Cas9", "Dox", "20211202_CTL_Gal259_PB"),
    "noFc_2": join(SUBF, "noFc", "Cas9", "Dox", "20220223_CTL_GalDox103_PB"),
    "noFc_n2": join(SUBF, "noFc", "Cas9", "Dox", "sum_n2"),
    #
    "Fc0_1": join(SUBF, "TG0", "20211202_Fc0_Gal259_PB"),
    "Fc0_2": join(SUBF, "TG0", "20220223_Fc0_Gal103-134_PB"),
    "Fc0_n2": join(SUBF, "TG0", "Fc0_sum_n2_Gal259"),
    #
    "Fc20_1": join(SUBF, "TG20", "Dox", "20220223_Fc20_GalDox103-134"),
    "Fc20_2": join(SUBF, "TG20", "Dox", "20220624_Fc20_Gal108-159"),
    "Fc20_n2": join(SUBF, "TG20", "Dox", "Fc20_sum2"),
    #
    "Fc30_1": join(SUBF, "TG30", "Dox", "20200626_Fc30_Gal120-159_PB"),
    "Fc30_2": join(SUBF, "TG30", "Dox", "20200709_Fc30_Gal122-159_PB"),
    "Fc30_n2": join(SUBF, "TG30", "Dox", "Fc30_sum2"),
    #
    "Fc40_1": join(SUBF, "TG40", "20201125_Fc40_Gal119_PB"),
    "Fc40_2": join(SUBF, "TG40", "20210806_Fc40_Gal129-160_PB"),
    "Fc40_n2": join(SUBF, "TG40", "FC40_Sum2"),
    #
    "Fc50_1": join(SUBF, "TG50", "20200626_Fc50_Gal120_PB"),
    "Fc50_2": join(SUBF, "TG50", "20210806_Gal129_PB"),
    "Fc50_n2": join(SUBF, "TG50", "TG50_n2"),
    #
    "Fc70_1": join(SUBF, "TG70", "20210806_Fc70_DoxGal129-160_PB"),
    "Fc70_2": join(SUBF, "TG70", "20230701_Fc70_DoxGal134-172"),
    "Fc70_n2": join(SUBF, "TG70", "Fc70_n2"),
}

# For each experiment: indexes of the frames (i.e. times [10 min]) at which Dox
# was added and Gal added and removed.
IDXS_CDT = {  # [t_Dox, t_Gal, t_noGal]
    "noFc_noCas9_noDox": [None, 82, 121],
    "noFc_1": [222, 259, 298],
    "noFc_2": [71, 103, 134],
    "Fc0_1": [222, 259, 298],
    "Fc0_2": [71, 103, 134],
    "Fc20_1": [71, 103, 134],
    "Fc20_2": [74, 108, 159],
    "Fc30_1": [91, 120, 159],
    "Fc30_2": [93, 122, 159],
    "Fc40_1": [88, 119, 154],
    "Fc40_2": [89, 129, 160],
    "Fc50_1": [91, 120, 159],
    "Fc50_2": [89, 129, 160],
    "Fc70_1": [89, 129, 160],
    "Fc70_2": [100, 134, 172],
}


# Posttreat functions
# -------------------


def postreat_experimental_lineages_from_path(
    data_path, data_key, threshold, gen_count_min, par_multiple_thresholds=None
):
    data = sio.loadmat(data_path)
    data = data[data_key]
    return pst.postreat_experimental_lineages(
        data, threshold, gen_count_min, par_multiple_thresholds
    )


def extract_cycles_dataset(data, gal_delay=30, dox_frame_idx=None, saved_key=None):
    """Extract and return from `data` the cell cycles [10 min], from strictly
    after DOX addition to the end of measurements, of all the lineages longer
    than `gen_count_by_lineage_min`. Besides, compute and return information
    on each lineage (generations at which event triggered and type).

    """
    lineage_count = len(data["endlineage"][0])

    # Array, for each lineage, of the time at which DOX has been added.
    dox_time = dox_frame_idx or 0
    gal_times = data["DOXaddition"][0].astype(int)
    nogal_times = gal_times + gal_delay

    # Extraction of cycles duration times after DOX addition saving some info.
    # > Initialization of lists:
    # cycles_dox = [] # cycles[i]: array of cycle durations of the ith lineage.
    cycles = {"dox": [], "gal": [], "raf": [], "raf_dox": []}

    # > Iteration on all lineages.
    for i in range(lineage_count):
        # Extraction of times of division/death (except last one) in ith lin.
        # div_times = data['track'][0, i].astype(int)[:, :2]
        # birth_times = np.append(np.zeros((len(div_times), 1)),
        #                         div_times[:, :-1], axis=1)

        birth_times = data["track"][0, i].astype(int)[:, 0]
        div_times = data["track"][0, i].astype(int)[:, 1]
        # We keep only times after Dox addition.
        # is_dox = div_times[:, 0] > dox_time
        is_dox = birth_times >= dox_time
        # div_times_dox = div_times[is_dox]
        # Or during Galactose addition.
        # is_gal = np.logical_and(div_times[:, 0] > gal_times[i],
        #                         birth_times[:, 0] < nogal_times[i])
        # # Option 1: Gal-cycle as soon as there is Gal in a part of the cycle.
        # is_gal = np.logical_or(np.logical_and(birth_times >= gal_times[i],
        #                                       birth_times < nogal_times[i]),
        #                        np.logical_and(div_times > gal_times[i],
        #                                       div_times <= nogal_times[i]))
        # Option 2: Gal-cycle cycles that starts with Gal.
        is_gal = np.logical_and(
            birth_times >= gal_times[i], birth_times < nogal_times[i]
        )

        # div_times_gal = div_times[is_gal, :]
        # Or when there is no Galactose.
        # div_times_raf = div_times[~is_gal, :]
        is_raf_dox = np.logical_and(~is_gal, is_dox)
        # div_times_raf_dox = div_times[is_raf_dox, :]
        # We turn "times of division" to cycle duration times.
        # cycles['dox'].extend(div_times_dox[:, 1] - div_times_dox[:, 0])
        # cycles['gal'].extend(div_times_gal[:, 1] - div_times_gal[:, 0])
        # cycles['raf'].extend(div_times_raf[:, 1] - div_times_raf[:, 0])
        # cycles['raf_dox'].extend(div_times_raf_dox[:, 1] -
        #                            div_times_raf_dox[:, 0])
        cycles["dox"].extend(div_times[is_dox] - birth_times[is_dox])
        cycles["gal"].extend(div_times[is_gal] - birth_times[is_gal])
        cycles["raf"].extend(div_times[~is_gal] - birth_times[~is_gal])
        cycles["raf_dox"].extend(div_times[is_raf_dox] - birth_times[is_raf_dox])
        # print(div_times[:, 1] - div_times[:, 0])
        # print('dox', div_times_dox[:, 1] - div_times_dox[:, 0])
        # print('gal', div_times_gal[:, 1] - div_times_gal[:, 0])
        # print('raf', div_times_raf[:, 1] - div_times_raf[:, 0])
        # print('raf_dox', div_times_raf_dox[:,1] - div_times_raf_dox[:,0])

    # Saving.
    if saved_key is not None:
        sfolder = os.path.join(DIR, saved_key)
        if not os.path.exists(sfolder):
            os.makedirs(sfolder)
        for key, cdts in cycles.items():
            path = join(sfolder, f"cycles_{key}.csv")
            pd.DataFrame(cdts).to_csv(path, header=None, index=None)
    return cycles


def compute_cycle_threshold(cycles, is_printed=False, key=""):
    cst = 2
    if "raf" in key:
        cst = 2
    threshold = np.mean(cycles) + cst * np.std(cycles)
    if is_printed:
        print("Threshold: ", threshold)
        print("Mean: ", np.mean(cycles))
        print("STD: ", np.std(cycles), "\n")
    return threshold


def compute_cycle_thresholds(cycles, is_printed=False):
    thresholds = {}
    for key, cdts in cycles.items():
        if is_printed:
            print(key)
        thresholds[key] = compute_cycle_threshold(cdts, is_printed, key)
    return thresholds


def gather_cycle_dataset(
    data_key_to_gather,
    skey=None,
    keys=["dox", "gal", "raf", "raf_dox"],
    is_printed=False,
):
    folders_to_gather = [os.path.join(DIR, key) for key in data_key_to_gather]
    cycles = {key: [] for key in keys}
    # Extracting and concatenating.
    for folder in folders_to_gather:
        for key in keys:
            cycles[key].extend(np.loadtxt(join(folder, f"cycles_{key}.csv")))
    if is_printed:
        for key in keys:
            if is_printed:
                print(key)
            compute_cycle_threshold(cycles[key], is_printed, key)
    # Saving.
    if skey is not None:
        sfolder = os.path.join(DIR, skey)
        if not os.path.exists(sfolder):
            os.makedirs(sfolder)
        for key, cdts in cycles.items():
            path = join(sfolder, f"cycles_{key}.csv")
            pd.DataFrame(cdts).to_csv(path, header=None, index=None)
    return cycles


def make_cycles_dataset(data_key, thresholds=None, is_printed=False, is_saved=True):
    folder = os.path.join(DIR, data_key)
    cycles, ncycles, lcycles = {}, {}, {}
    if thresholds is None:
        thresholds = {"gal": None, "raf_dox": None}

    # Extracting and concatenating.
    for key in ["gal", "raf_dox"]:
        cycles[key] = np.loadtxt(join(folder, f"cycles_{key}.csv"))
        if thresholds[key] is None:
            thresholds[key] = compute_cycle_threshold(cycles[key], is_printed, key)
        ncycles[key] = cycles[key][cycles[key] <= thresholds[key]]
        lcycles[key] = cycles[key][cycles[key] > thresholds[key]]

        if is_saved:  # Saving.
            lpath = join(folder, f"cycles_arrest_{key}.csv")
            npath = join(folder, f"cycles_normal_{key}.csv")
            pd.DataFrame(ncycles[key]).to_csv(npath, header=None, index=None)
            pd.DataFrame(lcycles[key]).to_csv(lpath, header=None, index=None)
    if is_printed:
        print(thresholds)
    return ncycles, lcycles


def gather_postreated_output(output_1, output_2):
    # ({'cycle': cycles}, gtrigs, lineage_types, is_unseen_htypes,
    #      is_accidental_deaths, lcycle_per_seq_counts)
    gen1 = len(output_1[0]["cycle"][0])
    gen2 = len(output_2[0]["cycle"][0])
    gen = max(gen1, gen2)
    cycles1 = af.reshape_with_nan(output_1[0]["cycle"], gen, axis=-1)
    cycles2 = af.reshape_with_nan(output_2[0]["cycle"], gen, axis=-1)
    output = [{"cycle": np.append(cycles1, cycles2, 0)}]

    gtrigs = {}
    for key in output_1[1]:
        gtrigs1 = output_1[1][key]
        gtrigs2 = output_2[1][key]
        if key == "nta":
            nta_count = max(len(gtrigs1[0]), len(gtrigs2[0]))
            gtrigs1 = af.reshape_with_nan(gtrigs1, nta_count, axis=-1)
            gtrigs2 = af.reshape_with_nan(gtrigs2, nta_count, axis=-1)
        gtrigs[key] = np.append(gtrigs1, gtrigs2, 0)
    output.append(gtrigs)

    for i in range(2, 5):
        if output_1[i] is None:
            output.append(None)
        else:
            output.append(np.append(output_1[i], output_2[i], 0))

    lc_per_seq = {}
    for key in output_1[5]:
        lc_per_seq1 = output_1[1][key]
        lc_per_seq2 = output_2[1][key]
        if key == "nta":
            lc_per_seq1 = af.reshape_with_nan(lc_per_seq1, nta_count, axis=-1)
            lc_per_seq2 = af.reshape_with_nan(lc_per_seq2, nta_count, axis=-1)
        lc_per_seq[key] = np.append(lc_per_seq1, lc_per_seq2, 0)
    output.append(lc_per_seq)
    output = pst.sort_lineages(output, "len")
    return output


def print_if_true(string_to_print, is_printed):
    if is_printed:
        print(string_to_print)


def compute_prop_arrest(data_key, thresholds):
    # Extract data.
    ncycles, lcycles = make_cycles_dataset(
        data_key, thresholds=thresholds, is_printed=False, is_saved=False
    )
    counts, props = {}, {}
    for key in ["gal", "raf_dox"]:
        counts[key] = {"nor": len(ncycles[key]), "arr": len(lcycles[key])}
        props[key] = (
            100 * counts[key]["arr"] / (counts[key]["arr"] + counts[key]["nor"])
        )
    print(counts)
    print(props)
    return props
