#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 15:45:52 2022

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
import os
import math
import numpy as np
from os.path import join

import telomeres.auxiliary.figures_properties as fp
import telomeres.model.parameters as par

from telomeres.dataset.extract_processed_dataset import write_parameters_linit

absolute_path = os.path.abspath(__file__)  # Path to extract_processed_dataset.
current_dir = os.path.dirname(absolute_path)  # Path to auxiliary directory.
parent_dir = os.path.dirname(current_dir)  # Path to telomeres directory.
projet_dir = os.path.dirname(parent_dir)

FOLDER_SIM = join(projet_dir, "simulations")
FOLDER_FIG = join(projet_dir, "figures")
FOLDER_DATA = join(projet_dir, "data")

FOLDER_L = "lineage"
FOLDER_P = "population"
FOLDER_FC = "finalCut"
FOLDER_DAT = "dataset"

# PAR_DEFAULT_LIN = {'is_htype_accounted': par.HTYPE_CHOICE,
#                    'fit': par.PAR,
#                    'p_exit': par.P_EXIT,
#                    'finalCut': None}
# PAR_DEFAULT_POP = {'is_htype_accounted': par.HTYPE_CHOICE,
#                    'p_exit': par.P_EXIT,
#                    'fit': par.PAR,
#                    'sat': par.PAR_SAT}

# -------------------
# Auxiliary functions
# -------------------


def list_to_strings(list_to_write, is_last_int=False, decimal_count=None):
    """Same as `list_to_string` except that a list of well formatted float
    is returned rather than one string (with float turned to str and separated
    by '-').

    """
    list_cp = list(deepcopy(list_to_write))
    element_formatted_count = len(list_cp)
    if is_last_int:
        list_cp[-1] = int(np.round(list_cp[-1]))
        element_formatted_count -= 1
    if decimal_count is not None:
        for i in range(element_formatted_count):
            if decimal_count == 2:
                list_cp[i] = f"{list_cp[i]:3.2f}"
            elif decimal_count == 3:
                list_cp[i] = f"{list_cp[i]:3.3f}"
            elif decimal_count == 4:
                list_cp[i] = f"{list_cp[i]:5.4f}"
            else:
                raise Exception(
                    "Please update `list_to_string' function to "
                    f"allow `decimal_count` to be {decimal_count}"
                )
    return list_cp


def list_to_string(list_to_write, is_last_int=False, decimal_count=None):
    """Convert a list of strings or float/integer into one string.

    Parameters
    ----------
    list_to_write : list
        List of strings float or int to convert to a unique string with
        separator `-`.
    is_last_int : bool, optional
        If True, the last element of the list is written under integer format.
        The default is False.
    decimal_count : int or NoneType
        If None (default value) no change of format, otherwise the elements
        (except the last one if `is_last_int` is True), assumed to be floats,
        are returned in a decimal format, with `decimal_count` decimals after
        point.

    """
    list_cp = list_to_strings(
        list_to_write, is_last_int=is_last_int, decimal_count=decimal_count
    )
    # Concatenation of elements of the list in one string.
    string = ""
    for element in list_cp[:-1]:
        string += str(element) + "-"
    string += str(list_cp[-1])
    return string


def write_path_directory_from_file(file_path, make_dir=False):
    """Return the string corresponding to the path of the directory in which
    the file with path given as argument in saved.

    """
    idx = len(file_path) - 1 - file_path[::-1].find(os.path.sep)
    file_dir = file_path[:idx]
    if (not os.path.exists(file_dir)) and make_dir:
        os.makedirs(file_dir)
    return file_dir


def write_parameters_onset(parameters):
    """Generate a string with the parameters of the law of entry in a sequence
    of arrests (non-terminal and senescence) given as argument.

    Parameters
    ----------
    parameters : list
        parameters[0]: list w the parameter (a,b, lmin) of the law of entry nta
        parameters[1]: list w the parameter (a,b) of the law of entry in
           senescence of type A cells (`parameters[1][0]`) and type B (...[1]).

    Returns
    -------
    par_string : string
        NB: if same parameters for type A and type B, the parameters are
        written only once.

    """
    is_int = True
    parameters_sen = parameters[1]
    par_sen0_string = list_to_string(parameters_sen[0], is_last_int=is_int)
    if list(parameters_sen[0]) == list(parameters_sen[1]):
        par_string = "parSEN" + par_sen0_string
    else:
        par_string = (
            "parSENA"
            + par_sen0_string
            + "_parSENB"
            + list_to_string(parameters_sen[1], is_last_int=is_int)
        )
    par_string = f"parNTA{parameters[0][0]}-{parameters[0][1]}_" + par_string
    return par_string


def write_parameters_exit(p_exit):
    p_exit_tmp = deepcopy(p_exit)
    for key in p_exit:
        if p_exit[key] is None:
            # None used to write path of figures with varying `p_exit`.
            p_exit_tmp[key] = "_variable"
    par_string = (
        f"pdeath{p_exit_tmp['accident']}-{p_exit_tmp['death']}"
        + f"_prepair{p_exit_tmp['repair']}"
    )
    if p_exit["sen_limit"] != math.inf:
        if p_exit["sen_limit"] is None:
            par_string = par_string + "_maxSEN_variable"
        else:
            par_string = par_string + f"_maxSEN{int(p_exit['sen_limit'])}"
    return par_string


def write_parameters_to_fit(parameters):
    string = write_parameters_onset(parameters)
    to_add = write_parameters_linit(parameters[2])
    if to_add != "":
        string = string + "_" + to_add
    return string


def write_parameters_finalCut(par_fc):
    l_cut = par_fc["lcut"]
    idxs_frame = par_fc["idxs_frame"]
    if l_cut is None:
        lcut_string = "noFc"
    else:
        lcut_string = f"fc{int(l_cut)}"
    return (
        f"{lcut_string}-"
        f"gal{idxs_frame[1] - idxs_frame[0]}-{idxs_frame[2] - idxs_frame[0]}"
        + f"_delay{par_fc['delay']}"
    )


# --------------------
# Lineages simulations
# --------------------

# Auxiliary functions
# -------------------


def characteristics_to_string(characteristics):
    """Generate a string with the different types lineage characterizations
    given as argument.

    Parameters
    ----------
    characteristics : list
        List of strings among 'atype' btype' 'htype' 'arrested1' 'arrested2'...
        'senescent' 'dead' dead_accidentally' 'dead_naturally'.
        See `is_as_expected_lineage` docstring.
    is_htype_seen : bool
        See `is_as_expected_lineage` docstring.

    Returns
    -------
    par_string : string
        NB: types are returned ordered by alphabetical order. If several
        `arrested_i` we keep only the one with the biggest `i`.

    """
    characteristics_cp = deepcopy(characteristics)
    characteristics_cp.sort()
    # Concatenation of all characteristics in one string.
    is_w_arrested = np.array(["arrested" in c for c in characteristics_cp])
    if any(is_w_arrested):
        nta_idxs = []
        for i in range(len(is_w_arrested)):
            if is_w_arrested[i]:
                nta_idxs.append(int(characteristics_cp[i][-1]))
        nta_idxs.sort()
        for i in nta_idxs[:-1]:  # We keep only the biggest arrest.
            characteristics_cp.remove("arrested" + str(i))
    characteristics_string = list_to_string(characteristics_cp)
    return characteristics_string


def types_of_sort_to_string(types_of_sort):
    """Generate a string with the different ways of sorting lineages given in
    argument.

    Parameters
    ----------
    types_of_sort : list
        List of strings among 'gdeath' 'lmin' 'gnta1' 'gnta2'... 'gsen'...

    Returns
    -------
    par_string : string
        NB: types are returned ordered by alphabetical order.

    """
    types_of_sort_cp = deepcopy(types_of_sort)
    types_of_sort_cp.sort()
    # Concatenation of all types of sort in one string.
    types_of_sort_string = list_to_string(types_of_sort_cp)
    return types_of_sort_string


# Simulation paths
# ----------------


def write_stat_path(
    simulation_count,
    lineage_count,
    types_of_sort,
    characteristics,
    par_update=None,
    par_sim_update=None,
    make_dir=False,
):
    """Return the path at which lineage simulations are saved.

    Returns
    -------
    path : string
        The path at which outputs of `simulate_n_average_lineages` (obtained
        with all the parameters given in argument) are be saved.

    """
    p = deepcopy(par.PAR_DEFAULT_LIN)
    if isinstance(par_update, dict):
        p.update(par_update)
    psim = deepcopy(par.PAR_DEFAULT_SIM_LIN)
    if isinstance(par_sim_update, dict):
        psim.update(par_sim_update)

    # List of strings put in alphabetical/reverse order.
    types_of_sort_string = types_of_sort_to_string(types_of_sort)
    characteristics_string = characteristics_to_string(characteristics)
    # Construction of the path.
    if p["finalCut"] is None:
        folder = join(FOLDER_SIM, FOLDER_L)
        fc_data = ""
    else:
        folder = join(FOLDER_SIM, FOLDER_FC, FOLDER_L)
        fc_data = write_parameters_finalCut(p["finalCut"]) + "_"
    path = join(folder, write_parameters_to_fit(p["fit"]), fc_data)

    if p["p_exit"] is not None:
        path = path + write_parameters_exit(p["p_exit"])
    path = join(path, characteristics_string + "_lineages")
    if not p["is_htype_accounted"]:
        path = join(path, "no_htype")
    elif p["is_htype_seen"]:
        path = join(path, "htype_seen")
    else:
        path = join(path, "htype_unseen")
    if (not os.path.exists(path)) and make_dir:
        os.makedirs(path)
    path = join(path, f"stat_by_{types_of_sort_string}_s{simulation_count}_")
    if lineage_count is not None:
        path = path + f"l{lineage_count}_"
    path = path + f"p{fp.PERCENT}.npy"
    if psim["is_lcycle_count_saved"]:
        path = path.replace(".npy", "_w-hist-lc.npy")
    if psim["hist_lmins_axis"] is not None:
        path = path.replace(".npy", "_w-hist-lmin.npy")
    pdt = psim["postreat_dt"]
    is_time_postreat = pdt is not None
    if psim["is_evo_saved"] or is_time_postreat:
        if is_time_postreat:
            path = path.replace(".npy", f"_w-evo-dt{pdt}.npy")
        else:
            path = path.replace(".npy", "_w-evo.npy")
    return path


def write_lineages_paths(
    simulation_count,
    lineage_count,
    characteristics,
    is_lcycle_count_saved,
    is_evo_saved,
    par_update=None,
    make_dir=True,
):
    """Return a list of possible path at which find the data.

    The first path of the list is the most natural one.

    """
    par_sim_update = {
        "is_lcycle_count_saved": is_lcycle_count_saved,
        "is_evo_saved": is_evo_saved,
    }
    path = write_stat_path(
        simulation_count,
        lineage_count,
        [""],
        characteristics,
        par_update=par_update,
        par_sim_update=par_sim_update,
        make_dir=make_dir,
    )
    path = path.replace("stat_by_", "lineages")
    paths = [path]
    if not (is_lcycle_count_saved or is_evo_saved):
        paths.append(path.replace(".npy", "_w-hist-lc_w-evo.npy"))
        paths.append(path.replace(".npy", "_w-hist-lc.npy"))
        paths.append(path.replace(".npy", "_w-evo.npy"))
    elif is_lcycle_count_saved and not is_evo_saved:
        paths.append(path.replace(".npy", "_w-evo.npy"))
    elif not is_lcycle_count_saved and is_evo_saved:
        paths.append(path.replace("w-evo.npy", "w-hist-lc_w-evo.npy"))
    return paths


# Plot paths
# ----------


def write_hist_lc_path(
    simulation_count,
    lineage_count,
    characteristics,
    lcycle_types,
    par_update=None,
    make_dir=True,
    subdirectory="",
):
    """Return the string chain corresponding to the path at which the histo-
    gram plots obtained from data simulated with all the parameters given as
    argument should be saved.
    In addition, check if the directory is created or not, if not creates it.

     NB: par_update corresponds to a dict updating key is_htype_accounted,
        is_htype_seen, parameters, postreat_dt, is_evo_saved of
        PAR_DEFAULT_LIN.

    """
    path_stat = write_stat_path(
        simulation_count, lineage_count, [""], characteristics, par_update=par_update
    )
    if par_update["finalCut"] is None:
        path = path_stat.replace(
            join(FOLDER_SIM, FOLDER_L),
            join(FOLDER_FIG, subdirectory, FOLDER_L, "gcurves_n_hists"),
        )
    else:
        path = path_stat.replace(
            join(FOLDER_SIM, FOLDER_FC, FOLDER_L),
            join(FOLDER_FIG, FOLDER_FC, subdirectory, FOLDER_L, "gcurves_n_hists"),
        )
    lcycle_types_string = types_of_sort_to_string(lcycle_types)
    path = path.replace("stat_", f"hist_lc_{lcycle_types_string}_")
    path = path.replace("by__", "")
    path = path.replace(".npy", ".pdf")
    # Creation of the directory if not existing.
    directory_path = write_path_directory_from_file(path)
    if (not os.path.exists(directory_path)) and make_dir:
        os.makedirs(directory_path)
    return path


def write_hist_lmin_path(
    simulation_count,
    lineage_count,
    width,
    characteristics,
    par_update=None,
    make_dir=True,
    subdirectory="",
):
    path = write_hist_lc_path(
        simulation_count,
        lineage_count,
        characteristics,
        [""],
        par_update=par_update,
        make_dir=False,
        subdirectory=subdirectory,
    )
    path = path.replace("_lc_", f"_lmin_w{width}")
    path = path.replace("_p95", "")
    # Creation of the directory if not existing.
    directory_path = write_path_directory_from_file(path)
    if (not os.path.exists(directory_path)) and make_dir:
        os.makedirs(directory_path)
    return path


def write_hist_finalcut_path(
    simulation_count,
    lineage_count,
    bin_width,
    bin_max,
    characteristics,
    par_update=None,
    make_dir=True,
    subdirectory="",
):
    path = write_hist_lmin_path(
        simulation_count,
        lineage_count,
        bin_width,
        characteristics,
        par_update=par_update,
        make_dir=make_dir,
        subdirectory=subdirectory,
    )
    path = path.replace("_lmin_", f"_len_af_cut_m{bin_max}_")
    # Creation of the directory if not existing.
    directory_path = write_path_directory_from_file(path)
    if (not os.path.exists(directory_path)) and make_dir:
        os.makedirs(directory_path)
    return path


def write_hist_lc_exp_path(lineage_count, characteristics, subdirectory=""):
    """Return the string chain corresponding to the path at which the histo-
    gram plots obtained from data simulated with all the parameters given as
    argument should be saved.

    """
    # List of strings put in alphabetical/reverse order.
    characteristics_string = characteristics_to_string(characteristics)
    # Construction of the path.
    path = join(
        FOLDER_FIG,
        subdirectory,
        FOLDER_L,
        "gcurves_n_hists",
        "exp",
        f"hist_exp_lc_{characteristics_string}_lineages_l{lineage_count}.pdf",
    )
    # Creation of the directory if not existing.
    directory_path = write_path_directory_from_file(path)
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    return path


def write_gcurve_path(
    simulation_count,
    lineage_count,
    types_of_sort,
    characteristics,
    par_update=None,
    make_dir=True,
    subdirectory="",
):
    path = write_stat_path(
        simulation_count,
        lineage_count,
        types_of_sort,
        characteristics,
        par_update=par_update,
    )
    folder = FOLDER_SIM
    if "finalCut" in (par_update or {}).keys():
        par_fc = par_update["finalCut"]
    else:
        par_fc = par.PAR_DEFAULT_LIN["finalCut"]
    if par_fc is not None:  # par instead of par_update?
        folder = join(folder, FOLDER_FC)
    path = path.replace(
        join(folder, FOLDER_L),
        join(FOLDER_FIG, subdirectory, FOLDER_L, "gcurves_n_hists"),
    )
    path = path.replace("stat_", "gcurves_")
    path = path.replace(".npy", ".pdf")
    # Creation of the directory if not existing.
    directory_path = write_path_directory_from_file(path)
    if (not os.path.exists(directory_path)) and make_dir:
        os.makedirs(directory_path)
    return path


def write_gcurves_path(par_update=None, make_dir=True, subdirectory=""):
    """Return the string chain corresponding to the path at which outputs of
    `simulate_n_average_lineages` (obtained with all the parameters given in
    argument) should be saved.

    """
    path = write_gcurve_path(
        0,
        0,
        [""],
        [""],
        par_update=par_update,
        make_dir=False,
        subdirectory=subdirectory,
    )
    path = path.replace(join("_lineages", "sort_by_"), "all_lineages")
    path = path.replace("_s0_l0", "")
    directory_path = write_path_directory_from_file(path)
    if (not os.path.exists(directory_path)) and make_dir:
        os.makedirs(directory_path)
    return path


def write_cycles_path(
    lineage_count,
    is_exp,
    lineage_types=None,
    is_dead=None,
    evo_avg=None,
    is_htype_seen=True,
    subdirectory="",
):
    """Return the string chain corresponding to the path at which the
    figure of cycle duration times should be saved.

    evo_avg dict with entries 'simu_count' and 'type_of_sort' containing the
    info when data represents an average on several sorted lineage simulations.
    """
    path = join(FOLDER_FIG, subdirectory, FOLDER_L, "cycles", "cycles_")
    if is_exp:
        path = path + "exp_"
    if evo_avg is not None:
        path = path + f"s{evo_avg['simu_count']}_"
    path = path + f"l{lineage_count}"
    if not is_htype_seen:
        path = path + "_htype_unseen"
    if lineage_types is not None:
        path = path + "_w_types"
    if is_dead is not None:
        path = path + "_w_alive"
    if evo_avg is not None:
        sort_string = types_of_sort_to_string([evo_avg["type_of_sort"]])
        path = path + "_by_" + sort_string
    path = path + ".pdf"
    # Creation of the directory if not existing.
    directory_path = write_path_directory_from_file(path)
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    return path


def write_propB_path(lineage_count, evo_avg, subdirectory=""):
    path = write_cycles_path(
        lineage_count, False, evo_avg=evo_avg, subdirectory=subdirectory
    )
    path = path.replace(join("cycles", "cycles_"), join("prop", "prop_"))
    directory_path = write_path_directory_from_file(path)
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    return path


# ----------------------
# Population simulations
# ----------------------

# Simulation paths
# ----------------


def write_simu_pop_directory(par_update=None):
    """Return the string corresponding to the path to the directory assumed
    to contain any simulation generated from `simu_parallel` of
    `population_simulation.py` with the parameters (of `parameters.py`) defined
    above by `PAR_DEFAULT_POP` updated through `par_update`.

    NB: value of `par_update` with usual type (see `parameter.py`) except for
       'p_exit' and 'fit'[2] that can be None (no reference in the path
       returned). This is useful for example to create the folder path of
       figures gathering simulations obtained w different 'p_exit' or 'fit'[2]'

    Parameters
    ----------
    par_update : dict
        A dictionary containing parameter updates. Default None (no update).

    """
    p = deepcopy(par.PAR_DEFAULT_POP)
    if isinstance(par_update, dict):
        p.update(par_update)

    # if p['finalCut'] is None:
    folder = join(FOLDER_SIM, FOLDER_P)
    fc_data = ""
    # else:
    #     folder = join(FOLDER_SIM, FOLDER_FC, FOLDER_P)
    #     fc_data = write_parameters_finalCut(p['finalCut']) + '_'

    path = join(folder, write_parameters_onset(p["fit"][:2]))
    if p["sat"]["choice"][0] == "time":
        path = join(path, "tsat")
        for key, tsat in p["sat"]["times"].items():
            path = path + f"{key}{tsat}-"
        path = path[:-1]
    else:
        if p["sat"]["prop"] is None:  # (For figure paths).
            path = join(path, "psat_varying")
        else:
            path = join(path, f"psat{int(p['sat']['prop'])}")
    if not p["is_htype_accounted"]:
        path = path + "_wo_htype"
    path = join(path, fc_data)
    if p["p_exit"] is not None:
        path = path + write_parameters_exit(p["p_exit"])
        path = join(path, write_parameters_linit(p["fit"][2]))
    return path


def write_simu_pop_subdirectory(cell=None, para=None, par_update=None):
    # !!! par_sim_update to add? and update in demo.
    """Add to the directory path (string) returned by
    `write_simu_pop_directory(par_update)` the subfolders corresponding
    to the simulations of a specific number of parallelization and cell given
    as argument (default None for no subfolder extension).

    """
    path = write_simu_pop_directory(par_update)
    if cell is not None:
        path = join(path, f"cell_{cell}")
        if para is not None:
            path = join(path, f"para_{para}")
    return path


def write_simu_pop_file(cell, para, output_index, par_update=None):
    sub_dir_path = write_simu_pop_subdirectory(cell, para, par_update=par_update)
    return join(sub_dir_path, f"output_{output_index:02d}.npy")


# Post-treat path
# ---------------


def write_sim_pop_postreat_average(folder_name, simu_count, is_stat=True):
    if is_stat:
        return join(folder_name, f"postreat_s{simu_count}_evo_statistics.npy")
    return join(folder_name, f"postreat_s{simu_count}_evo_as_one_simu.npy")


def write_sim_c_csv(folder_name, simu_count):
    return join(folder_name, f"output_s{simu_count}_cOD.csv")


def write_sim_lmode_csv(folder_name, simu_count):
    return join(folder_name, f"output_s{simu_count}_lmode.csv")


def write_sim_pop_postreat_evo_from_path(file_name):
    return file_name.replace(".npy", "_p_from_c.npy")


def write_sim_pop_postreat_evo(cell_count, para_count, output_index, par_update=None):
    output_path = write_simu_pop_file(
        cell_count, para_count, output_index, par_update=par_update
    )
    return write_sim_pop_postreat_evo_from_path(output_path)


def write_sim_pop_postreat_perf(folder_name, simu_count):
    return join(folder_name, f"postreat_s{simu_count}_performances.npy")


# Plotting
# --------


def write_fig_pop_directory(cell=None, para=None, par_update=None, subdirectory=""):
    """Return the string corresponding to the path to the directory assumed
    to contain a set of figures from simulations sharing same parameters
    (defined by `par_update`, see `write_simu_pop_directory` docstring)
    and possible same initial number of cells (cell) and parallelization
    (para) if not set to None (default value).

    Also, create the directories corresponding to the returned path if not
    already created.

    """
    path = write_simu_pop_subdirectory(cell, para, par_update)
    path = path.replace("simulations", join("figures", subdirectory))
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def write_fig_pop_name_end(simu=1, cell=None, para=1, tmax=None, is_stat=None):
    # tmax expected to be in day.
    path = ""
    if simu > 1:
        path = path + f"_s{simu}"
    if cell is not None:
        path = path + f"_c{cell}"
    if para > 1:
        path = path + f"_p{para}"
    if tmax is not None:
        path = path + f"_d{int(tmax)}"

    stat_names = []
    if is_stat is not None:
        for key, is_ in is_stat.items():
            if is_:  # NB: key in 'ext', 'per' and 'std'.
                stat_names.append(key)
    if stat_names != []:
        path = path + "_w_" + characteristics_to_string(stat_names)
    return path + ".pdf"
