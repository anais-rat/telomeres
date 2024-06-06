#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 15:45:52 2022

@author: arat
"""

import copy
# import imp
import os
import math
import numpy as np

import aux_figures_properties as fp
import parameters as par

# imp.reload(fp)
# imp.reload(par)


FOLDER_SIM = "simulations/"
FOLDER_L = "lineage/"
FOLDER_P = "population/"
FOLDER_FC = "final_cut/"

# DEFAULT_PARAMETERS_L = {'is_htype_accounted': par.HTYPE_CHOICE,
#                         'is_htype_seen': True,
#                         'parameters': par.PAR,
#                         'postreat_dt': None,
#                         'hist_lmins_axis': None,
#                         'is_lcycle_counts': False,
#                         'is_evo_stored': False,
#                         'p_exit': par.P_EXIT}
# DEFAULT_PARAMETERS_P = {'htype_choice': par.HTYPE_CHOICE,
#                         'p_exit': par.P_EXIT,
#                         'p_onset': par.P_ONSET,
#                         'par_l_init': par.PAR_L_INIT,
#                         'sat_choice': par.SAT_CHOICE,
#                         'times_sat': par.TIMES_SAT,
#                         'prop_sat': par.PROP_SAT}
# if __name__ == "__main__":
#     print(DEFAULT_PARAMETERS_L, DEFAULT_PARAMETERS_P)

# -------------------
# Auxiliary functions
# -------------------

def list_to_strings(list_to_write, is_last_int=False, decimal_count=None):
    """
    Same as `list_to_string` except that a list of well formatted float
    is returned rather than one string (with float turned to str and separated
    by '-').

    """
    list_cp = list(copy.deepcopy(list_to_write))
    element_formatted_count = len(list_cp)
    if is_last_int:
        list_cp[-1] = int(np.round(list_cp[-1]))
        element_formatted_count -= 1
    if not isinstance(decimal_count, type(None)):
        for i in range(element_formatted_count):
            if decimal_count == 2:
                list_cp[i] = f'{list_cp[i]:3.2f}'
            elif decimal_count == 3:
                list_cp[i] = f'{list_cp[i]:3.3f}'
            elif decimal_count == 4:
                list_cp[i] = f'{list_cp[i]:5.4f}'
            else:
                raise Exception("Please update `list_to_string' function to "
                                f"allow `decimal_count` to be {decimal_count}")
    return list_cp

def list_to_string(list_to_write, is_last_int=False, decimal_count=None):
    """
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
    list_cp = list_to_strings(list_to_write, is_last_int=is_last_int,
                              decimal_count=decimal_count)
    # Concatenation of elements of the list in one string.
    string = ''
    for element in list_cp[:-1]:
        string += str(element) + '-'
    string += str(list_cp[-1])
    return string

def write_path_directory_from_file(file_path):
    """ Return the string corresponding to the path of the directory in which
    the file with path given as argument in saved.

    """
    idx = len(file_path) - 1 - file_path[::-1].find("/")
    return file_path[:idx]

def write_parameters_onset(parameters):
    """ Generate a string with the parameters of the law of entry in a sequence
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
        par_string = "parSENA" + par_sen0_string + "_parSENB" + \
                     list_to_string(parameters_sen[1], is_last_int=is_int)
    par_string = f"parNTA{parameters[0][0]}-{parameters[0][1]}_" + par_string
    return par_string

def write_parameters_exit(p_exit):
    par_string = f'pdeath{p_exit[0]}-{p_exit[1]}_prepair{p_exit[2]}'
    if p_exit[3] != math.inf:
        par_string = par_string + f'_maxSEN{p_exit[3]}'
    return par_string

def write_parameters_to_fit(parameters):
    string = write_parameters_onset(parameters)
    to_add = write_parameters_linit(parameters[2])
    if to_add != '':
        string = string + '_' + to_add
    return string + '/'

def write_parameters_linit(par_linit):
    ltrans, l0, l1 = np.round(par_linit).astype(int)
    return f'linit{ltrans}-{l0}-{l1}'

def write_parameters_finalCut(par_fc):
    l_cut, g_delay, p_escape, idxs_frame = par_fc
    if isinstance(l_cut, type(None)):
        lcut_string = 'noFc'
    else:
        lcut_string = f'fc{int(l_cut)}'
    return f'{lcut_string}-{g_delay}-{p_escape}' \
         f'-gal{idxs_frame[1]-idxs_frame[0]}-{idxs_frame[2]-idxs_frame[0]}'

# --------------------
# Lineages simulations
# --------------------

# Auxiliary functions
# -------------------

def characteristics_to_string(characteristics):
    """ Generate a string with the different types lineage characterizations
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
        NB: types are returned orderred by alphabetical order. If several
        `arrested_i` we keep only the one with the biggest `i`.

    """
    characteristics_cp = copy.deepcopy(characteristics)
    characteristics_cp.sort()
    # Concatenation of all characteristics in one string.
    is_w_arrested = np.array(['arrested' in c for c in characteristics_cp])
    if any(is_w_arrested):
        nta_idxs = []
        for i in range(len(is_w_arrested)):
            if is_w_arrested[i]:
                nta_idxs.append(int(characteristics_cp[i][-1]))
        nta_idxs.sort()
        for i in nta_idxs[:-1]: # We keep only the biggest arrest.
            characteristics_cp.remove('arrested' + str(i))
    characteristics_string = list_to_string(characteristics_cp)
    return characteristics_string

def types_of_sort_to_string(types_of_sort):
    """ Generate a string with the different ways of sorting lineages given in
    argument.

    Parameters
    ----------
    types_of_sort : list
        List of strings among 'gdeath' 'lmin' 'gnta1' 'gnta2'... 'gsen'...

    Returns
    -------
    par_string : string
        NB: types are returned orderred by alphabetical order.

    """
    types_of_sort_cp = copy.deepcopy(types_of_sort)
    types_of_sort_cp.sort()
    # Concatenation of all types of sort in one string.
    types_of_sort_string = list_to_string(types_of_sort_cp)
    return types_of_sort_string



# Simulation paths
# ----------------


def write_stat_path(simulation_count, lineage_count, types_of_sort,
                    characteristics, par_update=None, make_dir=False):
    """ Return the string  chain corresponding to the path at which outputs of
    `simulate_n_average_lineages` (obtained with all the parameters given in
    argument) should be saved.

    """
    p = par.DEFAULT_PARAMETERS_L.copy()
    if isinstance(par_update, dict):
        p.update(par_update)

    # List of strings put in alphabetical/reverse order.
    types_of_sort_string = types_of_sort_to_string(types_of_sort)
    characteristics_string = characteristics_to_string(characteristics)
    # Construction of the path.
    if isinstance(p['par_finalCut'], type(None)):
        folder = FOLDER_SIM + FOLDER_L
        fc_data = ''
    else:
        folder = FOLDER_SIM + FOLDER_FC + FOLDER_L
        fc_data = write_parameters_finalCut(p['par_finalCut']) + '_'
    path = folder + write_parameters_to_fit(p['parameters']) + fc_data
    if not isinstance(p['p_exit'], type(None)):
        if 'p_death_acc' in list(p.keys()):
            if not isinstance(p['p_death_acc'], type(None)):
               path = path + write_parameters_exit(p['p_exit']) + '/'
        else:
            path = path + write_parameters_exit(p['p_exit']) + '/'
    path = path + characteristics_string + "_lineages"
    if not p['is_htype_accounted']:
        path = path + "/no_htype/"
    elif p['is_htype_seen']:
        path = path + "/htype_seen/"
    else:
        path = path + "/htype_unseen/"
    # path = path + f"sort_by_{types_of_sort_string}/"
    if (not os.path.exists(path)) and make_dir:
        os.makedirs(path)
    path = path + f"stat_by_{types_of_sort_string}_s{simulation_count}_"
    if not isinstance(lineage_count, type(None)):
        path = path + f"l{lineage_count}_"
    path = path + f"p{fp.PERCENT}.npy"
    if p['is_lcycle_counts']:
        path = path.replace('.npy', '_w-hist-lc.npy')
    if not isinstance(p['hist_lmins_axis'], type(None)):
        path = path.replace('.npy', '_w-hist-lmin.npy')
    pdt = p['postreat_dt']
    is_time_postreat = not isinstance(pdt, type(None))
    if p['is_evo_stored'] or is_time_postreat:
        if is_time_postreat:
            path = path.replace('.npy', f'_w-evo-dt{pdt}.npy')
        else:
            path = path.replace('.npy', '_w-evo.npy')
    return path


def write_lineages_paths(simulation_count, lineage_count, characteristics,
                         is_lcycle_counts, is_evos, par_update=None,
                         make_dir=True):
    """ Return a list of possible path at which find the data. The first one
    beging the natural one.

    """
    par_update['is_lcycle_counts'] = is_lcycle_counts
    par_update['is_evo_stored'] = is_evos

    path = write_stat_path(simulation_count, lineage_count, [''],
                           characteristics, par_update, make_dir=make_dir)
    path = path.replace('stat_by_', 'lineages')
    paths = [path]
    if not (is_lcycle_counts or is_evos):
        paths.append(path.replace('.npy', '_w-hist-lc_w-evo.npy'))
        paths.append(path.replace('.npy', '_w-hist-lc.npy'))
        paths.append(path.replace('.npy', '_w-evo.npy'))
    elif is_lcycle_counts and not is_evos:
        paths.append(path.replace('.npy', '_w-evo.npy'))
    elif not is_lcycle_counts and is_evos:
        paths.append(path.replace('w-evo.npy', 'w-hist-lc_w-evo.npy'))
    return paths


# Plot paths
# ----------

def write_hist_lc_sim_path(simulation_count, lineage_count, characteristics,
                           lcycle_types, par_update=None, make_dir=True,
                           supdirectory='figures'):
    """ Return the string chain corresponding to the path at which the histo-
    gram obtained from data simulated with all the parameters given as argument
    should be saved.
    In addition, check if the directory is created or not, if not creates it.

     NB: par_update corresponds to a dict updting key is_htype_accounted,
        is_htype_seen, parameters, postreat_dt, is_evo_stored of
        DEFAULT_PARAMETERS_L.

    """
    path = write_stat_path(simulation_count, lineage_count, [''],
                           characteristics, par_update)
    path = path.replace(FOLDER_SIM, supdirectory + "/gcurves_n_hists/")
    lcycle_types_string = types_of_sort_to_string(lcycle_types)
    path = path.replace("stat_", f"hist_lc_{lcycle_types_string}_")
    path = path.replace('by__', '')
    path = path.replace(".npy", ".pdf")
    # Creation of the directory if not existing.
    directory_path = write_path_directory_from_file(path)
    if (not os.path.exists(directory_path)) and make_dir:
        os.makedirs(directory_path)
    return path

def write_hist_lmin_sim_path(simulation_count, lineage_count, width,
                             characteristics, par_update=None, make_dir=True,
                             supdirectory='figures'):
    path = write_hist_lc_sim_path(simulation_count, lineage_count,
                                  characteristics, [''], par_update=par_update,
                                  make_dir=False, supdirectory=supdirectory)
    path = path.replace('_lc_', f'_lmin_w{width}')
    # Creation of the directory if not existing.
    directory_path = write_path_directory_from_file(path)
    if (not os.path.exists(directory_path)) and make_dir:
        os.makedirs(directory_path)
    return path

def write_hist_lc_exp_path(lineage_count, characteristics,
                           supdirectory='figures'):
    """ Return the string chain corresponding to the path at which the histo-
    gram obtained from data simulated with all the parameters given as argument
    should be saved.

    """
    # List of strings put in alphabetical/reverse order.
    characteristics_string = characteristics_to_string(characteristics)
    # Construction of the path.
    path = supdirectory + "/lineage/gcurves_n_hists/exp/hist_exp_lc_" + \
        characteristics_string + f"_lineages_l{lineage_count}.pdf"
    # Creation of the directory if not existing.
    directory_path = write_path_directory_from_file(path)
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    return path

def write_gcurve_path(simulation_count, lineage_count, types_of_sort,
                      characteristics, par_update=None, make_dir=True,
                      supdirectory='figures'):
    """ Return the string chain corresponding to the path at which outputs of
    `simulate_n_average_lineages` (obtained with all the parameters given in
    argument) should be saved.

    """
    path = write_stat_path(simulation_count, lineage_count, types_of_sort,
                           characteristics, par_update)
    folder = FOLDER_SIM
    if not isinstance(par_update['par_finalCut'], type(None)):
        folder = folder + FOLDER_FC
    path = path.replace(folder + FOLDER_L,
                        f"{supdirectory}/{FOLDER_L}gcurves_n_hists/")
    path = path.replace("stat_", "gcurves_")
    path = path.replace(".npy", ".pdf")
    # Creation of the directory if not existing.
    directory_path = write_path_directory_from_file(path)
    if (not os.path.exists(directory_path)) and make_dir:
        os.makedirs(directory_path)
    return path

def write_gcurves_path(par_update=None, make_dir=True, supdirectory='figures'):
    """ Return the string chain corresponding to the path at which outputs of
    `simulate_n_average_lineages` (obtained with all the parameters given in
    argument) should be saved.

    """
    path = write_gcurve_path(0, 0, [''], [''], par_update, make_dir=False,
                             supdirectory=supdirectory)
    path = path.replace('_lineages/sort_by_', 'all_lineages')
    path = path.replace('_s0_l0', '')
    directory_path = write_path_directory_from_file(path)
    if (not os.path.exists(directory_path)) and make_dir:
        os.makedirs(directory_path)
    return path

def write_cycles_path(lineage_count, is_exp, lineage_types=None, is_dead=None,
                      evo_avg=None, is_htype_seen=True,
                      supdirectory='figures'):
    """ Return the string chain corresponding to the path at which the
    figure of cycle duration times should be saved.

    evo_avg dict with entries 'simu_count' and 'type_of_sort' containing the
    info when data represents an average on several sorted lineage simulations.
    """
    path = supdirectory + "/lineage/cycles/cycles_"
    if is_exp:
        path = path + "exp_"
    if not isinstance(evo_avg, type(None)):
        path = path + f"s{evo_avg['simu_count']}_"
    path = path + f"l{lineage_count}"
    if not is_htype_seen:
        path = path + "_htype_unseen"
    if not isinstance(lineage_types, type(None)):
        path = path + "_w_types"
    if not isinstance(is_dead, type(None)):
        path = path + "_w_alive"
    if not isinstance(evo_avg, type(None)):
        sort_string = types_of_sort_to_string([evo_avg['type_of_sort']])
        path = path + "_by_" + sort_string
    path = path + ".pdf"
    # Creation of the directory if not existing.
    directory_path = write_path_directory_from_file(path)
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    return path

def write_propB_path(lineage_count, evo_avg, supdirectory='figures'):
    path = write_cycles_path(lineage_count, False, evo_avg=evo_avg,
                             supdirectory=supdirectory)
    path = path.replace('cycles/cycles_', 'prop/prop_')
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
    """ Return the string corresponding to the path to the directory assumed
    to contain any simulation generated from `simu_parallel`of
    `population_simulation.py` with the parameters (of `parameters.py`) defined
    above by `DEFAULT_PARAMETERS_P` updated through `par_update`.

    NB: value of `par_update` with usual type (see `parameter.py`) except for
       'p_exit' and 'l_init' that can be None (no reference in the path
       returned). This is useful for example to create the folder path of
       figures gathering simulations obtained w different 'p_exit' or 'l_init'.

    Parameters
    ----------
    par_update : dict
        A dictionary containing parameter updates. Default None (no update).

    """
    p = par.DEFAULT_PARAMETERS_P.copy()
    if isinstance(par_update, dict):
        p.update(par_update)

    if isinstance(p['par_finalCut'], type(None)):
        folder = FOLDER_SIM + FOLDER_P
        fc_data = ''
    else:
        folder = FOLDER_SIM + FOLDER_FC + FOLDER_P
        fc_data = write_parameters_finalCut(p['par_finalCut']) + '_'
    # path = folder + write_parameters_to_fit(p['parameters']) + fc_data #

    path = folder + write_parameters_onset(p['p_onset'])
    if p['sat_choice'][0] == 'time':
        path = path + '/tsat'
        for key, tsat in p['times_sat'].items():
            path = path + f'{key}{tsat}-'
        path = path[:-1]
    else:
        if isinstance(p['prop_sat'], type(None)):
            path = path + "/psat_varying"
        else:
            path = path + f"/psat{int(p['prop_sat'])}"
    if not p['htype_choice']:
        path = path + "_wo_htype"
    path = path + "/" + fc_data
    if not isinstance(p['p_exit'], type(None)):
        path = path + write_parameters_exit(p['p_exit'])
        if not isinstance(p['par_l_init'], type(None)):
            path = path + '/' + write_parameters_linit(p['par_l_init'])
        path = path  + "/"
    return path

def write_simu_pop_subdirectory(cell=None, para=None, par_update=None):
    """ Add to the directory path (string) returned by
    `write_simu_pop_directory(par_update)` the subfolders corresponding
    to the simulations of a specific number of parallelization and cell given
    as argument (default None for no subfolder extension).

    """
    path = write_simu_pop_directory(par_update)
    if not isinstance(cell, type(None)):
        path = path + f'cell_{cell}/'
        if not isinstance(para, type(None)):
            path = path + f'para_{para}/'
    return path


# Postreat path
# -------------

def write_sim_pop_postreat_average(folder_name, simu_count, is_stat=True):
    if is_stat:
        return folder_name + f'postreat_s{simu_count}_evo_statistics.npy'
    return folder_name + f'postreat_s{simu_count}_evo_as_one_simu.npy'

def write_sim_pop_postreat_evo(file_name):
    return file_name.replace('.npy', '_p_from_c.npy')

def write_sim_pop_postreat_perf(folder_name, simu_count):
    return f'{folder_name}postreat_s{simu_count}_performances.npy'


# Plotting
# --------

def write_fig_pop_directory(cell=None, para=None, par_update=None,
                            supdirectory='figures'):
    """ Return the string corresponding to the path to the directory assumed
    to contain a set of figures from simulations sharing same parameters
    (defined by `par_update`, see `write_simu_pop_directory` docstring)
    and possible same initial number of cells (cell) and parallelization
    (para) if not set to None (default value).

    Also, create the directories corresponding to the returned path if not
    already created.

    """
    path = write_simu_pop_subdirectory(cell, para, par_update)
    path = path.replace('simulations', supdirectory)
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def write_fig_pop_name_end(simu=1, cell=None, para=1, tmax=None, is_stat=None):
    # tmax expected to be in day.
    path = ''
    if simu > 1:
        path = path + f'_s{simu}'
    if not isinstance(cell, type(None)):
        path = path + f'_c{cell}'
    if para > 1:
        path = path + f'_p{para}'
    if not isinstance(tmax, type(None)):
        path = path + f'_d{int(tmax)}'

    stat_names = []
    if not isinstance(is_stat, type(None)):
        for key, is_ in is_stat.items():
            if is_: # NB: key in 'ext', 'per' and 'std'.
                stat_names.append(key)
    if stat_names != []:
        path = path + '_w_' + characteristics_to_string(stat_names)
    return path + '.pdf'
