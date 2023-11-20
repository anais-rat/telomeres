#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 16:30:04 2022

@author: arat
"""

import numpy as np
import os
import seaborn as sns
import warnings

import aux_figures_properties as fp
import aux_functions as fct
import aux_keys as ks
import aux_write_paths as wp
import parameters as par


# Parameters for plottings
# ------------------------
STD_CHOICE = False
PERCENT_CHOICE = True
EVO_DEATH_CHOICE = False

ANC_GROUP_COUNT = 2


# Parameters of the plots.
# ------------------------

# > A few settings for seaborn theme.
sns.set_context("paper", font_scale = 1.4)
sns.set_style("darkgrid", {"axes.facecolor": ".94"})
sns.set_palette(fp.MY_COLORS_3_ROCKET)

# To test :sns.set_style("ticks") # Backgroud. NB: whitegrid for hgrid.
# sns.set_context("notebook", font_scale = 1)

# Auxiliary functions.
# --------------------

def statistics(arr):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        stats = {'mean': np.nanmean(arr, axis=0)}
        stats['std'] = np.nanstd(arr, axis=0)
        stats['perup'] = np.nanpercentile(arr, fp.P_UP, axis=0)
        stats['perdown'] = np.nanpercentile(arr, fp.P_DOWN, axis=0)
        stats['min'] = np.nanmin(arr, axis=0)
        stats['max'] = np.nanmax(arr, axis=0)
    return stats


# Postreat functions.
# -------------------

def make_hist_lmin_gsen_from_evo(evo_lmin_gsen, x_axis=par.HIST_LMIN_X_AXIS):
    hist = {'hist_lmin_all': {}, 'hist_lmin_per_day': {}}
    for key in ks.type_keys:
        lmins_gsen = []
        lmins_gsen_per_day = []
        for evo_day in evo_lmin_gsen[key]:
            temp = []
            for evo in evo_day:
                temp.extend(evo)
            lmins_gsen_per_day.append(temp)
            lmins_gsen.extend(temp)
        hist['hist_lmin_all'][key] = fct.make_hist_from_data(lmins_gsen,
                                                             x_axis, False)
        hist['hist_lmin_per_day'][key] = [fct.make_hist_from_data(lmins,x_axis,
                                                                  False)
                                          for lmins in lmins_gsen_per_day]
    return hist

def postreat_from_evo_c(file_name):
    """ Return postreated data from data of file 'file_name'.
    More specifically converts evolution array from concentration of cells to
    proportion of cells and create histograms of lmin at senescence onset.

    """
    p = {}

    # We create vectors needeed to create `evo_p` arrays.
    # > Load.
    evo_c_anc = np.load(file_name, allow_pickle='TRUE').any().get('evo_c_ancs')
    evo_c_sen_anc = np.load(file_name, allow_pickle='TRUE').any().get(
                            'evo_c_sen_ancs')
    evo_lmin_gsen = np.load(file_name, allow_pickle='TRUE').any().get(
                            'evo_lmin_gsen')

    # > Summing on ancestors and saving.
    evo_c = fct.nansum(evo_c_anc, axis=1)
    evo_c_sen = fct.nansum(evo_c_sen_anc, axis=1)
    p['evo_c'] = evo_c
    p['evo_c_sen'] = evo_c_sen
    # > We turn zeros to ones (in order to divide 0 by nan or 1 and not 0).
    # evo_c[evo_c == 0] = np.nan
    # evo_c_sen[evo_c_sen == 0] = 1
    for key in ks.evo_c_anc_keys:
        evo = np.load(file_name, allow_pickle='TRUE').any().get(key)
        # > Creation of 'evo_*' from 'evo_*_anc' by summing on ancestors.
        sum_key = key.replace('_ancs', '')
        if not(sum_key in ['evo_c', 'evo_c_sen']): # if not already computed.
            p[sum_key] = fct.nansum(evo, axis=1)
        # > Creation of 'evo_p_*_anc' from 'evo_c_*_anc' and of 'evo_p_*' from
        #   'evo_c_*' by dividing.
        prop_key = key.replace('_c_', '_p_')
        prop_sum_key = sum_key.replace('_c_', '_p_')
        # NB: avoid printing error for 0 / 0, just returns 0.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            # NB: `evo_p_B/H_sen...` represent the proportion among senescent
            #      cells, thus we divide by the number of senescent cells
            if key == 'evo_c_B_sen_ancs':
                p[prop_key] = evo / evo_c_sen[:, None]
                p[prop_sum_key] = p[sum_key] / evo_c_sen
            else: 
                if 'ancs' in prop_key:
                    p[prop_key] = evo / p[sum_key][:, None]
                else:
                    p[prop_key] = evo / evo_c[:, None]
                if prop_sum_key != 'evo_c':
                    p[prop_sum_key] = p[sum_key] / evo_c
                # We also create prop of H among senescent (new key generated!).
                if key == 'evo_c_H_ancs':
                    p['evo_p_H_sen_ancs'] = evo / evo_c_sen[:, None]
                    p['evo_p_H_sen'] = p['evo_c_H'] / evo_c_sen
    # > gen key.
    p['evo_p_gen'] = np.load(file_name, allow_pickle='TRUE').any(
                              ).get('evo_c_gens') / evo_c[:, None]
    
    # > Time evolution of telomere lengths.
    for key in ks.evo_l_to_sum_keys:
        if 'sum' in key:
            evo = np.load(file_name, allow_pickle='TRUE').any().get(key)
            avg_key = key.replace('_sum', '_avg')
            p[avg_key] = evo / evo_c

    # > Histogram of lmin triggering senescence.
    p.update(make_hist_lmin_gsen_from_evo(evo_lmin_gsen))

    # Saving.
    file_postreated_name = wp.write_sim_pop_postreat_evo(file_name)
    np.save(file_postreated_name, p)

def postreat_from_evo_c_if_not_saved(file_name):
    file_postreated_name = wp.write_sim_pop_postreat_evo(file_name)
    if not(os.path.exists(file_postreated_name)):
        postreat_from_evo_c(file_name)

def postreat_from_evo_c_if_not_saved_whole_folder(generic_name, simu_count):
    for i in range(1, simu_count + 1):
        file_name = generic_name + f'{i:02d}.npy'
        file_postreated_name = wp.write_sim_pop_postreat_evo(file_name)
        if not(os.path.exists(file_postreated_name)):
            print(f'Postreat from evo_c simulation n°{i}')
            postreat_from_evo_c(file_name)

def postreat_performances(folder_name, simu_count): #, is_memory):
    saving_path = wp.write_sim_pop_postreat_perf(folder_name, simu_count)
    if not os.path.exists(saving_path):
        simus = np.arange(simu_count)
        # Load paths to all simulations in a list.
        s = [f'{folder_name}output_{i:02d}.npy' for i in simus + 1]
        # Initialization of a dictionary with Performance related postreat data.
        p = {}
        # > Computation times.
        p['computation_time'] = statistics([np.load(s[i],
            allow_pickle='TRUE').any().get('computation_time') for i in simus])
        # > Allocated memory.
        # if is_memory:
        p['memory'] = statistics([np.load(s[i], allow_pickle='TRUE').any().get(
                                 'memory') for i in simus])
        np.save(saving_path, p)
    return

def statistics_simus(folder, simu_count):
    """ Postreat saved data, computing and saving mean, std, min, max... on all
    of the `simu_count` first simulations present in the folder at path
    `folder`.

    """
    simus = np.arange(simu_count)
    # Load paths to all simulations in a list.
    s = [f'{folder}output_{i:02d}.npy' for i in simus + 1]
    s_postreat = [f'{folder}output_{i:02d}_p_from_c.npy' for i in simus + 1]
    
    # Initialization of a dictionary with postreat data.
    es = {} # Statistics on simulateds time evolution data.

    # > Extinction times and longuest time array.
    textinct_s = [np.load(s[i], allow_pickle='TRUE').any().get(
                  'extinction_time') for i in simus]
    extinct_prop = np.sum(~np.isnan(textinct_s)) / len(textinct_s)
    if extinct_prop == 0:
        tmax_sim_idx = 0
    else:
        tmax_sim_idx = np.nanargmax(textinct_s)
    es['extinction_time'] = statistics(textinct_s)
    times = np.load(s[tmax_sim_idx], allow_pickle='TRUE').any().get('times')
    if not np.isnan(textinct_s[tmax_sim_idx]):
        times = times[times <= textinct_s[tmax_sim_idx]]
    time_count = len(times)
    es['times'] = times

    # > Saturation times.
    sat_times_s = [np.load(s[i], allow_pickle='TRUE').any().get('sat_time')
                   for i in simus]
    sat_day_count = max([np.argmax(sat_times_s[i]) for i in simus])
    sat_times_s = np.array([fct.reshape1D_with_nan(sat_times_s[i],
                            max(2, sat_day_count)) for i in simus])
    es['sat_time'] = statistics(sat_times_s)
    # > Proportion of saturated subsimulations.
    sat_props_s = [np.load(s[i], allow_pickle='TRUE').any().get('sat_prop') for
                   i in simus]
    sat_props_s = np.array([fct.reshape1D_with_nan(sat_props_s[i],
                            max(2, sat_day_count)) for i in simus])
    es['sat_prop'] = statistics(sat_props_s)

    # > Time of senescence of the population.
    tsen_idx_s = [np.nanargmin(np.load(s_postreat[i], allow_pickle='TRUE'
                                       ).any().get('evo_c')
                               - np.load(s_postreat[i], allow_pickle='TRUE'
                                         ).any().get('evo_c_sen'))
                      for i in simus]
    tsen_s = [np.load(s[i], allow_pickle='TRUE').any().get('times'
               )[tsen_idx_s[i]] for i in simus]
    es['sen_time'] = statistics(tsen_s)

    # Computation of evo_gen/anc len (subsimus' gen distrib reshape to common).
    gen_count_s = np.array([len(np.load(s[i], allow_pickle='TRUE').any().get(
                            'evo_c_gens')[1]) for i in simus])
    gen_count = np.max(gen_count_s)
    # Ordering of ancestors by increasing shortest telomere.
    lmin_init_s = [np.min(np.load(s[i], allow_pickle='TRUE').any().get(
                  'day_init_data')['lengths'][0], axis=(1,2)) for i in simus]
    # Ordering of ancestors by increasing average telomere.
    lavg_init_s = [np.mean(np.load(s[i], allow_pickle='TRUE').any().get(
                  'day_init_data')['lengths'][0], axis=(1,2)) for i in simus]
    anc_s = {'by_lmin': [np.argsort(lmin_init_s[i]) for i in simus],
             'by_lavg': [np.argsort(lavg_init_s[i]) for i in simus]}

    # Reshape and computation of evolution arrays.
    cell_count = len(np.load(s[0], allow_pickle='TRUE').any().get(
                     'day_init_data')['lengths'][0])
    # > For all key to postreat.
    for key in ks.evo_c_keys_to_postreat:
        print(key)
        # We reshape to common shape for all simulations.
        if key in ks.evo_1Dkeys_new:
            evo_s = [np.load(s_postreat[i], allow_pickle='TRUE').any().get(key)
                     for i in simus]
            evo_s = [fct.reshape1D_with_nan(evo_s[i], time_count) for i in
                     simus]
        elif key in ks.evo_c_anc_keys: # Ancestors orderred by increasing lmin.
            evo_s = [fct.reshape2D_along0_w_NaN_along1_w_0_or_NaN(np.load(s[i],
                     allow_pickle='TRUE').any().get(key), time_count,
                     cell_count) for i in simus]
            # Create evo array wrt to anc orderred by lavg.
            tmp_s = [evo_s[i][:, anc_s['by_lavg'][i]] for i in simus]
            es[key + '_lavg'] =  statistics(tmp_s)
            # key = key + '_lmin'
            evo_s = [evo_s[i][:, anc_s['by_lmin'][i]] for i in simus]
        elif key in ks.evo_p_anc_keys:
            evo_s = [fct.reshape2D_along0_w_NaN_along1_w_0_or_NaN(np.load(
                     s_postreat[i], allow_pickle='TRUE').any().get(
                     key), time_count, cell_count) for i in simus]
            # Create evo array wrt to anc orderred by lavg.
            tmp_s = [evo_s[i][:, anc_s['by_lavg'][i]] for i in simus]
            es[key + '_lavg'] =  statistics(tmp_s)
            evo_s = [evo_s[i][:, anc_s['by_lmin'][i]] for i in simus]
        elif key in ks.evo_c_gen_keys:
            evo_s = [np.load(s[i], allow_pickle='TRUE').any().get(key) for i in
                     simus]
            evo_s = [fct.reshape2D_along0_w_NaN_along1_w_0_or_NaN(evo_s[i],
                     time_count, gen_count) for i in simus]
        # Computation of statistics (es) on all subsimu.
        es[key] = statistics(evo_s)

    # > Time evolution of telomere lengths.
    for key in ks.evo_l_keys_af_postreat:
        if '_avg' in key:
            evo_s = [fct.reshape1D_with_nan(np.load(s_postreat[i], allow_pickle
                     ='TRUE').any().get(key), time_count) for i in simus]
        else:
            evo_s = [fct.reshape1D_with_nan(np.load(s[i], allow_pickle='TRUE'
                     ).any().get(key), time_count) for i in simus]
        es[key] = statistics(evo_s)

    # > Histogram of lmin triggering senescence.
    hist_s = [np.load(s_postreat[i], allow_pickle='TRUE').any().get(
                      'hist_lmin_all') for i in simus]
    es['hist_lmin_all'] = {key: statistics([hist[key] for hist in hist_s]) for
                           key in ks.type_keys}
    hist_s = [np.load(s_postreat[i], allow_pickle='TRUE').any().get(
                      'hist_lmin_per_day') for i in simus]
    days = np.arange(len(hist_s[0]['atype']))
    es['hist_lmin_per_day'] = {key: [statistics([hist[key][day] for hist in
                             hist_s]) for day in days] for key in ks.type_keys}

    np.save(wp.write_sim_pop_postreat_average(folder, simu_count), es)
    return

def statistics_simus_if_not_saved(folder_name, simu_count):
    evo_stat_path = wp.write_sim_pop_postreat_average(folder_name, simu_count)
    if not os.path.exists(evo_stat_path):
        print('\n Averaging on all simulations...')
        statistics_simus(folder_name, simu_count)

def postreat_cgen(is_stat, folder, simu_count):
    """ Compute evolution of the avg, max, min... generation from the
    evolution of the concentration by generation.

    """
    precision = 10
    spath = wp.write_sim_pop_postreat_average(folder + '/', simu_count)
    path = spath.replace('statistics.npy', f'gens_p{precision}.npy')
    if os.path.exists(path):
        print("Load: ", path)
        return np.load(path, allow_pickle='TRUE').item()

    evo = np.load(spath, allow_pickle='TRUE').any().get('evo_c_gens')['mean']
    time_count, gen_count = np.shape(evo)
    # Need to compute evolution of the avg, max, min... generation.
    evo_gmin = np.nanargmax(evo > 0, axis=1)
    evo_gmax = gen_count - np.nanargmax(evo[:, ::-1] > 0, axis=1)
    evo_gavg = np.zeros(time_count)
    evo_g_pdown = np.zeros(time_count)
    evo_g_pup = np.zeros(time_count)
    evo_g_std = np.zeros(time_count)
    for t in range(time_count):
        gens = np.array([]) # array of all the generations present at time t.
        for gen in range(int(evo_gmin[t]), int(evo_gmax[t])):
            gens = np.append(gens, gen * np.ones(int(precision * evo[t,gen])))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            evo_gavg[t] = np.mean(gens)
            if is_stat['per']:
                evo_g_pdown[t] = np.nanpercentile(gens, fp.P_DOWN)
                evo_g_pup[t] = np.nanpercentile(gens, fp.P_UP)
            if is_stat['std']:
                evo_g_std[t] = np.nanstd(gens)
    d = {'avg': evo_gavg, 'min':evo_gmin, 'max':evo_gmax,
         'perdown': evo_g_pdown, 'perup': evo_g_pup, 'std': evo_g_std}
    np.save(path, d)
    return d
