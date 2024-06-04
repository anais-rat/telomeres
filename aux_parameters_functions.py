#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 26 14:00:06 2021

@author: arat
"""

# import imp
from copy import deepcopy
import math
# import matplotlib
import matplotlib.pylab as plt
import numpy as np
import seaborn as sns
from scipy import interpolate

import aux_figures_properties as fp
# imp.reload(fp)

DECIMAL_COUNT = 2
FONT_SCALE = 1.2 # To remove (initially in fp) !!!!
FDIR_PAR = 'parameters'

LABELS = {'ax_proba': 'Probability',
          'ax_proba_nta': r'Probability $p_{nta}$ of ' +
              '\nsignalling non-terminal arrest',
          'ax_proba_sen': r'Probability $p_{sen}$ of ' +
              '\nsignalling senescence',
          #
          'best': "Best law",
          'pnta': r'$p_{nta}$',
          'psen': r'$p_{sen}$',
          'psenA': r'$p_{sen_A}$',
          'psenB': r'$p_{sen_B}$',
          'linit': None}
LABELS.update(fp.LABELS)


def list_to_strings(list_to_write, is_last_int=False, decimal_count=None):
    """
    Same as `list_to_string` except that a list of well formatted float
    is returned rather than one string (with float turned to str and separated
    by '-') .

    """
    list_cp = list(deepcopy(list_to_write))
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

def write_exp_law(parameters, law_name):
    a, b  = parameters[:2]
    string = rf'$a_{{ {law_name} }} = {a}, ~$' + \
             rf'$b_{{ {law_name} }} = {b}, ~$'
    # rf'$p_{{ {law_name} }} ( {{ \ell }} ) {{\,}} {{\approx}}' + \
    #           rf'{{\,}} {b} {{\;}} {{\exp ( - {a} {{ \ell }} ) }}$'
    if len(parameters) == 3:
        l = parameters[2]
        name = law_name[-1]
        name = name.replace('n', '')
        string = string + rf'${{ \ell }}_{{ {{min}}_{{ {name} }} }} = {l}$'
        # rf'$ {{\,}} {{ \mathds{{1}} }}_{{ {{\ell}} \geq {l} }}$'
    return string

def write_linit_law(parameters):
    parameters_rounded = np.round(parameters)
    ltrans, l0, l1  = parameters_rounded.astype('int')
    string = rf'${{\ell}}_{{trans}} = {ltrans}, ~$' + \
             rf'${{\ell}}_{{0}} = {l0}, ~$' + \
             rf'${{\ell}}_{{1}} = {l1}$'
    return string

def write_laws_s(parameters, decimal_count=DECIMAL_COUNT):
    par_nta, par_sen, par_l_init = parameters
    par_sen = [list(par_by_type) for par_by_type in par_sen]
    nta_formatted = list_to_strings(par_nta, decimal_count=decimal_count)
    laws = {'pnta': write_exp_law(nta_formatted, 'nta')}
    senA_formatted = list_to_strings(par_sen[0], is_last_int=True,
                                        decimal_count=decimal_count)
    if par_sen[0] != par_sen[1]:
        laws['psenA'] = write_exp_law(senA_formatted, 'senA')
        senB_formatted = list_to_strings(par_sen[1], is_last_int=True,
                                            decimal_count=decimal_count)
        laws['psenB'] = write_exp_law(senB_formatted, 'senB')
    else:
        laws['psen'] = write_exp_law(senA_formatted, 'sen')
    # laws = laws + '\n' + laws + '\n' + gtrig + \
    #       ' ' + characteristics[0] + ' by ' + gcurve +
    laws['linit'] = None
    if par_l_init != [0, 0, 0]:
        laws['linit'] = write_linit_law(par_l_init)
    return laws

def write_laws(parameters, decimal_count=DECIMAL_COUNT):
    par_nta, par_sen, par_l_init = parameters
    par_sen = [list(par_by_type) for par_by_type in par_sen]
    nta_formatted = list_to_strings(par_nta, decimal_count=decimal_count)
    laws = write_exp_law(nta_formatted, 'nta')
    senA_formatted = list_to_strings(par_sen[0], is_last_int=True,
                                        decimal_count=decimal_count)
    if par_sen[0] != par_sen[1]:
        laws = laws + '\n' + write_exp_law(senA_formatted, 'senA')
        senB_formatted = list_to_strings(par_sen[1], is_last_int=True,
                                            decimal_count=decimal_count)
        laws = laws + '\n' + write_exp_law(senB_formatted, 'senB')
    else:
        laws = laws + '\n' + write_exp_law(senA_formatted, 'sen')
    # laws = laws + '\n' + laws + '\n' + gtrig + \
    #       ' ' + characteristics[0] + ' by ' + gcurve +
    laws = laws + '\n' + write_linit_law(par_l_init)
    return laws

# Plot functions
# --------------
# NB: parameters of plottings (labels, line colors...) are defined within the
#   functions but should be rather accessible (see PARAMETERS to locate them).

def print_data_on_special_initial_distributions(l_short, l_medium, l_long):
    print('\n Generation of short, medium and long intial distribution of '
          'telomeres: \n -----------------------------------------------------'
          '-----------------')
    print('               |   l_long   |   l_medium   |   l_short   |')
    print(f'Average length |  {np.mean(l_long)} |   '
          f'{np.mean(l_medium)}   |   {np.mean(l_short)}     |')
    print(f'Minimum length |    {np.min(l_long)}   |'
          f'     {np.min(l_medium)}    |'
          f'     {np.min(l_short)}    |')
    print(f'Maximum length |    {np.max(l_long)}   |'
          f'     {np.max(l_medium)}    |'
          f'    {np.max(l_short)}    |')
    return None


def extract_evo_c():
    dox_p = np.genfromtxt("data/population/raw-DOX+.csv", delimiter=',')
    dox_m = np.genfromtxt("data/population/raw-DOX-.csv", delimiter=',')

    # First transformation (Zhou).
    t_dox_p, t_dox_m = 1e5 * 2 ** dox_p, 1e5 * 2 ** dox_m
    avg_p1, std_p1 = np.mean(t_dox_p, axis=1), np.std(t_dox_p, axis=1)
    avg_m1, std_m1 = np.mean(t_dox_m, axis=1), np.std(t_dox_m, axis=1)
    # Second transformation (Teresa).
    t_dox_p, t_dox_m = 3e7 * dox_p, 3e7 * dox_m
    avg_p2, std_p2 = np.mean(t_dox_p, axis=1), np.std(t_dox_p, axis=1)
    avg_m2, std_m2 = np.mean(t_dox_m, axis=1), np.std(t_dox_m, axis=1)
    # No transformation.
    avg_p3, std_p3 = np.mean(dox_p, axis=1), np.std(dox_p, axis=1)
    avg_m3, std_m3 = np.mean(dox_m, axis=1), np.std(dox_m, axis=1)
    return (avg_p1, std_p1, avg_m1, std_m1, avg_p2, std_p2, avg_m2, std_m2,
            avg_p3, std_p3, avg_m3, std_m3)

def plot_data_exp_concentration_curves(x, y, x_sat, y_sat, sem, is_saved):
    # colors = fp.MY_COLORS_2_BIS_ROCKET # colors = fp.MY_COLORS_2_ROCKET
    colors = fp.MY_COLORS[:2]
    sns.set_palette(colors)

    # Pop. doublings per day w.r.t. pop. doublings when no saturation.
    plt.figure(dpi=fp.DPI)
    plt.errorbar(x[:, 0], y[:, 0], yerr=y[:, 2], capsize=2, fmt='.-',
                 label=LABELS['telo+'])
    plt.errorbar(x[:, 1], y[:, 1], yerr=y[:, 3], capsize=2,
                 label=LABELS['telo-'])
    plt.legend(borderaxespad=.9)
    plt.xlabel('Cumulative population doubling number')
    plt.ylabel('Population doubling number')
    sns.despine()
    if is_saved:
        plt.savefig('figures/parameters/pop_doublings_no_sat.pdf',
                    bbox_inches='tight')

    # Concentration w.r.t days when no saturation.
    plt.figure(dpi=fp.DPI)
    x = np.arange(len(y[:, 0]))
    plt.plot(x, 1e5 * 2**y[:, 0], '.-', label=LABELS['telo+'])
    plt.plot(x, 1e5 * 2**y[:, 1], '+-', label=LABELS['telo-'])
    plt.legend()
    plt.xlabel(LABELS['ax_time'])
    plt.ylabel(LABELS['ax_cexp'])
    plt.xticks(x)
    sns.despine()
    if is_saved:
        plt.savefig('figures/parameters/concentration_no_sat.pdf',
                    bbox_inches='tight')

    # Population doublings per day w.r.t. population doublings.
    plt.figure(dpi=fp.DPI)
    plt.errorbar(x_sat[:, 0], y_sat[:, 0], yerr=y_sat[:, 2], capsize=2,
                 fmt='.-', label=LABELS['telo-'])
    plt.errorbar(x_sat[:, 1], y_sat[:, 1], yerr=y_sat[:, 3], capsize=2,
                 label=LABELS['telo+'])
    plt.legend()
    plt.xlabel('Cumulative population doubling number')
    plt.ylabel('Population doubling number')
    sns.despine()
    if is_saved:
        plt.savefig('figures/parameters/pop_doublings.pdf',
                    bbox_inches='tight')

    # Concentration w.r.t days.
    w, h = plt.figaspect(.7)
    plt.figure(dpi=fp.DPI, figsize=(w,h))
    x = np.arange(len(y_sat[:, 0])) + 1
    # First version (on manuscript).
    plt.errorbar(x, 1e5 * 2**y_sat[:, 1], yerr=sem[:, 1], capsize=2,
                  label=LABELS['telo+'])
    plt.errorbar(x, 1e5 * 2**y_sat[:, 0], yerr=sem[:, 0], capsize=2, fmt='.-',
                  label=LABELS['telo-'])
    # Second version (on paper).
    # plt.errorbar(x, 3 * 10 ** 7 * y_sat[:, 1], yerr=sem[:, 1], capsize=2,
    #              label=LABELS['telo+'])
    # plt.errorbar(x,  3 * 10 ** 7 * y_sat[:, 0], yerr=sem[:, 0], capsize=2, fmt='.-',
    #              label=LABELS['telo-'])
    plt.legend(loc="lower left")
    plt.xlabel(LABELS['ax_time'])
    plt.ylabel(LABELS['ax_cexp'])
    plt.xticks(x)
    sns.despine()
    if is_saved:
        plt.savefig('figures/parameters/concentration.pdf',
                    bbox_inches='tight')
    return

LENGTHS = np.linspace(0, 250, 2501)

def law_sen_usual(lmins, a, b, lmin):
    densities = np.array([min(1, b * math.exp(- a * lmin)) for lmin in
                          lmins])
    densities[lmins < lmin] = 1
    return densities

def law_nta_usual(lmins, a, b):
    densities = np.array([min(1, b * math.exp(- a * lmin)) for lmin in lmins])
    return densities

def plot_laws_nta_various_a(a_to_test, b, lengths=LENGTHS,
                            law_nta=law_nta_usual, is_saved=False,
                            font_scale=FONT_SCALE):
    """ Plot on `lengths` the laws for non-terminal arrests defined by
    parameter `b` fixed and parameter a in `a_to_test` and save it if asked.

    """
    sns.set_context(fp.CONTEXT, font_scale=font_scale)
    # Plotting.
    plt.figure(dpi=fp.DPI)
    for a in a_to_test:
        plt.plot(lengths, law_nta(lengths, a, b), label=rf'$a_{{nta}} = {a}$',
                 linewidth=2.5)
        plt.xlabel(LABELS['ax_lmin_min'], labelpad=6)
        plt.ylabel(r'$p_{nta}$', labelpad=8) #LABELS['ax_proba_nta'])
    ax = plt.gca()
    plt.text(.92, 1.02, rf'$b_{{nta}} = {b}$', horizontalalignment='right',
             transform=ax.transAxes)
    plt.legend()
    sns.despine()

    # Saving.
    if is_saved:
        plt.savefig(f'figures/parameters/law_nta_A{a_to_test}_B{b}.pdf',
                    bbox_inches='tight')
    sns.set_context(fp.CONTEXT, font_scale=FONT_SCALE)
    return None

def plot_laws_nta_various_b(b_to_test, a, lengths=LENGTHS,
                            law_nta=law_nta_usual, is_saved=False,
                            font_scale=FONT_SCALE):
    """ Plot on `lengths` the laws for non-terminal arrests defined by
    parameter `a` fixed and parameter b in `b_to_test` and save it if asked.

    """
    sns.set_context(fp.CONTEXT, font_scale=font_scale)
    # Plotting.
    plt.figure(dpi=fp.DPI)
    for b in b_to_test:
        plt.plot(lengths, law_nta(lengths, a, b), label=rf'$b_{{nta}} = {b}$',
                 linewidth=2.5)
        plt.xlabel(LABELS['ax_lmin_min'], labelpad=6)
        plt.ylabel(LABELS['pnta'], labelpad=8)
    ax = plt.gca()
    plt.text(.92,1.02, rf'$a_{{nta}} = {a}$', horizontalalignment='right',
             transform=ax.transAxes)
    plt.legend()
    sns.despine()
    sns.set_context(fp.CONTEXT, font_scale=FONT_SCALE)

    # Saving.
    if is_saved:
        plt.savefig(f'figures/parameters/law_nta_A{a}_B{b_to_test}.pdf',
                    bbox_inches='tight')
    return None

def stat_all(arr_stat, p_up=fp.P_UP, p_down=fp.P_DOWN, axis=0):
    """ Copy from `functions_auxiliary.py`.

    """
    stats = {'mean': np.mean(arr_stat, axis=axis),
             'per': [np.percentile(arr_stat, p_down, axis=axis),
                     np.percentile(arr_stat, p_up, axis=axis)],
             'ext': [np.min(arr_stat, axis=axis),
                     np.max(arr_stat, axis=axis)]}
    return stats

def plot_laws_s(parameters_s, idx_best=None, lengths=LENGTHS,
                law_nta=law_nta_usual, law_sen=law_sen_usual, fig_name='',
                fig_supdirectory=None, is_zoomed=False):
    is_senA_neq_senB = parameters_s[0][1][0] != parameters_s[0][1][1]
    colors = sns.color_palette('viridis', len(parameters_s))
    probas = {}
    pnta_s = [law_nta(lengths, *par[0]) for par in parameters_s]
    probas['pnta'] = stat_all(pnta_s)
    probas['pnta']['all'] = pnta_s
    if not isinstance(idx_best, type(None)):
        colors[idx_best] = 'black'
        probas['pnta']['best'] = pnta_s[idx_best]
    psen_s = [law_sen(lengths, *par[1][0]) for par in parameters_s]
    if is_senA_neq_senB:
        fig_sizes = [(14, 3.5)] # default: (6.4, 4.8)
        if is_zoomed:
            fig_sizes.append((9, 2))
        probas['psenA'] = stat_all(psen_s)
        probas['psenA']['all'] = psen_s
        psenB_s = [law_sen(lengths, *par[1][1]) for par in parameters_s]
        probas['psenB'] = stat_all(psenB_s)
        probas['psenB']['all'] = psenB_s
        if not isinstance(idx_best, type(None)):
            probas['psenA']['best'] = psen_s[idx_best]
            probas['psenB']['best'] = psenB_s[idx_best]
    else:
        fig_sizes = [(11, 3.5)] # default: (6.4, 4.8)
        if is_zoomed:
            fig_sizes.append((8, 2))
        probas['psen'] = stat_all(psen_s)
        probas['psen']['all'] = psen_s
        if not isinstance(idx_best, type(None)):
            probas['psen']['best'] = psen_s[idx_best]

    j = 0
    for fig_size in fig_sizes:
        fig_all, axes_all = plt.subplots(1, len(probas), sharex=True,
                                         sharey=True, dpi=fp.DPI,
                                         figsize=fig_size)
        fig, axes = plt.subplots(1, len(probas), sharex=True, sharey=True,
                                 dpi=fp.DPI, figsize=fig_size)
        i = 0
        for key, proba in probas.items():
            count = 0
            for proba_individual in proba['all']:
                axes_all[i].plot(lengths, proba_individual, color=colors[count])
                count += 1
            if not isinstance(idx_best, type(None)):
                axes[i].plot(lengths, proba['best'], label=LABELS['best'],
                             color='black')
            axes_all[i].text(.95,.95, LABELS[key], ha='right',
                             va='top', transform=axes_all[i].transAxes,
                             bbox=dict(boxstyle='round', fc="w", ec="w"))
            axes[i].plot(lengths, proba['mean'], label=LABELS['avg'],
                         color='darkorange')
            axes[i].fill_between(lengths, *proba['ext'], alpha=fp.ALPHA,
                                 color='gray', label=LABELS['ext'])
            axes[i].fill_between(lengths, *proba['per'], alpha=fp.ALPHA,
                                 label=LABELS['per'], color='orange')
            axes[i].text(.95,.95, LABELS[key], horizontalalignment='right',
                         verticalalignment='top', transform=axes[i].transAxes,
                         bbox=dict(boxstyle='round', fc="w", ec="w"))
            i += 1
        axes[-1].legend(loc='upper right', bbox_to_anchor=(1, 0.85))
        for figure in [fig, fig_all]:
            figure.add_subplot(111, frameon=False)
            ax = figure.gca()
            ax.tick_params(labelcolor='none', which='both', top=False,
                           bottom=False, left=False, right=False)
            ax.grid(False)
            ax.set_xlabel(LABELS['ax_lmin_min'], labelpad=9)
            ax.set_ylabel(LABELS['ax_proba'], labelpad=16)
            sns.despine(fig=figure)

        if not isinstance(fig_supdirectory, type(None)):
            path = f"{fig_supdirectory}/{FDIR_PAR}/"
            if j == 1:
                path = path + f"fit_{fig_name}_sensitivity_zoomed.pdf"
            else:
                path = path + f"fit_{fig_name}_sensitivity.pdf"
            path = path.replace('__', '_')
            fig.savefig(path, bbox_inches='tight')
            fig_all.savefig(path.replace('.pdf', '_all.pdf'), bbox_inches='tight')
        plt.show()
        j +=1
    return None

def plot_law_sen(a, b, lmin, lengths=LENGTHS, law_nta=law_nta_usual,
                 law_sen=law_sen_usual, is_saved=False,
                 font_scale=FONT_SCALE):
    """ Plot for all `lengths` as argument their corresponding `densities` and
    save it if asked.

    """
    # Plotting.
    sns.set_context(fp.CONTEXT, font_scale=font_scale)
    plt.figure(dpi=fp.DPI)
    plt.clf()
    plt.plot(lengths, law_sen(lengths, a, b, lmin), linewidth=2.5, label=
           rf'$(a_{{sen}}, b_{{sen}}, {{\ell}}_{{min}}) = ({a}, {b}, {lmin})$')
    plt.xlabel(LABELS['ax_lmin_min'], labelpad=6)
    plt.ylabel(r'$p_{sen}$', labelpad=8) # LABELS['ax_proba_sen'],
    plt.legend()
    sns.despine()
    # Saving.
    if is_saved:
        plt.savefig(f'figures/parameters/law_sen_A{a}_B{b}.pdf',
                    bbox_inches='tight')
    plt.show()
    sns.set_context(fp.CONTEXT, font_scale=font_scale)
    return None

def plot_laws(parameters, lengths=LENGTHS, law_nta=law_nta_usual,
              law_sen=law_sen_usual, fig_name='', is_par_plot=False,
              fig_supdirectory=None, fig_size=(6.2,3.6),
              decimal_count=DECIMAL_COUNT):
    if isinstance(parameters[1][0], list):
        is_sen_common_AnB = parameters[1][0] == parameters[1][1]
    else:
        is_sen_common_AnB = all(parameters[1][0] == parameters[1][1])
    # Plotting.
    curve_count = 2 * is_sen_common_AnB + 3 * (not is_sen_common_AnB)
    COLORS = sns.color_palette('rocket', curve_count)
    if is_par_plot:
        labels = write_laws_s(parameters, decimal_count=decimal_count)
        name = f'_{fig_name}'
    else:
        labels = LABELS
        name = f'_wo_values_{fig_name}'
    if is_par_plot:
        plt.figure(dpi=fp.DPI, figsize=fig_size)
    else:
        plt.figure(dpi=fp.DPI)
    plt.plot(lengths, law_nta(lengths, *parameters[0]), label=labels['pnta'],
             color=COLORS[0])
    if len(parameters[1]) == 3:
        plt.plot(lengths, law_sen(lengths, *parameters[1]),
                 label=labels['psen'], color=COLORS[0])
    elif is_sen_common_AnB:
        plt.plot(lengths, law_sen(lengths, *parameters[1][0]),
                 label=labels['psen'], color=COLORS[1])
    else:
        plt.plot(lengths, law_sen(lengths, *parameters[1][0]),
                 label=labels['psenA'], color=COLORS[1])
        plt.plot(lengths, law_sen(lengths, *parameters[1][1]),
                 label=labels['psenB'], color=COLORS[2])
    plt.ylim(-.1, 1.1)
    plt.xlabel(LABELS['ax_lmin_min'])
    plt.ylabel(LABELS['ax_proba'])
    if is_par_plot:
        plt.legend(title=labels['linit'], # loc="upper right")
                   bbox_to_anchor=(1, 1), loc="lower right")
    else:
        plt.legend(title=labels['linit'])
    sns.despine()
    # Saving.
    if not isinstance(fig_supdirectory, type(None)):
        path = f"{fig_supdirectory}/{FDIR_PAR}/laws_par{name}.pdf"
        plt.savefig(path, bbox_inches='tight')
    plt.show()
    return None


def plot_exp_data(days, concentration, length):
    plt.clf()
    day_count = len(days)
    fig, axes = plt.subplots(day_count, 1, sharex=True, sharey=False,
                             dpi=fp.DPI) # figsize=(7, 15),
    # Concentration w.r.t days.
    count = len(concentration[:, 0])
    days_t = np.linspace(0, count - 1, count)

    axes[0].errorbar(days_t, 1e5 * 2**concentration[0][:, 1],
                 yerr=concentration[1][:, 1], capsize=2, label=LABELS['telo+'])
    axes[0].errorbar(days_t, 1e5 * 2**concentration[:, 0],
                 yerr=concentration[:, 0], capsize=2, fmt='.-',
                 label=LABELS['telo-'])
    axes[0].text(.85, .76, fontsize=11, borderaxespad=.9, facecolor="w",
                 edgecolor="w")
    axes[0].xlabel(LABELS['ax_t'])
    axes[0].ylabel(LABELS['ax_cexp'])

    axes[1].errorbar(days, length[0], yerr=length[1], capsize=2, fmt='.-')
    axes[1].ylabel(LABELS['ax_lavg'], labelpad=8)
    sns.despine()
    # plt.xlabel(LABELS['ax_t'], labelpad=8)
    # if is_saved:
    #     plt.savefig('figures/parameters/evo_lengths.pdf', bbox_inches='tight')
    # # Plotting of axis titles.
    # fig.add_subplot(111, frameon=False) # Add big axes, hide frame.
    # plt.tick_params(labelcolor='none', # Hide tick of the big axes.
    #                 top=False, bottom=False, left=False, right=False)
    # plt.grid(False) # And hide grid.
    # plt.xlabel(LABELS['ax_count'], fontsize=axes_fontsize, labelpad=6) # Set common titles
    # plt.ylabel(LABELS['ax_count'], fontsize=axes_fontsize, labelpad=9)
    # if is_saved:
    #     path = wp.write_fig_pop_path(cell_count, para_count) + 'hist_per_day.pdf'
    #     plt.savefig(path, bbox_inches='tight')
    return
    # plt.show()


# Utilitary functions
# -------------------

def postreat_l_init_exp(density_function_exp, linit_translation=0):
    """ Postreats data loaded from the file etat_asymp_val_juillet'.

    Parameters
    ----------
    density_function_exp : ndarray
        Density function loaded from 'etat_asymp_val_juillet'.
        NB: structured s.t. P(Linit = i) is on the 2nd half of 'L_init_EXP' for
          all 'i'in the 1st half ('i' not neccessarily an integer).

    Returns
    -------
    lengths : ndarray
        x-axis of the denstity function given by L_INIT_EXP with only integer
        lengths.
    densities : ndarray
        Corresponding y-axis
    support : ndarray
        Support of 'densities'.
    positive_densities : ndarray
        Corresponding y-axis (positive values of 'densities').

    """
    # Separation of the file.
    length_count = int(len(density_function_exp) / 4)
    lengths = np.linspace(1, length_count, length_count)
    densities_all = density_function_exp[2 * length_count:]

    densities = np.zeros(length_count)
    # We remove lengths that are not integers.
    for i in range(length_count):
        densities[i] = densities_all[2 * i] + densities_all[2 * i + 1]

    # We make sure it is a probability.
    mass = np.sum(np.diff(np.append(0, lengths)) * densities)
    densities = densities / mass

    # And translate the distribution by `l_init`.
    lengths = lengths + linit_translation

    # We remove proba equals to 0 from <densities>
    #  and lengths with proba 0 from <lengths>.
    support = np.array([])
    positive_densities = np.array([])
    for i in range(len(densities)):
        if densities[i] != 0:
            support  = np.append(support, lengths[i])
            positive_densities = np.append(positive_densities, densities[i])
    return lengths, densities, support, positive_densities

def transform_l_init(density_function_exp, ltrans=0, l0=0, l1=0):
    """ Postreats data loaded from the file etat_asymp_val_juillet'.

    Parameters
    ----------
    density_function_exp : ndarray
        Density function loaded from 'etat_asymp_val_juillet'.
        NB: structured s.t. P(Linit = i) is on the 2nd half of 'L_init_EXP' for
          all 'i'in the 1st half ('i' not neccessarily an integer).

    """
    # Separation of the file.
    length_count = int(len(density_function_exp) / 4)
    lengths = np.arange(1, length_count + 1)
    densities_all = density_function_exp[2 * length_count:]

    densities = np.zeros(length_count)
    # We remove lengths that are not integers.
    for i in range(length_count):
        densities[i] = densities_all[2 * i] + densities_all[2 * i + 1]
    # We make sure it is a probability renormalizing.
    diff = np.diff(np.append(0, lengths))
    densities = densities / np.sum(diff * densities)
    lmod_idx = np.where(lengths == np.argmax(densities))[0][0]
    lmod = lengths[lmod_idx]

    linf_idx = np.where(densities > 0)[0][0]
    linf = int(lengths[linf_idx])

    lsup_idx = np.where(densities > 0)[0][-1]
    lsup = int(lengths[lsup_idx])

    # Transformation of the distribution.
    lengths_tf = np.copy(lengths)
    lengths_tf[linf_idx:lmod_idx] = (lengths_tf[linf_idx:lmod_idx] - linf) \
        * (lmod - linf - l0) / (lmod - linf) + linf + l0
    lengths_tf[lmod_idx:] = (lengths_tf[lmod_idx:] - lmod) * \
        (lsup + l1 - lmod) / (lsup - lmod) + lmod
    lengths_tf[:linf_idx] = np.linspace(1,lengths_tf[linf_idx],linf_idx+1)[:-1]

    lengths_new = np.arange(1, np.round(lengths_tf[-1]) + 1)
    densities_new = interpolate.interp1d(lengths_tf, densities)(lengths_new)
    diff_new = np.diff(np.append(0, lengths_new))
    densities_new = densities_new / np.sum(densities_new * diff_new)
    lengths_new = lengths_new + ltrans
    return lengths_new, densities_new

def transform_l_init_old(density_function_exp, ltrans=0, l0=0, l1=0):
    """ Postreats data loaded from the file etat_asymp_val_juillet'.

    Parameters
    ----------
    density_function_exp : ndarray
        Density function loaded from 'etat_asymp_val_juillet'.
        NB: structured s.t. P(Linit = i) is on the 2nd half of 'L_init_EXP' for
          all 'i'in the 1st half ('i' not neccessarily an integer).

    Returns
    -------
    lengths : ndarray
        x-axis of the denstity function given by L_INIT_EXP with only integer
        lengths.
    densities : ndarray
        Corresponding y-axis
    support : ndarray
        Support of 'densities'.
    positive_densities : ndarray
        Corresponding y-axis (positive values of 'densities').

    """
    # Separation of the file.
    length_count = int(len(density_function_exp) / 4)
    lengths = np.linspace(1, length_count, length_count)
    densities_all = density_function_exp[2 * length_count:]

    densities = np.zeros(length_count)
    # We remove lengths that are not integers.
    for i in range(length_count):
        densities[i] = densities_all[2 * i] + densities_all[2 * i + 1]

    # We make sure it is a probability renormalizing.
    diff = np.diff(np.append(lengths, lengths[-1] + 1))
    densities = densities / np.sum(diff * densities)
    # densities = np.append(densities, 1 - sum(densities))

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


def arr_remove(arr, min_value):
    """ Return the array 'arr' with all values lower (strictly) to 'min_value'
    removed.

    """
    arr_new = np.array([])
    for value in arr:
        if value >= min_value:
            arr_new = np.append(arr_new, value)
    return arr_new

def postreat_cycles_exp(cycles_data, cycle_min, threshold):
    """ Postreats data loaded from 'CC_durations.mat' where a postreatment
    (a classification between cycles) was alredy made.

    Parameters
    ----------
    cycles_data : dict
        Dictionnary of experimental cycle duration times.
    cycle_min : int
        Minimum cycle duration time in 10 minutes (any lower experimental one
        is taken back from the data.
    threshold : int
        threshold in minutes between normal and long cycles.

    """
    cdts = {}
    # Extraction of the data in separate variables.
    cdts['nor-telo+'] = cycles_data['normalCC_durations'][0]
    cdts['btype'] = cycles_data['typebCC_durations'][0] # Cdts af 1st seq nta.
    cdts['nta1'] = cycles_data['firstarrestseqCC_durations'][0] # 1st seq nta.
    cdts['sen'] = cycles_data['senescenceCC_durations'][0]

    # We convert to min remove anormal cycle durations (< cycle_min).
    cdts['nor-telo+'] = 10 * arr_remove(cdts['nor-telo+'] , cycle_min)
    cdts['btype'] = 10 * arr_remove(cdts['btype'], cycle_min)
    cdts['nta1'] = 10 * arr_remove(cdts['nta1'], cycle_min)
    cdts['sen'] = 10 * arr_remove(cdts['sen'], cycle_min)

    # Separation of cdts['btype'] in 2 distribution: <= 180 min & > 180 min.
    threshold_min = 10 * threshold
    cdts['norB'] = cdts['btype'][cdts['btype'] <= threshold_min]
    at_theeshold_count = len(cdts['btype'][cdts['btype'] == threshold_min])
    cdts['nta2+'] = cdts['btype'][cdts['btype'] > threshold_min]
    return cdts, at_theeshold_count

def extract_cycles_dataset(folder="data/microfluidic/"):
    """ Extract the data postreated by our script, through
    `aux_make_cycle_dataset.py`.

    """
    cdts = {}
    cdts['arr'] = np.loadtxt(folder + "lcycles.csv")
    cdts['nta'] = np.loadtxt(folder + "lcycles_nta.csv")
    cdts['sen'] = np.loadtxt(folder + "lcycles_sen.csv")
    cdts['senA'] = np.loadtxt(folder + "lcycles_senA.csv")
    cdts['senB'] = np.loadtxt(folder + "lcycles_senB.csv")
    cdts['sen_last'] = np.loadtxt(folder + "lcycles_last_sen.csv")
    cdts['norA']  = np.loadtxt(folder + "ncycles_bf_arrest.csv")
    cdts['norB'] = np.loadtxt(folder + "ncycles_af_arrest.csv")
    return cdts

def extract_cycles_dataset_finalCut(folder="data_finalCut/"):
    """ Extract the data postreated by our script, through
    `finalCut_aux_make_cycle_dataset.py`.

    """
    cdts = {'raf': {}, 'gal': {}}
    cdts['raf']['nor'] = np.loadtxt(folder + "noFc_n2/ncycles_raf_dox.csv")
    cdts['raf']['arr'] = np.loadtxt(folder + "noFc_n2/lcycles_raf_dox.csv")
    cdts['gal']['nor'] = np.loadtxt(folder + "noFc_n2/ncycles_gal.csv")
    cdts['gal']['arr'] = np.loadtxt(folder + "noFc_n2/lcycles_gal.csv")
    return cdts

def make_time_arrays(times_per_day_count, times_saved_per_day_count, day_count,
                     step):
    times_day = np.array([])
    time_saved_idxs = {}
    q, r = np.divmod(times_per_day_count, times_saved_per_day_count)
    tspd_count_new = int(times_per_day_count / q)
    # if r != 0:
    #     print("WARNING: `TIMES_SAVED_PER_DAY_COUNT` the actual number of times"
    #           f" saved is not {times_saved_per_day_count} anymore but "
    #           f"{tspd_count_new}")

    for day in range(day_count):
        times_temp = np.linspace(day, day + 1, times_per_day_count)
        times_temp[-1] = times_temp[-1] - step
        idxs_temp = np.arange(len(times_day),
                              len(times_day) + times_per_day_count + 1,
                              times_per_day_count // tspd_count_new)
        idxs_temp[-1] -= 1
        times_day = np.append(times_day, times_temp)
        time_saved_idxs[day] = idxs_temp
    times_day = np.append(times_day, day + 1)
    times = times_day * 60 * 24 # Times in [min].
    # Indexes i s.t. t[i] is a time of dilution.
    dil_idxs = times_per_day_count * np.arange(1, day_count)
    dil_saved_idxs = tspd_count_new * np.arange(1, day_count)
    return times, time_saved_idxs, dil_idxs, dil_saved_idxs
