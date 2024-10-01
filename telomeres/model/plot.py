#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 26 14:00:06 2021

@author: arat

Script containing functions allowing to plot model related information.

"""

from os.path import join

# import imp
from copy import deepcopy
import math
import matplotlib.ticker as ticker
# import matplotlib.pylab as plt
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from textwrap import wrap

import telomeres.auxiliary.figures_properties as fp
import telomeres.auxiliary.write_paths as wp
import telomeres.model.functions as mfct


DECIMAL_COUNT = 2
FONT_SCALE = 1.2  # To remove (initially in fp) !!!!
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

CONTEXT = "notebook"


# Auxiliary

def list_to_strings(list_to_write, is_last_int=False, decimal_count=None):
    """Same as `list_to_string` except that a list of well formatted float
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
        lmin = parameters[2]
        name = law_name[-1]
        name = name.replace('n', '')
        string = string + rf'${{ \ell }}_{{ {{min}}_{{ {name} }} }} = {lmin}$'
        # rf'$ {{\,}} {{ \mathds{{1}} }}_{{ {{\ell}} \geq {lmin} }}$'
    return string


def write_linit_law(parameters):
    parameters_rounded = np.round(parameters)
    ltrans, l0, l1 = parameters_rounded.astype('int')
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


LENGTHS = np.linspace(0, 250, 2501)


def law_sen_usual(lmins, a, b, lmin):
    densities = np.array([min(1, b * math.exp(- a * lmin)) for lmin in lmins])
    densities[lmins < lmin] = 1
    return densities


def law_nta_usual(lmins, a, b):
    densities = np.array([min(1, b * math.exp(- a * lmin)) for lmin in lmins])
    return densities


def plot_laws_nta_various_a(a_to_test, b, lengths=LENGTHS,
                            law_nta=law_nta_usual, fig_subdirectory=None,
                            font_scale=FONT_SCALE):
    """Plot on `lengths` the laws for non-terminal arrests defined by
    parameter `b` fixed and parameter a in `a_to_test` and save it in
    `figure/<fig_subdirectory>` if `fig_subdirectory` is not None.

    """
    sns.set_context(CONTEXT, font_scale=font_scale)
    # Plotting.
    plt.figure()
    for a in a_to_test:
        plt.plot(lengths, law_nta(lengths, a, b), label=rf'$a_{{nta}} = {a}$',
                 linewidth=2.5)
        plt.xlabel(LABELS['ax_lmin_min'], labelpad=6)
        plt.ylabel(r'$p_{nta}$', labelpad=8)  # LABELS['ax_proba_nta'])
    ax = plt.gca()
    plt.text(.92, 1.02, rf'$b_{{nta}} = {b}$', horizontalalignment='right',
             transform=ax.transAxes)
    plt.legend()
    sns.despine()
    # Saving.
    if not isinstance(fig_subdirectory, type(None)):
        folder = join(wp.FOLDER_FIG, fig_subdirectory, FDIR_PAR)
        if not os.path.exists(folder):
            os.makedirs(folder)
        path = join(folder, f'law_nta_A{a_to_test}_B{b}.pdf')
        print("\n Saved at: ", path)
        plt.savefig(path, bbox_inches='tight')
    plt.show()
    sns.set_context(CONTEXT, font_scale=FONT_SCALE)
    return None


def plot_laws_nta_various_b(b_to_test, a, lengths=LENGTHS,
                            law_nta=law_nta_usual, fig_subdirectory=None,
                            font_scale=FONT_SCALE):
    """Plot on `lengths` the laws for non-terminal arrests defined by
    parameter `a` fixed and parameter b in `b_to_test` and save it in
    `figure/<fig_subdirectory>` if `fig_subdirectory` is not None.

    """
    sns.set_context(CONTEXT, font_scale=font_scale)
    # Plotting.
    plt.figure()
    for b in b_to_test:
        plt.plot(lengths, law_nta(lengths, a, b), label=rf'$b_{{nta}} = {b}$',
                 linewidth=2.5)
        plt.xlabel(LABELS['ax_lmin_min'], labelpad=6)
        plt.ylabel(LABELS['pnta'], labelpad=8)
    ax = plt.gca()
    plt.text(.92, 1.02, rf'$a_{{nta}} = {a}$', horizontalalignment='right',
             transform=ax.transAxes)
    plt.legend()
    sns.despine()
    # Saving.
    if not isinstance(fig_subdirectory, type(None)):
        folder = join(wp.FOLDER_FIG, fig_subdirectory, FDIR_PAR)
        if not os.path.exists(folder):
            os.makedirs(folder)
        path = join(folder, f'law_nta_A{a}_B{b_to_test}.pdf')
        print("\n Saved at: ", path)
        plt.savefig(path, bbox_inches='tight')
    plt.show()
    sns.set_context(CONTEXT, font_scale=FONT_SCALE)
    return None


def stat_all(arr_stat, p_up=fp.P_UP, p_down=fp.P_DOWN, axis=0):
    """Copy from `functions_auxiliary.py`."""
    stats = {'mean': np.mean(arr_stat, axis=axis),
             'per': [np.percentile(arr_stat, p_down, axis=axis),
                     np.percentile(arr_stat, p_up, axis=axis)],
             'ext': [np.min(arr_stat, axis=axis),
                     np.max(arr_stat, axis=axis)]}
    return stats


def plot_laws_s(parameters_s, idx_best=None, lengths=LENGTHS,
                law_nta=law_nta_usual, law_sen=law_sen_usual, fig_name='',
                fig_subdirectory=None, is_zoomed=False, tick_spacing=None):
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
        fig_sizes = [(14, 3.5)]  # default: (6.4, 4.8)
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
        fig_sizes = [(11, 3.5)]  # default: (6.4, 4.8)
        if is_zoomed:
            fig_sizes.append((8, 2))
        probas['psen'] = stat_all(psen_s)
        probas['psen']['all'] = psen_s
        if not isinstance(idx_best, type(None)):
            probas['psen']['best'] = psen_s[idx_best]

    j = 0
    for fig_size in fig_sizes:
        fig_all, axes_all = plt.subplots(1, len(probas), sharex=True,
                                         sharey=True, figsize=fig_size)
        fig, axes = plt.subplots(1, len(probas), sharex=True, sharey=True,
                                 figsize=fig_size)
        i = 0
        for key, proba in probas.items():
            count = 0
            for proba_individual in proba['all']:
                axes_all[i].plot(lengths, proba_individual,
                                 color=colors[count])
                count += 1
            if not isinstance(idx_best, type(None)):
                axes[i].plot(lengths, proba['best'], label=LABELS['best'],
                             color='black')
            axes_all[i].text(.95, .95, LABELS[key], ha='right',
                             va='top', transform=axes_all[i].transAxes,
                             bbox=dict(boxstyle='round', fc="w", ec="w"))
            axes[i].plot(lengths, proba['mean'], label=LABELS['avg'],
                         color='darkorange')
            axes[i].fill_between(lengths, *proba['ext'], alpha=fp.ALPHA,
                                 color='gray', label=LABELS['ext'])
            axes[i].fill_between(lengths, *proba['per'], alpha=fp.ALPHA,
                                 label=LABELS['per'], color='orange')
            axes[i].text(.95, .95, LABELS[key], horizontalalignment='right',
                         verticalalignment='top', transform=axes[i].transAxes,
                         bbox=dict(boxstyle='round', fc="w", ec="w"))
            if not isinstance(tick_spacing, type(None)):
                fig.tight_layout()
                axes[i].xaxis.set_major_locator(
                    ticker.MultipleLocator(tick_spacing))
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

        if not isinstance(fig_subdirectory, type(None)):
            folder = join(wp.FOLDER_FIG, fig_subdirectory, FDIR_PAR)
            if not os.path.exists(folder):
                os.makedirs(folder)
            if j == 1:
                path = join(folder, f"fit_{fig_name}_sensitivity_zoomed.pdf")
            else:
                path = join(folder, f"fit_{fig_name}_sensitivity.pdf")
            path = path.replace('__', '_')
            print("\n Saved at: ", path)
            fig.savefig(path, bbox_inches='tight')
            fig_all.savefig(path.replace('.pdf', '_all.pdf'),
                            bbox_inches='tight')
        plt.show()
        j += 1
    return None


def plot_laws(parameters, lengths=LENGTHS, law_nta=law_nta_usual,
              law_sen=law_sen_usual, fig_name='', is_par_plot=False,
              fig_subdirectory=None, fig_size=(6.2, 3.6),
              decimal_count=DECIMAL_COUNT, tick_spacing=None):
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
        fig, axes = plt.subplots(1, 1, figsize=fig_size)
    else:
        fig, axes = plt.subplots(1, 1)
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
    plt.tight_layout()
    plt.xlabel(LABELS['ax_lmin_min'])
    plt.ylabel(LABELS['ax_proba'], wrap=True)
    if not isinstance(tick_spacing, type(None)):
        axes.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    if is_par_plot:
        plt.legend(title=labels['linit'],  # loc="upper right")
                   bbox_to_anchor=(1, 1), loc="lower right")
    else:
        plt.legend(title=labels['linit'])
    sns.despine()
    # Saving.
    if not isinstance(fig_subdirectory, type(None)):
        folder = join(wp.FOLDER_FIG, fig_subdirectory, FDIR_PAR)
        if not os.path.exists(folder):
            os.makedirs(folder)
        path = join(folder, f"laws_par{name}.pdf")
        print("\n Saved at: ", path)
        plt.savefig(path, bbox_inches='tight')
    plt.show()
    return None


# Initial distribution of telomere lengths.
# -----------------------------------------

PALETTE = 'viridis'


def distribution_shortest(cell_count, is_plotted=False, par_l_init=None):
    lengths, probas = mfct.transform_l_init(par_l_init=par_l_init)
    distrib = np.array([0])
    for i in range(1, len(lengths)):
        tmp = 0
        for j in [2*i, 2*i + 1]:
            tmp += (1 - sum(probas[:j-1])) ** (32 * cell_count) - \
                   (1 - sum(probas[:j])) ** (32 * cell_count)
        distrib = np.append(distrib, tmp)
    if is_plotted:
        plt.plot(lengths, probas)
        plt.plot(lengths, distrib)
    return lengths, distrib


def plot_distributions_shortest_wrt_linit(par_l_inits, cell_count,
                                          is_rescaled=False,
                                          fig_supdirectory=None):
    curve_count = len(par_l_inits)
    plt.figure()
    plt.xlabel(LABELS['ax_l'], labelpad=8)
    plt.ylabel("Density", labelpad=6)
    colors = sns.color_palette(PALETTE, curve_count)
    for i in range(curve_count):
        lengths, distrib = distribution_shortest(cell_count,
                                                 par_l_init=par_l_inits[i])
        if is_rescaled:
            distrib = distrib / np.max(distrib)
        if cell_count == 1e5:
            plt.plot(lengths[:350], distrib[:350], label=r"$N_{exp}$",
                     color=colors[i])
        else:
            plt.plot(lengths[:350], distrib[:350],
                     label=rf"${par_l_inits[i]}$", color=colors[i])
    plt.legend(title=r"$(\ell_T, \ell_0, \ell_1)$")
    sns.despine()
    plt.title(rf"$N_{{init}} = {cell_count}$")
    plt.show()

    plt.figure()
    plt.xlabel(LABELS['ax_l'], labelpad=8)
    plt.ylabel("Density", labelpad=6)
    for i in range(curve_count):
        lengths, distrib = distribution_shortest_max(cell_count,
                                                     par_l_init=par_l_inits[i])
        if is_rescaled:
            distrib = distrib / np.max(distrib)
        if cell_count == 1e5:
            plt.plot(lengths[:350], distrib[:350], label=r"$N_{exp}$",
                     color=colors[i])
        else:
            plt.plot(lengths[:350], distrib[:350],
                     label=rf"${par_l_inits[i]}$", color=colors[i])
    plt.legend(title=r"$(\ell_T, \ell_0, \ell_1)$")
    sns.despine()
    plt.title(rf"$N_{{init}} = {cell_count}$")
    plt.show()


def plot_distributions_shortest_min(cell_counts, is_rescaled=False,
                                    fig_subdirectory=None):
    """Plot the initial distribution of the shortest telomere length in several
    populations, with varying sizes (`cell_counts`).

    """
    cell_counts = np.sort(cell_counts).astype(int)
    curve_count = len(cell_counts)
    plt.figure()
    plt.xlabel(LABELS['ax_l'], labelpad=8)
    plt.ylabel("Density", labelpad=6)
    colors = sns.color_palette(PALETTE, curve_count)
    for i in range(curve_count):
        distrib = distribution_shortest(cell_counts[i])[1]
        if is_rescaled:
            distrib = distrib / np.max(distrib)
        if cell_counts[i] == 1e5:
            plt.plot(distrib[:350], label=r"$N_{exp}$", color=colors[i])
        else:
            plt.plot(distrib[:350], label=cell_counts[i], color=colors[i])
    plt.legend(title=LABELS['leg_cell_count'])
    sns.despine()
    if not isinstance(fig_subdirectory, type(None)):
        folder = join(wp.FOLDER_FIG, fig_subdirectory, FDIR_PAR)
        if not os.path.exists(folder):
            os.makedirs(folder)
        fig_name = "distributions_shortest_min_c" + \
            wp.list_to_string(cell_counts) + ".pdf"
        path = join(folder, fig_name)
        print("\n Saved at: ", path)
        plt.savefig(path, bbox_inches='tight')
    plt.show()


def distribution_shortest_max(cell_count, is_plotted=False, par_l_init=None):
    lengths, probas = mfct.transform_l_init(par_l_init=par_l_init)
    distrib = np.array([0])
    for i in range(1, len(lengths)):
        tmp = 0
        for j in [2*i, 2*i + 1]:
            tmp += (1 - (1 - sum(probas[:j])) ** 32) ** cell_count - \
                   (1 - (1 - sum(probas[:j-1])) ** 32) ** cell_count
        distrib = np.append(distrib, tmp)
    if is_plotted:
        plt.plot(lengths, probas)
        plt.plot(lengths, distrib)
    return lengths, distrib


def plot_distributions_shortest(cell_counts, is_rescaled=False,
                                fig_subdirectory=None):
    cell_counts = np.sort(cell_counts).astype(int)
    curve_count = len(cell_counts)
    fig, axes = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(9, 5))
    colors = sns.color_palette(PALETTE, curve_count)
    for i in range(curve_count):
        distrib_min = distribution_shortest(cell_counts[i])[1]
        distrib_max = distribution_shortest_max(cell_counts[i])[1]
        if is_rescaled:
            distrib_min = distrib_min / np.max(distrib_min)
            distrib_max = distrib_max / np.max(distrib_max)
        if cell_counts[i] == 1e5:
            axes[0].plot(distrib_min[:350], label=r"$N_{exp}$",
                         color=colors[i])
            axes[1].plot(distrib_max[:350], color=colors[i])
        else:
            axes[0].plot(distrib_min[:350], label=cell_counts[i],
                         color=colors[i])
            axes[1].plot(distrib_max[:350], color=colors[i])
    axes[0].legend(title=LABELS['leg_cell_count'])
    fig.add_subplot(111, frameon=False)  # Add big axes, hide frame.
    plt.tick_params(labelcolor='none',  # Hide tick of the big axes.
                    top=False, bottom=False, left=False, right=False)
    plt.grid(False)  # And hide grid.
    plt.xlabel(LABELS['ax_l'], labelpad=8)  # Set common titles
    plt.ylabel("Density", labelpad=23)
    if not isinstance(fig_subdirectory, type(None)):
        folder = join(wp.FOLDER_FIG, fig_subdirectory, FDIR_PAR)
        if not os.path.exists(folder):
            os.makedirs(folder)
        fig_name = "distributions_shortest_c" + wp.list_to_string(cell_counts)
        path = join(folder, fig_name + ".pdf")
        print("\n Saved at: ", path)
        plt.savefig(path, bbox_inches='tight')
    plt.show()
