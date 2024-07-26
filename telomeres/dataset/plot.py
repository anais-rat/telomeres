#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 19:17:55 2023

@author: arat
"""

from os.path import join
from textwrap import wrap
import math
import matplotlib.pylab as plt
import numpy as np
import os
import seaborn as sns

from telomeres.dataset.extract_processed_dataset import \
    extract_distribution_cycles
from telomeres.auxiliary.figures_properties import LABELS as LABELS_FP
from telomeres.auxiliary.figures_properties import MY_PALETTE, MY_COLORS
from telomeres.model.posttreat import transform_l_init
from telomeres.auxiliary.write_paths import characteristics_to_string, \
    FOLDER_FIG


DIR_DAT = 'dataset'
DIR_PAR = 'parameters'

# Global labels.
# > Maximal number of caracters per line. (24-7 for article plotting).
LABEL_MAX = 32
TO_ADD = 18  # For tex text use (unplotted caracters countted as plottedd one).


LABELS = {'ax_dens': 'Density',
          #
          'arr': 'Long cycle',  # Arrest
          'nta': 'Non-terminal arrest',
          'sen': 'Terminal arrest',
          'senA': 'Terminal arrest of type A',
          'senB': 'Terminal arrest of type B',
          'sen_last': 'Terminal arrest before death',
          'nor': 'Normal cycle',
          'norA': 'Normal cycle of type A',
          'norB': 'Normal cycle of type B',
          'norA_short': r"$nor_A$",
          'norB_short': r"$nor_B$",
          'nta_short': r"$nta$",
          'sen_short': r"$sen$"}
LABELS.update(LABELS_FP)
LABELS_ = {}
for key, label in LABELS.items():
    LABELS_[key] = "\n".join(wrap(label[::-1], LABEL_MAX))[::-1]


def plot_ltelomere_init_wrt_par(ltrans=0, l0=0, l1=0, distribution=None,
                                labels=None, legend_key=None,
                                fig_subdirectory=None, fig_size=None):
    """ltrans, l0 and l1 should be lists (of same length) of values to plot
    a flot for fixed value, common to all plot.

    """
    # density_function_exp = np.loadtxt('data/etat_asymp_val_juillet')
    ltrans_s, l0_s, l1_s = [ltrans], [l0], [l1]
    if isinstance(ltrans, list) or isinstance(ltrans, np.ndarray):
        ltrans_s = ltrans
    if isinstance(l0, list) or isinstance(l0, np.ndarray):
        l0_s = l0
    if isinstance(l1, list) or isinstance(l1, np.ndarray):
        l1_s = l1
    a, b, c = len(ltrans_s), len(l0_s), len(l1_s)
    par_count = max([a, b, c])

    colors = sns.color_palette('rocket', par_count)[::-1]
    plt.figure(figsize=fig_size)
    for i in range(par_count):
        par_l_init = [ltrans_s[min(i, a-1)], l0_s[min(i, b-1)],
                      l1_s[min(i, c-1)]]
        lengths, densities = transform_l_init(distribution=distribution,
                                              par_l_init=par_l_init)
        if isinstance(labels, type(None)):
            plt.plot(lengths, densities, color=colors[i])
        else:
            plt.plot(lengths, densities, label=labels[i], color=colors[i])
    if isinstance(legend_key, type(None)):
        plt.legend()
    else:
        plt.legend(title=LABELS[legend_key])
    plt.xlabel('Telomere length (bp)', labelpad=6)
    plt.ylabel('Density', labelpad=8)
    sns.despine()
    if not isinstance(fig_subdirectory, type(None)):
        folder = join(FOLDER_FIG, fig_subdirectory, DIR_DAT)
        if not os.path.exists(folder):
            os.makedirs(folder)
        path = join(folder, f'linit_wrt_{legend_key}.pdf')
        print("Saved at ", path, '\n')
        plt.savefig(path, bbox_inches='tight')
    plt.show()
    return


def plot_data_exp_length_curves(x, y, std, fig_subdirectory):
    w, h = plt.figaspect(.85)
    plt.figure(figsize=(w, h))
    plt.errorbar(x, y, yerr=std, capsize=2, fmt='.-')
    plt.ylabel(LABELS_['ax_lmode'])
    plt.xlabel(LABELS['ax_time'])
    plt.xticks(x)
    sns.despine()
    if not isinstance(fig_subdirectory, type(None)):
        folder = join(FOLDER_FIG, fig_subdirectory, DIR_DAT)
        if not os.path.exists(folder):
            os.makedirs(folder)
        path = join(folder, 'evo_lengths.pdf')
        print("Saved at ", path, '\n')
        plt.savefig(path, bbox_inches='tight')
    plt.show()


def plot_data_exp_concentration_curves_final(stat_telo_m, stat_telo_p,
                                             fig_subdirectory,
                                             bbox_to_anchor=None,
                                             ylabel=LABELS_['ax_cexp'],
                                             fig_name=None):
    colors = MY_COLORS[:2]
    sns.set_palette(colors)
    # Concentration w.r.t days.
    plt.figure()
    x = np.arange(len(stat_telo_m['avg'])) + 1
    plt.errorbar(x, stat_telo_p['avg'], yerr=stat_telo_p['std'], capsize=2,
                 label=LABELS['telo+'])
    plt.errorbar(x, stat_telo_m['avg'], yerr=stat_telo_m['std'], capsize=2,
                 label=LABELS['telo-'])
    plt.legend(bbox_to_anchor=bbox_to_anchor)
    plt.xlabel(LABELS['ax_time'], labelpad=6)
    plt.ylabel(ylabel, labelpad=8)
    plt.xticks(x)
    sns.despine()
    if not isinstance(fig_subdirectory, type(None)):
        folder = join(FOLDER_FIG, fig_subdirectory, DIR_DAT)
        if not os.path.exists(folder):
            os.makedirs(folder)
        path = join(folder, f"{fig_name or 'concentration_final'}.pdf")
        print("Saved at ", path, '\n')
        plt.savefig(path, bbox_inches='tight')
    plt.show()
    return


def weighted_std(values, weights, origin=None):
    """Return the weighted standard deviation with respect to `origin` (the
    average if None).

    values, weights -- Numpy ndarrays with the same shape.

    """
    # np.sqrt(np.cov(lengths, aweights=densities))
    if isinstance(origin, type(None)):
        origin = np.sum(values * weights)
    variance = np.sum(weights * (values - origin) ** 2)
    return math.sqrt(variance)


def plot_n_print_l_init(distribution, fig_subdirectory, fig_name=None,
                        color='grey'):
    """Print useful information on the initial distribution of telomere
    lengths and plot it.

    """
    lengths, densities = distribution

    # Support of the distribution.
    linf_idx = np.where(densities > 0)[0][0]
    linf = int(lengths[linf_idx])
    lsup_idx = np.where(densities > 0)[0][-1]
    lsup = int(lengths[lsup_idx])

    # Compute useful quantities.
    # > Cumulative probabilities.
    cumul_prob = np.array([sum(densities[:i]) for i in range(len(densities))])
    # > Index where cumulative probabilities first exceeds 0.5.
    mean = int(np.round(sum(densities * lengths)))
    std_mean = int(np.round(weighted_std(lengths, densities, mean)))
    median_idx = int(np.argmin(cumul_prob < 0.5))
    median = int(np.round(lengths[median_idx]))
    std_median = int(np.round(weighted_std(lengths, densities, median)))
    mode_idx = np.argmax(densities)
    mode = int(np.round(lengths[mode_idx]))
    std_mode = int(np.round(weighted_std(lengths, densities, mode)))
    # Printing.
    print('\n Distribution of telomere lengths: \n'
          ' ---------------------------------')
    print(f'Average initial length: {mean} +/- {std_mean}')
    print(f'Median initial length: {median} +/- {std_median}')
    print(f'Mode: {mode} +/- {std_mode}')
    print(f'Support of distribution fonction: [{linf},{lsup}]')
    diff = np.diff(np.append(lengths[0]-1, lengths))
    print('int L_init(x) dx: ', np.sum(densities * diff))
    print('Last values of <L_init_prob>: ', densities[-4:])

    # Plot.
    plt.figure()
    plt.plot(lengths, densities, color=color)
    ax = plt.gca()
    text = rf"mean = {int(np.round(mean))}" + '\n' \
        + rf"median = {int(np.round(median))}" + '\n' \
        + rf"mode = {mode}" + '\n' \
        + rf"support = [{int(linf)}, {int(lsup)}]"
    plt.text(.45, .6, text, transform=ax.transAxes, bbox=dict(facecolor='w'))
    plt.xlabel('Telomere length (bp)', labelpad=6)
    plt.ylabel('Density', labelpad=8)
    sns.despine()
    # Save.
    if not isinstance(fig_subdirectory, type(None)):
        folder = join(FOLDER_FIG, fig_subdirectory, DIR_DAT)
        if not os.path.exists(folder):
            os.makedirs(folder)
        fig_name = fig_name or 'linit_original'
        path = join(folder, f"{fig_name}.pdf")
        print("Saved at ", path, '\n')
        plt.savefig(path, bbox_inches='tight')
    plt.show()
    return None


def plot_ltelomere_init(fig_subdirectory, distribution=None, par_l_init=None,
                        bbox_to_anchor=None, labels=[None, None]):
    """Postreats data loaded from the file etat_asymp_val_juillet'.

    Parameters
    ----------
    density_function_exp : ndarray
        Density function loaded from 'etat_asymp_val_juillet'.
        NB: structured s.t. P(Linit = i) is on the 2nd half of 'L_init_EXP' for
          all 'i'in the 1st half ('i' not neccessarily an integer).

    """
    if isinstance(distribution, type(None)):
        distribution = transform_l_init()
        if isinstance(par_l_init, type(None)):
            return plot_n_print_l_init(distribution, fig_subdirectory)
    distribution_new = transform_l_init(distribution=distribution,
                                        par_l_init=par_l_init)
    plot_n_print_l_init(distribution_new, fig_subdirectory,
                        fig_name='linit_transformation', color='darkorange')
    plt.figure()
    plt.xlabel('Telomere length (bp)')
    plt.ylabel('Density')
    plt.plot(*distribution, label=labels[0], color='grey')
    plt.plot(*distribution_new, '--', label=labels[1], color='darkorange')
    if isinstance(bbox_to_anchor, type(None)):
        plt.legend(loc="upper right")
    else:
        plt.legend(bbox_to_anchor=bbox_to_anchor)
    sns.despine()
    if not isinstance(fig_subdirectory, type(None)):
        folder = join(FOLDER_FIG, fig_subdirectory, DIR_DAT)
        if not os.path.exists(folder):
            os.makedirs(folder)
        path = join(folder, 'linit_comparison.pdf')
        print("Saved at ", path, '\n')
        plt.savefig(path, bbox_inches='tight')
    plt.show()


def plot_hist_s(datas, binwidth, fig_ratio, labels=None, saving_path=None,
                xlabel=LABELS['cycle'], ylabel=LABELS['ax_count']):
    fig_count = len(datas)
    if not isinstance(binwidth, list):
        binwidth = [binwidth] * fig_count
    w, h = plt.figaspect(fig_ratio)
    fig, axes = plt.subplots(1, fig_count, figsize=(w, h))
    for idx in range(fig_count):
        sns.histplot(datas[idx], ax=axes[idx], binwidth=binwidth[idx],
                     kde=True, kde_kws={'bw_adjust': 1.4}, legend=False,
                     color="black", )
        if not isinstance(labels, type(None)):
            if fig_count > 3:
                x, y = .95, 1.1
            else:
                x, y = .95, .95
            axes[idx].text(x, y, labels[idx], ha='right', va='top',
                           transform=axes[idx].transAxes,
                           bbox=dict(boxstyle='round', fc="w", ec="w"))
        axes[idx].set_ylabel(None)
        ymax = int(axes[idx].get_ylim()[1] * 1.25)
        axes[idx].set_ylim([0, ymax])
    sns.despine()
    fig.add_subplot(111, frameon=False)
    ax = fig.gca()
    ax.tick_params(labelcolor='none', which='both', top=False, bottom=False,
                   left=False, right=False)
    ax.grid(False)
    ax.set_xlabel(xlabel, labelpad=9)
    ax.set_ylabel(ylabel, labelpad=16)
    if not isinstance(saving_path, type(None)):
        plt.savefig(saving_path, bbox_inches='tight')
    plt.show()


def plot_cycles_from_dataset(fig_subdirectory, is_saved=False):
    if is_saved:
        folder = join(FOLDER_FIG, fig_subdirectory, DIR_DAT)
        if not os.path.exists(folder):
            os.makedirs(folder)
        print(f"Figures saved in {folder} \n")
    cycles = extract_distribution_cycles()
    path = None

    # Histograms of cdt per categories of the model.
    keys_to_plot_ = [['norA', 'norB', 'arr'],
                     ['norA', 'norB', 'nta', 'sen'],
                     ['senA', 'senB'],
                     ['sen', 'sen_last'],
                     ['nta', 'sen', 'arr']]
    fig_ratios = [0.35, 0.32, 0.28]
    for keys in keys_to_plot_:
        if is_saved:
            name = f'cycles_hists_{characteristics_to_string(keys)}'
            path = join(folder, f'{name}.pdf')
        if keys == ['norA', 'norB', 'nta', 'sen']:
            binwidth = [10, 10, 40, 40]
            path_ = None
            if is_saved:
                path_ = path.replace('.pdf', '_acronym.pdf')
            plot_hist_s([cycles[key] for key in keys], binwidth=binwidth,
                        fig_ratio=fig_ratios[2], labels=[LABELS[key+'_short']
                        for key in keys], saving_path=path_)
        else:
            binwidth = 10
        plot_hist_s([cycles[key] for key in keys], binwidth=binwidth,
                    fig_ratio=fig_ratios[len(keys) - 2],
                    labels=[LABELS[key] for key in keys], saving_path=path)

    # Distributions in kernel density estimate (KDE).
    # > Classical.
    keys = ['norA', 'norB', 'nta', 'sen']
    sns.kdeplot(data=[cycles[key] for key in keys][::-1], log_scale=False,
                fill=True, common_norm=False, palette=MY_PALETTE, alpha=.5,
                linewidth=0, bw_adjust=2, legend=False)
    plt.ticklabel_format(axis="y", style='sci', scilimits=(0, 0))
    plt.legend(labels=[LABELS[key] for key in keys], bbox_to_anchor=(1.1, 1))
    plt.xlabel(LABELS['cycle'], labelpad=6)
    plt.ylabel(LABELS_['ax_dens'], labelpad=8)
    sns.despine()
    if is_saved:
        name = f'cycles_kde_{characteristics_to_string(keys)}'
        path = join(folder, f'{name}.pdf')
        plt.savefig(path, bbox_inches='tight')
    plt.show()

    # > Log-scale.
    keys = ['norA', 'norB', 'nta', 'sen']
    sns.kdeplot(data=[cycles[key] for key in keys][::-1], log_scale=True,
                fill=True, common_norm=True, palette=MY_PALETTE, alpha=.5,
                linewidth=0, bw_adjust=2, legend=False)
    plt.legend(labels=[LABELS[key] for key in keys], bbox_to_anchor=(1.2, 1))
    plt.xlabel(LABELS['cycle_log'], labelpad=6)
    plt.ylabel(LABELS_['ax_dens'], labelpad=8)
    sns.despine()
    if is_saved:
        name = f'cycles_kde_{characteristics_to_string(keys)}_logscale'
        path = join(folder, f'{name}.pdf')
        plt.savefig(path, bbox_inches='tight')
    plt.show()

    # > Other (to remore?).
    # >> Legend in full text.
    keys = ['norA', 'norB', 'nta', 'sen']
    sns.kdeplot(data=[cycles[key] for key in keys][::-1], log_scale=False,
                fill=True, common_norm=False, palette=MY_PALETTE, alpha=.5,
                linewidth=0, bw_adjust=2.5, legend=False)
    plt.ticklabel_format(axis="y", style='sci', scilimits=(0, 0))
    plt.xlim(0, 1000)
    plt.legend(labels=[LABELS[key] for key in keys])
    plt.xlabel(LABELS['cycle'], labelpad=6)
    plt.ylabel(LABELS_['ax_dens'], labelpad=8)
    sns.despine()
    if is_saved:
        name = f'cycles_kde_{characteristics_to_string(keys)}_xlim100'
        path = join(folder, f'{name}.pdf')
        plt.savefig(path, bbox_inches='tight')
    plt.show()
    # >> Legend with accronyms.
    keys = ['norA', 'norB', 'nta', 'sen']
    sns.kdeplot(data=[cycles[key] for key in keys][::-1], log_scale=False,
                fill=True, common_norm=False, palette=MY_PALETTE, alpha=.5,
                linewidth=0, bw_adjust=2.5, legend=False)
    plt.ticklabel_format(axis="y", style='sci', scilimits=(0, 0))
    plt.xlim(0, 1000)
    plt.legend(labels=[LABELS[key + '_short'] for key in keys])
    plt.xlabel(LABELS['cycle'], labelpad=6)
    plt.ylabel(LABELS_['ax_dens'], labelpad=8)
    sns.despine()
    if is_saved:
        name = f'cycles_kde_{characteristics_to_string(keys)}_xlim100'
        path = join(folder, f'{name}_acronym.pdf')
        plt.savefig(path, bbox_inches='tight')
    plt.show()
