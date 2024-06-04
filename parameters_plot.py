#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 23:54:53 2023

@author: arat
"""

import imp
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

import aux_figures_properties as fp
import aux_write_paths as wp
import parameters as par
import aux_parameters_functions as parf
import population_plot as pp

imp.reload(fp)
imp.reload(par)


# Initial distribution of telomere lengths.
# -----------------------------------------

PALETTE = 'viridis'

def distribution_shortest(cell_count, is_plotted=False,
                          par_l_init_update=None):
    length_count = int(len(par.L_INIT_EXP) / 4)
    if isinstance(par_l_init_update, type(None)):
        lengths = np.linspace(1, length_count, length_count)
        probas = par.L_INIT_EXP[2 * length_count:]
    else:
        lengths, probas = parf.transform_l_init(par.L_INIT_EXP,
                                                *par_l_init_update)
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
    plt.xlabel(pp.LABELS['ax_l'], labelpad=8)
    plt.ylabel("Density", labelpad=6)
    colors = sns.color_palette(PALETTE, curve_count)
    for i in range(curve_count):
        lengths, distrib = distribution_shortest(cell_count,
                                        par_l_init_update=par_l_inits[i])
        if is_rescaled:
            distrib = distrib / np.max(distrib)
        if cell_count == 1e5:
            plt.plot(lengths[:350], distrib[:350], label=r"$N_{exp}$",
                     color=colors[i])
        else:
            plt.plot(lengths[:350], distrib[:350], label=rf"${par_l_inits[i]}$",
                     color=colors[i])
    plt.legend(title=r"$(\ell_T, \ell_0, \ell_1)$")
    sns.despine()
    plt.title(rf"$N_{{init}} = {cell_count}$")
    plt.show()

    plt.figure()
    plt.xlabel(pp.LABELS['ax_l'], labelpad=8)
    plt.ylabel("Density", labelpad=6)
    for i in range(curve_count):
        lengths, distrib = distribution_shortest_max(cell_count,
                                        par_l_init_update=par_l_inits[i])
        if is_rescaled:
            distrib = distrib / np.max(distrib)
        if cell_count == 1e5:
            plt.plot(lengths[:350], distrib[:350], label=r"$N_{exp}$",
                     color=colors[i])
        else:
            plt.plot(lengths[:350], distrib[:350], label=rf"${par_l_inits[i]}$",
                     color=colors[i])
    plt.legend(title=r"$(\ell_T, \ell_0, \ell_1)$")
    sns.despine()
    plt.title(rf"$N_{{init}} = {cell_count}$")
    plt.show()


def plot_distributions_shortest_min(cell_counts, is_rescaled=False,
                                    fig_supdirectory=None):
    cell_counts = np.sort(cell_counts).astype(int)
    curve_count = len(cell_counts)
    plt.figure()
    plt.xlabel(pp.LABELS['ax_l'], labelpad=8)
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
    plt.legend(title=pp.LABELS['leg_cell_count'])
    sns.despine()

    if not isinstance(fig_supdirectory, type(None)):
        directory_path = fig_supdirectory + '/dataset'
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        fig_name = directory_path + "/distributions_shortest_min_c" + \
            wp.list_to_string(cell_counts) + ".pdf"
        plt.savefig(fig_name, bbox_inches='tight')
    plt.show()

def distribution_shortest_max(cell_count, is_plotted=False,
                              par_l_init_update=None):
    length_count = int(len(par.L_INIT_EXP) / 4)
    if isinstance(par_l_init_update, type(None)):
        lengths = np.linspace(1, length_count, length_count)
        probas = par.L_INIT_EXP[2 * length_count:]
    else:
        lengths, probas = parf.transform_l_init(par.L_INIT_EXP,
                                                *par_l_init_update)
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
                                fig_supdirectory=None):
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
            axes[0].plot(distrib_min[:350], label=r"$N_{exp}$", color=colors[i])
            axes[1].plot(distrib_max[:350], color=colors[i])
        else:
            axes[0].plot(distrib_min[:350], label=cell_counts[i],
                         color=colors[i])
            axes[1].plot(distrib_max[:350], color=colors[i])
    axes[0].legend(title=pp.LABELS['leg_cell_count'])
    fig.add_subplot(111, frameon=False) # Add big axes, hide frame.
    plt.tick_params(labelcolor='none', # Hide tick of the big axes.
                    top=False, bottom=False, left=False, right=False)
    plt.grid(False) # And hide grid.
    plt.xlabel(pp.LABELS['ax_l'], labelpad=8) # Set common titles
    plt.ylabel("Density", labelpad=23)
    if not isinstance(fig_supdirectory, type(None)):
        directory_path = fig_supdirectory + '/dataset'
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        fig_name = directory_path + "/distributions_shortest_c" + \
            wp.list_to_string(cell_counts) + ".pdf"
        plt.savefig(fig_name, bbox_inches='tight')
    plt.show()

