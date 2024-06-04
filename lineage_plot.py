#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 15:55:41 2022

@author: arat

Console:
pip install latex

Terminal:
sudo apt install texlive texlive-latex-extra texlive-fonts-recommended dvipng
 cm-super texlive-fonts-extra

(or sudo apt install texlive-full)

"""

# import imp
from copy import deepcopy
import os
import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
import scipy.io as sio
import time

from scipy import interpolate
from textwrap import wrap

import aux_figures_properties as fp
# imp.reload(fp)
import aux_functions as fct
import aux_parameters_functions as parf
import aux_write_paths as wp
# imp.reload(wp)
import lineage_simulation as sim
import parameters as par
# imp.reload(par)
import population_plot as pp


FORMAT = 'article'  # 'manuscript' or 'article'.

# ------------------------------------
# Definition of name-related variables
# ------------------------------------

# - nta_total: number of long cycle per lineage.
# - nta_by_idx: number of long cycle per sequence of nta distinguishing
#             between the 1st, 2nd, ... `seq_count_max`th sequence of nta.
# - nta: number of long cycle per sequence of nta.
# - ntai: number of long cycle of the ith sequence of nta.
# - sen: number of senescent cycles.

# > Maximal number of caracters per line. (24-7 for article plotting).
if FORMAT == 'manuscript':
    LABEL_MAX = 32
    TO_ADD = 18 # For tex text use (unplotted caracters countted as plottedd one).
else:
    LABEL_MAX = 32
    TO_ADD = 18

LABELS = {'ax_gen': "Generation",
          'ax_gen_arr': "Generation of arrest",
          'btype_exp': "experimental type B",
          'xtype': "unknown",
          'alive': "lineage alive", # at the end of the measurements",
          'sen': "Senescent",
          'sen_short': "sen",
           # 'ax_lin_inc': 'Lineage (by increasing generation)',
          'ax_lin': 'Lineage',
          'ax_lin_norm': 'Empirical cumulative distribution',
          'lmin': pp.LABELS['lmin_avg'],
          'lavg': pp.LABELS['lavg_avg'],
          'lmin_min': pp.LABELS['lmin_min'],
          'propA': 'Proportion of type A',
          'propB': 'Proportion of type B',
          'propH': 'Proportion of type H',
          'gsen': 'Generation of senescence onset',
          'gnta': 'Generation of arrest',
          'gnta1': 'Generation of the first non-terminal arrest',
          'gnta2': 'Generation of the second non-terminal arrest',
          'gdeath': 'Generation of death'}
          # 'death': 'death',
          # 'nta': [r'$\mathrm{1^{st}~non}$' + '-' + r'$\mathrm{terminal~arrest}$',
          #         r'$\mathrm{2^{nd}~arrest}$',
          #         r'$\mathrm{3^{rd}~arrest}$', r'$\mathrm{4^{th}~arrest}$',
          #         r'$\mathrm{5^{th}~arrest}$', r'$\mathrm{6^{th}~arrest}$'],
          # 'sen': r'$\mathrm{senescence}$'} #!!!! possible wrong confusion w 'sen'
LABELS.update(fp.LABELS)

LABELS_ = {}
for key, label in LABELS.items():
    LABELS_[key] = "\n".join(wrap(label[::-1] , LABEL_MAX))[::-1]

LABEL_HIST_X = {'nta': 'Number of consecutive arrests',
                'nta_total': r'Number of non-terminal arrests'}
LABEL_HIST_Y = {'nta_total': r'Percent of (non-terminally arrested) lineages',
                'nta': r'Percent of sequences of non-terminal arrests',
                'nta_by_idx': r'Percent of sequences' +'\n' +
                r'of non-terminal arrests',
                'nta1': r'Percent of $1^{st}$ sequences of non-terminal arrests',
                'nta2': r'Percent of $2^{nd}$ sequences of non-terminal arrests',
                'nta3': r'Percent of $3^{rd}$ sequences of non-terminal arrests',
                'nta4': r'Percent of $4^{th}$ sequences of non-terminal arrests',
                'nta5': r'Percent of $5^{th}$ sequences of non-terminal arrests',
                'sen': 'Percent of senescent lineages'}
LEGEND_NTA_BY_IDX = "Index of the\nsequence"

def type_of_sort_to_label_string(type_of_sort):
    """ Return the readable string that indicates the type of sort given in
    argument.

    Parameters
    ----------
    type_of_sort : string
        Indicates how data is sorted:
        - 'lmin' : by length of the initial shortest telomere length.
            WARNING: does not work for experimental data.
        - 'gntai' : by generation of the ith non-terminal arrest.
        - 'gsen' : by generation of senescence.
        - 'gdeath' : by generation of death.

    """
    if type_of_sort == 'lmin':
        return r"initial $\ell_{min}$"
    if  type_of_sort == 'lavg':
        return r"initial $\ell_{avg}$"
    if 'gnta' in type_of_sort:
        nta = int(type_of_sort[-1])
        if nta == 1:
            return "generation of\n" + r"$1^{st}$ nta"
        # r"$\mathrm{generation}$" + '\n' + \
        #         r"\smash{$\mathrm{of~1^{st}~nta}$}"
        elif nta == 2:
            return r"onset of $2^{nd}$ sequence of nta"
        elif nta == 3:
            return "$\mathrm{generation~of~3^{nd}~sequence~of~nta}$"
        return r"generation of " + rf"${nta}$" + r"th~arrest"
    if type_of_sort == 'gsen':
        return "generation of senescence"
    if type_of_sort == 'gdeath':
        return "generation of death"
    if type_of_sort == 'length': # !!!!!!!!!! ça sert quand ?
        print('see type_of_sort_to_string')
        return r"length"

def write_ylin_label(type_of_sort):
    string = type_of_sort_to_label_string(type_of_sort)
    if plt.rcParams['text.usetex']:
        ylabel = "Lineage\n" + rf"\textit{{(by increasing {string})}}"
    else:
        ylabel = "Lineage\n" + rf"(by increasing {string})"
    return ylabel

def write_sim_avg(simulation_count):
    if plt.rcParams['text.usetex']:
        string = rf'\textit{{(average on {simulation_count} simulations)}}'
    else:
        string = rf'(average on {simulation_count} simulations)'
    return '\n' + string

def write_simlabel_w_count(simulation_count):
    return r'Simulation' + write_sim_avg(simulation_count)


def write_typelabel_w_count(characteristics, lineage_count):
    """

    Parameters
    ----------
    characteristics : list
        List of strings among 'atype' btype' 'htype' 'arrested1' 'arrested2'...
        'senescent' 'dead' dead_accidentally' 'dead_naturally'.
        See `is_as_expected_lineage` docsting.
    lineage_count : int
        Number of experimental lineages with the characteristics given / of
        lineage simulated.

    Returns
    -------
    label : string
        String to plot in the legend.

    """
    characteristics_cp = copy.deepcopy(characteristics)
    characteristics_cp.sort() # Orderring in alphabetical order.

    label = ""
    for charac in characteristics_cp:
        label = label + ""
    return label + '\n' + r'$\mathit{(}$' + \
            f'${lineage_count}$'+ r'$\mathit{~lineages)}$'


# --------------
# Plot functions
# --------------

# Fitting - Error on gcurves
# --------------------------

# Definition of parameters.

# > Distance(s) for the error between exp an sim gcurves.
def distance_l2(arr_1, arr_2):
    """ Return the distance between arrays 'arr_1' and 'arr_2' in L2 norm.

    """
    return np.sqrt(np.sum((arr_1 - arr_2) ** 2))

# > Time parameters.
LINEAGE_COUNT_PER_TCOMPUT = 15
CHARAC_S = {'atype': ['atype', 'senescent'],
            'btype': ['btype']}
if __name__ == "__main__":
    tcomputs = np.empty((0, len(CHARAC_S)))
    for i in range(100):
        tcomputs_temp = np.array([])
        for characteristics in CHARAC_S:
            tstart = time.time()
            sim.simulate_lineages_evolution(LINEAGE_COUNT_PER_TCOMPUT,
                                            characteristics, False)
            tend = time.time()
            tcomputs_temp = np.append(tcomputs_temp, [tend - tstart], axis=0)
        tcomputs = np.append(tcomputs, [tcomputs_temp], axis=0)
    T_COMPUT_MAX = max(np.mean(tcomputs, 0) + 3 * np.std(tcomputs, 0))
    print(T_COMPUT_MAX)
T_COMPUT_MAX = .5
PARAMETERS_COMPUT = [LINEAGE_COUNT_PER_TCOMPUT, T_COMPUT_MAX]
PARAMETERS_COMPUT = None

# > Plot parameters.
LABEL_ERRORS = [r"$L^2$-error for ",
                r"Error between proportions of type B for ",
                r"Error between polynomal fits for"]


# Simulation and computation of errors.
# -------------------------------------

def compute_n_plot_gcurve_error(exp_data_raw, lineage_count_on_all_simu,
                                gcurves, characteristics, par_update=None,
                                is_plotted=False, error_types=[0, 1, 2],
                                distance=distance_l2, simulation_count=None,
                                proc_count=1, is_printed=False):
    """ Warning: the number of experimental lineages having the given
    characteristics must be identical to all the types of sort given.

    """
    # Experimental data.
    # > We keep only lineages having required characteristics.
    exp_data_raw = sim.select_exp_lineages(exp_data_raw, characteristics)
    # > And sort them according to all the gcurves we want to fit.
    exp_data_s = {}
    for gcurve in gcurves:
        exp_data_s[gcurve] = sim.sort_lineages(exp_data_raw, gcurve)

    # Simulated data.
    # > Computation of the parameters of simulation.
    is_htype_seen = False
        # NB: we fit on experimental date where H type, if existing, are unseen
        #     thus htype unseen for simulations as well.
    lineage_count = len(exp_data_s[gcurves[0]][0]['cycle'])
    lineages = np.arange(lineage_count)
    if isinstance(simulation_count, type(None)):
        simulation_count = int(lineage_count_on_all_simu / lineage_count)
    p_update = deepcopy(par_update) or {}
    p_update['is_htype_seen'] = is_htype_seen

    # > Simulation.
    sim_data_s = sim.simulate_n_average_lineages(lineage_count,
                    simulation_count, gcurves, characteristics,
                    par_update=p_update, parameters_comput=PARAMETERS_COMPUT,
                    proc_count=proc_count)
    # Computation of error for all types of sort.
    error_s = {}
    for gcurve in gcurves:
        # Default value.
        error_s[gcurve] = np.nan * np.ones(3)
        # If computation was too long for `PARAMETERS_COMPUT` parameters.
        if isinstance(sim_data_s[gcurve], type(None)):
            # No need to compute: parameters s.t. lineages with given charac.
            # are not enough probable.
            pass
        else:
            error_s[gcurve] = np.nan * np.ones(3)
            # Experimental data.
            exp_data = exp_data_s[gcurve]
            exp_gtrigs = exp_data[1]
            # Simulated data.
            sim_data = sim_data_s[gcurve]
            sim_gtrigs = sim_data[1]

            # Computation of error.
            gtrig = gcurve[1:]
            if 0 in error_types:
                if 'nta' in gtrig:
                    nta_idx = int(gtrig[-1]) - 1
                    gtrig = 'nta'
                    exp_gtrigs[gtrig] = exp_gtrigs[gtrig][:, nta_idx]
                    for key, data in sim_gtrigs[gtrig].items():
                        sim_gtrigs[gtrig][key] = data[:, nta_idx]
                error_s[gcurve][0] = distance(exp_gtrigs[gtrig],
                                              sim_gtrigs[gtrig]['mean'])
            if 1 in error_types:
                exp_bprop = np.mean(exp_data[2][~np.isnan(exp_data[2])] == 1)
                sim_bprop = sim_data[2][1]['btype']
                if is_printed:
                    print('bprop exp/sim: ', exp_bprop, sim_bprop['mean'])
                    print('percentile bprop sim: ', sim_bprop['per'])
                    print('extremum bprop sim: ', sim_bprop['ext'])
                error_s[gcurve][1] = abs(exp_bprop - sim_bprop['mean'])
            if 2 in error_types:
                exp_fit = np.polyfit(exp_gtrigs[gtrig], lineages, 1)
                sim_fit = np.polyfit(sim_gtrigs[gtrig]['mean'], lineages, 1)
                error_s[gcurve][2] = abs(exp_fit[0] - sim_fit[0]) + \
                    abs(exp_fit[1] - sim_fit[1])
            if is_plotted:
                w, h = plt.figaspect(.6) + .3
                plt.figure(figsize=(w,h))
                plt.xlabel(LABELS[gcurve], labelpad=6)
                plt.ylabel(LABELS['ax_lin'], labelpad=8)
                plt.plot(exp_gtrigs, lineages, '--', label=LABELS['exp'],
                         color='black')
                sim_label = write_simlabel_w_count(simulation_count)
                plt.plot(sim_gtrigs['mean'], lineages, label=sim_label,
                         color='darkorange')
                plt.fill_betweenx(lineages, sim_gtrigs['perdown'],
                                  sim_gtrigs['perup'], alpha=fp.ALPHA,
                                  label=LABELS['per'], color='darkorange')
                plt.legend(loc='lower right')
                # xfit = np.linspace(0, max(exp_gtrigs[gtrig]), 100)
                # plt.plot(xfit, np.poly1d(exp_fit)(xfit))
                # plt.plot(xfit, sim_fit[1] + xfit * sim_fit[0])
                ax = plt.gca()
                plt.text(.01, .97, parf.write_laws(p_update['parameters']),
                          transform=ax.transAxes, horizontalalignment='left',
                          verticalalignment='top')
                sns.despine()
                plt.show()
    return error_s


# Plot 2D matrices (lineage_count, gen_count)
# -------------------------------------------

# > Cycle duration times.

def plot_lineages_cycles(cycles, is_exp, fig_supdirectory, font_size,
                         curve_to_plot=None, lineage_types=None, is_dead=None,
                         evo_avg=None, gmax=None, add_to_name=None,
                         bbox_to_anchor=None, fig_size=None):
    """ Plot cycle duration times (in the order given) indicating the type of
    each lineages or if dead before the end of the experiment if given in
    argument. If not None also plot `curve_to_plot` (can be e.g. `gtrig_sen`).
    If average of cycles among several simulations specifiying the
    type of data orderring use `type_of_sort` will specifyin it in y-axis.
    If 'saved_as' is True, the plot is saved.

    """
    sns.set_style("darkgrid")
    plt.rcParams.update({'axes.facecolor': ".94",
                         'legend.frameon': True,
                         'legend.framealpha': 1,
                         'legend.facecolor': 'white',
                         'legend.edgecolor': '#EAEAF2',
                         'legend.fancybox': True
                        })

    lineage_count, generation_count = np.shape(cycles)
    if isinstance(gmax, type(None)):
       gmax = generation_count
    generations = np.arange(gmax + 3)
    lineages = np.arange(lineage_count)
    cycles = fct.reshape_with_nan(cycles, len(generations), -1)

    # Default value
    is_htype_seen = True

    # Plot.
    plt.figure(figsize=fig_size) # default: (6.4, 4.8)
    df = pd.DataFrame(data=cycles, columns=generations, index=lineages)
    if isinstance(evo_avg, type(None)):
        ylabel = LABELS['ax_lin']
        blabel = LABELS['cycle']
    else:
        ylabel = write_ylin_label(evo_avg['type_of_sort'])
        blabel = LABELS['cycle'] + write_sim_avg(evo_avg['simu_count'])
    if lineage_count < 15:
        yticklabels = 1
    elif lineage_count < 25:
        yticklabels = 2
    else:
        yticklabels = 25
    sns.heatmap(df, cmap=fp.CMAP_LINEAGE, vmin=60, vmax=250,
                xticklabels=10, yticklabels=yticklabels, zorder=0,
                cbar_kws={'ticklocation': 'left', 'extend': 'max',
                          'label': blabel})
    # Legend.
    gen_counts = np.sum(~np.isnan(cycles), 1)
    fs = plt.rcParams['legend.fontsize']
    if not isinstance(lineage_types, type(None)):
        is_htype_seen = np.any(np.isnan(lineage_types))
        btype_idxs = lineage_types == 1
        if is_exp:
            leg_btype = LABELS['btype_exp'].replace('type B', '\ntype B')
            leg_htype = LABELS['xtype']
        elif par.HYBRID_CHOICE:
            if is_htype_seen:
                leg_btype = LABELS['btype']
                leg_htype = LABELS['htype']
            else:
                leg_btype = LABELS['btype_exp'].replace('type B', '\ntype B')

                leg_htype = None
        else:
            leg_btype = LABELS['btype']
            leg_htype = None
        plt.scatter(gen_counts[btype_idxs], lineages[btype_idxs]+.5, s=2*fs,
                    marker=r'$--$', color='black', label=leg_btype)
        htype_idxs = np.isnan(lineage_types)
        plt.scatter(gen_counts[htype_idxs], lineages[htype_idxs]+.5, s=2*fs,
                    marker=r'$\times$', color='black', label=leg_htype)
        if isinstance(bbox_to_anchor, type(None)):
            plt.legend(loc='lower right', facecolor="w", edgecolor="w")
        else:
            plt.legend(bbox_to_anchor=bbox_to_anchor, loc="lower right")
    if not isinstance(is_dead, type(None)):
        is_alive = np.isnan(is_dead)
        plt.scatter(gen_counts[is_alive]+.85, lineages[is_alive]+.5, s=1.8*fs,
                    marker='>', color='black', label=LABELS['alive'])
        if isinstance(bbox_to_anchor, type(None)):
            plt.legend(loc='lower right', facecolor="w", edgecolor="w")
        else:
            plt.legend(bbox_to_anchor=bbox_to_anchor, loc="upper left")
    if not isinstance(curve_to_plot, type(None)):
        plt.plot(curve_to_plot, lineages, color='black', zorder=10000,
                 label='Generation of \n senescence onset')
        if isinstance(bbox_to_anchor, type(None)):
            plt.legend(loc='lower right', facecolor="w", edgecolor="w")
        else:
            plt.legend(bbox_to_anchor=bbox_to_anchor, loc="upper left")
    # Axis.
    plt.xlabel(LABELS['ax_gen'], labelpad=6, size=font_size)
    plt.ylabel(ylabel, labelpad=8, size=font_size)
    plt.ylim(0, lineage_count)
    sns.despine()
    if not isinstance(fig_supdirectory, type(None)):
        path = wp.write_cycles_path(lineage_count, is_exp,
                                    is_htype_seen=is_htype_seen,
                                    lineage_types=lineage_types,
                                    is_dead=is_dead, evo_avg=evo_avg,
                                    supdirectory=fig_supdirectory)
        if not isinstance(add_to_name, type(None)):
            path = path.replace('.pdf', f'_{add_to_name}.pdf')
        print("Saved at ", path, '\n')
        plt.savefig(path, bbox_inches='tight')
    plt.show()
    return

# > Cycle duration times.

def plot_lineage_avg_proportions(props, is_exp, fig_supdirectory, font_size,
                                 curve_to_plot=None, evo_avg=None):
    lineage_count, generation_count = np.shape(props)
    generations = np.arange(generation_count + 3)
    lineages = np.arange(lineage_count)
    cycles = np.append(props, np.nan * np.ones((lineage_count, 3)), 1)
    # Plot.
    # w, h = plt.figaspect(1.2)
    plt.figure(figsize=(5.8, 9.5)) # default: (6.4, 4.8)
    df = pd.DataFrame(data=cycles, columns=generations, index=lineages)
    if isinstance(evo_avg, type(None)):
        ylabel = LABELS['ax_lin']
        blabel = LABELS['propB']
    else:
        ylabel = write_ylin_label(evo_avg['type_of_sort'])
        blabel = LABELS['propB'] + write_sim_avg(evo_avg['simu_count'])
    sns.heatmap(df, vmin=0, vmax=1,
                xticklabels=10, yticklabels=20, zorder=0,
                cbar_kws={'ticklocation': 'left', 'label': blabel})
    # Legend.
    if not isinstance(curve_to_plot, type(None)):
        plt.plot(curve_to_plot, lineages, color='black', zorder=10000)
    # Axis.
    plt.xlabel(LABELS['ax_gen'], labelpad=6, size=font_size)
    plt.ylabel(ylabel, labelpad=8, size=font_size)
    plt.ylim(0, lineage_count)
    sns.despine()
    if not isinstance(fig_supdirectory, type(None)):
        path = wp.write_propB_path(lineage_count, evo_avg,
                                   supdirectory=fig_supdirectory)
        print('\n Saved at ', path)
        plt.savefig(path, bbox_inches='tight')
    plt.show()
    return


# Plot generation curves.
# -----------------------

def plot_gcurves_exp(exp_data, characteristics_s, fig_supdirectory,
                     labels=None, is_gathered=False, add_to_name=None,
                     fig_size=(6.4, 4.8), title=None):
    char_count = len(characteristics_s)
    labels = labels or [''] * char_count

    if not isinstance(fig_supdirectory, type(None)):
        dir_name = fig_supdirectory + '/lineage/gcurves_n_hists/exp/'
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

    gtrigs_s, lineage_counts = [], []
    for chars in characteristics_s:
        type_of_sort = sim.type_of_sort_from_characteristics(chars)
        # Experimental data.
        gtrigs = sim.select_exp_lineages(exp_data, chars)
        gtrigs = sim.sort_lineages(gtrigs, type_of_sort)[1]
        if 'nta' in type_of_sort:
            gtrigs_s.append(gtrigs['nta'][:, int(type_of_sort[-1]) - 1])
        else:
            gtrigs_s.append(gtrigs[type_of_sort[1:]])
        lineage_counts.append(len(gtrigs_s[-1]))

    colors = sns.color_palette('rocket', char_count)[::-1]
    if is_gathered:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
        for i in range(char_count):
            lineages = np.arange(lineage_counts[i])
            axes[0].plot(gtrigs_s[i], lineages, color=colors[i],
                         label=labels[i])
            axes[1].plot(gtrigs_s[i], lineages / lineage_counts[i],
                         label=labels[i].replace('G^{-1}', 'F'),
                         color=colors[i])
        axes[0].set_ylabel(LABELS['ax_lin'], labelpad=8)
        axes[1].set_ylabel(LABELS['ax_lin_norm'], labelpad=8)
        axes[0].legend()
        axes[1].legend()
        fig.add_subplot(111, frameon=False)
        plt.title(title)
        plt.tick_params(labelcolor='none', which='both', top=False,
                        bottom=False, left=False, right=False)
        plt.grid(False)
        plt.xlabel(LABELS['ax_gen'], labelpad=6)
        sns.despine()
        if not isinstance(fig_supdirectory, type(None)):
            path = dir_name + 'gcurves_exp.pdf'
            print("Saved at ", path, '\n')
            plt.savefig(path, bbox_inches='tight')
        plt.show()
    else:
        plt.figure(figsize=fig_size)
        plt.title(title)
        for i in range(char_count):
            plt.plot(gtrigs_s[i], np.arange(lineage_counts[i]),
                     color=colors[i], label=labels[i])
        plt.ylabel(LABELS['ax_lin'], labelpad=8)
        plt.xlabel(LABELS['ax_gen_arr'], labelpad=6)
        plt.legend()
        sns.despine()
        if not isinstance(fig_supdirectory, type(None)):
            path = dir_name + 'gcurves_exp.pdf'
            print("Saved at ", path, '\n')
            plt.savefig(path, bbox_inches='tight')
        plt.show()

        plt.figure(figsize=fig_size)
        plt.title(title)
        for i in range(char_count):
            plt.plot(gtrigs_s[i],
                     np.arange(lineage_counts[i]) / lineage_counts[i],
                     label=labels[i], color=colors[i])
        plt.ylabel(LABELS_['ax_lin_norm'], labelpad=8)
        plt.xlabel(LABELS['ax_gen_arr'], labelpad=6)
        plt.legend()
        sns.despine()
        if not isinstance(fig_supdirectory, type(None)):
            path = dir_name + 'gcurves_normalized_exp.pdf'
            if not isinstance(add_to_name, type(None)):
                path = path.replace('.pdf', f'_{add_to_name}.pdf')
            print("Saved at ", path, '\n')
            plt.savefig(path, bbox_inches='tight')
        plt.show()
    return

def compute_n_plot_gcurve(exp_data, simu_count, characteristics,
                          fig_supdirectory, gcurve=None, type_of_sort=None,
                          par_update=None, is_exp_plotted=False,
                          bbox_to_anchor=None, title=None, is_propB=False,
                          proc_count=1):
    p_update = {'is_htype_seen': False,
                'parameters': par.PAR}
    p_update.update(par_update or {})

    gtrig = sim.type_of_sort_from_characteristics(characteristics)
    gcurve = gcurve or gtrig
    type_of_sort = type_of_sort or gtrig

    # Compute.
    out = sim.compute_gtrigs(exp_data, simu_count, characteristics, gcurve,
                             type_of_sort, par_update=p_update,
                             is_propB=is_propB, proc_count=proc_count)
    lineages, gtrigs_exp, gtrigs_sim = out[:3]

    # Plot.
    w, h = plt.figaspect(.6) + .3
    plt.figure(figsize=(w,h))
    plt.xlabel(LABELS[gcurve], labelpad=6)
    if type_of_sort == gcurve:
        ylabel = LABELS['ax_lin']
    else:
        ylabel = write_ylin_label(type_of_sort)
    plt.ylabel(ylabel, labelpad=8)
    if is_exp_plotted:
        plt.plot(gtrigs_exp, lineages, '--', label=LABELS['exp'],
                 color='black')
    sim_label = write_simlabel_w_count(simu_count)
    plt.plot(gtrigs_sim['mean'], lineages, label=sim_label, color='darkorange')
    plt.fill_betweenx(lineages, gtrigs_sim['perdown'], gtrigs_sim['perup'],
                      alpha=fp.ALPHA, label=LABELS['per'], color='darkorange')
    plt.legend(loc='lower right', bbox_to_anchor=bbox_to_anchor)
    if is_propB:
        print(p_update)
        plt.title(title + rf"\quad $r_{{B\% , exp}}={out[3][0]:3.2f}, \quad"
                   rf" r_{{B \%,sim}}={out[3][1]:3.2f}, \quad \ell_{{min_A}}"
                   rf" = {int(p_update['parameters'][1][0][2])}$")
    else:
        plt.title(title)
    sns.despine()
    if not isinstance(fig_supdirectory, type(None)):
        # p_update['is_htype_seen'] = True
        path = wp.write_gcurve_path(simu_count, len(lineages), [type_of_sort],
                                    characteristics, par_update=p_update,
                                    supdirectory=fig_supdirectory)
        print("\n Saved at ", path, '\n')
        plt.savefig(path, bbox_inches='tight')
    plt.show()
    return

def compute_n_plot_gcurves_wrt_charac(exp_data, simu_count, characteristics_s,
                                      fig_supdirectory, labels=None,
                                      par_update=None, path=None, proc_count=1,
                                      bbox_to_anchor=None, add_to_name=None,
                                      fig_size=(5.5, 11), xticks=None):
    gcurve_s = [sim.type_of_sort_from_characteristics(characs) for characs in
                characteristics_s]
    fig_count = len(characteristics_s)
    fig, ax = plt.subplots(fig_count, 1, sharex='col', figsize=fig_size)
    for i in range(fig_count):
        if not isinstance(xticks, type(None)):
            ax[i].set_xticks(xticks)
        characteristics = characteristics_s[i]
        gcurve = gcurve_s[i]

        lineages, gtrigs_exp, gtrigs_sim = sim.compute_gtrigs(exp_data,
            simu_count, characteristics, gcurve, gcurve, par_update=par_update,
            proc_count=proc_count)

        ax[i].plot(gtrigs_exp, lineages, label=LABELS['exp'], color='black')
        sim_label = LABELS['sim'] # write_simlabel_w_count(simu_count)
        ax[i].plot(gtrigs_sim['mean'], lineages, label=sim_label,
                   color='dimgrey')
        ax[i].fill_betweenx(lineages, gtrigs_sim['perdown'],
                            gtrigs_sim['perup'], alpha=fp.ALPHA,
                            label=LABELS['per'], color='gray')
        if not isinstance(labels, type(None)):
            ax[i].text(.96,.85, labels[i], horizontalalignment='right',
                       verticalalignment='top', transform=ax[i].transAxes)
        ax[i].set_ylabel(None)
    if isinstance(bbox_to_anchor, type(None)):
        ax[0].legend(borderaxespad=.2)
    else:
        ax[0].legend(bbox_to_anchor=bbox_to_anchor, loc="upper left")
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False,
                    left=False, right=False)
    plt.grid(False)
    plt.xlabel(LABELS['ax_gen'], labelpad=9)
    plt.ylabel(LABELS['ax_lin'], labelpad=16)
    sns.despine()
    if not isinstance(fig_supdirectory, type(None)):
        path = path or wp.write_gcurves_path(par_update=par_update,
                                             supdirectory=fig_supdirectory)
        if not isinstance(add_to_name, type(None)):
            path = path.replace('.pdf', f'_{add_to_name}.pdf')
        plt.savefig(path, bbox_inches='tight')
        print("\n Saved at ", path, '\n')
    plt.show()
    return

def compute_n_plot_gcurves_wrt_sort_n_gen(exp_data, simulation_count,
                                          characteristics, types_of_sort,
                                          gcurves, fig_supdirectory,
                                          bbox_to_anchor,
                                          labels=None, par_update=None,
                                          is_exp_plotted=False):
    is_unique_gcurve = len(dict.fromkeys(gcurves)) == 1
    p_update = {'is_htype_seen': False}
    p_update.update(par_update or {})

    plt.figure(figsize=(6.2, 3.4))#(9, 4.8)) # default: (6.4, 4.8)
    colors = sns.color_palette('rocket', len(gcurves))[::-1]
    for i in range(len(gcurves)):
        gcurve, type_of_sort = gcurves[i], types_of_sort[i]
        if is_unique_gcurve:
            legend = type_of_sort_to_label_string(type_of_sort)
        else:
            legend = type_of_sort_to_label_string(gcurve)
            print(legend)
            legend = legend.replace('onset~of~', '')
            legend = legend.replace('generation~of~', '')
        legend = legend.replace('senescence', '\n senescence')
        legend = legend.replace('sequence of nta', '\n sequence of nta')

        lineages, gtrigs_exp, gtrigs_sim = sim.compute_gtrigs(exp_data,
            simulation_count, characteristics, gcurve, type_of_sort,
            par_update=p_update)
        LINESTYLE = '-'
        if is_exp_plotted:
            plt.plot(gtrigs_exp, lineages, color=colors[i])
            plt.fill_betweenx(lineages, gtrigs_sim['perdown'],
                              gtrigs_sim['perup'], alpha=0.11,
                              color=colors[i])
            LINESTYLE = '--'
        plt.plot(gtrigs_sim['mean'], lineages, LINESTYLE, label=legend,
                 color=colors[i])
    if is_unique_gcurve:
        xlabel = LABELS[gcurves[0]]
        title = "Lineages ordered by"
    else:
        xlabel = LABELS['ax_gen']
        title = None
        # title = r"$\mathrm{Onset~of}$" #r"$\mathrm{}$"
    plt.legend(title=title, fontsize='small',
               bbox_to_anchor=bbox_to_anchor)
    # borderaxespad=.2, fontsize=13,  title_fontsize=13)
    plt.xlabel(xlabel, labelpad=6)
    plt.ylabel(LABELS['ax_lin'], labelpad=8)
    sns.despine()
    if not isinstance(fig_supdirectory, type(None)):
        if is_unique_gcurve:
            path = wp.write_gcurve_path(simulation_count, len(lineages),
                      types_of_sort, characteristics,
                      par_update=p_update, supdirectory=fig_supdirectory)
            path = path.replace('/gcurves_s', '/gcurves_wrt_order_s')
            path = path.replace('_p95', '')
        else:
            path = wp.write_gcurve_path(simulation_count, len(lineages),
                      [types_of_sort[0]], characteristics,
                      par_update=p_update, supdirectory=fig_supdirectory)
            path = path.replace('/gcurves_s', '/gcurves_wrt_arrest_s')
        if is_exp_plotted:
            path = path.replace('/gcurves_w', '/gcurves_w_exp_w')
        print('\n Saved at', path + '\n')
        plt.savefig(path, bbox_inches='tight')
    plt.show()
    return

def plot_gcurves_wrt_par(exp_data, simu_count, characteristics,
                         varying_par_updates, varying_key, fig_supdirectory,
                         curve_labels=None, is_exp_plotted=False,
                         linestyles=None, add_to_name=None,
                         bbox_to_anchor=None, fig_size=None):
    type_of_sort = sim.type_of_sort_from_characteristics(characteristics)
    gcurve = type_of_sort
    par_count = len(varying_par_updates)

    plt.figure(figsize=fig_size)
    colors = sns.color_palette('rocket', par_count)[::-1]
    if isinstance(linestyles, type(None)):
        linestyles = ['-' for i in range(par_count)]
    for i in range(par_count):
        lineages, gtrigs_exp, gtrigs_sim = sim.compute_gtrigs(exp_data,
                simu_count, characteristics, gcurve, type_of_sort,
                par_update=varying_par_updates[i])
        plt.plot(gtrigs_sim['mean'], lineages, color=colors[i],
                 label=curve_labels[i], linestyle=linestyles[i])
        plt.fill_betweenx(lineages, gtrigs_sim['perdown'],
                          gtrigs_sim['perup'], alpha=0.12, color=colors[i])
    if isinstance(is_exp_plotted, bool):
        if is_exp_plotted == True:
            plt.plot(gtrigs_exp, lineages, color='black', label=LABELS['exp'],
                     linestyle='-.')
    else:
        lineages_exp = np.arange(len(is_exp_plotted))
        lineages_exp = lineages_exp * lineages[-1] / lineages_exp[-1]
        plt.plot(is_exp_plotted, lineages_exp, color='black',
                 label=LABELS['exp'], linestyle='--')
    if not isinstance(curve_labels, type(None)):
        if isinstance(bbox_to_anchor, type(None)):
            plt.legend(title=fp.LABELS[varying_key], loc="lower right")
        else:
            plt.legend(title=fp.LABELS[varying_key],
                       bbox_to_anchor=bbox_to_anchor)
    plt.xlabel(LABELS_[gcurve], labelpad=6)
    plt.ylabel(LABELS['ax_lin'], labelpad=8)
    sns.despine()
    if not isinstance(fig_supdirectory, type(None)):
        par_update = varying_par_updates[linestyles.index('--')]
        par_update[varying_key] = None
        path = wp.write_gcurve_path(simu_count, len(lineages), [type_of_sort],
                                    characteristics, par_update=par_update,
                                    supdirectory=fig_supdirectory)
        path = path.replace('gcurves_b', f'gcurves_wrt_{varying_key}_b')
        if not isinstance(add_to_name, type(None)):
            path = path.replace('.pdf', f'_{add_to_name}.pdf')
        print('\n Saved at', path)
        plt.savefig(path, bbox_inches='tight')
    plt.show()
    return

def plot_gcurves_wrt_par_n_char(exp_data, simu_count, characteristics_s,
                                varying_par_updates, varying_key,
                                fig_supdirectory, curve_labels=None,
                                texts=None, linestyles=None, fig_size=(6.5, 12)):
    type_of_sort = sim.type_of_sort_from_characteristics(characteristics_s[0])
    gcurve = type_of_sort
    par_count = len(varying_par_updates)
    colors = sns.color_palette('rocket', par_count)[::-1]


    if isinstance(linestyles, type(None)):
        linestyles = ['-' for i in range(par_count)]
    fig_count = len(characteristics_s)
    fig, ax = plt.subplots(fig_count, 1, sharex='col', figsize=fig_size)
    for f_idx in range(fig_count):
        for i in range(par_count):
            lineages, gtrigs_exp, gtrigs_sim = sim.compute_gtrigs(exp_data,
                    simu_count, characteristics_s[f_idx], gcurve, type_of_sort,
                    par_update=varying_par_updates[i])
            ax[f_idx].plot(gtrigs_sim['mean'], lineages, color=colors[i],
                           label=curve_labels[i], linestyle=linestyles[i])
            ax[f_idx].fill_betweenx(lineages, gtrigs_sim['perdown'],
                              gtrigs_sim['perup'], alpha=0.12, color=colors[i])
            if not isinstance(texts, type(None)):
                ax[f_idx].text(.95, .1, texts[f_idx], ha='right', va='bottom',
                               transform=ax[f_idx].transAxes)
    if not isinstance(curve_labels, type(None)):
        ax[0].legend(title=fp.LABELS[varying_key], loc="lower right")
    fig.add_subplot(111, frameon=False) # Add big axes, hide frame.
    plt.tick_params(labelcolor='none', # Hide tick of the big axes.
                    top=False, bottom=False, left=False, right=False)
    plt.grid(False) # And hide grid.
    plt.xlabel(LABELS[gcurve], labelpad=9)
    plt.ylabel(LABELS['ax_lin'], labelpad=16)
    sns.despine()
    if not isinstance(fig_supdirectory, type(None)):
        characteristics = characteristics_s[np.argmin([len(c) for c in
                                                       characteristics_s])]
        par_update = varying_par_updates[linestyles.index('--')]
        par_update[varying_key] = None
        path = wp.write_gcurve_path(simu_count, 0, [type_of_sort],
                                    characteristics, par_update=par_update,
                                    supdirectory=fig_supdirectory)
        path = path.replace('_l0_', '_by_type_')
        path = path.replace('gcurves_b', f'gcurves_wrt_{varying_key}_b')
        print('\n Saved at', path)
        plt.savefig(path, bbox_inches='tight')
    plt.show()
    return

def plot_medians_wrt_par(exp_data, simu_count, characteristics, varying_pars,
                         varying_par_updates, varying_key, fig_supdirectory,
                         x_label='', curve_labels=None, y_exp=None,
                         linestyles=None, add_to_name=None):
    type_of_sort = sim.type_of_sort_from_characteristics(characteristics)
    gcurve = type_of_sort
    par_count = len(varying_par_updates)

    plt.figure()
    median_means = np.zeros(par_count)
    median_perdowns, median_perups = np.zeros(par_count), np.zeros(par_count)
    medians = np.zeros(par_count)
    if isinstance(linestyles, type(None)):
        linestyles = ['-' for i in range(par_count)]
    for i in range(par_count):
        lineages, gtrigs_exp, gtrigs_sim = sim.compute_gtrigs(exp_data,
                simu_count, characteristics, gcurve, type_of_sort,
                par_update=varying_par_updates[i])
        print(gtrigs_sim)
        median_means[i] = np.median(gtrigs_sim['mean'])
        median_perdowns[i] = np.median(gtrigs_sim['perdown'])
        median_perups[i] = np.median(gtrigs_sim['perup'])
        medians[i] = gtrigs_sim['mean'][int(len(gtrigs_sim['mean']) / 2)]
    plt.plot(varying_pars, median_means, '-+', color='black')  #, label='old')
    # plt.plot(varying_pars, medians, '-+', color='black', label='new')
    fct_interpol_inv_medians = interpolate.interp1d(median_means, varying_pars)
    # fct_interpol_inv_medians = interpolate.interp1d(medians, varying_pars)
    x_exp = fct_interpol_inv_medians(y_exp)
    axes = plt.gca()
    xmin, ymin = axes.get_xlim()[0], axes.get_ylim()[0]
    axes.plot([x_exp], [y_exp], 'go', color='darkorange',
              label=rf'$({x_exp:3.1f}, {y_exp:3.1f})$')
    plt.hlines(y=y_exp, xmin=xmin, xmax=x_exp, color='darkorange',
               ls='--', label=LABELS['exp'])
    plt.vlines(x=x_exp, ymin=ymin, ymax=y_exp, color='darkorange', ls='--')
    plt.xticks(varying_pars)
    plt.legend(fontsize='small')
    plt.xlabel(x_label + LABELS[varying_key], labelpad=6)
    plt.ylabel('Median generation of\nsenescent onset', labelpad=8)
    sns.despine()
    if not isinstance(fig_supdirectory, type(None)):
        par_update = varying_par_updates[0]
        par_update[varying_key] = None
        path = wp.write_gcurve_path(simu_count, len(lineages), [type_of_sort],
                                    characteristics, par_update=par_update,
                                    supdirectory=fig_supdirectory)
        path = path.replace('gcurves_b', f'medians_wrt_{varying_key}_b')
        if not isinstance(add_to_name, type(None)):
            path = path.replace('.pdf', f'_{add_to_name}.pdf')
        print('\n Saved at', path)
        plt.savefig(path, bbox_inches='tight')
    plt.show()
    return medians, median_means

#  Histograms
# ------------

def plot_histogram(x_axis, y_axis, width=1, normalized=True, ylim=None,
                   fig_supdirectory=None, title=None):
    plt.figure()
    if not isinstance(ylim, type(None)):
        plt.ylim(ylim)
    plt.xlabel(LABELS['gsen'], labelpad=6)
    if normalized:
        plt.ylabel(LABELS['ax_per'], labelpad=8)
    else:
        plt.ylabel(LABELS['ax_count'], labelpad=8)
    plt.bar(x_axis, y_axis, width=width, color='darkorange')
    plt.legend()
    plt.title(title)
    sns.despine()
    # if not isinstance(fig_supdirectory, type(None)):
    #     print('\n Saved at ', path)
    #     plt.savefig(path, bbox_inches='tight')
    plt.show()

def compute_n_plot_hist_gen_coupure(gsen_exp, fig_supdirectory,
                                    x_axis=np.arange(7), width=1,
                                    normalized=True, ylim=None, title=None):
    y_axis = fct.make_hist_from_data_wo_nan(gsen_exp, x_axis=x_axis,
                                            normalized=normalized)
    plot_histogram(x_axis, y_axis, width=width, normalized=normalized,
                   fig_supdirectory=fig_supdirectory, ylim=ylim, title=title)
    return y_axis

def compute_n_plot_hist_lmin(exp_data, simulation_count, characteristics_s,
                             hist_lmins_axis, fig_supdirectory,
                             parameters=par.PAR, width=1, is_htype_seen=False,
                             lineage_count_on_all_simu=None,
                             is_old_data=False):
    """ Warning: the number of experimental lineages having the given
    characteristics must be identical to all the types of sort given.

    """
    p_update = {'parameters': parameters,
                'is_htype_seen': is_htype_seen}

    char_count = len(characteristics_s)
    colors = []
    legends = []
    legends_short = []
    i = 0
    for char in characteristics_s:
        if 'btype' in f'{char}':
            colors.append(fp.COLORS_TYPE['btype'])
            legends.append(LABELS['btype'])
            legends_short.append(LABELS['btype_short'])
            idx_b = i
        elif 'atype' in f'{char}':
            colors.append(fp.COLORS_TYPE['atype'])
            legends.append(LABELS['atype'])
            legends_short.append(LABELS['atype_short'])
            idx_a = i
        elif char == ['senescent']:
            colors.append(fp.COLORS_TYPE['all'])
            legends.append(LABELS['sen'])
            legends_short.append(LABELS['sen_short'])
            idx_sen = i
        else:
            colors.append(fp.COLORS_TYPE['all'])
        i +=1

    # Simulated data.
    lineage_counts, data_s = sim.compute_lmin_histogram_data(exp_data,
                    simulation_count, characteristics_s, par_update=p_update,
                    hist_lmins_axis=hist_lmins_axis,
                    lineage_count_on_all_simu=lineage_count_on_all_simu)
    hist_s = []
    for i in range(char_count):
        axis, hist = data_s[i]
        axis_new, hist_new = fct.rescale_histogram_bin(axis,hist['mean'],width)
        if i == 0:
            x_count = len(axis_new)
            hist_all = [0 * axis_new]
        hist_s.append(hist_new)
        x_count = min(x_count, len(axis_new))
        hist_all.append(hist_all[-1][:x_count] + hist_new[:x_count])
        # width = np.min(np.diff(axis))
        plt.figure()
        plt.xlabel(LABELS['ax_lsen'], labelpad=6)
        plt.ylabel(LABELS['ax_per'], labelpad=8)
        if is_old_data:
            # NB: should be removed with new simu corrected.
            # Here corrects an error in lineage simulation that was saving g+1
            # instead of g as gen of arrest and thus was translating the
            # histogram in lineage simulation.
            axis = axis_new - par.OVERHANG
        else:
            axis = axis_new
        plt.bar(axis, hist_new, width=width, color=colors[i], label=legends[i])
        print('prop > 27 bp: ', np.sum(hist_new[axis > 27 ]),
              np.sum(hist_new), '\n')
        plt.legend()
        sns.despine()
        if not isinstance(fig_supdirectory, type(None)):
            par_update = {'parameters': parameters,
                          'is_htype_seen': is_htype_seen}
            path = wp.write_hist_lmin_sim_path(simulation_count,
                                                lineage_counts[i],
                                                width, characteristics_s[i],
                                                par_update=par_update,
                                                supdirectory=fig_supdirectory)
            print('\n Saved at ', path)
            plt.savefig(path, bbox_inches='tight')
        plt.show()

    if char_count > 1:
        plt.figure()
        plt.xlabel(LABELS['ax_lsen'], labelpad=6)
        plt.ylabel(LABELS['ax_per'], labelpad=8)
        for i in range(char_count):
            plt.bar(axis[:x_count], hist_all[i+1][:x_count],
                    bottom=hist_all[i][:x_count],
                    width=width, color=colors[i], label=legends_short[i])
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(handles[::-1], labels[::-1], title='Cell type')
        sns.despine()
        if not isinstance(fig_supdirectory, type(None)):
            par_update = {'parameters': parameters,
                          'is_htype_seen': is_htype_seen}
            path = wp.write_hist_lmin_sim_path(simulation_count,
                                               None, width, [''],
                                               par_update=par_update,
                                               supdirectory=fig_supdirectory)
            print('\n Saved at ', path)
            plt.savefig(path, bbox_inches='tight')
        plt.show()
        if (['btype','senescent'] in characteristics_s
            and ['atype','senescent'] in characteristics_s
            and ['senescent'] in characteristics_s):
            # Previous stack plotting of B type  on top of B type is wrong
            # since no ponderation is applied. Here we try to evaluate the
            # right weight (that is the proportion of type B/A among sen)
            # as the pondetation w that  minimize the error between hist of
            # senescent and w * hist_senA + (1- w) * hist_senB
            weights = np.linspace(0, 1, 1001)
            x_max = max([len(hist) for hist in hist_s])
            hist_s = [np.append(hist, np.zeros(x_max - len(hist))) for hist in hist_s]
            hist_s = [hist / sum(hist) for hist in hist_s]
            errors = [np.sum(np.abs(hist_s[idx_sen]- w * hist_s[idx_a]
                             - (1-w) * hist_s[idx_b])) for w in weights]
            plt.plot(weights, errors)
            plt.show()
            hist_s = [hist[:x_count] for hist in hist_s]
            weight = weights[np.argmin(errors)]
            print('Estimated prop of Btype: ', 1 - weight, errors, np.argmin(errors))
            plt.bar(axis[:x_count], weight * hist_s[idx_a])
            plt.show()
            plt.figure()
            plt.xlabel(LABELS['ax_lsen'], labelpad=6)
            plt.ylabel(LABELS['ax_per'], labelpad=8)
            plt.bar(axis[:x_count], (1-weight) * hist_s[idx_b],
                    bottom=0 * hist_s[0], width=width,
                    color=fp.COLORS_TYPE['btype'], label=LABELS['btype'])
            plt.bar(axis[:x_count], weight * hist_s[idx_a]
                                    + (1-weight) * hist_s[idx_b],
                    bottom=(1-weight) * hist_s[idx_b], width=width,
                    color=fp.COLORS_TYPE['atype'], label=LABELS['atype'])
            handles, labels = plt.gca().get_legend_handles_labels()
            plt.legend(handles[::-1], labels[::-1], title='Cell type')
            sns.despine()
            if not isinstance(fig_supdirectory, type(None)):
                par_update = {'parameters': parameters,
                              'is_htype_seen': is_htype_seen}
                path = wp.write_hist_lmin_sim_path(simulation_count,
                                                   None, width, [''],
                                                   par_update=par_update,
                                                   supdirectory=fig_supdirectory)
                print('\n Saved at ', path)
                plt.savefig(path, bbox_inches='tight')
            plt.show()
    return


def plot_histogram_from_lcycle_counts(lcycle_per_seq_counts, lcycle_type,
                                      seq_count_max=None, count_max=None,
                                      path_to_save=None, x_max=-1,
                                      fig_size=None):
    """ Plot the histogram from the data given in `lcycle_per_seq_counts`
    arranged according to the parameters specified as argument and save at
    `path_to_save` if `path_to_save` is not None.

    Parameters
    ----------
    lcycle_per_seq_counts : dict
        Output of `sim.compute_exp_lcycle_counts`: dictionnary of the number of
        long cycles per sequence of long cycles gathered by entries st.:
        nta : ndarray
            2D array with same shape as `gtrigs['nta']` s.t.
            `lcycle_per_seq_counts['nta'][i, j]` is the number of successive
            long cycles of the jth nta of the ith lineage.
        sen : ndarray
            1D array (lineage_count, ) (ie with shape of `gtrigs['sen']`) s.t.
            `lcycle_per_seq_counts['sen'][i]` is the number of successive
            senescent (long)  cycles of the ith lineage.
        NB: Nan value whenever there is if no such sequence.
    lcycle_type : string
        String indicating which data is represented and how, i.e. the x-axis:
        - nta_total: number of non-terminal arrests per lineage.
        - nta_by_idx: number of long cycle per sequence of nta distinguishing
            between the 1st, 2nd, ... `seq_count_max`th sequence of nta.
        - nta: number of long cycle per sequence of nta.
        - ntai: number of long cycle of the ith sequence of nta.
        - sen: number of senescent cycles.
        NB: if simulated data, what is returned is the average histogram among
        all simulations, with error bars (except for 'nta_by_idx').
    seq_count_max : int, optional
        Used only for `lcycle_type` equal to 'nta_by_idx', histogram composed
        of stacks for 1st, 2nd, ... `seq_count_max`th seq of nta.
        The default is None, if so the value is the maximum number of nta seq.
    path_to_save : string, optional
        If none the histogram is not saved, otherwise indicates the path at
        which should be saved. The default is None.
    font_scale : list
        font_scale[0]: default font scale ()

    """
    colors = sns.color_palette("YlOrRd", 4)[-2:]
    is_sim_data = len(np.shape(lcycle_per_seq_counts['sen'])) == 2
    if isinstance(fig_size, type(None)):
        plt.figure()
    else:
        plt.figure(figsize=fig_size)

    label_x = LABEL_HIST_X['nta']
    hist = None
    if is_sim_data:
        simu_count, lineage_count, nta_count = np.shape(
                                                lcycle_per_seq_counts['nta'])
        if lcycle_type == 'nta_total':
            label_x = LABEL_HIST_X['nta_total']
            hist = fct.make_average_histogram(
                       np.nansum(lcycle_per_seq_counts['nta'], axis=-1))
        elif lcycle_type == 'nta_by_idx':
            lcycle_per_nta_count_max = np.nanmax(lcycle_per_seq_counts['nta'])
            axis = np.arange(1, int(lcycle_per_nta_count_max) + 1)

            hist = fct.make_stacked_average_histogram(
                        lcycle_per_seq_counts['nta'], axis)
            bottom = 0 * hist[0]

            seq_count = np.shape(lcycle_per_seq_counts['nta'])[-1]
            if seq_count_max is None:
                seq_count_max = seq_count
            colors = sns.color_palette("YlOrRd", seq_count_max)[::-1]
            # sns.cubehelix_palette(seq_count_max, start=0, rot=-.3,
                                           # dark=.7, light=0.4, as_cmap=False)
            for i in range(seq_count_max):
                if i < seq_count_max - 1:
                    plt.bar(hist[0][:x_max], hist[1][i][:x_max], width=1,
                            bottom=bottom[:x_max], alpha=.7, label=rf"{i+1}",
                            color=colors[i])
                    bottom = hist[1][i] + bottom
                else:
                    plt.bar(hist[0][:x_max],
                            np.nansum(hist[1][i:], 0)[:x_max],
                            width=1, bottom=bottom[:x_max], alpha=.7,
                            label=rf"$\geq {i+1}$", color=colors[i])
                plt.legend(title=LEGEND_NTA_BY_IDX)
        elif lcycle_type == 'nta':
            hist = fct.make_average_histogram(
                       np.reshape(lcycle_per_seq_counts['nta'],
                                  (simu_count, lineage_count * nta_count)))
        elif lcycle_type[:-1] == 'nta':
            nta_idx = int(lcycle_type[-1]) - 1
            hist = fct.make_average_histogram(
                       lcycle_per_seq_counts['nta'][:,nta_idx])
        elif lcycle_type == 'sen':
            hist = fct.make_average_histogram(lcycle_per_seq_counts['sen'])
        else:
            raise ValueError("ERROR: wrong `lcycle_type` argument for "
                              "`plot_histogram_lcycle_count` function")
        if lcycle_type != 'nta_by_idx':
            lineage_count = hist[3]
            plt.bar(hist[0][:x_max], hist[1][:x_max], width=1, alpha=.7,
                    color=colors[0])
            plt.errorbar(hist[0][:x_max], hist[1][:x_max],
                         yerr=hist[2][:x_max], linestyle='', capsize=5,
                         color=colors[1])
    else:
        lineage_count = len(lcycle_per_seq_counts['sen'])
        if lcycle_type == 'nta_total':
            lcycle_counts = np.nansum(lcycle_per_seq_counts['nta'], axis=-1)
        elif lcycle_type == 'nta_by_idx':
            if seq_count_max is None:
                seq_count_max = np.shape(lcycle_per_seq_counts['nta'])[-1]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                new = np.append(lcycle_per_seq_counts['nta'][:,
                                                             :seq_count_max-1],
                                np.nanmean(lcycle_per_seq_counts['nta'][:,
                                        seq_count_max-1:], axis=1)[:, None], 1)
            labels = [str(i+1) for i in range(seq_count_max)]
            labels[-1] = rf"$\geq {labels[-1]}$"

            df = pd.DataFrame(data=new, columns=labels)
            palette = sns.color_palette("YlOrRd", seq_count_max)
            sns.histplot(df, stat="percent", discrete=True, multiple='stack',
                          hue_order=labels[::-1], palette=palette,
                          legend=False)
            plt.legend(title=LEGEND_NTA_BY_IDX, labels=labels) # loc='upper left')
            is_nan = np.isnan(lcycle_per_seq_counts['nta'][:, 0])
            lcycle_counts = lcycle_per_seq_counts['nta'][:, 0][~is_nan]
        elif lcycle_type == 'nta':
            lcycle_counts = lcycle_per_seq_counts['nta'].flatten()
            lcycle_counts = lcycle_counts[~np.isnan(lcycle_counts)]
        elif lcycle_type[:-1] == 'nta':
            nta_idx = int(lcycle_type[-1]) - 1
            is_nan = np.isnan(lcycle_per_seq_counts['nta'][:, nta_idx])
            lcycle_counts = lcycle_per_seq_counts['nta'][:, nta_idx][~is_nan]
        elif lcycle_type == 'sen':
            is_nan = np.isnan(lcycle_per_seq_counts['sen'])
            lcycle_counts = lcycle_per_seq_counts['sen'][~is_nan]
        else:
            raise ValueError("ERROR: wrong `lcycle_type` argument for "
                              "`plot_histogram_lcycle_count` function")
        if lcycle_type != 'nta_by_idx':
            lineage_count = min(lineage_count, len(lcycle_counts))
            sns.histplot(lcycle_counts, stat="percent", discrete=True,
                         color=colors[0])
    # > Axis and legend.
    plt.xlabel(label_x, labelpad=6)
    plt.ylabel(LABEL_HIST_Y[lcycle_type], labelpad=8)
    sns.despine()
    if not isinstance(path_to_save, type(None)):
        print("Saved at ", path_to_save, '\n')
        plt.savefig(path_to_save, bbox_inches='tight')
    plt.show()
    return hist, colors


def compute_n_plot_lcycle_hist(exp_data, simulation_count, characteristics,
                               lcycle_types, fig_supdirectory,
                               par_update=None, seq_count=None,
                               is_exp_support=False, fig_size=None):
    is_saved = not isinstance(fig_supdirectory, type(None))
    p_update = deepcopy(par_update) or {}

    # Definition of some parameters
    (lineage_count, lcycle_counts_exp, lcycle_counts_sim,
     lcycle_counts_sim_h) = sim.compute_lcycle_histogram_data(exp_data,
                                   simulation_count, characteristics,
                                   par_update=p_update)
    if is_saved:
        # Concatenation of some arguments.
        kwarg = (lineage_count, characteristics, lcycle_types)
        p_update['is_htype_seen'] = False
        path_exp = wp.write_hist_lc_exp_path(lineage_count, characteristics,
                                             supdirectory=fig_supdirectory)
        path_sim = wp.write_hist_lc_sim_path(simulation_count, *kwarg,
                                             par_update=p_update,
                                             supdirectory=fig_supdirectory)
        if par.HYBRID_CHOICE:
            p_update['is_htype_seen'] = True
            path_sim_h = wp.write_hist_lc_sim_path(simulation_count, *kwarg,
                       par_update=p_update, supdirectory=fig_supdirectory)
    else:
        path_exp = None
        path_sim = None
        path_sim_h = None

    #Plotting.
    path_sim_temp = None
    path_sim_h_temp = None
    for lcycle_type in lcycle_types:
        if is_exp_support:
            if 'nta' in lcycle_type:
                x_max = int(np.nanmax(lcycle_counts_exp['nta']))
            else:
                x_max = int(max(lcycle_counts_exp['sen']))
            if is_saved:
                new = f'_xmax{x_max}.pdf'
                path_sim_temp = path_sim.replace('.pdf', new)
                path_sim_h_temp = path_sim_h.replace('.pdf', new)
        else:
            x_max = -1
            path_sim_temp = path_sim
            path_sim_h_temp = path_sim_h
        plot_histogram_from_lcycle_counts(lcycle_counts_exp, lcycle_type,
                                          seq_count_max=seq_count,
                                          path_to_save=path_exp,
                                          fig_size=fig_size)
        plot_histogram_from_lcycle_counts(lcycle_counts_sim, lcycle_type,
                                          seq_count_max=seq_count,
                                          path_to_save=path_sim_temp,
                                          x_max=x_max, fig_size=fig_size)
        if par.HYBRID_CHOICE:
            plot_histogram_from_lcycle_counts(lcycle_counts_sim_h, lcycle_type,
                                              seq_count_max=seq_count,
                                              path_to_save=path_sim_h_temp,
                                              x_max=x_max, fig_size=fig_size)
    return


# Statistics on postreated data
# -----------------------------

def compute_n_plot_postreat_time_vs_gen(exp_data, simu_count, characteristics,
                                        postreat_dt, is_htype_seen):
    par_update = {'is_htype_seen': is_htype_seen}
    out = sim.compute_postreat_data(exp_data, simu_count, characteristics,
                                    postreat_dt, par_update=par_update)
    LEG_TITLE = "Cell type"
    COLORS_R = sns.color_palette("rocket", 3)

    for key in ['gen', 'time']:
        x_ax = out[key][0]
        if key  == 'time':
            x_ax = x_ax / (60 * 24)

        # Telomere lengths.
        plt.figure()
        data_keys = ['lavg', 'lmin', 'lmin_min']
        for i in range(3):
            data_key = data_keys[i]
            y_axs = out[key][1][data_key]
            if data_key == 'lmin_min':
                plt.plot(x_ax, y_axs, label=LABELS[data_key],
                         color=COLORS_R[i])
            else:
                plt.plot(x_ax, y_axs['mean'], label=LABELS[data_key],
                         color=COLORS_R[i])
                plt.fill_between(x_ax, y_axs['perdown'], y_axs['perup'],
                                 alpha=fp.ALPHA, color=COLORS_R[i])
        plt.xlabel(LABELS['ax_' + key], labelpad=6)
        plt.ylabel(LABELS['ax_l'], labelpad=8)
        sns.despine()
        plt.legend()
        plt.show()

        # Proportion of cells.
        # > Senescent cells.
        plt.figure()
        y_axs = out[key][1]['prop_sen']
        for type_idx in range(2):
            plt.plot(x_ax, y_axs['mean'])
            plt.fill_between(x_ax, y_axs['perdown'], y_axs['perup'],
                             alpha=fp.ALPHA)
        plt.xlabel(LABELS['ax_' + key], labelpad=6)
        plt.ylabel(LABELS['ax_p_sen'], labelpad=8)
        sns.despine()
        plt.show()

        keys = ['atype', 'btype', 'htype']
        colors = [fp.COLORS_TYPE[key] for key in keys]
        labels = [LABELS[key + '_short'] for key in keys]
        print(labels)
        # > Of each type among all cells
        plt.figure()
        data_key = 'prop_type'
        y_axs = out[key][1][data_key]
        for i in range(3):
            plt.plot(x_ax, y_axs['mean'][i], label=labels[i], color=colors[i])
            plt.fill_between(x_ax, y_axs['perdown'][i], y_axs['perup'][i],
                             alpha=fp.ALPHA, color=colors[i])
        plt.xlabel(LABELS['ax_' + key], labelpad=6)
        plt.ylabel(LABELS['ax_prop'], labelpad=8)
        plt.legend(bbox_to_anchor=(1, 1), title=LEG_TITLE)
        sns.despine()
        plt.show()

        # > Of each type among senescent cells
        plt.figure()
        data_key = 'prop_type_sen'
        y_axs = out[key][1][data_key]
        for i in range(3):
            plt.plot(x_ax, y_axs['mean'][i], label=labels[i], color=colors[i])
        plt.xlabel(LABELS['ax_' + key], labelpad=6)
        plt.ylabel(LABELS['ax_p_sen'], labelpad=8)
        plt.legend(bbox_to_anchor=(1, 1), title=LEG_TITLE)
        sns.despine()
        plt.show()

        # Cell cycle duration times.
        plt.figure()
        y_axs = out[key][1]['cycle']
        for i in range(3):
            plt.plot(x_ax, y_axs['mean'])
            plt.fill_between(x_ax, y_axs['perdown'], y_axs['perup'],
                             alpha=fp.ALPHA)
        plt.xlabel(LABELS['ax_' + key], labelpad=6)
        plt.ylabel(LABELS['cycle'], labelpad=8)
        sns.despine()
        plt.show()
    return
