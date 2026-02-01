#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 17:00:13 2022

@author: arat

Script containing functions allowing to plot (and compute in some cases)
population data.

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

import telomeres.dataset.extract_processed_dataset as xtd
import telomeres.auxiliary.figures_properties as fp
import telomeres.auxiliary.functions as fct
import telomeres.auxiliary.keys as ks
import telomeres.auxiliary.write_paths as wp
import telomeres.model.parameters as par
import telomeres.population.posttreat as pps

from copy import deepcopy

import matplotlib
import matplotlib.pyplot as plt
import mpl_axes_aligner  # pip install mpl-axes-aligner

from os.path import join
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from scipy import interpolate

import numpy as np
import seaborn as sns


# Parameters of the plots.
# ------------------------

# Global labels.
P_ANC_STRINGS = "Proportion of descendants"
LABELS = {
    "ax_c_init": "Initial number of cells (log-scale)",
    "ax_c": "Number of cells",
    "ax_c_log10": r"Log$_{10}$-number of cells",
    "ax_c_norm": "Normalized number of cells",
    "ax_c_sen": "Number of senescent cells",
    "ax_c_B": "Number of type B cells",
    "ax_gen": "Generation",
    "ax_gen_avg": "Average generation",
    "ax_p_B": "Proportion of type B cells    ",
    "ax_prop_B_bf_dil": "Proportion of type B cells before dilution",
    "ax_prop_sen_bf_dil": "Proportion of senescent cells before dilution",
    "ax_p_ancs": P_ANC_STRINGS,
    "ax_p_B_ancs": P_ANC_STRINGS + " among type B cells",
    "ax_p_sen_ancs": P_ANC_STRINGS + " among senescent cells",
    "ax_t_comput": "Computation time (hour)",
    "ax_t_sat": "Saturation time since last dilution (day)",
    "ax_textinct": "Time of extinction (day)",
    "ax_tsen": "Time at which the population became senescent (day)",
    "ax_mem_comput": "Allocated memory (Mo)",
    "sen": "senescent",
    "leg_prop_anc": "Ancestor index",
    "leg_prop_anc_long": "Ancestor index (by increasing shortest telomere)",
    "leg_day": "Day",
}
LABELS.update(fp.LABELS)


def write_avg_label(simu_count):
    """Write a global label for the legend of average simulated curves
    depending on `simu_count`, the number of simulations averaged.

    """
    return rf"$\mathrm{{Average~on~{simu_count}}}$" + " \n" + r"$\mathrm{simulations}$"


IS_STAT_STD = {"per": False, "std": True, "ext": False}
IS_STAT_PER = {"per": True, "std": False, "ext": False}
IS_STAT_NONE = {"per": False, "std": False, "ext": False}
IS_STAT_DEFAULT = {"per": True, "std": False, "ext": True}

KWARGS_PLOT = {
    "y_format": None,
    "y_scale": None,
    "alpha": fp.ALPHA,
    "fig_path": None,
    "linestyles": None,
    "colors": None,
    "legend_title": None,
    "legend_loc": "best",
    "general_labels": LABELS,
    "xticks": None,
    "yticks": None,
    "bbox_to_anchor": None,
    "figsize": None,
    "idxs_no_stat": [],
    "curve_labels": None,
    "leg_fontsize": "medium",
    "legend_frameon": plt.rcParams["legend.frameon"],
}


def write_ylabel_anc(anc_prop):
    """Write a global label for the y axis depending on `anc_prop`, the
    proportion of ancestors whose offsprings' evolution is plotted.

    """
    label = (
        P_ANC_STRINGS
        + f" of the {int(anc_prop * 100)}"
        + r"$\%$ of initial cells with the longest $\ell_{min} $ "
    )
    return label  # "\n".join(wrap(label[::-1], LABEL_MAX + TO_ADD))[::-1]


def define_xticks_from_counts(count, xticks=None):
    xticks_ = xticks or count
    xleft = xticks_[0] - (xticks_[0] - 1) / 2
    xright = xticks_[-1] + (xticks_[-1] - xticks_[-2]) / 2
    return xticks_, xleft, xright


def plot_evo_curves_w_stats(x, y_s, axis_labels, is_stat, kwargs=None):
    """Plot (with possible customization) and save evolution curves with
    statistics.

    Parameters
    ----------
    x : ndarray
        1D array of the x-axis values
    y_s : list
        List (length `curve_count`) of dictionaries (each dictionary
        corresponds to a set of associated y-axis statistical data to plot)
        with key 'mean' (and possibly 'min', 'max', 'perdown', 'perup',
        'std') each associated with a 1D arrays of fixed length.
    axis_labels : list
        List of strings: the x-axis and y-axis labels, respectively.
    is_stat : list or dict
        List of dictionaries of booleans (with keys 'ext', 'per', 'std' as
        e.g. `IS_STAT_STD`) indicating whether to show statistics for
        each curve.
        > If list (length `curve_count`): indicating for each 'mean' curve
            whether to show other statistics or not.
        > If dict same indication when assumed common to all curves.
    curve_labels : list or None, optional
       List (length `curve_count`) of strings: the labels associated to each
       `y_s[i]['mean']` curve. Default is None (no label).
    y_format : str, optional
        The format for the y-axis tick labels (e.g. 'sci' for scientific
        notation, cf plt doc for other). Default is None for no special format.
    y_scale : str, optional
        The scale for the y-axis (e.g. 'log' for logarithmic scale, cf
        matplotlib documentation for others). Default is None.
    alpha : float, optional
        The transparency level for the shaded areas representing statistics.
        Default is fp.ALPHA.
    fig_name : str, optional
        The filename to save the plot. Default is None (plot not saved).
    linestyles : list, optional
        A list (length `curve_count`) of linestyles for each `y_s[i]['mean']`
        curve. Default is None (solid lines).
    colors : list, optional
        Same with line color. Default is None (default colors from matplotlib).
    legend_title : str, optional
        The title for the legend. Default is None (no legend title).
    legend_loc : str, optional
        The location of the legend. Default is 'best' (see matplotlib).
    general_labels : dict, optional
        Dictionary (with keys 'avg', 'min', 'max', 'perdown'..) of strings or
        None: the labels of the legend of statistic curves. Set to {key: None}
        to disable the legend associated to the curve key. Default is LABELS.
    xticks : ndarray, optional
        Values to use for x-axis tick marks. Default is None (uses default).
    yticks : ndarray, optional
        Values to use for y-axis tick marks. Default is None (uses default).
    bbox_to_anchor : tuple, optional
        The bounding box coordinates for the legend. Default is None (uses
        default legend placement).
    figsize : dict or tuple, optional
        Size of the figure in inches. Default is None (uses default).
    idxs_no_stat : list, optional
        List of indices indicating which curves should not display statistics.
        Default is an empty list.

    """
    # Key optional arguments.
    kw = deepcopy(KWARGS_PLOT)
    if isinstance(kwargs, dict):
        kw.update(deepcopy(kwargs))

    # Plot.
    plt.clf()
    if kw["figsize"] is None:
        # plt.figure()
        fig, ax = plt.subplots(1, 1)  # plt.figure()
    else:
        fig, ax = plt.subplots(1, 1, figsize=kw["figsize"])
        # plt.figure(figsize=kw['figsize'])
    plt.xlabel(axis_labels[0], labelpad=6, wrap=True)
    plt.ylabel(axis_labels[1], labelpad=8, wrap=True)
    plt.tight_layout()
    if not isinstance(kw["xticks"], dict):
        plt.xticks(kw["xticks"])
    if not isinstance(kw["yticks"], dict):
        plt.yticks(kw["yticks"])

    # Define plot options.
    if isinstance(is_stat, dict):
        is_stat = [is_stat] * len(y_s)
    if kw["linestyles"] is None:
        kw["linestyles"] = ["-"] * len(y_s)
    if kw["colors"] is None:
        # kw['colors'] = plt.rcParams['axes.prop_cycle'].by_key()['color']
        # while len(kw['colors']) < len(y_s):
        #     kw['colors'].extend(plt.rcParams['axes.prop_cycle'].by_key(
        #                         )['color'])
        kw["colors"] = sns.color_palette("rocket", n_colors=len(y_s))
    if kw["y_format"] is not None:
        plt.ticklabel_format(
            style=kw["y_format"], axis="y", scilimits=(0, 0), useMathText=True
        )
    if kw["y_scale"] is not None:
        plt.yscale(kw["y_scale"])
    imax = len(x)

    # > First set of data with legend for statistics.
    if kw["curve_labels"] is None:
        kw["curve_labels"] = [None] * len(y_s)
    if kw["curve_labels"][0] is None:
        legend_0 = kw["general_labels"]["avg"]
    else:
        legend_0 = kw["curve_labels"][0]
    plt.plot(
        x,
        y_s[0]["mean"][:imax],
        label=legend_0,
        linestyle=kw["linestyles"][0],
        color=kw["colors"][0],
    )
    if is_stat[0]["ext"] and (0 not in kw["idxs_no_stat"]):
        plt.fill_between(
            x,
            y_s[0]["min"][:imax],
            y_s[0]["max"][:imax],
            alpha=kw["alpha"],
            label=kw["general_labels"]["ext"],
            color="gray",
        )
    if is_stat[0]["per"] and (0 not in kw["idxs_no_stat"]):
        plt.fill_between(
            x,
            y_s[0]["perdown"][:imax],
            y_s[0]["perup"][:imax],
            alpha=kw["alpha"],
            label=kw["general_labels"]["per"],
            color=kw["colors"][0],
        )
    if is_stat[0]["std"] and (0 not in kw["idxs_no_stat"]):
        plt.fill_between(
            x,
            y_s[0]["mean"][:imax] - y_s[0]["std"][:imax],
            y_s[0]["mean"][:imax] + y_s[0]["std"][:imax],
            alpha=kw["alpha"],
            label=kw["general_labels"]["std"],
            color=kw["colors"][0],
        )
    # Plot remaining data, without legend for statistics.
    for idx in range(1, len(y_s)):
        plt.plot(
            x,
            y_s[idx]["mean"][:imax],
            label=kw["curve_labels"][idx],
            linestyle=kw["linestyles"][idx],
            color=kw["colors"][idx],
        )
        if is_stat[idx]["ext"] and (idx not in kw["idxs_no_stat"]):
            plt.fill_between(
                x,
                y_s[idx]["min"][:imax],
                y_s[idx]["max"][:imax],
                alpha=kw["alpha"],
                color="gray",
            )
        if is_stat[idx]["per"] and (idx not in kw["idxs_no_stat"]):
            plt.fill_between(
                x,
                y_s[idx]["perdown"][:imax],
                y_s[idx]["perup"][:imax],
                alpha=kw["alpha"],
                color=kw["colors"][idx],
            )
        if is_stat[idx]["std"] and (idx not in kw["idxs_no_stat"]):
            plt.fill_between(
                x,
                y_s[idx]["mean"][:imax] - y_s[idx]["std"][:imax],
                y_s[idx]["mean"][:imax] + y_s[idx]["std"][:imax],
                alpha=kw["alpha"],
                color=kw["colors"][idx],
            )
    is_legend = [g is not None for g in kw["general_labels"].values()]
    if kw["curve_labels"] != [None] * len(y_s) or any(is_legend):
        if kw["bbox_to_anchor"] is None:
            plt.legend(
                title=kw["legend_title"],
                loc=kw["legend_loc"],
                frameon=kw["legend_frameon"],
                fontsize=kw["leg_fontsize"],
            )
        else:
            plt.legend(
                title=kw["legend_title"],
                bbox_to_anchor=kw["bbox_to_anchor"],
                loc="upper left",
                frameon=kw["legend_frameon"],
                fontsize=kw["leg_fontsize"],
            )
    sns.despine()  # Remove and top and right axis.
    if kw["fig_path"] is not None:  # Save.
        print("\n Saved at: ", kw["fig_path"])
        plt.savefig(kw["fig_path"], bbox_inches="tight")
    plt.show()


def reposition_yoffset(axes, x_pos=0.01, y_pos=0.98, pad=None):  # -0.11, y_pos=0.97
    """Reposition (at coordinates `x_pos, y_pos`) the offset of the y-axis
    appearing after asking axes.ticklabel_format(style='sci')`.
    Inspired from @edsmith's solution.

    """
    yaxis = axes.yaxis
    if pad is not None:
        plt.tight_layout(pad=pad)
    # Get the offset value.
    offset = yaxis.get_offset_text().get_text()

    if len(offset) > 0:
        # Turn off the offset text that's calculated automatically.
        yaxis.offsetText.set_visible(False)
        # Add in a text box at the top of the y axis.
        axes.text(x_pos, y_pos, offset, transform=axes.transAxes)


# --------------------------------
# Plot statistics on several simus
# --------------------------------

# > At cell_count and para_count fixed.
# -------------------------------------


def add_subplot_axes(ax, rect, axisbg="w"):
    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]  # <= Typo was here
    subax = fig.add_axes([x, y, width, height], axisbg=axisbg)
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2] ** 0.5
    y_labelsize *= rect[3] ** 0.5
    subax.xaxis.set_tick_params(labelsize=x_labelsize)
    subax.yaxis.set_tick_params(labelsize=y_labelsize)
    return subax


def plot_hist_lmin_at_sen(
    cell_count,
    para_count,
    simu_count,
    fig_subdirectory,
    day_count,
    x_axis=par.X_AXIS_HIST_LMIN,
    width=1,
    bbox_to_anchor=None,
    fig_size=None,
    par_update=None,
):
    # Path strings.
    sim_path = wp.write_simu_pop_subdirectory(
        cell_count, para_count, par_update=par_update
    )
    stat_path = wp.write_sim_pop_postreat_average(sim_path, simu_count)
    is_saved = fig_subdirectory is not None
    if is_saved:
        fig_directory = wp.write_fig_pop_directory(
            cell=cell_count,
            para=para_count,
            subdirectory=fig_subdirectory,
            par_update=par_update,
        )
        fig_end_name = f"_w{width}" + wp.write_fig_pop_name_end(
            simu=simu_count, tmax=day_count
        )
    # Data extraction.
    days = np.arange(day_count)
    hist = np.load(stat_path, allow_pickle=True).item().get("hist_lmin_all")
    hist_day = np.load(stat_path, allow_pickle=True).item().get("hist_lmin_per_day")
    y_s, yday_s, sup_idx = {}, {}, {}
    xmax = 120
    for key in ks.type_keys:  # Iteration on types.
        x_ax, y_ax = fct.rescale_histogram_bin(x_axis, hist[key]["mean"], width)
        y_s[key] = y_ax[x_ax < xmax]
        sup_idx[key] = len(y_ax) - np.argmax(y_ax[::-1] > 0)
        # tmp_idx  = 0
        yday_s[key] = []
        for d in days:
            x_tmp, y_tmp = fct.rescale_histogram_bin(
                x_axis, hist_day[key][d]["mean"], width
            )
            # tmp_idx = max(tmp_idx, len(y_tmp) - np.argmax(y_tmp[::-1] > 0))
            yday_s[key].append(y_tmp)
        # sup_idx[key] = tmp_idx
    # WARNING: here htype corresponds to the "True" htype (as defined in the
    # manuscript) WITHOUT the "True" mtypes, i.e. all the htypes having at
    # least 2 seq of nta .Therefore:
    # 'h+mtype': "True" htype with htype classification.
    y_s["h+mtype"] = y_s["htype"] + y_s["mtype"]
    yday_s["h+mtype"] = [yday_s["htype"][d] + yday_s["mtype"][d] for d in days]
    sup_idx["h+mtype"] = min(sup_idx["htype"], sup_idx["mtype"])
    # 'b+htype': "True" btype with mtype classification.
    y_s["b+htype"] = y_s["btype"] + y_s["htype"]
    yday_s["b+htype"] = [yday_s["btype"][d] + yday_s["htype"][d] for d in days]
    sup_idx["b+htype"] = min(sup_idx["btype"], sup_idx["htype"])

    # Visualization options.
    LEGENDS = [rf"$d = {int(day + 1)}$" for day in days]
    # > Limit of rescaled histograms.
    ymax_rescale = 7000
    xmax_rescale = 100

    # Plot bars in stack manner
    x_ax = x_ax[x_ax < xmax]
    keys_ = []  # [key] for key in ks.type_keys]
    keys_.extend(
        [
            ["mtype", "atype"],  # In order of plotting.
            ["b+htype", "mtype"],
            ["btype", "h+mtype", "atype"],  # All categories w htype.
            ["b+htype", "mtype", "atype"],  # ... w mtype classification.
            ["btype", "htype", "mtype", "atype"],
        ]
    )
    for keys in keys_:  # Iteration on all the data to plot in a common fig.
        # 1. Sum on all days.
        bottom = 0 * x_ax
        fig, ax1 = plt.subplots()
        for key in keys:
            plt.bar(
                x_ax,
                y_s[key],
                bottom=bottom,
                width=x_ax[1],
                color=fp.COLORS_TYPE[key],
                label=LABELS[key + "_short"],
            )
            bottom = bottom + y_s[key]
        ax1.ticklabel_format(style="sci", axis="y", scilimits=(0, 0), useMathText=True)
        reposition_yoffset(ax1)
        plt.xlabel(LABELS["ax_lsen"], labelpad=6, wrap=True)
        plt.ylabel(LABELS["ax_count"], labelpad=8)
        plt.tight_layout()
        # Plot Rescaled ("zoomed") histogram in some cases.
        if keys in [
            ["btype", "htype", "mtype", "atype"],
            ["btype", "h+mtype", "atype"],
            ["b+htype", "mtype", "atype"],
        ]:
            # Manually set the position and relative size of the inset axes ax2
            # created from ax1 according to [(x, y, pW, pH)]; where (x,y) coor-
            # dinate of ax2 left bottom corner, p proportion of the parent axes ax1.
            ax2 = ax1.inset_axes([0.44, 0.4, 0.56, 0.56], transform=ax1.transAxes)
            sns.despine(ax=ax2, top=True, right=True)
            # Draw the connecting lines between the inset axes and the parent axes.
            mark_inset(ax1, ax2, loc1=2, loc2=4, alpha=0.3, fc="none", ec="0.5")
            bottom = 0 * x_ax[x_ax < xmax_rescale]
            for key in keys:
                ax2.bar(
                    x_ax[x_ax < xmax_rescale],
                    y_s[key][x_ax < xmax_rescale],
                    bottom=bottom,
                    width=x_ax[1],
                    color=fp.COLORS_TYPE[key],
                    label=LABELS[key + "_short"],
                )
                bottom = bottom + y_s[key][x_ax < xmax_rescale]
            ax2.set_ylim(ymax=ymax_rescale)
            ax2.ticklabel_format(
                style="sci", axis="y", scilimits=(0, 0), useMathText=True
            )
            reposition_yoffset(ax2)  # ax2.set_facecolor('ghostwhite')
            # Add lengend in revert order and outside the figure frame.
            handles, labels = ax1.get_legend_handles_labels()
            plt.legend(
                handles[::-1],
                labels[::-1],
                title="Cell type",
                bbox_to_anchor=(0.7, 1.3),
                loc="upper left",
                fancybox=True,
                facecolor="white",
                fontsize="small",
            )
            to_add = f"_xlim{xmax_rescale}_ylim{ymax_rescale}"
        else:
            to_add = ""
            handles, labels = plt.gca().get_legend_handles_labels()
            plt.legend(
                handles[::-1],
                labels[::-1],
                title="Cell type",
                fancybox=True,
                facecolor="white",
            )
        sns.despine()
        if is_saved:
            type_strs = [key[0] for key in keys]
            type_strs.sort()
            fig_name = "hist_" + wp.list_to_string(type_strs) + "-type"
            path = fig_directory + fig_name + to_add + fig_end_name
            print("\n Saved at: ", path)
            fig.savefig(path, bbox_inches="tight")
        plt.show()

        # 2. Day to day evolution.
        fig, axes = plt.subplots(
            day_count, 1, sharex=True, sharey=False, figsize=fig_size
        )  # default: [6.4, 4.8]
        idx = np.max([sup_idx[key] for key in keys])
        x_ax_new = x_tmp[:idx]
        for i in days:  # Iteration on the subplots (one for each day).
            bottom = 0 * x_ax_new
            for key in keys:
                y_ax = yday_s[key][i][:idx]
                axes[i].bar(
                    x_ax_new,
                    y_ax,
                    bottom=bottom,
                    width=x_ax_new[1],
                    color=fp.COLORS_TYPE[key],
                    label=LABELS[key + "_short"],
                )
                bottom = bottom + y_ax
            axes[i].text(0.85, 0.76, LEGENDS[i], transform=axes[i].transAxes)
            axes[i].set_ylim(bottom=0, top=None)
            # Format 'sci' is for big data for no ytick and ylabel overlap.
            axes[i].ticklabel_format(
                style="sci", axis="y", scilimits=(-4, 4), useMathText=True
            )
            reposition_yoffset(axes[i], x_pos=-0.16, y_pos=0.91, pad=0.5)
        if len(keys) > 1:
            handles, labels = axes[0].get_legend_handles_labels()
            axes[0].legend(
                handles[::-1],
                labels[::-1],
                title="Cell type",
                bbox_to_anchor=bbox_to_anchor,
                fancybox=True,
                facecolor="white",
            )
        sns.despine()
        # Plotting of axis titles.
        fig.add_subplot(111, frameon=False)  # Add big axes, hide frame.
        plt.tick_params(
            labelcolor="none",  # Hide tick of the big axes.
            top=False,
            bottom=False,
            left=False,
            right=False,
        )
        plt.grid(False)  # Hide grid and set common titles.
        plt.xlabel(LABELS["ax_lsen"], labelpad=6)
        plt.ylabel(LABELS["ax_count"], labelpad=22)
        if is_saved:
            path = join(fig_directory, fig_name + fig_end_name)
            path = path.replace("hist_", "hist_per_day_")
            print("\n Saved at: ", path)
            plt.savefig(path, bbox_inches="tight")
        plt.show()


def plot_evo_c_n_p_pcfixed_from_stat(
    c,
    p,
    simu_count,
    fig_subdirectory,
    t_max,
    is_stat_update=None,
    par_update=None,
    strain="TetO2-TLC1",
    par_update_bis=None,
    is_only_exp=False,
    is_all_points=False,
    ysim_scale=None,
    yexp_scale=None,
):
    """Plot time-evolution of cell-number-related data from simulations run
    at `p` and `c` fixed.

    Parameters
    ----------
    c : int
        Initial (and after dilution) number of cells per simulation.
    p : int
        Number of parallelization per simulation.
    simu_count : int
        Number of simulation.
    fig_subdirectory : str or None
        Figures plotted are saved in the folder `figures/fig_subdirectory`
        unless `fig_subdirectory` is None, in which case figures are not saved.
    t_max : float
        Maximum time (in day) up to which evolution curves are plotted.
    strain : str
        'TetO2-TLC1' for the classical experimental data, or 'RAD51' or 'POL32'.
    is_only_exp : bool
        True to plot only curves of comparison with experimental data.
    ysim_scale : str or None
        None for usual y-axis scale (number of cell in simulation) for
        simulations.
        'log' for a log(10)-scale of the ticks axis (does not transform the
        data, but only the spacing between ticks, for a more readable).
        'log10' for a transformation of the data of the form `np.log10(data)`.
    yexp_scale : str or None
         Same as ysim_scale for experimental data.

    """
    # Import experimental evolution of cell concentration.
    # > Telomerase negative / doxycycline positive.
    # NB: `CSIM`, `CEXP` ADJUSTED MANUALLY!
    LABELS_SIM, FTM_SIM = [LABELS["sim"]], ["x-"]
    if strain == "POL32":  # Pol32- and Pol32+.
        CSTAT = xtd.extract_population_concentration_pol32()
        LABELS_SIM[0] = LABELS_SIM[0] + r" (tlc1$\Delta$ pol32$\Delta$)"
        LABELS_EXP = {
            "tlc1": LABELS["exp"] + r" (tlc1$\Delta$)",
            "tlc1_pol32": LABELS["exp"] + r" (tlc1$\Delta$ pol32$\Delta$)",
        }
        FTM = {"tlc1": "--", "tlc1_pol32": "-"}
        CSIM = {"c": 5}  # {'PD': 5, 'c': 5,  'OD': 1.2}
        CEXP = {"c": 1.5}  # {'PD': 1.5, 'c': 1.5,  'OD': 1.12}
        if fig_subdirectory == "article":  # format is 'article'
            LEGEND_POS = (0.95, 1.25)
        else:  # format is 'manuscript' or None
            LEGEND_POS = (0.83, 1.2)
    elif strain == "RAD51":
        if ysim_scale == "log":
            CSIM = {"OD": 15}
            CEXP = {"OD": 1.1}
        else:
            CSIM = {"OD": 1.35}
            CEXP = {"OD": 1.1}
        CSTAT = xtd.extract_population_concentration_rad51()
        CSTAT = {key: CSTAT[key] for key in ["RAD51"]}  # Comment for all data.
        LABELS_EXP = {
            "RAD51": LABELS["exp"] + r" (rad51$\Delta$)",
            "RAD51_sep": r"rad51_sept",
            "RAD51_oct": r"rad51_oct",
        }
        FTM = {"RAD51": "-", "RAD51_sep": "-.", "RAD51_oct": "--"}
        if fig_subdirectory == "article":
            LEGEND_POS = (0.9, 1.15)
        else:
            LEGEND_POS = (0.81, 1.12)
        if par_update_bis is not None:
            LABELS_SIM = [
                LABELS["sim"] + r" (rad51$\Delta$)",
                LABELS["sim"] + r" (rad51)",
            ]
            FTM_SIM.append("x--")
    else:
        CSTAT = {"DOX+": xtd.extract_population_concentration_doxP()}  # Telo-.
        LABELS_EXP = {"DOX+": LABELS["exp"]}
        FTM = {"DOX+": "-"}
        if fig_subdirectory == "article":
            LEGEND_POS = (0.88, 1.05)
        else:
            LEGEND_POS = (0.88, 0.88)
        CSIM = {"OD": 1.2}
        CEXP = {"OD": 1.1}

    # General `kwargs` (see plot_evo_curves_w_stats) options.
    if par_update is None or "sat" not in par_update.keys():
        p_sat = par.PAR_SAT["prop"]
    elif "sat" in par_update.keys():
        p_sat = par_update["sat"]["prop"]
    # > Style dependent parameters (e.g. legend position).
    if fig_subdirectory is None or fig_subdirectory == "manuscript":
        LEG_POS = (1, 1)  # LEG_POS_L =  (0.6, 1)
    elif fig_subdirectory == "article":
        LEG_POS = (0.97, 1)  # LEG_POS_L = (0.6, 1.05)
    else:
        raise Exception("Parameters of plotting to adjust manually should bespecified")
    # > Statistical curves to plot (default updated with `is_stat_update`).
    is_stat = deepcopy(IS_STAT_DEFAULT)
    if isinstance(is_stat_update, dict):
        is_stat.update(is_stat_update)
    # > Figure name (None if figures should not be saved).
    is_saved = fig_subdirectory is not None
    if is_saved:
        directory = wp.write_fig_pop_directory(
            c, p, par_update=par_update, subdirectory=fig_subdirectory
        )
        end_name = wp.write_fig_pop_name_end(
            simu=simu_count, tmax=t_max, is_stat=is_stat
        )
        end_name_none = wp.write_fig_pop_name_end(simu=simu_count, tmax=t_max)
        end_name_std = wp.write_fig_pop_name_end(
            simu=simu_count, tmax=t_max, is_stat=IS_STAT_STD
        )

        def fpath(name, stat_type=None, ysim_scale=None, yexp_scale=None):
            if stat_type == "std":
                path = join(directory, name + end_name_std)
            if stat_type == "none":
                path = join(directory, name + end_name_none)
            else:
                path = join(directory, name + end_name)
            if ysim_scale is not None:
                path = path.replace(".pdf", f"_ysim{ysim_scale}.pdf")
            if yexp_scale is not None:
                path = path.replace(".pdf", f"_yexp{yexp_scale}.pdf")
            return path

    else:

        def fpath(*_):
            return None

    # General data.
    # > Paths to data.
    sim_paths = [wp.write_simu_pop_subdirectory(c, p, par_update)]
    if par_update_bis is not None:
        sim_paths.append(wp.write_simu_pop_subdirectory(c, p, par_update_bis))
    stat_data_paths = [
        wp.write_sim_pop_postreat_average(path, simu_count) for path in sim_paths
    ]
    # > Times array (only up to `t_max`).
    times = np.load(stat_data_paths[0], allow_pickle=True).item().get("times")
    t_max = min(t_max, times[-1])
    times = times[times <= t_max]
    # > Days arrays.
    key0 = list(CSTAT.keys())[0]
    # NB: following to improve for shorter and clearer code.
    if strain == "POL32":
        # Pol32 data start 3 days after telomerase inactivation.
        days = np.arange(5, len(CSTAT["tlc1"]["c"]["avg"]) + 5)
        i0_exp = 0  # Index 0 of exp data corresponds to days[0].
    elif strain == "RAD51":
        i0_exp = 1
        days = np.arange(1, len(CSTAT[key0]["c"]["avg"]))
    else:
        i0_exp = 0
        days = np.arange(1, len(CSTAT[key0]["c"]["avg"]) + 1)
    idxs_bf_dil = np.array(
        [np.where(times == day)[0][0] - 1 for day in days[days <= times[-1]]]
    ).astype("int")
    if len(days) > len(idxs_bf_dil):
        days_sim = days[: len(idxs_bf_dil)]
    else:
        days_sim = days
    # > Extraction simulated data in the dictionary `d`.
    d = np.load(stat_data_paths[0], allow_pickle=True).item()
    # > And only evolution of concentration but on all simu_count simulations.
    if is_all_points:
        #   Index 0 of `np.genfromtxt(wp.write_sim_c_csv...` is day 1 of simu.
        i0 = days[0] - 1
        evo_c = [
            np.genfromtxt(wp.write_sim_c_csv(path, simu_count), delimiter=",")[
                :, i0 : i0 + len(days_sim)
            ]
            for path in sim_paths
        ]
        # > Compute quantities to adjust graphs.
        ysat_sim = c * p_sat
        if ysim_scale == "log10":
            evo_c = [np.log10(evo_cc) for evo_cc in evo_c]  # / c0) c0 = c * 5
            ysat_sim = np.log10(ysat_sim)
        evo_c_mean = [np.mean(ev, axis=0) for ev in evo_c]
        evo_c_std = [np.std(ev, axis=0) for ev in evo_c]
    else:
        evo_c_mean = [d["evo_c"]["mean"][idxs_bf_dil]]
        evo_c_std = [d["evo_c"]["std"][idxs_bf_dil]]

    # ..... Plot .....
    # Day-to-day evolution of the concentration.
    # ------------------------------------------
    # > Discrete comparison to experimental curves: First version.
    # NB: MANUALLY ADJUSTED (from ymax_sim).
    if "PD" in CSIM.keys():
        fig, ax1 = plt.subplots()
        ax1.set_xlabel(LABELS["ax_time"], labelpad=6)
        ax1.set_xticks(days)
        # >> Right axis: simulated data.
        y_color_sim = [fp.COLORS_SIM_VS_EXP[0], "black"]
        for idx in range(len(evo_c_mean)):
            ax1.errorbar(
                days_sim,
                evo_c_mean[idx],
                yerr=evo_c_std[idx],
                capsize=2,
                fmt=FTM_SIM[-idx],
                label=LABELS_SIM[-idx],
                color=fp.COLORS_SIM_VS_EXP[-idx],
            )
            if ysim_scale == "log10":
                y_label = LABELS["ax_c_log10"]
            else:
                y_label = LABELS["ax_c"]
            if is_all_points:
                for i in range(simu_count):
                    ax1.scatter(
                        days_sim, evo_c[idx][i], s=30, color=fp.COLORS_SIM_VS_EXP[-idx]
                    )
        ax1.set_ylabel(y_label, color=y_color_sim[idx], labelpad=8)
        ax1.tick_params(axis="y", labelcolor=y_color_sim[idx])
        ax1.set_ylim(ymax=CSIM["PD"] * ysat_sim)
        if ysim_scale == "log":
            ax1.set_yscale("log")
        else:
            ax1.ticklabel_format(
                style="sci", axis="y", scilimits=(0, 0), useMathText=True
            )
        # >> Left axis: experimental data.
        idx_color = 1
        ysat_exp, ymax_exp = [], 0
        for key, cstat in CSTAT.items():
            y_exp = cstat["PD"]["avg"][i0_exp:]
            y_err = cstat["PD"]["std"][i0_exp:]
            ysat_exp.append(np.mean(y_exp[:3]))
            if key == key0:  # Set parameters for axis 2.
                ax2 = ax1.twinx()
                ymax_exp = max(ymax_exp, np.max(y_exp + y_err))  # ADJUST!
                y_color_exp = fp.COLORS_SIM_VS_EXP[1]
            else:
                y_color_exp = "black"
            ax2.errorbar(
                days,
                y_exp,
                yerr=y_err,
                capsize=2,
                fmt=FTM[key],
                color=fp.COLORS_SIM_VS_EXP[idx_color],
                label=LABELS_EXP[key],
            )
            if is_all_points:
                for i in range(len(cstat["PD"]["all"][0])):
                    ax2.scatter(
                        days,
                        cstat["PD"]["all"][i0_exp:, i],
                        s=30,
                        color=fp.COLORS_SIM_VS_EXP[idx_color],
                    )
            idx_color += 1
        ax2.set_ylim(ymax=CEXP["PD"] * ymax_exp)
        ax2.grid(False)
        if yexp_scale == "log":
            ax2.set_yscale("log")
        else:
            ax2.ticklabel_format(
                style="sci", axis="y", scilimits=(0, 0), useMathText=True
            )
        ax2.set_ylabel(LABELS["ax_cexp"], color=y_color_exp, labelpad=9)
        ax2.tick_params(axis="y", labelcolor=y_color_exp)
        mpl_axes_aligner.align.yaxes(ax1, ysat_sim, ax2, np.mean(ysat_exp), 0.9)
        fig.legend(bbox_to_anchor=LEGEND_POS, fontsize="small")
        sns.despine(top=True, right=False)
        plt.show()
        if is_saved:
            path = fpath("evo_c_by_day_w_exp1", "std", ysim_scale, yexp_scale)
            if par_update_bis is not None:
                path = path.replace("_w_exp1_", "_w_exp1_x2_")
            print("\n Saved at: ", path)
            plt.savefig(path, bbox_inches="tight")
        plt.show()

    # > Discrete comparison to experimental curves: Second version.
    if "c" in CSIM.keys():
        fig, ax1 = plt.subplots()
        ax1.set_xlabel(LABELS["ax_time"], labelpad=6)
        ax1.set_xticks(days)
        # >> Right axis: simulated data.
        y_color_sim = [fp.COLORS_SIM_VS_EXP[0], "black"]
        for idx in range(len(evo_c_mean)):
            ax1.errorbar(
                days_sim,
                evo_c_mean[idx],
                yerr=evo_c_std[idx],
                capsize=2,
                fmt=FTM_SIM[-idx],
                label=LABELS_SIM[-idx],
                color=fp.COLORS_SIM_VS_EXP[-idx],
            )
            if ysim_scale == "log10":
                y_label = LABELS["ax_c_log10"]
            else:
                y_label = LABELS["ax_c"]
            if is_all_points:
                for i in range(simu_count):
                    ax1.scatter(
                        days_sim, evo_c[idx][i], s=30, color=fp.COLORS_SIM_VS_EXP[-idx]
                    )
        ax1.set_ylabel(y_label, color=y_color_sim[idx], labelpad=8)
        ax1.tick_params(axis="y", labelcolor=y_color_sim[idx])
        if ysim_scale == "log":
            ax1.set_yscale("log")
        else:
            ax1.ticklabel_format(
                style="sci", axis="y", scilimits=(0, 0), useMathText=True
            )
        ax1.set_ylim(ymax=CSIM["c"] * ysat_sim)
        # >> Left axis: experimental data.
        idx_color = 1
        ymax_exp = 0
        for key, cstat in CSTAT.items():
            y_exp = cstat["c"]["avg"][i0_exp:]
            y_err = cstat["c"]["std"][i0_exp:]
            ymax_exp = max(ymax_exp, np.max(y_exp + y_err))
            if key == key0:
                ax2 = ax1.twinx()
                y_color_exp = fp.COLORS_SIM_VS_EXP[1]
            else:
                y_color_exp = "black"
            ax2.errorbar(
                days,
                y_exp,
                yerr=y_err,
                capsize=2,
                fmt=FTM[key],
                color=fp.COLORS_SIM_VS_EXP[idx_color],
                label=LABELS_EXP[key],
            )
            if is_all_points:
                for i in range(len(cstat["c"]["all"][0])):
                    ax2.scatter(
                        days,
                        cstat["c"]["all"][i0_exp:, i],
                        s=30,
                        color=fp.COLORS_SIM_VS_EXP[idx_color],
                    )
            idx_color += 1
        ax2.grid(False)
        ax2.set_ylabel(LABELS["ax_cexp"], color=y_color_exp, labelpad=9)
        ax2.tick_params(axis="y", labelcolor=y_color_exp)
        if yexp_scale == "log":
            ax2.set_yscale("log")
        else:
            ax2.ticklabel_format(
                style="sci", axis="y", scilimits=(0, 0), useMathText=True
            )
        ax2.set_ylim(ymax=CEXP["c"] * ymax_exp)
        mpl_axes_aligner.align.yaxes(ax1, 0, ax2, 0, 0.05)
        fig.legend(bbox_to_anchor=LEGEND_POS, fontsize="small")
        sns.despine(top=True, right=False)
        if is_saved:
            path = fpath("evo_c_by_day_w_exp2", "std", ysim_scale, yexp_scale)
            if par_update_bis is not None:
                path = path.replace("_w_exp2_", "_w_exp2_x2_")
            print("\n Saved at: ", path)
            plt.savefig(path, bbox_inches="tight")
        plt.show()

    # > Discrete comparison to experimental curves: OD.
    if "OD" in CSIM.keys():
        fig, ax1 = plt.subplots()
        ax1.set_xlabel(LABELS["ax_time"], labelpad=6)
        ax1.set_xticks(days)
        # >> Right axis: simulated data.
        y_color_sim = [fp.COLORS_SIM_VS_EXP[0], "black"]
        ymax_sim = 0
        for idx in range(len(evo_c_mean)):
            ys, ys_err = evo_c_mean[idx], evo_c_std[idx]
            ymax_sim = max(ymax_sim, np.max(ys + ys_err))
            ax1.errorbar(
                days_sim,
                ys,
                yerr=ys_err,
                capsize=2,
                fmt=FTM_SIM[-idx],
                label=LABELS_SIM[-idx],
                color=fp.COLORS_SIM_VS_EXP[-idx],
            )
            if ysim_scale == "log10":
                y_label = LABELS["ax_c_log10"]
            else:
                y_label = LABELS["ax_c"]
            if is_all_points:
                for i in range(simu_count):
                    ax1.scatter(
                        days_sim, evo_c[idx][i], s=30, color=fp.COLORS_SIM_VS_EXP[-idx]
                    )
        ax1.set_ylabel(y_label, color=y_color_sim[idx], labelpad=8)
        ax1.tick_params(axis="y", labelcolor=y_color_sim[idx])
        ax1.set_ylim(ymax=CSIM["OD"] * ymax_sim)
        if ysim_scale == "log":
            ax1.set_yscale("log")
        else:
            ax1.ticklabel_format(
                style="sci", axis="y", scilimits=(0, 0), useMathText=True
            )
        # >> Left axis: experimental data.
        idx_color = 1
        ymax_exp = 0
        for key, cstat in CSTAT.items():
            y_exp = cstat["OD"]["avg"][i0_exp:]
            y_err = cstat["OD"]["std"][i0_exp:]
            ymax_exp = max(ymax_exp, np.max(y_exp + y_err))
            if key == key0:
                ax2 = ax1.twinx()
                y_color_exp = fp.COLORS_SIM_VS_EXP[1]
            else:
                y_color_exp = "black"
            ax2.errorbar(
                days,
                y_exp,
                yerr=y_err,
                capsize=2,
                fmt=FTM[key],
                color=fp.COLORS_SIM_VS_EXP[idx_color],
                label=LABELS_EXP[key],
            )
            if is_all_points:
                for i in range(len(cstat["OD"]["all"][0])):
                    ax2.scatter(
                        days,
                        cstat["OD"]["all"][i0_exp:, i],
                        s=30,
                        color=fp.COLORS_SIM_VS_EXP[idx_color],
                    )
            idx_color += 1
        ax2.set_ylim(ymax=CEXP["OD"] * ymax_exp)
        ax2.grid(False)
        ax2.set_ylabel(LABELS["ax_OD"], color=y_color_exp, labelpad=9)
        ax2.tick_params(axis="y", labelcolor=y_color_exp)
        if yexp_scale == "log":
            ax2.set_yscale("log")
        mpl_axes_aligner.align.yaxes(ax1, 0, ax2, 0, 0.05)
        fig.legend(bbox_to_anchor=LEGEND_POS, fontsize="small")
        sns.despine(top=True, right=False)
        if is_saved:
            path = fpath("evo_c_by_day_w_exp3", "std", ysim_scale, yexp_scale)
            if par_update_bis is not None:
                path = path.replace("_w_exp3_", "_w_exp3_x2_")
            print("\n Saved at: ", path)
            plt.savefig(path, bbox_inches="tight")
        plt.show()
    # ---------------------------------

    if not is_only_exp:
        # Evolutions at all times.
        # > Concentration of cells.
        XTICKS = np.array([0, *days_sim])
        # >> Simulated data alone with type and senescencent.
        custom_args = {
            "curve_labels": [None, LABELS["btype"], LABELS["sen"]],
            "y_format": "sci",  # 'linestyles': ['-', '--', '-.'],
            "fig_path": fpath("evo_c"),
            "bbox_to_anchor": LEG_POS,
            "xticks": XTICKS,
            "leg_fontsize": "small",
        }
        AXIS_LABELS = [LABELS["ax_time"], LABELS["ax_c"]]
        plot_evo_curves_w_stats(
            times,
            [d["evo_c"], d["evo_c_B"], d["evo_c_sen"]],
            AXIS_LABELS,
            is_stat,
            kwargs=custom_args,
        )

        evo_c_bnhtype = {}
        for stat_key, evo_c_btype in d["evo_c_B"].items():
            evo_c_bnhtype[stat_key] = evo_c_btype + d["evo_c_H"][stat_key]
        custom_args["fig_path"] = fpath("evo_c_woH")
        plot_evo_curves_w_stats(
            times,
            [d["evo_c"], evo_c_bnhtype, d["evo_c_sen"]],
            AXIS_LABELS,
            is_stat,
            kwargs=custom_args,
        )

        # > Concentration of senescent cells.
        custom_args.update(
            {
                "curve_labels": [LABELS["sen"], LABELS["btype"], LABELS["htype"]],
                "fig_path": fpath("evo_c_sen"),
            }
        )
        plot_evo_curves_w_stats(
            times,
            [d["evo_c_sen"], d["evo_c_B_sen"], d["evo_c_H"]],
            [LABELS["ax_time"], LABELS["ax_c_sen"]],
            is_stat,
            kwargs=custom_args,
        )

        evo_c_bnhtype_sen = {}
        for stat_key, c_sen_btype in d["evo_c_B_sen"].items():
            evo_c_bnhtype_sen[stat_key] = c_sen_btype + d["evo_c_H"][stat_key]
        custom_args["fig_path"] = fpath("evo_c_sen_woH")
        plot_evo_curves_w_stats(
            times,
            [d["evo_c_sen"], evo_c_bnhtype_sen],
            [LABELS["ax_time"], LABELS["ax_c_sen"]],
            is_stat,
            kwargs=custom_args,
        )

        # > Proportion of cells.
        evo_p_atype = {}
        evo_p_bnhtype = {}
        for stat_key, evo_p_btype in d["evo_p_B"].items():
            evo_p_atype[stat_key] = 1 - evo_p_btype - d["evo_p_H"][stat_key]
            evo_p_atype[stat_key][evo_p_atype[stat_key] < 0] = 0
            evo_p_atype[stat_key][evo_p_atype[stat_key] > 1] = 1
            evo_p_bnhtype[stat_key] = evo_p_btype + d["evo_p_H"][stat_key]
            evo_p_bnhtype[stat_key][evo_p_bnhtype[stat_key] < 0] = 0
            evo_p_bnhtype[stat_key][evo_p_bnhtype[stat_key] > 1] = 1
        evo_p_sen_atype = {}
        evo_p_sen_bnhtype = {}
        for stat_key, evo_p_sen_btype_sen in d["evo_p_B_sen"].items():
            evo_p_sen_atype[stat_key] = (
                1 - evo_p_sen_btype_sen - d["evo_p_H_sen"][stat_key]
            )
            # evo_p_sen_atype[stat_key][evo_p_sen_atype[stat_key] < 0] = 0
            # evo_p_sen_atype[stat_key][evo_p_sen_atype[stat_key] > 1] = 1
            evo_p_sen_bnhtype[stat_key] = (
                evo_p_sen_btype_sen + d["evo_p_H_sen"][stat_key]
            )
        AXIS_LABELS = [LABELS["ax_time"], LABELS["ax_prop"]]

        # >> By type.
        # is_stat_s = [{'per': False, 'std': False, 'ext': True}, is_stat,
        #              is_stat]
        keys = ["atype", "btype", "htype"]
        custom_args = {
            "curve_labels": [LABELS[key] for key in keys],
            "colors": [fp.COLORS_TYPE[key] for key in keys],
            # 'bbox_to_anchor': LEG_POS,
            "fig_path": fpath("evo_p_type"),
            "xticks": XTICKS,
            "leg_fontsize": "small",
            "general_labels": {"per": None, "ext": None},
        }
        plot_evo_curves_w_stats(
            times,
            [evo_p_atype, d["evo_p_B"], d["evo_p_H"]],
            AXIS_LABELS,
            is_stat,
            kwargs=custom_args,
        )
        keys = ["atype", "btype"]
        custom_args = {
            "curve_labels": [LABELS[key] for key in keys],
            "colors": [fp.COLORS_TYPE[key] for key in keys],
            "bbox_to_anchor": LEG_POS,
            "leg_fontsize": "small",
            "fig_path": fpath("evo_p_type_woH"),
            "xticks": XTICKS,
        }
        plot_evo_curves_w_stats(
            times,
            [evo_p_atype, evo_p_bnhtype],
            AXIS_LABELS,
            is_stat,
            kwargs=custom_args,
        )

        # >> Among senescent cells.
        custom_args = {
            "curve_labels": [LABELS["sen"]],
            "colors": [fp.COLORS_TYPE["sen"]],
            "fig_path": fpath("evo_p_sen"),
            "xticks": XTICKS,
            "leg_fontsize": "small",
        }
        plot_evo_curves_w_stats(
            times, [d["evo_p_sen"]], AXIS_LABELS, is_stat, kwargs=custom_args
        )

        # >> Both previous.
        keys = ["atype", "btype", "htype", "sen"]
        custom_args = {
            "curve_labels": [LABELS[key] for key in keys],
            "colors": [fp.COLORS_TYPE[key] for key in keys],
            "bbox_to_anchor": LEG_POS,
            "leg_fontsize": "small",
            "linestyles": ["-", "-", "-", "--"],
            "xticks": XTICKS,
            "fig_path": fpath("evo_p_type_n_sen"),
        }
        plot_evo_curves_w_stats(
            times,
            [evo_p_atype, d["evo_p_B"], d["evo_p_H"], d["evo_p_sen"]],
            AXIS_LABELS,
            is_stat,
            kwargs=custom_args,
        )

        keys = ["atype", "btype", "sen"]
        custom_args = {
            "curve_labels": [LABELS[key] for key in keys],
            "colors": [fp.COLORS_TYPE[key] for key in keys],
            "bbox_to_anchor": LEG_POS,
            "linestyles": ["-", "-", "--"],
            "xticks": XTICKS,
            "fig_path": fpath("evo_p_type_n_sen_woH"),
            "leg_fontsize": "small",
        }
        plot_evo_curves_w_stats(
            times,
            [evo_p_atype, evo_p_bnhtype, d["evo_p_sen"]],
            AXIS_LABELS,
            is_stat,
            kwargs=custom_args,
        )

        imax = len(times)
        plt.figure()
        plt.plot(
            times,
            evo_p_atype["mean"][:imax],
            color=fp.COLORS_TYPE["atype"],
            label=LABELS["atype"],
        )
        plt.plot(
            times,
            evo_p_bnhtype["mean"][:imax],
            color=fp.COLORS_TYPE["btype"],
            label=LABELS["btype"],
        )
        plt.plot(
            times,
            d["evo_p_sen"]["mean"][:imax],
            "--",
            color=fp.COLORS_TYPE["sen"],
            label=LABELS["sen"],
        )
        evo_p_sen_atype_sen = (
            evo_p_sen_atype["mean"][:imax] * d["evo_p_sen"]["mean"][:imax]
        )
        plt.fill_between(
            times,
            0 * times,
            evo_p_sen_atype_sen,
            alpha=fp.ALPHA,
            color=fp.COLORS_TYPE["atype"],
            label=LABELS["atype_sen"],
        )
        plt.fill_between(
            times,
            evo_p_sen_atype_sen,
            d["evo_p_sen"]["mean"][:imax],
            alpha=fp.ALPHA,
            color=fp.COLORS_TYPE["btype"],
            label=LABELS["atype_sen"],
        )
        plt.legend(bbox_to_anchor=LEG_POS, loc="upper left")
        plt.xticks(XTICKS)
        plt.xlabel(AXIS_LABELS[0], labelpad=6)
        plt.ylabel(AXIS_LABELS[1], labelpad=8, wrap=True)
        sns.despine()
        fig_path = fpath("evo_p_type_n_sen_woH_fill")
        if fig_path is not None:
            print("\n Saved at: ", fig_path)
            plt.savefig(fig_path, bbox_inches="tight")
        plt.show()

        # > Proportion of senescent cells by type.
        keys = ["atype", "btype", "htype", "sen"]
        custom_args = {
            "curve_labels": [LABELS[key] for key in keys],
            "colors": [fp.COLORS_TYPE[key] for key in keys],
            "bbox_to_anchor": LEG_POS,
            "leg_fontsize": "small",
            "linestyles": ["-", "-", "-", "--"],
            "xticks": XTICKS,
            "fig_path": fpath("evo_p_type_sen_n_sen", "none"),
        }
        AXIS_LABELS = [LABELS["ax_time"], LABELS["ax_p_sen"]]
        plot_evo_curves_w_stats(
            times,
            [evo_p_sen_atype, d["evo_p_B_sen"], d["evo_p_H_sen"], d["evo_p_sen"]],
            AXIS_LABELS,
            IS_STAT_NONE,
            kwargs=custom_args,
        )

        keys = ["atype", "btype", "htype"]
        custom_args = {
            "curve_labels": [LABELS[key + "_short"] for key in keys],
            "colors": [fp.COLORS_TYPE[key] for key in keys],
            "xticks": XTICKS,
            "legend_title": "Cell type",
            "bbox_to_anchor": LEG_POS,
            "leg_fontsize": "small",
            "fig_path": fpath("evo_p_type_sen", "none"),
        }
        plot_evo_curves_w_stats(
            times,
            [evo_p_sen_atype, d["evo_p_B_sen"], d["evo_p_H_sen"]],
            AXIS_LABELS,
            IS_STAT_NONE,
            kwargs=custom_args,
        )
        keys = ["atype", "btype", "sen"]
        custom_args = {
            "curve_labels": [LABELS[key] for key in keys],
            "colors": [fp.COLORS_TYPE[key] for key in keys],
            "bbox_to_anchor": LEG_POS,
            "leg_fontsize": "small",
            "linestyles": ["-", "-", "--"],
            "xticks": XTICKS,
            "fig_path": fpath("evo_p_type_sen_n_sen_woH", "none"),
        }
        plot_evo_curves_w_stats(
            times,
            [evo_p_sen_atype, evo_p_sen_bnhtype, d["evo_p_sen"]],
            AXIS_LABELS,
            IS_STAT_NONE,
            kwargs=custom_args,
        )
        imax = len(times)
        plt.figure()
        plt.xlabel(AXIS_LABELS[0], labelpad=6)
        plt.ylabel(AXIS_LABELS[1], labelpad=8, wrap=True)
        plt.tight_layout()
        plt.plot(
            times,
            evo_p_sen_atype["mean"][:imax],
            color=fp.COLORS_TYPE["atype"],
            label=LABELS["atype"],
        )
        plt.plot(
            times,
            evo_p_sen_bnhtype["mean"][:imax],
            color=fp.COLORS_TYPE["btype"],
            label=LABELS["btype"],
        )
        plt.plot(
            times,
            d["evo_p_sen"]["mean"][:imax],
            "--",
            color=fp.COLORS_TYPE["sen"],
            label=LABELS["sen"],
        )
        evo_p_sen_atype_sen = (
            evo_p_sen_atype["mean"][:imax] * d["evo_p_sen"]["mean"][:imax]
        )
        plt.fill_between(
            times,
            0 * times,
            evo_p_sen_atype_sen,
            alpha=fp.ALPHA,
            color=fp.COLORS_TYPE["atype"],
        )
        plt.fill_between(
            times,
            evo_p_sen_atype_sen,
            d["evo_p_sen"]["mean"][:imax],
            alpha=fp.ALPHA,
            color=fp.COLORS_TYPE["btype"],
        )
        plt.legend(bbox_to_anchor=LEG_POS, loc="upper left")
        plt.xticks(XTICKS)
        sns.despine()
        fig_path = fpath("evo_p_type_sen_n_sen_woH_fill")
        if fig_path is not None:
            print("\n Saved at: ", fig_path)
            plt.savefig(fig_path, bbox_inches="tight")
        plt.show()


def plot_evo_l_pcfixed_from_stat(
    c,
    p,
    simu_count,
    fig_subdirectory,
    t_max,
    is_stat_update=None,
    par_update=None,
    is_all_points=False,
):
    # General `kwargs` (see plot_evo_curves_w_stats) options.
    # > Style dependent parameters (e.g. legend position).
    if (
        fig_subdirectory is None or fig_subdirectory == "manuscript"
    ):  # format = 'manuscript'
        LEG_POS = (1, 1)
    elif fig_subdirectory == "article":  # format = 'article'
        LEG_POS = (0.97, 1)
    else:
        raise Exception("Parameters of plotting to adjust manually should bespecified")
    # > Statistical curves to plot (default updated with `is_stat_update`).
    is_stat = deepcopy(IS_STAT_DEFAULT)
    if isinstance(is_stat_update, dict):
        is_stat.update(is_stat_update)
    # > Figure name (None if figures should not be saved).
    fig_path = None
    is_saved = fig_subdirectory is not None
    if is_saved:
        directory = wp.write_fig_pop_directory(
            cell=c, para=p, subdirectory=fig_subdirectory, par_update=par_update
        )
        fig_path = join(
            directory,
            "evo_l"
            + wp.write_fig_pop_name_end(simu=simu_count, tmax=t_max, is_stat=is_stat),
        )
        fig_path_w_exp = join(
            directory,
            "evo_l_w_exp"
            + wp.write_fig_pop_name_end(
                simu=simu_count, tmax=t_max, is_stat=IS_STAT_STD
            ),
        )
    # Data.
    # > Paths to data.
    sim_path = wp.write_simu_pop_subdirectory(c, p, par_update)
    stat_data_path = wp.write_sim_pop_postreat_average(sim_path, simu_count)
    # > Experimental data.
    evo_l_exp = xtd.extract_population_lmode()
    # > Times array up to `t_max`.
    times = np.load(stat_data_path, allow_pickle=True).item().get("times")
    t_max = min(t_max, times[-1])
    times = times[times <= t_max]
    # > Days arrays.
    days_exp = np.arange(len(evo_l_exp[0]))
    idxs_bf_dil = np.array(
        [np.where(times == day)[0][0] for day in days_exp[days_exp <= times[-1]]]
    )
    day_max = min(len(days_exp), len(idxs_bf_dil))
    idxs_bf_dil = idxs_bf_dil[:day_max]
    days = days_exp[:day_max]
    XTICKS = np.array([0, *days])

    # > Evolution data.
    d = {}
    ckeys = [
        "evo_lavg_avg",
        "evo_lmode",
        "evo_lmin_max",
        "evo_lmin_avg",
        "evo_lmin_min",
    ]
    for key in ckeys:
        d[key] = np.load(stat_data_path, allow_pickle=True).item().get(key)
    if is_all_points:
        evo_mode = np.genfromtxt(
            wp.write_sim_lmode_csv(sim_path, simu_count), delimiter=","
        )

    # Plot.
    # > Simulated data, all in one graph.
    custom_args = {
        "curve_labels": [LABELS[key.replace("evo_", "")] for key in ckeys],
        "general_labels": {"per": None, "ext": None},
        "bbox_to_anchor": LEG_POS,
        "idxs_no_stat": [1],
        "fig_path": fig_path,
        "xticks": XTICKS,
        "leg_fontsize": "small",
    }
    plot_evo_curves_w_stats(
        times,
        [d[key] for key in ckeys],
        [LABELS["ax_time"], LABELS["ax_l"]],
        is_stat,
        kwargs=custom_args,
    )
    # > Mode, comparison with experiment.
    evo_l_exp_stat = pps.statistics(evo_l_exp)
    plt.figure()
    plt.xticks(days)
    plt.xlabel(LABELS["ax_time"])
    plt.ylabel(LABELS["ax_lmode"], wrap=True)
    plt.errorbar(
        days,
        evo_l_exp_stat["mean"][:day_max],
        yerr=evo_l_exp_stat["std"][:day_max],
        capsize=2,
        fmt="x-",
        label=LABELS["exp"],
        color=fp.COLORS_SIM_VS_EXP[1],
    )
    if is_all_points:
        for i in range(simu_count):
            plt.scatter(days, evo_mode[i, days], s=20, color=fp.COLORS_SIM_VS_EXP[0])
        for i in range(len(evo_l_exp)):
            print(len(days), np.shape(evo_l_exp))
            plt.scatter(
                days, evo_l_exp[i, :day_max], s=20, color=fp.COLORS_SIM_VS_EXP[1]
            )
    plt.errorbar(
        days,
        d["evo_lmode"]["mean"][idxs_bf_dil],
        yerr=d["evo_lmode"]["std"][idxs_bf_dil],
        capsize=2,
        fmt="-",
        color=fp.COLORS_SIM_VS_EXP[0],
        label=LABELS["sim"],
    )
    sns.despine()
    # plt.errorbar(days, d['evo_lavg_avg']['mean'][idxs_bf_dil],
    #              yerr=d['evo_lavg_avg']['std'][idxs_bf_dil],
    #              capsize=2, fmt='-', label=LABELS['sim'] + r"$\mathrm{~-~}$"
    #              + LABELS['lavg_avg'])
    # Add lengend in revert order.
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[::-1], labels[::-1])
    if is_saved:
        print("\n Saved at: ", fig_path_w_exp)
        plt.savefig(fig_path_w_exp, bbox_inches="tight")
    plt.show()


def plot_evo_p_anc_pcfixed_from_stat(
    c,
    p,
    simu_count,
    group_count,
    fig_subdirectory,
    t_max,
    is_stat_update=None,
    par_update=None,
    is_old_sim=False,
):
    # General configuration.
    # > Style dependent parameters (e.g. legend position).
    if (
        fig_subdirectory is None or fig_subdirectory == "manuscript"
    ):  # format = 'manuscript'
        LEG_POS = (1, 1.06)
    elif fig_subdirectory == "article":  # format = 'article'
        LEG_POS = (0.97, 1.1)
    else:
        raise Exception("Parameters of plotting to adjust manually should bespecified")
    # > Statistical curves to plot (default updated with `is_stat_update`).
    is_stat = deepcopy(IS_STAT_DEFAULT)
    if isinstance(is_stat_update, dict):
        is_stat.update(is_stat_update)
    # > Legends.
    if group_count > c:
        raise Exception(
            "`group_count` is too big for the number of cells  given as argument."
        )
    group_size = c // group_count
    ALABELS = [
        f"{i * group_size + 1} to {(i + 1) * group_size}" for i in range(group_count)
    ]
    # > Colors.
    COLORS = sns.color_palette("viridis", group_count)[::-1]
    # > Figure name.
    is_saved = fig_subdirectory is not None
    if is_saved:
        directory = wp.write_fig_pop_directory(
            cell=c, para=p, subdirectory=fig_subdirectory
        )
        end_name = wp.write_fig_pop_name_end(
            simu=simu_count, tmax=t_max, is_stat=is_stat
        )

    # Paths to data.
    sim_path = wp.write_simu_pop_subdirectory(c, p, par_update)
    stat_data_path = wp.write_sim_pop_postreat_average(sim_path, simu_count)

    # Load data and plot.
    times = np.load(stat_data_path, allow_pickle=True).item().get("times")
    t_max = min(t_max, times[-1])
    XTICKS = np.arange(int(t_max) + 1)
    times = times[times <= t_max]
    time_count = len(times)
    keys = ["evo_p_ancs", "evo_p_B_ancs", "evo_p_sen_ancs"]
    if not is_old_sim:
        keys.extend([key + "_lavg" for key in keys])
    for key in keys:
        evo = (
            np.load(stat_data_path, allow_pickle=True)
            .item()
            .get(key)["mean"][:time_count]
        )
        evo_group = np.array(
            [
                fct.nansum(evo[:, i * group_size : (i + 1) * group_size], axis=1)
                for i in range(group_count)
            ]
        )
        plt.figure()
        plt.xlabel(LABELS["ax_time"])
        plt.xticks(XTICKS)
        plt.ylim(-0.05, 1.05)
        plt.ylabel(LABELS[key.replace("evo", "ax").replace("_lavg", "")], wrap=True)
        for i in range(group_count):
            plt.plot(times, evo_group[i], label=ALABELS[i], color=COLORS[i])
        if key == "evo_p_ancs":
            plt.legend(title=LABELS["leg_prop_anc"], bbox_to_anchor=LEG_POS)
        sns.despine()
        if is_saved:
            fig_path = join(directory, f"{key}_by_group{group_count}{end_name}")
            print("\n Saved at: ", fig_path)
            plt.savefig(fig_path, bbox_inches="tight")
        plt.show()


def plot_evo_gen_pcfixed_from_stat(
    c,
    p,
    simu_count,
    fig_subdirectory,
    t_max,
    is_stat_update=None,
    par_update=None,
    bbox_to_anchor=None,
    fig_size=None,
):
    # Statistical curves to plot (default updated with `is_stat_update`).
    is_stat = deepcopy(IS_STAT_DEFAULT)
    if isinstance(is_stat_update, dict):
        is_stat.update(is_stat_update)
    # Data import from `evo_statistics.py file`.
    sim_path = wp.write_simu_pop_subdirectory(c, p, par_update)
    stat_data_path = wp.write_sim_pop_postreat_average(sim_path, simu_count)
    # > Time
    times = np.load(stat_data_path, allow_pickle=True).item().get("times")
    t_max = min(t_max, times[-1])
    times = times[times <= t_max]
    time_count = len(times)
    XTICKS = np.arange(int(t_max))

    # Need to compute evolution of the avg, max, min... generation.
    stat_directory = wp.write_path_directory_from_file(stat_data_path)
    d = pps.postreat_cgen(is_stat, stat_directory, simu_count)
    plt.figure(figsize=fig_size)
    plt.xlabel(LABELS["ax_time"])
    plt.xticks(XTICKS)
    plt.ylabel(LABELS["ax_gen"])
    plt.fill_between(
        times,
        d["min"][:time_count],
        d["max"][:time_count],
        alpha=fp.ALPHA,
        label=LABELS["ext"],
        color="gray",
    )
    if is_stat["per"]:
        plt.fill_between(
            times,
            d["perdown"][:time_count],
            d["perup"][:time_count],
            alpha=fp.ALPHA,
            label=LABELS["per"],
            color="orange",
        )
    if is_stat["std"]:
        plt.fill_between(
            times,
            d["avg"][:time_count] - d["std"][:time_count],
            d["avg"][:time_count] + d["std"][:time_count],
            alpha=fp.ALPHA,
            label=LABELS["std"],
            color="orange",
        )
    plt.plot(
        times[:time_count], d["avg"][:time_count], label=LABELS["avg"], color="black"
    )
    plt.legend(bbox_to_anchor=bbox_to_anchor)
    sns.despine()
    if fig_subdirectory is not None:
        directory = wp.write_fig_pop_directory(
            cell=c, para=p, subdirectory=fig_subdirectory
        )
        end_name = wp.write_fig_pop_name_end(
            simu=simu_count, tmax=t_max, is_stat=is_stat
        )
        path = join(directory, "evo_c_gens" + end_name)
        print("\n Saved at: ", path)
        plt.savefig(path, bbox_inches="tight")
    plt.show()


# > At cell_count variable and para_count fixed.
# ----------------------------------------------


def write_fig_pfixe_path(
    simu_counts, para_count, tmax, is_stat, par_update, fig_subdirectory
):
    simu_count = simu_counts[0]
    if np.any(simu_counts != simu_count):
        simu_count = 1  # Equivalent to None, see wp.write_fig_pop_name_end.
    fig_directory = wp.write_fig_pop_directory(
        par_update=par_update, subdirectory=fig_subdirectory
    )
    fig_name_end = wp.write_fig_pop_name_end(
        simu=simu_count, para=para_count, tmax=tmax, is_stat=is_stat
    )
    return fig_directory, fig_name_end


def plot_performances_pfixed(
    cell_counts,
    simu_counts,
    fig_subdirectory,
    para_count=1,
    par_update=None,
    xticks=None,
    fig_size=None,
):
    idxs = np.arange(len(cell_counts))
    cell_counts = cell_counts.astype(int)
    simu_counts = simu_counts.astype(int)
    xticks_, xleft, xright = define_xticks_from_counts(cell_counts, xticks)

    pp = [
        pps.postreat_performances(
            para_count,
            cell_counts[i],
            simu_counts[i],
            par_update=par_update,
            is_loaded=True,
        )
        for i in idxs
    ]

    # Plot with right and left y-axis.
    fig, ax1 = plt.subplots(figsize=fig_size)
    ax1.set_xscale("log")
    ax1.set_xticks(xticks_)
    ax1.set_xlim(left=xleft, right=xright)
    ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    # Right axis.
    colors = fp.MY_COLORS_2_ROCKET
    ax1.set_xlabel(LABELS["ax_c_init"])
    ax1.set_ylabel(LABELS["ax_t_comput"], color=colors[0])
    ax1.errorbar(
        cell_counts,
        [pp[i]["computation_time"]["mean"] / 60 for i in idxs],
        yerr=[pp[i]["computation_time"]["std"] / 60 for i in idxs],
        fmt="-",
        capsize=2,
        color=colors[0],
    )
    ax1.tick_params(axis="y", labelcolor=colors[0])
    ymax = 120
    ax1.set_ylim(ymax=ymax)
    # Left axis.
    ax2 = ax1.twinx()
    ax2.set_ylim(ymax=ymax)
    if "memory_in_mo" not in pp[0].keys():
        raise Exception(
            "Need to run `postreat_performances` again, to add "
            " the key `'memory_in_mo'`."
        )
    ax2.set_ylabel(LABELS["ax_mem_comput"], color=colors[1])
    ax2.errorbar(
        cell_counts,
        [pp[i]["memory_in_mo"]["mean"] for i in idxs],
        yerr=[pp[i]["memory_in_mo"]["std"] for i in idxs],
        fmt="--",
        capsize=2,
        color=colors[1],
    )
    ax2.tick_params(axis="y", labelcolor=colors[1], color=colors[1])
    ax2.grid(False)
    sns.despine(top=True, right=False)
    # >>> Align y=0 of ax1 and ax2 with the `pos` (last argument) of figure.
    mpl_axes_aligner.align.yaxes(ax1, 0, ax2, 0, 0.05)
    # fig.tight_layout() # Otherwise the right y-label is slightly clipped.

    # Plot with right and left y-axis.
    plt.figure(figsize=fig_size)  # default: (6.4, 4.8)
    plt.xscale("log")
    plt.xticks(xticks_)
    plt.xlim(left=xleft, right=xright)
    ax = plt.gca()
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    colors = fp.MY_COLORS_2_ROCKET
    plt.xlabel(LABELS["ax_c_init"], labelpad=6)
    plt.errorbar(
        cell_counts,
        [pp[i]["computation_time"]["mean"] / 60 for i in idxs],
        yerr=[pp[i]["computation_time"]["std"] / 60 for i in idxs],
        fmt="-",
        capsize=2,
        color=colors[0],
        label=LABELS["ax_t_comput"],
    )
    # Left axis.
    plt.errorbar(
        cell_counts,
        [pp[i]["memory_in_mo"]["mean"] for i in idxs],
        yerr=[pp[i]["memory_in_mo"]["std"] for i in idxs],
        fmt="--",
        capsize=2,
        color=colors[1],
        label=LABELS["ax_mem_comput"],
    )
    plt.legend()
    sns.despine()
    if fig_subdirectory is not None:
        directory, name_end = write_fig_pfixe_path(
            simu_counts, para_count, None, IS_STAT_STD, par_update, fig_subdirectory
        )
        fig_path = join(directory, "performances_wrt_c" + name_end)
        print("\n Saved at: ", fig_path)
        plt.savefig(fig_path, bbox_inches="tight")
    plt.show()


def plot_extinct_pfixed(
    cell_counts,
    simu_counts,
    fig_subdirectory,
    para_count=1,
    par_update=None,
    xticks=None,
    fig_size=None,
):
    idxs = np.arange(len(cell_counts))
    cell_counts = cell_counts.astype(int)
    simu_counts = simu_counts.astype(int)
    xticks_, xleft, xright = define_xticks_from_counts(cell_counts, xticks)

    sim_path_s = [
        wp.write_simu_pop_subdirectory(c, para_count, par_update) for c in cell_counts
    ]
    stat_path_s = [
        wp.write_sim_pop_postreat_average(sim_path_s[i], simu_counts[i]) for i in idxs
    ]

    textinct_s = [
        np.load(path, allow_pickle=True).item().get("extinction_time")
        for path in stat_path_s
    ]
    textinct_sen_s = [
        np.load(path, allow_pickle=True).item().get("sen_time") for path in stat_path_s
    ]
    prop_s = [
        np.load(path, allow_pickle=True).item().get("extinct_prop")
        for path in stat_path_s
    ]
    is_saved = fig_subdirectory is not None

    # Printing.
    for i in idxs:
        print("\n cell_counts: ", cell_counts[i])
        print("pextinct: ", prop_s[i])

    # Plotting.
    fig, ax = plt.subplots(1, 1, figsize=fig_size)
    plt.xscale("log")
    plt.xticks(xticks_)
    plt.xlim(left=xleft, right=xright)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.errorbar(
        cell_counts,
        [textinct_s[i]["mean"] for i in idxs],
        yerr=[textinct_s[i]["std"] for i in idxs],
        capsize=2,
    )
    plt.ylabel(LABELS["ax_textinct"], labelpad=8)
    plt.xlabel(LABELS["ax_c_init"], labelpad=6)
    sns.despine()
    if is_saved:
        directory, name_end = write_fig_pfixe_path(
            simu_counts, para_count, None, IS_STAT_STD, par_update, fig_subdirectory
        )
        fig_path = join(directory, "textinct_wrt_c" + name_end)
        print("\n Saved at: ", fig_path)
        plt.savefig(fig_path, bbox_inches="tight")
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=fig_size)  # default: (6.4, 4.8)
    fig.tight_layout()
    ax.set_xscale("log")
    ax.set_xticks(xticks_)
    ax.set_xlim(left=xleft, right=xright)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.errorbar(
        cell_counts,
        [textinct_sen_s[i]["mean"] for i in idxs],
        yerr=[textinct_sen_s[i]["std"] for i in idxs],
        capsize=2,
    )
    plt.ylabel(LABELS["ax_tsen"], labelpad=8, wrap=True)
    plt.xlabel(LABELS["ax_c_init"], labelpad=6)
    sns.despine()
    if is_saved:
        directory, name_end = write_fig_pfixe_path(
            simu_counts, para_count, None, IS_STAT_STD, par_update, fig_subdirectory
        )
        fig_path = join(directory, "tsen_wrt_c" + name_end)
        print("\n Saved at: ", fig_path)
        plt.savefig(fig_path, bbox_inches="tight")
    plt.show()


def plot_sat_pfixed(
    cell_counts,
    simu_counts,
    fig_subdirectory,
    para_count=1,
    par_update=None,
    dsat_count_max=None,
    xticks=None,
    fig_size=None,
):
    idxs = np.arange(len(cell_counts))
    cell_counts = cell_counts.astype(int)
    simu_counts = simu_counts.astype(int)
    xticks_, xleft, xright = define_xticks_from_counts(cell_counts, xticks)

    sim_path_s = [
        wp.write_simu_pop_subdirectory(c, para_count, par_update) for c in cell_counts
    ]
    stat_path_s = [
        wp.write_sim_pop_postreat_average(sim_path_s[i], simu_counts[i]) for i in idxs
    ]
    tsat_s = [
        np.load(path, allow_pickle=True).item().get("sat_time") for path in stat_path_s
    ]
    psat_s = [
        np.load(path, allow_pickle=True).item().get("sat_prop") for path in stat_path_s
    ]
    dsat_count = np.max([len(tsat_s[i]["mean"]) for i in idxs])
    if dsat_count_max is not None:
        dsat_count = min(dsat_count_max, dsat_count)
    tsat_s_avg = np.transpose(
        [fct.reshape_with_nan(tsat_s[i]["mean"], dsat_count) for i in idxs]
    )
    tsat_s_std = np.transpose(
        [fct.reshape_with_nan(tsat_s[i]["std"], dsat_count) for i in idxs]
    )

    legends = {i: f"{i + 1}" for i in range(dsat_count)}
    colors = sns.color_palette("rocket", dsat_count)

    fig, ax = plt.subplots(1, 1, figsize=fig_size)
    plt.xscale("log")
    plt.tight_layout()
    plt.xticks(xticks_)
    plt.xlim(left=xleft, right=xright)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.xlabel(LABELS["ax_c_init"], labelpad=6)
    plt.ylabel(LABELS["ax_t_sat"], labelpad=8, wrap=True)
    for day in range(dsat_count):
        plt.errorbar(
            cell_counts,
            tsat_s_avg[day] - day,
            yerr=tsat_s_std[day],
            capsize=2,
            label=legends[day],
            color=colors[day],
        )
    plt.legend(title=LABELS["leg_day"], loc="upper left", bbox_to_anchor=(1, 0.95))
    sns.despine()
    # Saving.
    if fig_subdirectory is not None:
        directory, name_end = write_fig_pfixe_path(
            simu_counts, para_count, None, IS_STAT_STD, par_update, fig_subdirectory
        )
        fig_path = join(directory, "tsat_wrt_c" + name_end)
        print("\n Saved at: ", fig_path)
        plt.savefig(fig_path, bbox_inches="tight")
    plt.show()
    for i in idxs:
        print("\n cell_counts: ", cell_counts[i])
        print("psat: ", psat_s[i])


def plot_p_pfixed(
    cell_counts,
    simu_counts,
    fig_subdirectory,
    para_count=1,
    par_update=None,
    par_sim_update=None,
    xticks=None,
    fig_size=None,
    bbox_to_anchor=None,
):
    p_sim = deepcopy(par.PAR_DEFAULT_SIM_POP)
    if isinstance(par_sim_update, dict):
        p_sim.update(par_sim_update)

    dil_idxs = pps.make_time_arrays(p_sim, is_printed=False)[-1]
    idxs = np.arange(len(cell_counts))
    days = np.arange(p_sim["day_count"] - 1)
    cell_counts = cell_counts.astype(int)
    simu_counts = simu_counts.astype(int)
    colors_B = sns.color_palette("rocket", p_sim["day_count"] - 1)
    colors_sen = sns.color_palette("rocket", p_sim["day_count"] - 1)
    legends = {i: f"{i + 1}" for i in days}
    xticks_, xleft, xright = define_xticks_from_counts(cell_counts, xticks)

    sim_path_s = [
        wp.write_simu_pop_subdirectory(c, para_count, par_update) for c in cell_counts
    ]
    stat_path_s = [
        wp.write_sim_pop_postreat_average(sim_path_s[i], simu_counts[i]) for i in idxs
    ]
    prop_B_avg_s = np.transpose(
        [
            np.load(path, allow_pickle=True).item().get("evo_p_B")["mean"][dil_idxs]
            for path in stat_path_s
        ]
    )
    prop_sen_avg_s = np.transpose(
        [
            np.load(path, allow_pickle=True).item().get("evo_p_sen")["mean"][dil_idxs]
            for path in stat_path_s
        ]
    )
    prop_B_std_s = np.transpose(
        [
            np.load(path, allow_pickle=True).item().get("evo_p_B")["std"][dil_idxs]
            for path in stat_path_s
        ]
    )
    prop_sen_std_s = np.transpose(
        [
            np.load(path, allow_pickle=True).item().get("evo_p_sen")["std"][dil_idxs]
            for path in stat_path_s
        ]
    )

    is_saved = fig_subdirectory is not None

    fig, ax = plt.subplots(1, 1, figsize=fig_size)
    plt.xscale("log")
    plt.tight_layout()
    plt.xticks(xticks_)
    plt.xlim(left=xleft, right=xright)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    for day in days[: p_sim["day_count"] - 1]:
        plt.errorbar(
            cell_counts,
            prop_B_avg_s[day],
            yerr=prop_B_std_s[day],
            capsize=2,
            label=legends[day],
            color=colors_B[day],
        )
    plt.legend(title=LABELS["leg_day"], bbox_to_anchor=bbox_to_anchor)
    plt.xlabel(LABELS["ax_c_init"], labelpad=6)
    plt.ylabel(LABELS["ax_prop_B_bf_dil"], labelpad=8, wrap=True)
    sns.despine()
    if is_saved:
        directory, name_end = write_fig_pfixe_path(
            simu_counts,
            para_count,
            p_sim["day_count"] - 1,
            IS_STAT_STD,
            par_update,
            fig_subdirectory,
        )
        fig_path = join(directory, "prop_B_bf_dil_wrt_c" + name_end)
        print("\n Saved at: ", fig_path)
        plt.savefig(fig_path, bbox_inches="tight")
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=fig_size)
    plt.xscale("log")
    plt.tight_layout()
    plt.xticks(xticks_)
    plt.xlim(left=xleft, right=xright)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    for day in days[: p_sim["day_count"] - 1]:
        plt.errorbar(
            cell_counts,
            prop_sen_avg_s[day],
            yerr=prop_sen_std_s[day],
            capsize=2,
            label=legends[day],
            color=colors_sen[day],
        )
    plt.legend(title=LABELS["leg_day"], bbox_to_anchor=bbox_to_anchor)
    plt.xlabel(LABELS["ax_c_init"], labelpad=6)
    plt.ylabel(LABELS["ax_prop_sen_bf_dil"], labelpad=8, wrap=True)
    sns.despine()
    if is_saved:
        directory, name_end = write_fig_pfixe_path(
            simu_counts,
            para_count,
            p_sim["day_count"] - 1,
            IS_STAT_STD,
            par_update,
            fig_subdirectory,
        )
        fig_path = join(directory, "prop_sen_bf_dil_wrt_c" + name_end)
        print("\n Saved at: ", fig_path)
        plt.savefig(fig_path, bbox_inches="tight")
    plt.show()


def plot_evo_pfixed(
    cell_counts,
    simu_counts,
    anc_prop,
    fig_subdirectory,
    t_max,
    is_stat_update=None,
    para_count=1,
    par_update=None,
    bbox_to_anchor=None,
    fig_size=None,
):
    # Useful variables.
    cell_counts = cell_counts.astype(int)
    simu_counts = simu_counts.astype(int)
    curve_count = len(cell_counts)

    # General `kwargs` (see plot_evo_curves_w_stats) options.
    # > Curves
    CURVE_LABELS = [f"{cell_counts[i]}" for i in range(curve_count)]
    COLORS = sns.color_palette("viridis", curve_count)
    is_stat = deepcopy(IS_STAT_PER)
    if isinstance(is_stat_update, dict):
        is_stat.update(is_stat_update)
    ALPHA = 0.1
    # > Figure name (None if figures should not be saved).
    is_saved = fig_subdirectory is not None
    if is_saved:
        directory, name_end = write_fig_pfixe_path(
            simu_counts, para_count, t_max, is_stat, par_update, fig_subdirectory
        )
        name_end_std = write_fig_pfixe_path(
            simu_counts, para_count, t_max, IS_STAT_STD, par_update, fig_subdirectory
        )[1]
        name_end_none = write_fig_pfixe_path(
            simu_counts, para_count, t_max, IS_STAT_NONE, par_update, fig_subdirectory
        )[1]
        name_end_per = write_fig_pfixe_path(
            simu_counts, para_count, t_max, IS_STAT_PER, par_update, fig_subdirectory
        )[1]

        def fpath(name, stat_type=None):
            if stat_type == "std":
                return join(directory, name + "_wrt_c" + name_end_std)
            if stat_type == "none":
                return join(directory, name + "_wrt_c" + name_end_none)
            if stat_type == "per":
                return join(directory, name + "_wrt_c" + name_end_per)
            return join(directory, name + "_wrt_c" + name_end)

    else:

        def fpath(*_):
            return None

    # General data.
    # > Paths to simulated data.
    sim_paths = [
        wp.write_simu_pop_subdirectory(c, para_count, par_update) for c in cell_counts
    ]
    stat_paths = [
        wp.write_sim_pop_postreat_average(sim_paths[i], simu_counts[i])
        for i in range(curve_count)
    ]
    # > Experimental data.
    evo_l_exp = xtd.extract_population_lmode()
    # > Time array.
    times = np.load(stat_paths[0], allow_pickle=True).item().get("times")
    t_max = min(t_max, times[-1])
    times = times[times <= t_max]
    time_count = len(times)
    # > Days arrays.
    days_exp = np.arange(len(evo_l_exp[0]))
    idxs_bf_dil = np.array(
        [np.where(times == day)[0][0] for day in days_exp[days_exp <= times[-1]]]
    )
    day_max = min(len(days_exp), len(idxs_bf_dil))
    idxs_bf_dil = idxs_bf_dil[:day_max]
    days = days_exp[:day_max]
    XTICKS = days

    # Plot.
    # > Concentration of cells at all times.
    data = [
        np.load(stat_paths[i], allow_pickle=True).item().get("evo_c")
        for i in range(curve_count)
    ]
    custom_args = {
        "curve_labels": CURVE_LABELS,
        "colors": COLORS,
        "general_labels": {"per": None},
        "y_format": "sci",
        "alpha": 0.16,
        "legend_title": LABELS["leg_cell_count"],
        "fig_path": fpath("evo_c"),
        "xticks": XTICKS,
        "figsize": fig_size,
        "bbox_to_anchor": bbox_to_anchor,
        "leg_fontsize": "small",
    }
    plot_evo_curves_w_stats(
        times, data, [LABELS["ax_time"], LABELS["ax_c"]], is_stat, kwargs=custom_args
    )

    for i in range(curve_count):
        ratio = cell_counts[-1] / cell_counts[i]
        for key, d in data[i].items():
            data[i][key] = d * ratio
    custom_args = {
        "curve_labels": None,
        "colors": COLORS,
        "alpha": ALPHA,
        "fig_path": fpath("evo_c_norm"),
        "yticks": [],
        "general_labels": {"per": None, "avg": None},
        "xticks": XTICKS,
    }
    plot_evo_curves_w_stats(
        times,
        data,
        [LABELS["ax_time"], LABELS["ax_c_norm"]],
        is_stat,
        kwargs=custom_args,
    )

    # Plot telomere lengths evolution (average and mode).
    # > At all times.
    data = [
        np.load(stat_paths[i], allow_pickle=True).item().get("evo_lavg_avg")
        for i in range(curve_count)
    ]
    custom_args = {
        "curve_labels": None,
        "colors": COLORS,
        "alpha": ALPHA,
        "general_labels": {"per": None, "avg": None},
        "fig_path": fpath("evo_lavg"),
        "xticks": XTICKS,
    }
    plot_evo_curves_w_stats(
        times, data, [LABELS["ax_time"], LABELS["ax_lavg"]], is_stat, kwargs=custom_args
    )

    data = [
        np.load(stat_paths[i], allow_pickle=True).item().get("evo_lmode")
        for i in range(curve_count)
    ]
    custom_args = {
        "curve_labels": CURVE_LABELS,
        "colors": COLORS,
        "alpha": ALPHA,
        "legend_title": LABELS["leg_cell_count"],
        "bbox_to_anchor": bbox_to_anchor,
        "general_labels": {"per": None},
        "leg_fontsize": "small",
        "xticks": XTICKS,
        "fig_path": fpath("evo_lmode"),
    }
    plot_evo_curves_w_stats(
        times,
        data,
        [LABELS["ax_time"], LABELS["ax_lmode"]],
        is_stat,
        kwargs=custom_args,
    )

    data = [
        np.load(stat_paths[i], allow_pickle=True).item().get("evo_lmin_min")
        for i in range(curve_count)
    ]
    custom_args["fig_path"] = fpath("evo_lmin_min")
    plot_evo_curves_w_stats(
        times,
        data,
        [LABELS["ax_time"], LABELS["ax_lmin_min"]],
        is_stat,
        kwargs=custom_args,
    )

    data = [
        np.load(stat_paths[i], allow_pickle=True).item().get("evo_lmin_avg")
        for i in range(curve_count)
    ]
    custom_args["fig_path"] = fpath("evo_lmin_avg")
    plot_evo_curves_w_stats(
        times, data, [LABELS["ax_time"], LABELS["ax_lmin"]], is_stat, kwargs=custom_args
    )

    # > Before dilution.
    evo_l_exp = pps.statistics(evo_l_exp)
    fig, ax = plt.subplots(1, 1)
    plt.tight_layout()
    plt.xlabel(LABELS["ax_time"], labelpad=6)
    plt.xticks(XTICKS)
    plt.ylabel(LABELS["ax_lavg"], labelpad=8, wrap=True)
    plt.errorbar(
        days,
        evo_l_exp["mean"][:day_max],
        yerr=evo_l_exp["std"][:day_max],
        capsize=2,
        fmt="x-",
        label=LABELS["exp"],
        color="black",
    )
    for i in range(curve_count):
        data = np.load(stat_paths[i], allow_pickle=True).item().get("evo_lavg_avg")
        plt.errorbar(
            days,
            data["mean"][idxs_bf_dil],
            yerr=data["std"][idxs_bf_dil],
            capsize=2,
            label=CURVE_LABELS[i],
            color=COLORS[i],
        )
    plt.legend(
        title=LABELS["leg_cell_count"], bbox_to_anchor=bbox_to_anchor, fontsize="small"
    )
    sns.despine()
    if is_saved:
        path = fpath("evo_lavg_bf_dil", "std")
        print("\n Saved at: ", path)
        plt.savefig(path, bbox_inches="tight")
    plt.show()

    fig, ax = plt.subplots(1, 1)
    plt.tight_layout()
    plt.xticks(XTICKS)
    plt.xlabel(LABELS["ax_time"], labelpad=6)
    plt.ylabel(LABELS["ax_lmode"], labelpad=8, wrap=True)
    plt.errorbar(
        days,
        evo_l_exp["mean"][:day_max],
        yerr=evo_l_exp["std"][:day_max],
        capsize=2,
        fmt="x-",
        label=LABELS["exp"],
        color="black",
    )
    for i in range(curve_count):
        data = np.load(stat_paths[i], allow_pickle=True).item().get("evo_lmode")
        plt.errorbar(
            days,
            data["mean"][idxs_bf_dil],
            yerr=data["std"][idxs_bf_dil],
            capsize=2,
            fmt="-",
            label=CURVE_LABELS[i],
            color=COLORS[i],
        )
    plt.legend(
        title=LABELS["leg_cell_count"], bbox_to_anchor=bbox_to_anchor, fontsize="small"
    )
    sns.despine()
    if is_saved:
        path = fpath("evo_lmode_bf_dil", "std")
        print("\n Saved at: ", path)
        plt.savefig(path, bbox_inches="tight")
    plt.show()

    # Proportions.
    # > Senescent cells.
    data = [
        np.load(stat_paths[i], allow_pickle=True).item().get("evo_p_sen")
        for i in range(curve_count)
    ]
    custom_args = {
        "curve_labels": CURVE_LABELS,
        "colors": COLORS,
        "alpha": ALPHA,
        "general_labels": {"per": None},
        "legend_title": LABELS["leg_cell_count"],
        "bbox_to_anchor": bbox_to_anchor,
        "leg_fontsize": "small",
        "fig_path": fpath("evo_psen"),
        "xticks": XTICKS,
    }
    plot_evo_curves_w_stats(
        times,
        data,
        [LABELS["ax_time"], LABELS["ax_p_sen"]],
        is_stat,
        kwargs=custom_args,
    )
    # > Type B cells.
    data = [
        np.load(stat_paths[i], allow_pickle=True).item().get("evo_p_B")
        for i in range(curve_count)
    ]
    custom_args = {
        "curve_labels": None,
        "colors": COLORS,
        "alpha": ALPHA,
        "general_labels": {"per": None, "avg": None},
        "fig_path": fpath("evo_ptypeB"),
        "xticks": XTICKS,
    }
    plot_evo_curves_w_stats(
        times, data, [LABELS["ax_time"], LABELS["ax_p_B"]], is_stat, kwargs=custom_args
    )

    # Ancestors.
    data = []
    group_sizes = cell_counts * anc_prop
    group_sizes = group_sizes.astype(int)
    for i in range(curve_count):
        evo = np.load(stat_paths[i], allow_pickle=True).item().get("evo_p_ancs")["mean"]
        data.append({"mean": fct.nansum(evo[:, -group_sizes[i] :], axis=1)})
    custom_args = {
        "curve_labels": CURVE_LABELS,
        "colors": COLORS,
        "bbox_to_anchor": bbox_to_anchor,
        "leg_fontsize": "small",
        "legend_title": LABELS["leg_cell_count"],
        "xticks": XTICKS,
        "fig_path": fpath(f"evo_anc_top{anc_prop}", "none"),
    }
    plot_evo_curves_w_stats(
        times,
        data,
        [LABELS["ax_time"], write_ylabel_anc(anc_prop)],
        IS_STAT_NONE,
        kwargs=custom_args,
    )

    # Generations over time.
    fig, ax = plt.subplots(1, 1)
    plt.tight_layout()
    plt.xlabel(LABELS["ax_time"], labelpad=6)
    plt.ylabel(LABELS["ax_gen_avg"], labelpad=8, wrap=True)
    plt.xticks(XTICKS)
    for i in range(curve_count):
        folder = wp.write_path_directory_from_file(stat_paths[i])
        d = pps.postreat_cgen(is_stat, folder, simu_counts[i])
        plt.fill_between(
            times,
            d["perdown"][:time_count],
            d["perup"][:time_count],
            alpha=ALPHA,
            label=LABELS["per"],
            color=COLORS[i],
        )
        plt.plot(times, d["avg"][:time_count], color=COLORS[i])
    sns.despine()
    if is_saved:
        path = fpath("evo_gen", "per")
        print("\n Saved at: ", path)
        plt.savefig(path, bbox_inches="tight")
    plt.show()


# Plot average curves obtained for variable parameters
# ----------------------------------------------------


def plot_evo_w_variable(
    c,
    p,
    simu_count,
    varying_par_updates,
    varying_key,
    curve_labels,
    anc_prop,
    fig_subdirectory,
    t_max,
    is_stat_update=None,
    shared_par_update=None,
    linestyles=None,
    is_interpolated=True,
):
    """
    par_updates : dict
    varying_key: string
        The name of the variable that varies.
        For example: 'accident' if this is the 'accidnet' key' of
        `varying_par_updates`.
        Exception for updates of 2nd component of 'fit': `varying_key` among
        'ltrans', 'l0', 'l1', 'lmode'.

    """
    # Useful variables.
    curve_count = len(varying_par_updates)
    varying_supkey = list(varying_par_updates[0].keys())[0]

    # Add the parameters to update shared by all simu (if some) to varying ones
    par_updates = deepcopy(varying_par_updates)
    if isinstance(shared_par_update, dict):
        for i in curve_count:
            par_updates[i].update(shared_par_update)

    # General `kwargs` (see plot_evo_curves_w_stats) options.
    # > Style dependent parameters (e.g. legend position).
    if (
        fig_subdirectory is None or fig_subdirectory == "manuscript"
    ):  # format = 'manuscript'
        LEG_POS = None
        LEG_POS_L = None
        LEG_POS_R = (1, 1)
    elif fig_subdirectory == "article":  # format = 'article'
        LEG_POS = (0.72, 1.1)
        LEG_POS_L = (0, 1.1)
        LEG_POS_R = (1, 1.1)
    else:
        raise Exception("Parameters of plotting to adjust manually should bespecified")
    # > Curves.
    COLORS = sns.color_palette("viridis", curve_count)
    if linestyles is None:
        linestyles = ["-" for i in range(curve_count)]

    is_stat = deepcopy(IS_STAT_NONE)
    if isinstance(is_stat_update, dict):
        is_stat.update(is_stat_update)
    # > Figure name (None if figures should not be saved).
    is_saved = fig_subdirectory is not None
    if is_saved:
        # Fig path with varying parameters set to None (no subfolder created).
        fig_par_update = deepcopy(par_updates[0])
        if varying_key == varying_supkey:
            fig_par_update[varying_key] = None
        elif varying_key in ["ltrans", "l0", "l1", "lmode"]:
            fig_par_update["fit"][2] = None
        else:
            fig_par_update[varying_supkey][varying_key] = None

        directory = wp.write_fig_pop_directory(
            par_update=fig_par_update, subdirectory=fig_subdirectory
        )
        name_end = wp.write_fig_pop_name_end(simu_count, c, p, t_max, is_stat)
        name_end_none = wp.write_fig_pop_name_end(simu_count, c, p, t_max)
        name_end_std = wp.write_fig_pop_name_end(simu_count, c, p, t_max, IS_STAT_STD)

        def fpath(name, stat_type=None):
            if stat_type == "std":
                return join(directory, name + f"_wrt_{varying_key}" + name_end_std)
            if stat_type == "none":
                return join(directory, name + f"_wrt_{varying_key}" + name_end_none)
            return join(directory, name + f"_wrt_{varying_key}" + name_end)

    else:

        def fpath(*_):
            return None

    # General data.
    # > Paths to simulated data.
    sim_paths = [wp.write_simu_pop_subdirectory(c, p, par_) for par_ in par_updates]
    stat_data_paths = [
        wp.write_sim_pop_postreat_average(path, simu_count) for path in sim_paths
    ]
    # > Experimental data.
    evo_l_exp = xtd.extract_population_lmode()
    # > Time array (up to t_max).
    times = np.load(stat_data_paths[0], allow_pickle=True).item().get("times")
    t_max = min(t_max, times[-1])
    times = times[times <= t_max]
    time_count = len(times)
    # > Days arrays.
    days_exp = np.arange(len(evo_l_exp[0]))
    idxs_af_dil = np.array(
        [np.where(times == day)[0][0] for day in days_exp[days_exp <= times[-1]]]
    )
    day_max = min(len(days_exp), len(idxs_af_dil))
    idxs_af_dil = idxs_af_dil[:day_max]
    idxs_bf_dil = idxs_af_dil[1:] - 1
    days = days_exp[:day_max]
    XTICKS = days

    # > Concentration of cells (with different characteristics).
    custom_args = {
        "curve_labels": curve_labels,
        "colors": COLORS,
        "linestyles": linestyles,
        "y_format": "sci",
        "legend_title": fp.LABELS[varying_key],
        "xticks": XTICKS,
        "leg_fontsize": "small",
        "bbox_to_anchor": LEG_POS,
    }
    for key in ["c", "c_sen", "c_B"]:
        evo_key = "evo_" + key
        data = [
            np.load(stat_path, allow_pickle=True).item().get(evo_key)
            for stat_path in stat_data_paths
        ]
        # Continuous version.
        custom_args["fig_path"] = fpath(evo_key)
        plot_evo_curves_w_stats(
            times,
            data,
            [LABELS["ax_time"], LABELS["ax_" + key]],
            is_stat,
            kwargs=custom_args,
        )
        # Discrete version.
        is_stat_d = deepcopy(is_stat)
        is_stat_d["std"] = True
        data = [
            {stat_key: d[stat_key][idxs_bf_dil] for stat_key in d.keys()} for d in data
        ]
        custom_args["fig_path"] = fpath(evo_key + "_bf_dil")

        if key == "c" and is_interpolated:
            means_s = [
                d["mean"][-len(d["mean"][d["mean"] < d["mean"][0]]) - 1 :] for d in data
            ]
            fct_interpol_inv_medians_s = [
                interpolate.interp1d(means, days[-len(means) :]) for means in means_s
            ]
            t_halflife_s = [
                fct_interpol_inv_medians_s[d](means_s[d][0] / 2)
                for d in range(len(data))
            ]
            print("Half times: ", t_halflife_s)
            xs = np.linspace(0.1e5, 3e5, 100)
            plt.figure()
            for fct_int in fct_interpol_inv_medians_s:
                plt.plot([fct_int(x) for x in xs], xs)
            plt.show()

        plot_evo_curves_w_stats(
            days[1:],
            data,
            [LABELS["ax_time"], LABELS["ax_" + key]],
            is_stat,
            kwargs=custom_args,
        )

    # > Telomere lengths evolution.
    evo_l_exp = pps.statistics(evo_l_exp)
    custom_args = {
        "curve_labels": curve_labels,
        "colors": COLORS,
        "linestyles": linestyles,
        "leg_fontsize": "small",
        "legend_title": fp.LABELS[varying_key],
        "bbox_to_anchor": LEG_POS,
        "xticks": XTICKS,
    }
    for key in ["lavg_avg", "lmin_avg", "lmode"]:
        evo_key = "evo_" + key
        key = key.replace("_avg", "")
        data = [
            np.load(stat_path, allow_pickle=True).item().get(evo_key)
            for stat_path in stat_data_paths
        ]
        custom_args["fig_path"] = fpath(evo_key)
        # Continuous version.
        plot_evo_curves_w_stats(
            times,
            data,
            [LABELS["ax_time"], LABELS["ax_" + key]],
            is_stat,
            kwargs=custom_args,
        )
        # Discrete version with experiment.
        if key != "lmin_avg":
            plt.figure()
            plt.xticks(XTICKS)
            plt.xlabel(LABELS["ax_time"], labelpad=6)
            plt.ylabel("  " + LABELS["ax_" + key] + "     ", labelpad=8, wrap=True)
            for i in range(curve_count):
                plt.errorbar(
                    days,
                    data[i]["mean"][idxs_af_dil],
                    yerr=data[i]["std"][idxs_af_dil],
                    capsize=2,
                    label=curve_labels[i],
                    color=COLORS[i],
                    linestyle=linestyles[i],
                )
            # plt.errorbar(days, evo_l_exp['mean'][:day_max],
            #               yerr=evo_l_exp['std'][:day_max],
            #               capsize=2, fmt='x-', label=LABELS['exp'],
            #               color='black')
            plt.legend(title=fp.LABELS[varying_key], bbox_to_anchor=LEG_POS_R)
            sns.despine()  # Remove and top and right axis.
            if is_saved:
                path = fpath(evo_key + "_bf_dil", "std")
                print("\n Saved at: ", path)
                plt.savefig(path, bbox_inches="tight")
            plt.show()

    # > Proportions.
    custom_args["bbox_to_anchor"] = LEG_POS_L
    for key in ["p_sen", "p_B"]:
        evo_key = "evo_" + key
        custom_args["fig_path"] = fpath(evo_key)
        data = [
            np.load(stat_path, allow_pickle=True).item().get(evo_key)
            for stat_path in stat_data_paths
        ]
        plot_evo_curves_w_stats(
            times,
            data,
            [LABELS["ax_time"], LABELS["ax_" + key]],
            is_stat,
            kwargs=custom_args,
        )

    # > Ancestors.
    data = []
    anc_count = int(c * anc_prop)
    for i in range(curve_count):
        evo = (
            np.load(stat_data_paths[i], allow_pickle=True)
            .item()
            .get("evo_p_ancs")["mean"]
        )
        data.append({"mean": fct.nansum(evo[:, -anc_count:], 1)})
    custom_args["fig_path"] = fpath(f"evo_anc_top{anc_prop}", "none")
    plot_evo_curves_w_stats(
        times,
        data,
        [LABELS["ax_time"], write_ylabel_anc(anc_prop)],
        IS_STAT_NONE,
        kwargs=custom_args,
    )

    # Generations over time.
    plt.figure()
    plt.xticks(XTICKS)
    plt.xlabel(LABELS["ax_time"], labelpad=6, wrap=True)
    plt.ylabel(LABELS["ax_gen_avg"], labelpad=8, wrap=True)
    for i in range(curve_count):
        folder = wp.write_path_directory_from_file(stat_data_paths[i])
        d = pps.postreat_cgen(is_stat, folder, simu_count)
        plt.plot(
            times,
            d["avg"][:time_count],
            label=curve_labels[i],
            color=COLORS[i],
            linestyle=linestyles[i],
        )
    plt.legend(loc="lower right", title=fp.LABELS[varying_key])
    sns.despine()
    if is_saved:
        print(fpath("evo_gen", "none"))
        plt.savefig(fpath("evo_gen", "none"), bbox_inches="tight")
    plt.show()
