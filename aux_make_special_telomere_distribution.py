#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 11:16:40 2024

@author: arat

The present script allows generating and saving special telomere length
distributions in a cell population, which can be later loaded (through the
parameter `L_INIT_CHOICE` in `parameter.py`) to initialize population
simulations.

For example, one can initialize the population with only two cells, both
generated from a distribution in a very large population: one cell being the
cell with the shortest telomere of the large population and the other cell
with the average telomere length of the cells in the large population.

The script and its output saved files, were not utilized in any published work.

"""

import imp
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import numpy.random as rd

import aux_parameters_functions as parf
import parameters as par
import aux_function as af
imp.reload(par)

def distibutions_cell_shortest_n_average(cell_count, simu_count,
                                         is_plotted=False):
    file_shortest = "data/data_init/distribution_shortest_" + \
                    f"c{cell_count}_s{simu_count}.csv"
    file_average = "data/data_init/distribution_average_" + \
                   f"c{cell_count}_s{simu_count}.csv"
    x_axis = parf.transform_l_init(par.L_INIT_EXP, *par.PAR_L_INIT)[0]
    if not os.path.exists(file_shortest):
        tmp_distrib_shortest = np.empty((0, len(x_axis)))
        tmp_distrib_average = np.empty((0, len(x_axis)))
        for s in range(simu_count):
            print(s)
            lengths = af.draw_cells_lengths(cell_count)
            tmp_shortest = np.min(lengths, axis=(1, 2))
            tmp_shortest = af.cdf_to_distribution(tmp_shortest, x_axis)[1]
            tmp_distrib_shortest = np.append(tmp_distrib_shortest,
                                             [tmp_shortest], 0)
            tmp_average = np.mean(lengths, axis=(1, 2))
            tmp_average = af.cdf_to_distribution(tmp_average, x_axis)[1]
            tmp_distrib_average = np.append(tmp_distrib_average, [tmp_average],
                                            0)
        distrib_shortest = np.mean(tmp_distrib_shortest, 0)
        distrib_average = np.mean(tmp_distrib_average, 0)
        std_shortest = np.std(tmp_distrib_shortest, 0)
        std_average = np.std(tmp_distrib_average, 0)
        pd.DataFrame(distrib_shortest).to_csv(file_shortest, header=None,
                                              index=None)
        pd.DataFrame(std_shortest).to_csv(file_shortest.replace('.csv',
                                                                '_std.csv'),
                                          header=None, index=None)
        pd.DataFrame(distrib_average).to_csv(file_average, header=None,
                                             index=None)
        pd.DataFrame(std_average).to_csv(file_average.replace('.csv',
                                         '_std.csv'), header=None, index=None)
    else:
        distrib_shortest = np.loadtxt(file_shortest)
        distrib_average = np.loadtxt(file_average)
    if is_plotted:
        plt.plot(x_axis, distrib_shortest, distrib_average)
    return  # distrib_shortest, distrib_average

#
def draw_lengths_one_long(cell_count, par_l_init=par.PAR_L_INIT):
    """ Generates telomere lengths of `cell_count` cells and return the
    telomere distribution of the cell with the biggest shortest telomere.

    Warning
    ------
    Argument `cell_count` is an integer >= 2.

    """
    lengths_init = af.draw_cells_lengths(cell_count, par_l_init)
    lengths_min = np.min(lengths_init, axis=(1, 2))
    longuest_cell_idx = np.argmax(lengths_min)
    return lengths_init[longuest_cell_idx]

#
def draw_lengths_one_short(cell_count, par_l_init=par.PAR_L_INIT):
    """ Generates telomere lengths of `cell_count` cells and return the
    telomere distribution of the cell with the smallest shortest telomere.

    Warning
    ------
    Argument `cell_count` is an integer >= 2.

    """
    lengths_init = af.draw_cells_lengths(cell_count, par_l_init)
    print('Generation of the population finished')
    lengths_min = np.min(lengths_init, axis=(1, 2))
    shortest_cell_idx = np.argmin(lengths_min)
    return lengths_init[shortest_cell_idx]

#
def find_nearest(arr, value):
    """ Return the index of the value of the array `arr` that is the (first)
    closest to `value`.

    """
    return (np.abs(arr - value)).argmin()

#
def draw_lengths_one_medium(cell_count, par_l_init=par.PAR_L_INIT):
    """ Generates telomere lengths of `cell_count` cells and return the
    telomere distribution of the cell with shortest telomere the closest to the
    average shortest telemore length in the generated initial population.

    Warning
    ------
    Argument `cell_count` is an integer >= 2.

    """
    lengths_init = af.draw_cells_lengths(cell_count, par_l_init)
    lmins = np.min(lengths_init, axis=(1, 2))
    lmin_avg = np.mean(lmins)
    average_cell_idx = find_nearest(lmins, lmin_avg)
    return lengths_init[average_cell_idx]


def draw_lengths_one_veryshort(cell_count, par_l_init=par.PAR_L_INIT):
    """ Generates telomere lengths of `cell_count` cells and return the
    telomere distribution of the cell composed of the (`par.CHROMOSOME_COUNT`)
    chromosomes of the whole population with the shortest telomeres.

    Warning
    -------
    Argument `cell_count` is an integer >= 2.

    """
    chromosome_count = par.CHROMOSOME_COUNT * cell_count
    lengths_init = af.draw_cell_lengths(chromosome_count, par_l_init)
    lengths_new = np.array([[], []])
    for chromosome in range(par.CHROMOSOME_COUNT):
        lmin_idx = np.mod(np.argmin(lengths_init), chromosome_count)
        chromosome_new = np.transpose([lengths_init[:, lmin_idx]])
        lengths_new = np.append(lengths_new, chromosome_new, axis=1)
        lengths_init = np.delete(lengths_init, lmin_idx, 1)
        chromosome_count -= 1
    return rd.permutation(lengths_new.T).T
