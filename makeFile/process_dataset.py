#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 16:30:23 2022

@author: arat

Run this script to process the experimental data contained in `data/raw`.
This will create the folder `data/processed`, filled with processed data.

"""

import os
import sys
# sys.path.append('..')

absolute_path = os.path.abspath(__file__)
current_dir = os.path.dirname(absolute_path)
projet_dir = os.path.dirname(current_dir)
sys.path.append(projet_dir)

import telomeres.dataset.make_processed_dataset as mk


# Create formated datasets.
if __name__ == "__main__":

    # MICROFLUIDIC DATA.
    # Coutelier at al. 2008.
    out = mk.make_microfluidic_dataset('TelomeraseNegative.mat',
                                       strain='TetO2-TLC1')
    # Rad 51.
    mk.make_microfluidic_dataset('TelomeraseNegMutantRAD51.mat',
                                 strain="RAD51")

    # INITIAL DISTRIBUTION OF TELOMERE LENGTHS.
    # Bourgeron et al. 2015
    mk.make_distribution_telomeres_init('etat_asymp_val_juillet')
    # Previous slightly modified (sse ltrans, l0, l1 parameters).
    mk.make_distribution_telomeres_init_fitted()

    # POPULATION DATA
    mk.make_population_dataset()
