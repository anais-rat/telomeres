#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 14:42:46 2024

@author: arat

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

import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import numpy as np


P_CUT = {  # Copied from data_finalCut/raw/EfficaciteCoupure2.xlsx
           # [0h, 3h, 6h, 9h]
         'Fc20_n2': [0.0000, 0.4816, 0.7845, 0.9000],
         'Fc30_n2': [0.0000, 0.4254, 0.8088, 0.8985],
         'Fc40_n2': [0.0000, 0.1608, 0.6315, 0.8232],
         'Fc50_n2': [0.0000, 0.5728, 0.8843, 0.9220],
         'Fc70_n2': [0.0094, 0.4463, 0.7904, 0.9607]
         }

# P_CUT = {  # Copied from data_finalCut/raw/EfficaciteCoupure.xlsx
#           'Fc20_n2': [0.00, 0.48, 0.78, 0.90],
#           'Fc30_n2': [0.00, 0.49, 0.89, 0.83],
#           'Fc40_n2': [0.00, 0.16, 0.63, 0.82],
#           'Fc50_n2': [0.00, 0.57, 0.88, 0.92],
#           'Fc70_n2': [0.04, 0.40, 0.69, 0.86]
#           }

P_CUT['Fc_avg'] = np.mean(np.concatenate([[p] for k, p in P_CUT.items()], 0),
                          0)  # Average on all Fc.

KEY_DATA = 'Fc20_n2'

# For fitting y = A + B log(x), just fit y against log(x).
# >>> x = numpy.array([1, 7, 20, 50, 79])
# >>> y = numpy.array([10, 19, 30, 35, 51])
# >>> numpy.polyfit(numpy.log(x), y, 1)
# array([ 8.46295607,  6.61867463])
# # y ≈ 8.46 log(x) + 6.62

# y = np.log(1 / (1 - np.array(P_CUT[KEY_DATA])))
# x = np.array([0, 3, 6, 9])



# x = np.array([0, 3, 6, 9])
# y = 1 - np.array(P_CUT[KEY_DATA])
# A, B = np.polyfit(np.log(x[1:]), y[1:], 1)  # Fit y against log(x): y=A+Blog(x)


x_s = np.linspace(0, 9.5, 100)
# y_s =  A * np.log(x_s) + B

# plt.plot(x, y, 'o')
# plt.plot(x_s, y_s)
# plt.plot(x_s, A * np.log(x_s))
# plt.show()

# plt.plot(x, y, 'o')
# plt.plot(x_s, y_s)
# plt.show()

# plt.plot(x, y, 'o')
# plt.plot(x_s, np.minimum(1, y_s))
# plt.show()

# plt.plot(x, 1-y, 'o')
# plt.plot(x_s, 1 - np.minimum(1, y_s))
# plt.show()


# plt.plot(x, y, 'o')
# lbd = np.log(y[1]) / x[1]
# lbd = np.log(A * np.log(x[1]) + B) / x[1]
# y_s_bis = np.exp(lbd * x_s)
# plt.plot(x_s, (x_s < 3) * y_s_bis + (x_s >= 3) * y_s)
# # plt.plot(x_s, out[0] * np.log(x_s))
# plt.show()




# plt.plot(x, y, 'o')
# y_s_bis = np.exp(lbd * x_s)
# y_s_ter = np.exp((x_s - B) / A)
# plt.plot(x_s, (x_s < 3) * y_s_bis + (x_s >= 3) * y_s_ter)
# # x_s = np.linspace(0, 9.5, 100)
# # plt.plot(x_s, y_s)
# # plt.plot(x_s, out[0] * np.log(x_s))
# plt.show()




# y_s = fit_cdf(x_s)
# y_s_bis = 1 - np.exp(lbd * x_s)


x_exp = np.array([0, 3, 6, 9])  # Time in hour.
y_exp = 1 - np.array(P_CUT[KEY_DATA])  # Proportion of uncut at times `x_exp`.
# Fit A, B s.t. x_exp[1:] approximates A ln(y_exp[1:]) + B
A, B = np.polyfit(np.log(y_exp[1:]), x_exp[1:],  1)

def fit_right_x1(x):
    return 1 - np.exp((x - B) / A)

# lbd = np.log(1 - fit(3)) / x[1]
lbd = (1 - B / x_exp[1]) / A
def fit_left_x1(x):
    return  1 - np.exp(lbd * x)


def fit_cdf(x):
    return (x < x_exp[1]) * fit_left_x1(x) + (x >= x_exp[1]) * fit_right_x1(x)


if __name__ == "__main__":
    x_s = np.linspace(0, 15, 100)

    fig, axes = plt.subplots(1, 1, dpi=600)
    plt.xlabel('Time (h)')
    plt.ylabel('Proportion of cut')
    plt.plot(x_s, fit_cdf(x_s), label='Simulation', color='darkorange')
    plt.plot(x_exp, 1 - y_exp, 'x', label='Experiment', color='black')
    axes.xaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.legend()

    plt.show()

    # y_s = (fit_cdf(x_s + 30 / 6) - fit_cdf(x_s)) / (1 - fit_cdf(x_s))
    # plt.plot(x_s, fit_cdf(y_s))
    # plt.show()
