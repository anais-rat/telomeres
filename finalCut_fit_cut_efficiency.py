#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 14:42:46 2024

@author: arat
"""

import numpy as np
import matplotlib.pyplot as plt


P_CUT = {# [0h, 3h, 6h, 9h]
         'Fc20_n2': [0.00, 0.48, 0.78, 0.90],
         'Fc30_n2': [0.00, 0.49, 0.89, 0.83],
         'Fc40_n2': [0.00, 0.16, 0.63, 0.82],
         'Fc50_n2': [0.00, 0.57, 0.88, 0.92],
         'Fc70_n2': [0.04, 0.40, 0.69, 0.86]
         }
P_CUT['Fc_avg'] = np.mean(np.concatenate([[p] for k, p in P_CUT.items()], 0), 0)

KEY_DATA = 'Fc20_n2'

# For fitting y = A + B log x, just fit y against (log x).

# >>> x = numpy.array([1, 7, 20, 50, 79])
# >>> y = numpy.array([10, 19, 30, 35, 51])
# >>> numpy.polyfit(numpy.log(x), y, 1)
# array([ 8.46295607,  6.61867463])
# # y ≈ 8.46 log(x) + 6.62

# y = np.log(1 / (1 - np.array(P_CUT[KEY_DATA])))
# x = np.array([0, 3, 6, 9])


# out = np.polyfit(np.log(x), y, 1)
x = np.array([0, 3, 6, 9])
y = 1 - np.array(P_CUT[KEY_DATA])
A, B = np.polyfit(np.log(x[1:]), y[1:], 1)


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


x_exp = np.array([0, 3, 6, 9])
y_exp = np.array(P_CUT[KEY_DATA])

# Fit A, B s.t x[1:] approximates A ln(1 - y[1:]) + B
A, B = np.polyfit(np.log(1 - y_exp[1:]), x_exp[1:],  1)

def fit_right_x1(x):
    return 1 - np.exp((x - B) / A)

# lbd = np.log(1 - fit(3)) / x[1]
lbd = (1 - B / x_exp[1]) / A
def fit_left_x1(x):
    return  1 - np.exp(lbd * x)

def fit_cdf(x):
    return (x < x_exp[1]) * fit_left_x1(x) + (x >= x_exp[1]) * fit_right_x1(x)

if __name__ == "__main__":
    x_s = np.linspace(0, 9.5, 100)

    plt.figure(dpi=600)
    plt.xlabel('Time (h)')
    plt.ylabel('Percentage of cut')
    plt.plot(x_exp, y_exp, 'o', label='Experiment')
    plt.plot(x_s, fit_cdf(x_s), label='Simulation')
    plt.legend()
    plt.show()

    y_s = (fit_cdf(x_s + 30 / 6) - fit_cdf(x_s)) / (1 - fit_cdf(x_s))
    plt.plot(x_s, fit_cdf(y_s))
    plt.show()
