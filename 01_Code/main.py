#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 15:08:26 2019

@author: adrian

Main Datei die die Simulation und damit den Vergleich der implementierten Schätzer
durchführt.
"""
import numpy as np
from sklearn.datasets import make_regression
from sklearn.preprocessing import MinMaxScaler
from data_gen import gen_data_Y
from constant import constant_estimate
from new_neural_network import new_neural_network_estimate
from nearest_neighbor import nearest_neighbor_estimate
from fc_neural_network import fc_neural_1_estimate 

#n = 100
#d = 4

N = 5
q = 4
R = 10
a = 3
M = 5
d = 2

X, Y = make_regression(n_samples=100, n_features=2, noise=0.1, random_state=1)
scalarX, scalarY = MinMaxScaler(), MinMaxScaler()
scalarX.fit(X)
scalarY.fit(Y.reshape(100,1))
X = scalarX.transform(X)
Y = scalarY.transform(Y.reshape(100,1))

#X = np.random.uniform(low=-1,high=1,size=(n,d))
#Y = gen_data_Y(X,0.05)

Y_pred_constant = constant_estimate(Y)
Y_pred_new_nn = new_neural_network_estimate(X, Y, N, q, R, d, M, a)
Y_pred_fc_nn_1 = fc_neural_1_estimate(X, Y)
Y_pred_nearest_neighbor = nearest_neighbor_estimate(X, Y) 