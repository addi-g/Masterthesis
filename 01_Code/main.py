#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 15:08:26 2019

@author: adrian

Main Datei die die Simulation und damit den Vergleich der implementierten Schätzer
durchführt.
"""
import numpy as np
import matplotlib . pyplot as plt
import pandas as pd
from scipy.stats import iqr
from data_gen import gen_data_Y
from constant import constant_estimate
from new_neural_network import new_neural_network_estimate
from nearest_neighbor import nearest_neighbor_estimate
from fc_neural_network import fc_neural_1_estimate 

'''
EINDIMENSIONALER FALL (d = 1) wird geplottet
'''
n = 10000

# Parameter für unseren neuen Neuronale-Netze-Regressionschätzer

N = 3
q = 2
R = 10
a = 3
M = 5
d = 1

sigma = 0.05

X_train = np.sort(np.random.uniform(low=-2,high=2,size=(int(n * 0.8),d)), axis = 0)
m_X_train, Y_train = gen_data_Y(X_train,sigma)

X_test = np.sort(np.random.uniform(low=-2,high=2,size=(int(n * 0.2),d)), axis = 0)

Y_pred_new_nn = new_neural_network_estimate(X_train, Y_train, X_test, N, q, R, d, M, a,)
Y_pred_fc_nn_1 = fc_neural_1_estimate(X_train, Y_train, X_test)
#Y_pred_nearest_neighbor = nearest_neighbor_estimate(X_train, Y_train, X_test)

m_X_test, Ü = gen_data_Y(X_test,sigma)

plt.plot(X_test, m_X_test, '-b', label='m_d') 
#plt.plot(X_test, Y_pred_nearest_neighbor, '-r', label='nearest_neigbhor')
plt.plot(X_test, Y_pred_fc_nn_1, '-g', label='fc_nn_1')
plt.plot(X_test, Y_pred_new_nn, '-y', label='new_nn')  
plt.legend(loc='upper left') 
plt.xlim(-2.0, 2.0)
#plt.show()
plt.savefig('foo.png')

'''
ZWEIDIMENSIONALER FALL (d = 2) wird ein Vergleich des emp. L2 Fehler gemacht 
'''
n = 100

# Parameter für unseren neuen Neuronale-Netze-Regressionschätzer

N = 3
q = 2
R = 10
a = 3
M = 5
d = 2

sigma = 0.05

scaled_error = np.empty((50, 3,))
scaled_error[:] = np.nan

e_L2_avg = np.zeros(50) 
e_L2_avg[:] = np.nan

for i in range(0,50,1):

    X_train = np.sort(np.random.uniform(low=-2,high=2,size=(int(n * 0.8),d)), axis = 0)
    m_X_train, Y_train = gen_data_Y(X_train,sigma)
    
    X_test = np.sort(np.random.uniform(low=-2,high=2,size=(int(n * 0.2),d)), axis = 0)
    
    #Y_pred_constant = constant_estimate(Y_train)
    Y_pred_new_nn = new_neural_network_estimate(X_train, Y_train, X_test, N, q, R, d, M, a,)
    Y_pred_fc_nn_1 = fc_neural_1_estimate(X_train, Y_train, X_test)
    Y_pred_nearest_neighbor = nearest_neighbor_estimate(X_train, Y_train, X_test)
    
    m_X_test, Ü = gen_data_Y(X_test,sigma)
    
    e_L2_new_nn = np.mean(sum (abs(Y_pred_new_nn - m_X_test) ** 2))
    e_L2_fc_nn_1 = np.mean(sum (abs(Y_pred_fc_nn_1 - m_X_test) ** 2))
    e_L2_nearest_neighbor = np.mean(sum (abs(Y_pred_nearest_neighbor - m_X_test) ** 2))
    
    for j in range(0,50,1):
        
        X = np.sort(np.random.uniform(low=-2,high=2,size=(n,d)), axis = 0)
        m_X, Y = gen_data_Y(X,sigma)
        Y_pred_constant = constant_estimate(Y)
        
        e_L2_avg[j] = np.mean(sum(abs(Y_pred_constant - m_X) ** 2))
    
    
    scaled_error[i,0] = e_L2_new_nn / np.median(e_L2_avg)
    scaled_error[i,1] = e_L2_fc_nn_1 / np.median(e_L2_avg)
    scaled_error[i,2] = e_L2_nearest_neighbor / np.median(e_L2_avg)
    
iqr_new_nn = iqr(scaled_error[:,0]) 
iqr_fc_nn_1 = iqr(scaled_error[:,1])
iqr_nearest_neighbor = iqr(scaled_error[:,2])

median_new_nn = np.median(scaled_error[:,0])
median_fc_nn_1 = np.median(scaled_error[:,1])
median_nearest_neighbor = np.median(scaled_error[:,2])

rows = ["noise","e_L2_avg","approach","new_nn", "fc_nn_1", "nearest_neighbor"]

series_noise_1 = pd.Series([repr(sigma)+'%',np.median(e_L2_avg),"(Median, IQR)",(median_new_nn, iqr_new_nn), (median_fc_nn_1, iqr_fc_nn_1), (median_nearest_neighbor, iqr_nearest_neighbor)], index=rows)
series_noise_1.name = ""
#series_noise_2 = pd.Series([repr(sigma)+'%',np.median(e_L2_avg),"(Median, IQR)",(median_new_nn, iqr_new_nn), (median_fc_nn_1, iqr_fc_nn_1), (median_nearest_neighbor, iqr_nearest_neighbor)], index=rows)
#series_noise_2.name = ""

error_df = pd.concat([series_noise_1], axis=1)
#print(error_df)
error_df.to_csv('out.csv',index = True)    

