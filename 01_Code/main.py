#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 15:08:26 2019

@author: adrian

Main Datei die die Simulation und damit den Vergleich der implementierten Schätzer
durchführt.
"""
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib . pyplot as plt
import pandas as pd
import tikzplotlib
from scipy.stats import iqr
from data_gen import gen_data_Y
from constant import constant_estimate
from new_neural_network import new_neural_network_estimate
from nearest_neighbor import nearest_neighbor_estimate
from fc_neural_network import fc_neural_1_estimate 

N = 10000
n_train = N * 0.8
n_test = N * 0.2

'''
EINDIMENSIONALER FALL (d = 1) wird geplottet
'''

N = 3
q = 2
R = 10 ** 6  
a = 2
M = 2
d = 1

sigma = 0.05

# Parameter für unseren neuen Neuronale-Netze-Regressionschätzer
X_train = np.random.uniform(low=-2,high=2,size=(int(n_train),d))
m_X_train, Y_train = gen_data_Y(X_train,sigma)
    
X_test = np.random.uniform(low=-2,high=2,size=(int(n_test),d))
    
Y_pred_new_nn = new_neural_network_estimate(X_train, Y_train, X_test, N, q, R, d, M, a,)
#Y_pred_fc_nn_1 = fc_neural_1_estimate(X_train, Y_train, X_test)
#Y_pred_nearest_neighbor = nearest_neighbor_estimate(X_train, Y_train, X_test)
m_X_test, dummy = gen_data_Y(X_test,sigma)
    

#plt.plot(X_test, Y_pred_nearest_neighbor, '-r', label='nearest_neigbhor')
#plt.plot(X_test, Y_pred_fc_nn_1, '-g', label='fc_nn_1')
#colors = (0,0,0)
area = 4
plt.scatter(X_test, m_X_test, s=area, color = 'blue', label='m_1', alpha=0.5)
plt.scatter(X_test, Y_pred_new_nn, s=area, color = 'red', label='new_neural_network_estimate', alpha=0.5)
plt.title('...')
plt.legend(loc='upper left') 
plt.xlabel('x')
plt.ylabel('y')
#plt.savefig('graph_d_1.png')
tikzplotlib.save("mytikz_d1.tex")
#plt.show()
#plt.plot(X_test, Y_pred_new_nn, 'ro-', label='new_nn')
#plt.plot(X_test, m_X_test, '-b', label='m_d')   

#plt.xlim(-2, 2)
#plt.xlim(-2,2)
#plt.show()


'''
ZEIDIMENSIONALER FALL (d = 2) wird geplottet
'''
N = 2
q = 2
R = 10 ** 6  
a = 2
M = 2
d = 2

sigma = 0.05


# Parameter für unseren neuen Neuronale-Netze-Regressionschätzer
X_train = np.random.uniform(low=-2,high=2,size=(int(n_train),d))
m_X_train, Y_train = gen_data_Y(X_train,sigma)
    
X_test = np.random.uniform(low=-2,high=2,size=(int(n_test),d))
    
Y_pred_new_nn = new_neural_network_estimate(X_train, Y_train, X_test, N, q, R, d, M, a,)
#Y_pred_fc_nn_1 = fc_neural_1_estimate(X_train, Y_train, X_test)
#Y_pred_nearest_neighbor = nearest_neighbor_estimate(X_train, Y_train, X_test)
m_X_test, dummy = gen_data_Y(X_test,sigma)
    


x = np.ravel(X_test[:,0])
y = np.ravel(X_test[:,1])

# so wie es sein soll
#z = m_X_test[:,0]
# was der SChätzer auswirft
z_new = Y_pred_new_nn[:,0]

ax = plt.axes(projection='3d')
ax.scatter(x, y, z_new, c=z_new, cmap='viridis', linewidth=0.5);
ax.view_init(40, 20)
plt.savefig('graph_d_2_new_estimate.png')

# so wie es sein soll
z = m_X_test[:,0]
# was der Schätzer auswirft

ax = plt.axes(projection='3d')
ax.scatter(x, y, z, c=z, cmap='viridis', linewidth=0.5);
ax.view_init(40, 20)
plt.savefig('test.png')
#tikzplotlib.save("mytikz_d2.tex")

postpro = np.asarray([ X_test[:,0], X_test[:,1], Y_pred_new_nn[:,0] ])
np.savetxt("plotpostpro.csv", np.transpose(postpro), delimiter=",")

#plt.savefig('graph_d_2_m_2.png')

'''
ein Vergleich des emp. L2 Fehler gemacht für d = 1
'''
#Parameter für unseren neuen Neuronale-Netze-Regressionschätzer

N = 3
q = 2
R = 10 ** 6 
a = 2
M = 2
d = 1

spreads = [0.05, 0.1]

scaled_error = np.empty((5, 3,))
scaled_error[:] = np.nan

e_L2_avg = np.zeros(5) 
e_L2_avg[:] = np.nan

for sigma in spreads:
    for i in range(0,50,1):
    
        X_train = np.random.uniform(low=-2,high=2,size=(int(n_train),d))
        m_X_train, Y_train = gen_data_Y(X_train,sigma)
        
        X_test = np.random.uniform(low=-2,high=2,size=(int(n_test),d))
        
        #Y_pred_constant = constant_estimate(Y_train)
        Y_pred_new_nn = new_neural_network_estimate(X_train, Y_train, X_test, N, q, R, d, M, a,)
        Y_pred_fc_nn_1 = fc_neural_1_estimate(X_train, Y_train, X_test)
        Y_pred_nearest_neighbor = nearest_neighbor_estimate(X_train, Y_train, X_test)
        
        m_X_test, not_needed = gen_data_Y(X_test,sigma)
        
        e_L2_new_nn = np.mean(abs(Y_pred_new_nn - m_X_test) ** 2)
        e_L2_fc_nn_1 = np.mean(abs(Y_pred_fc_nn_1 - m_X_test) ** 2)
        e_L2_nearest_neighbor = np.mean(abs(Y_pred_nearest_neighbor - m_X_test) ** 2)
        
        for j in range(0,25,1):
            
            X = np.random.uniform(low=-2,high=2,size=(n_test,d))
            m_X, Y = gen_data_Y(X,sigma)
            Y_pred_constant = constant_estimate(Y)
            
            e_L2_avg[j] = np.mean(abs(Y_pred_constant - m_X) ** 2)
        
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
    
    if sigma == 0.05:
        series_noise_1 = pd.Series([repr(sigma)+'%',np.median(e_L2_avg),"(Median, IQR)",(median_new_nn, iqr_new_nn), (median_fc_nn_1, iqr_fc_nn_1), (median_nearest_neighbor, iqr_nearest_neighbor)], index=rows)
        series_noise_1.name = ""
    else:
        series_noise_2 = pd.Series([repr(sigma)+'%',np.median(e_L2_avg),"(Median, IQR)",(median_new_nn, iqr_new_nn), (median_fc_nn_1, iqr_fc_nn_1), (median_nearest_neighbor, iqr_nearest_neighbor)], index=rows)
        series_noise_2.name = ""

error_df = pd.concat([series_noise_1, series_noise_2], axis=1)
#print(error_df)
error_df.to_csv('out_d_1.csv',index = True)

'''
ein Vergleich des emp. L2 Fehler gemacht für d = 2 
'''
# Parameter für unseren neuen Neuronale-Netze-Regressionschätzer

N = 2
q = 2
R = 10 ** 6 
a = 2
M = 2
d = 2

spreads = [0.05,0.1]

scaled_error = np.empty((5, 3,))
scaled_error[:] = np.nan

e_L2_avg = np.zeros(5) 
e_L2_avg[:] = np.nan

for sigma in spreads:
    for i in range(0,50,1):
    
        X_train = np.random.uniform(low=-2,high=2,size=(int(n_train),d))
        m_X_train, Y_train = gen_data_Y(X_train,sigma)
        
        X_test = np.random.uniform(low=-2,high=2,size=(int(n_test),d))
        
        #Y_pred_constant = constant_estimate(Y_train)
        Y_pred_new_nn = new_neural_network_estimate(X_train, Y_train, X_test, N, q, R, d, M, a,)
        Y_pred_fc_nn_1 = fc_neural_1_estimate(X_train, Y_train, X_test)
        Y_pred_nearest_neighbor = nearest_neighbor_estimate(X_train, Y_train, X_test)
        
        m_X_test, not_needed = gen_data_Y(X_test,sigma)
        
        e_L2_new_nn = np.mean(abs(Y_pred_new_nn - m_X_test) ** 2)
        e_L2_fc_nn_1 = np.mean(abs(Y_pred_fc_nn_1 - m_X_test) ** 2)
        e_L2_nearest_neighbor = np.mean(abs(Y_pred_nearest_neighbor - m_X_test) ** 2)
        
        for j in range(0,25,1):
            
            X = np.random.uniform(low=-2,high=2,size=(n_test,d))
            m_X, Y = gen_data_Y(X,sigma)
            Y_pred_constant = constant_estimate(Y)
            
            e_L2_avg[j] = np.mean(abs(Y_pred_constant - m_X) ** 2)
        
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
    
    if sigma == 0.05:
        series_noise_1 = pd.Series([repr(sigma)+'%',np.median(e_L2_avg),"(Median, IQR)",(median_new_nn, iqr_new_nn), (median_fc_nn_1, iqr_fc_nn_1), (median_nearest_neighbor, iqr_nearest_neighbor)], index=rows)
        series_noise_1.name = ""
    else:
        series_noise_2 = pd.Series([repr(sigma)+'%',np.median(e_L2_avg),"(Median, IQR)",(median_new_nn, iqr_new_nn), (median_fc_nn_1, iqr_fc_nn_1), (median_nearest_neighbor, iqr_nearest_neighbor)], index=rows)
        series_noise_2.name = ""

error_df = pd.concat([series_noise_1, series_noise_2], axis=1)
#print(error_df)
error_df.to_csv('out_d_2.csv',index = True)