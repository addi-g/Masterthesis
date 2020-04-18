"""
Main Datei die die Simulation und damit den Vergleich der implementierten Schaetzer
durchfuehrt.
"""
import numpy as np
import pandas as pd
from scipy.stats import iqr
from data_gen import gen_data_Y
from constant import constant_estimate
from new_neural_network import new_neural_network_estimate
from nearest_neighbor import nearest_neighbor_estimate
from fc_neural_network import fc_neural_1_estimate 

n = 1000
n_train = int(n * 0.8)
n_test = int(n * 0.2)

'''
ein Vergleich des emp. L2 Fehler gemacht fuer d = 1
'''
# Wahl der Parameter fuer unseren neuen Neuronale-Netze-Regressionschaetzer

N = 3
q = 2
R = 10 ** 6 
a = 2
M = 2
d = 1

spreads = [0.05, 0.1]

scaled_error = np.empty((50, 3,))
scaled_error[:] = np.nan

e_L2_avg = np.zeros(25) 
e_L2_avg[:] = np.nan

for sigma in spreads:
    for i in range(0,np.size(scaled_error,0),1):
    
        X_train = np.random.uniform(low=-a,high=a,size=(int(n_train),d))
        m_X_train, Y_train = gen_data_Y(X_train,sigma)
        
        X_test = np.random.uniform(low=-a,high=a,size=(int(n_test),d))
        
        Y_pred_new_nn = new_neural_network_estimate(X_train, Y_train, X_test, N, q, R, d, M, a,)
        Y_pred_fc_nn_1 = fc_neural_1_estimate(X_train, Y_train, X_test)
        Y_pred_nearest_neighbor = nearest_neighbor_estimate(X_train, Y_train, X_test)
        
        m_X_test, not_needed = gen_data_Y(X_test,sigma)
        
        e_L2_new_nn = np.mean(abs(Y_pred_new_nn - m_X_test) ** 2)
        e_L2_fc_nn_1 = np.mean(abs(Y_pred_fc_nn_1 - m_X_test) ** 2)
        e_L2_nearest_neighbor = np.mean(abs(Y_pred_nearest_neighbor - m_X_test) ** 2)
        
        for j in range(0,np.size(e_L2_avg),1):
            
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
        print("Der empirische L2 Fehler fuer d = 1 und sigma = 0.05 ist berechnet worden!")
        
    else:
        series_noise_2 = pd.Series([repr(sigma),np.median(e_L2_avg),"(Median, IQR)",(median_new_nn, iqr_new_nn), (median_fc_nn_1, iqr_fc_nn_1), (median_nearest_neighbor, iqr_nearest_neighbor)], index=rows)
        series_noise_2.name = ""
        print("Der empirische L2 Fehler fuer d = 1 und sigma = 0.1 ist berechnet worden!")


error_df = pd.concat([series_noise_1, series_noise_2], axis=1)
error_df.to_csv('out_d_1.csv',index = True)

'''
ein Vergleich des emp. L2 Fehler gemacht fuer d = 2
'''
# Wahl der Parameter fuer unseren neuen Neuronale-Netze-Regressionschaetzer

N = 2
q = 2
R = 10 ** 6 
a = 2
M = 2
d = 2

spreads = [0.05,0.1]

scaled_error = np.empty((50, 3,))
scaled_error[:] = np.nan

e_L2_avg = np.zeros(25) 
e_L2_avg[:] = np.nan

for sigma in spreads:
    for i in range(0,np.size(scaled_error,0),1):
    
        X_train = np.random.uniform(low=-a,high=a,size=(int(n_train),d))
        m_X_train, Y_train = gen_data_Y(X_train,sigma)
        
        X_test = np.random.uniform(low=-a,high=a,size=(int(n_test),d))
        
        Y_pred_new_nn = new_neural_network_estimate(X_train, Y_train, X_test, N, q, R, d, M, a,)
        Y_pred_fc_nn_1 = fc_neural_1_estimate(X_train, Y_train, X_test)
        Y_pred_nearest_neighbor = nearest_neighbor_estimate(X_train, Y_train, X_test)
        
        m_X_test, not_needed = gen_data_Y(X_test,sigma)
        
        e_L2_new_nn = np.mean(abs(Y_pred_new_nn - m_X_test) ** 2)
        e_L2_fc_nn_1 = np.mean(abs(Y_pred_fc_nn_1 - m_X_test) ** 2)
        e_L2_nearest_neighbor = np.mean(abs(Y_pred_nearest_neighbor - m_X_test) ** 2)
        
        for j in range(0,np.size(e_L2_avg),1):
            
            X = np.random.uniform(low=-a,high=a,size=(n_test,d))
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
        series_noise_1 = pd.Series([repr(sigma),np.median(e_L2_avg),"(Median, IQR)",(median_new_nn, iqr_new_nn), (median_fc_nn_1, iqr_fc_nn_1), (median_nearest_neighbor, iqr_nearest_neighbor)], index=rows)
        series_noise_1.name = ""
        print("Der empirische L2 Fehler fuer d = 2 und sigma = 0.05 ist berechnet worden!")

    else:
        series_noise_2 = pd.Series([repr(sigma)+'%',np.median(e_L2_avg),"(Median, IQR)",(median_new_nn, iqr_new_nn), (median_fc_nn_1, iqr_fc_nn_1), (median_nearest_neighbor, iqr_nearest_neighbor)], index=rows)
        series_noise_2.name = ""
        print("Der empirische L2 Fehler fuer d = 2 und sigma = 0.1 ist berechnet worden!")

error_df = pd.concat([series_noise_1, series_noise_2], axis=1)
error_df.to_csv('out_d_2.csv',index = True)
