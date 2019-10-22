#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 15:40:14 2019

@author: adrian

Um die Gewichte der Ausgabeschicht zu bestimmen lösen wir ein regularisiertes 
Kleinste-Quadrate Problem.

"""
import scipy.special
import numpy as np
import itertools
from NN_helpfunc import f_id, f_mult, f_hat 
from sklearn.datasets import make_regression
from sklearn.preprocessing import MinMaxScaler
import math

X, y = make_regression(n_samples=100, n_features=2, noise=0.1, random_state=1)
scalarX, scalarY = MinMaxScaler(), MinMaxScaler()
scalarX.fit(X)
scalarY.fit(y.reshape(100,1))
X = scalarX.transform(X)
y = scalarY.transform(y.reshape(100,1))




# Neuronales Netz welches die Funktion f(x) = (x^(1)- x_ik^(1))^j1 * ... * 
# (x^(d) - x_ik^(d))^jd * \prod_{j = 1}^d max((1 - (M/2a) * abs(x^(j) - x_ik^(j))),0)
#
# x: Eingabevektor für das Neuronale Netz x \in [-a,a]^d
# d: Ist die Dimension des Eingabevektors d > 0
# j_1_d: Ist ein d-dimensionaler Vektor j_1,...,j_d \in {0,1,...,N}
# k: k \in {1,...,(M+1)^d}
# X_i: Ist eine  d x (M+1)^d  Matrix. 
# N: Natürliche Zahl >= q
# q: 
# s: [log_2(N + d)]
# R: Zahl >= 1
# M: M \in N
# a: > 0         

def f_net (x, d, j_1_d, k, X_i, N, q, s, R, M, a):
#initialize f_l_k
    f_l_k = np.empty((s + 1, (1 + M) ** d,))
    f_l_k[:] = np.nan

    # k läuft hier ab j_1+....+j_d + d los da unsere Matrix ab Index 0 die Werte einträgt
    # daher wird hier das "+ 1" in der Summe weggelassen da unser Index früher starte.
    # hier k \in \{0,...,(M + 1)^d - 1|} und im Paper k \in \{1,...,(M + 1)^d|}
    # Mit der "range"-Funktion wird der Index bei dem gestoppt werden soll nicht angenommen
    
    for k in range(np.sum(j_1_d) + d, 2 ** s, 1):
        f_l_k[s, k] = 1
        
    for k in range(0, d, 1):
        f_l_k[s, np.sum(j_1_d) + k] = f_hat(x[k], X_i[k], R, M, a)
     
    for l in range(0, d, 1):
        k = j_1_d[range(0, l, 1)].sum()
        while k in range(j_1_d[range(0, l, 1)].sum(), j_1_d[range(0, l + 1, 1)].sum() + 1, 1):
            f_l_k[s, k] = f_id(f_id(x[l] - X_i[l], R), R)
            k += 1
            
    for l in range(s - 1, -1, -1):
        for k in range((2 ** l) - 1, -1, -1):
            # k = 0 separat betrachten da es sonst probleme mit dem Index 2 * k - 1 gibt
            if k == 0:
                f_l_k[l, k] = f_mult(f_l_k[l + 1, 0], f_l_k[l + 1, 1], R)
            else:
                f_l_k[l, k] = f_mult(f_l_k[l + 1, (2 * k) - 1], f_l_k[l + 1, 2 * k], R)
    return f_mult(f_l_k[0, 0],f_l_k[0, 0],R)

# Bestimmung der Gewichte der Ausgabeschicht durch lösen eines regularisierten
# Kleineste-Quadrate Problems
#
# X: Eingabevektoren der Form (X_1,...,X_n) für das Neuronale Netz aus dem Datensatz (X_1,Y_1),...,(X_n,Y_n)
# Y: Eingabevektoren der Form (Y_1,...,Y_n) für das Neuronale Netz aus dem Datensatz (X_1,Y_1),...,(X_n,Y_n)
# N: Natürliche Zahl >= q
# q: 
# R: Zahl >= 1
# d: Ist die Dimension des Eingabevektors d > 0
# M: M \in \N
# a: >0

def output_weights(X, Y, N, q, R, d, M, a):

    s = math.ceil(math.log2(N + d))
    
    # Anzahl der Eingabevektoren X_1,...,X_n
    
    n = np.size(X, 0)
    
    # Eine beliebige constante > 0
    
    c_3 = np.random.randint(1,10)
    
    # Anzahl der Spalten der Matrix für das Kleinste-Quadrate Problem
    # In den Spalten sind die Funktionswerte von f_net eingespeichert
    
    J = int(((1 + M) ** d) * scipy.special.binom(N + d, d))
    
    # Für die Konstruktion der Matrix brauchen wir erstmal alle Inputparameter
    # für f_net, da wir dort nur den Funktionswert für einen Vektor j_1,...,j_d einsetzen
    # müssen wir erstmals alle möglichen Vektoren dieser Art konstruieren die die Bedingung 0 <= j_1 + ... + j_d <= N erfüllen
    # X_ik hat in den Zeilen die Vektoren X_i aus dem Paper
    
    X_ik = np.transpose(np.empty((d, (1 + M) ** d,)))
    X_ik[:] = np.nan
    
    I_k = np.array(list(itertools.product(range(0, M + 1), repeat = d)))
    X_ik[:] = (I_k[:] * ((2 * a) / M)) - a
    
    all_j1_jd = np.array(list(itertools.product(range(0, N + 1), repeat = d)))
    all_j1_jd_by_cond = all_j1_jd[all_j1_jd.sum(axis=1) <= N]
    
    B = np.empty((n, J,))
    B[:] = np.nan
    
    for i in range(0, n):
        j = 0
        for k in range(0, ((M + 1) ** d)):
            for z in range(0, int(scipy.special.binom(N + d, d))):
                B[i,j] = f_net(X[i], d, all_j1_jd_by_cond[z], k, X_ik[k], N, q, s, R, M, a)
                j += 1
    
    weights = np.linalg.solve((1 / n) * np.dot(np.transpose(B),B) + (c_3 / n) * np.identity(J), (1 / n) * np.dot(np.transpose(B),Y))
    return (weights, J, all_j1_jd_by_cond, X_ik)

# Bestimmung des Funktionswert des Neuronale-Netzte-Regressionsschätzers
# 
# x: Eingabe für einen Vektor der Form [-a,a]^d für den eine Schätzung bestimmt werden soll
# X: Eingabevektoren der Form (X_1,...,X_n) für das Neuronale Netz aus dem Datensatz (X_1,Y_1),...,(X_n,Y_n)
# Y: Eingabevektoren der Form (Y_1,...,Y_n) für das Neuronale Netz aus dem Datensatz (X_1,Y_1),...,(X_n,Y_n)
# N: Natürliche Zahl >= q
# q: 
# s: [log_2(N + d)]
# R: Zahl >= 1
# d: Ist die Dimension des Eingabevektors d > 0
# M: M \in \N
# a: >0

def neural_network_estimate(x, X, Y, N, q, R, d, M, a):
    
    s = math.ceil(math.log2(N + d))
    
    weights, J, all_j1_jd_by_cond, X_ik = output_weights(X, Y, N, q, R, d, M, a)

    F_net = np.empty((1, J,))
    F_net[:] = np.nan
    
    
    j = 0
    while j < J:
        for k in range(0, ((M + 1) ** d)):
            for z in range(0, int(scipy.special.binom(N + d, d))):
                F_net[0,j] = f_net(x, d, all_j1_jd_by_cond[z], k, X_ik[k], N, q, s, R, M, a)
                j += 1 
                 
    return np.sum(np.transpose(weights) * F_net)

N = 2
q = 1
R = 1
d = 2
a = 1
M = 3

prediction_new_neuronal_network = np.empty((len(X), 1,))
prediction_new_neuronal_network[:] = np.nan


prediction_new_neuronal_network[0] = neural_network_estimate(X[3],X,y,N,q,R,d,M,a)



    
    

    
    
    
    
    
    
    
