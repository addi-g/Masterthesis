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
from NN_network_architecture import f_net

# Bestimmung der Gewichte der Ausgabeschicht durch lösen eines regularisierten
# Kleineste-Quadrate Problems
#
# X: Eingabevektoren der Form (X_1,...,X_n) für das Neuronale Netz aus dem Datensatz (X_1,Y_1),...,(X_n,Y_n)
# Y: Eingabevektoren der Form (Y_1,...,Y_n) für das Neuronale Netz aus dem Datensatz (X_1,Y_1),...,(X_n,Y_n)
# N: Natürliche Zahl >= q
# q: 
# s: [log_2(N + d)]
# R: Zahl >= 1
# d: Ist die Dimension des Eingabevektors d > 0
# M: M \in \N
# a: >0

def output_weights(X, Y, N, q, s, R, d, M, a):

    # Anzahl der Eingabevektoren X_1,...,X_n
    
    n = np.size(X, 0)
    
    # Eine beliebige constante > 0
    
    c_3 = np.random.randint(1,10)
    
    # Anzahl der Spalten der Matrix für das Kleinste-Quadrate Problem
    # In den Spalten sind die Funktionswerte von f_net eingespeichert
    
    J = ((1 + M) ** d) * scipy.special.binom(N + d, d)
    
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
            for z in range(0, scipy.special.binom(N + d, d)):
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

def neural_network_estimate(x, X, Y, N, q, s, R, d, M, a):
    
    a, J, all_j1_jd_by_cond, X_ik = output_weights(X, Y, N, q, s, R, d, M, a)

    F_net = np.empty((1, J,))
    F_net[:] = np.nan
    
    
    j = 0
    while j < J:
        for k in range(0, ((M + 1) ** d)):
            for z in range(0, scipy.special.binom(N + d, d)):
                F_net[0,j] = f_net(x, d, all_j1_jd_by_cond[z], k, X_ik[k], N, q, s, R, M, a)
                j += 1 
                 
    return np.sum(a * F_net)
    
    

    
    
    
    
    
    
    
