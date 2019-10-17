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



# Bestimmung der Gewichte der Ausgabeschicht
#
# x:
# N:
# d:
# M:
#
#



def output_weights(x, N, d, M, a):

    # Anzahl der Spalten der Matrix für das Kleinste-Quadrate Problem
    # In den Spalten sind die Funktionswerte von f_net eingespeichert
    J = ((1 + M) ** d) * scipy.special.binom(N + d, d)
    
    # Für die Konstruktion der Matrix brauchen wir erstmal alle Inputparameter
    # für f_net, da wir dort nur den Funktionswert für einen Vektor j_1,...,j_d einsetzen
    # müssen wir erstmals alle möglichen Vektoren dieser Art konstruieren die die Bedingung 0 <= j_1 + ... + j_d <= N erfüllen
    X_i = np.transpose(np.empty((d, (1 + M) ** d,)))
    X_i[:] = np.nan
    
    I_k = np.array(list(itertools.product(range(0, M + 1), repeat = d)))
    X_i[:] = (I_k[:] * ((2 * a) / M)) - a
    
    all_j1_jd = np.array(list(itertools.product(range(0, N + 1), repeat = d)))
    all_j1_jd_by_cond = all_j1_jd[all_j1_jd.sum(axis=1) <= N]
    
    B = np.empty((n, J,))
    B[:] = np.nan
    
    for i in range(0, n):
        j = 0
        for k in range(0, ((M + 1) ** d)):
            for z in range(0, scipy.special.binom(N + d, d)):
                B[i,j] = f_net(x[i], d, all_j1_jd_by_cond[z], k, X_i[k], N, q, s, R, M, a)
                j += 1 
    
    weights = np.linalg.solve((1 / n) * np.dot(np.transpose(B),B) + (c_3 / n) * np.identity(J), (1 / n) * np.dot(np.transpose(B),Y))
    return weights
    
    
    
    
