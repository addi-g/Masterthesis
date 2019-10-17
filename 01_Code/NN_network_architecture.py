#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 13:22:49 2019

@author: adrian

Implementation des Neuronalen-Netze-Regressionschätzers

Erst wird die Netzwerk Architektur bestimmt. Die Gewichte, außer die in der 
Ausgabeschicht werden fix gewählt.

Die Gewichte in der Ausgabeschit ermitteln wird mit Hilfe der Inputdaten in dem
wir ein regularisiertes Kleinste-Quadrate Problem lösen.
"""
import numpy as np
from NN_helpfunc import f_id, f_mult, f_relu, f_hat 
import itertools

# 
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
#initialize f_k_l
    f_k_l = np.empty((s + 1, (1 + M) ** d,))
    f_k_l[:] = np.nan
    
    X_i = np.transpose(np.empty((d, (1 + M) ** d,)))
    X_i[:] = np.nan
    
    I_k = np.array(list(itertools.product(range(0, M + 1), repeat = d)))
    X_i[:] = (I_k[:] * ((2 * a) / M)) - a

    for k in range(np.sum(j_1_d) + d, 2 ** s, 1):
        f_k_l[s, k] = 1
        
    for k in range(0, d, 1):
        f_k_l[s, np.sum(j_1_d) + k] = f_hat(x[k], X_i[k,k], R, M, a)
        
    for l in range(1, d + 1, 1):
        for k in range(j_1_d[range(0, l - 1, 1)].sum(), j_1_d[range(0, l, 1)].sum(), 1):
            f_k_l[s, k] = f_id(f_id(x[l] - X_i[l,k], R), R)
    
    for l in range(0, s):
        for k in range(0, 2 ** l, 1):
            f_k_l[l, k] = f_mult(f_k_l[l + 1, (2 * k) - 1], f_k_l[l + 1, 2 * k], R) 
              
    return f_k_l[0,0]



