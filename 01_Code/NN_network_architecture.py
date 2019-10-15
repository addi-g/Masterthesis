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


#
#
# x:
# d:
# j_1_d: 
# k:
# N:
# X_i:
# q:
# s:
# R:
# M:
# a:

for i in range(1,3,1):
    for j in range(1, (2 ** i)+ 1,1):
        print(str(i) + " / " + str(j))
        

def f_net (x, d, j_1_d, k, X_i, N, q, s):
#initialize f_k_l
    f_k_l = np.empty((s, 2 ** s,))
    f_k_l[:] = np.nan

    for k in range(np.sum(j_1_d) + d + 1, (2 ** s) + 1, 1):
        f_k_l[s, k] = 1
        
    for k in range(1, d + 1, 1):
        f_k_l[s,np.sum(j_1_d) + k] = f_hat(x[k],X_i[k,k],R,M,a)
        
    for l in range(1, d + 1, 1):
        for k in range(j_1_d[range(1, l - 1, 1)].sum() + 1, j_1_d[range(1, l, 1)].sum(), 1):
            f_k_l[s,k] = f_id(f_id(x[l] - X_i[l,k],R),R)
    
    for l in range(0, s - 1 + 1):
        for k in range(1, (2 ** l) + 1, 1):
            f_k_l[l,k] = f_mult(f_k_l[l + 1, (2 * k) - 1], f_k_l[l + 1, 2 * k], R) 
                    
    return f_k_l[0,1]



