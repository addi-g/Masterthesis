#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 13:22:49 2019

@author: adrian

Implementation des Neuronalen-Netze-Regressionsschätzers

Erst wird die Netzwerk Architektur bestimmt. Die Gewichte, außer die in der 
Ausgabeschicht werden fix gewählt.

Die Gewichte in der Ausgabeschit ermitteln wird mit Hilfe der Inputdaten in dem
wir ein regularisiertes Kleinste-Quadrate Problem lösen.
"""
import numpy as np
from NN_helpfunc import f_id, f_mult, f_hat 

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
        f_l_k[s, np.sum(j_1_d) + k] = f_hat(x[k], X_i[k,k], R, M, a)
        
    for l in range(1, d + 1, 1):
        for k in range(j_1_d[range(0, l - 1, 1)].sum(), j_1_d[range(0, l, 1)].sum(), 1):
            f_l_k[s, k] = f_id(f_id(x[l] - X_i[l,k], R), R)
    
    for l in range(0, s):
        for k in range(0, 2 ** l, 1):
            # k = 0 separat betrachten da es sonst probleme mit dem Index 2 * k - 1 gibt
            if k == 0:
                f_l_k[l, k] = f_mult(f_l_k[l + 1, 0], f_l_k[l + 1, 1], R)
            else:
                f_l_k[l, k] = f_mult(f_l_k[l + 1, (2 * k) - 1], f_l_k[l + 1, 2 * k], R) 
              
    return f_l_k[0,0]



