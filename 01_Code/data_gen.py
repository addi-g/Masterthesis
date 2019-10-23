#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 12:01:42 2019

@author: adrian

Generieren der Daten die wir für einen Vergleich von Regressionsschätzern benötigen
"""
# Wir wählen x gleichverteilt auf [-1,1]^d, wobei d die dimension des Inputs ist
# n is die Größe der Stichprobe

import numpy as np
from scipy.stats import iqr

n = 100
d = 4

X = np.random.uniform(low=-1,high=1,size=(n,d))

# Regressionsfunktionen
#
# x: Ein Vektor x \in [-1,-1]^d
# d: Dimension des Vektors x

def m_d (x, d):
    
    log = np.log
    pi = np.pi
    cos = np.cos
    sin = np.sin
    exp = np.exp
    tan = np.tan    
    
    if d == 2:
        return log(0.2 * x[0] + 0.9 * x[1]) + cos(pi / (log(0.5 * x[0] + 0.3 * x[1]))) + exp((1/50) * (0.7 * x[0] + 0.7 * x[1])) +  (tan(pi * (0.1 * x[0] + 0.3 * x[1]) ** 4))/((0.1 * x[0] + 0.3 * x[1]) ** 2)
    
    elif d == 4:
        return tan(sin(pi * (0.2 * x[0] + 0.5 * x[1] - 0.6 * x[2] + 0.2 * x[3]))) + (0.5 * (x[0] + x[1] + x[2] + x[3])) ** 3 + 1 / (((0.5 * x[0] + 0.3 * x[1] - 0.3 * x[2] + 0.25 * x[3]) ** 2) + 4)
    
    elif d == 5:
        return log(0.5 * (x[0] + 0.3 * x[1] + 0.6 * x[2] + x[3] - x[4]) ** 2) + sin(pi * (0.7 * x[0] + x[1] - 0.3 * x[2] - 0.4 * x[3] - 0.8 * x[4])) + cos(pi / (1 + sin(0.5 * (x[1] + 0.9 * x[2] - x[4]))))
    else:
        print("Your data has the wrong dimension!")

# Generiert den Vektor Y_1,...,Y_n für den Datensatz (X_1,Y_1),...,(X_n,Y_n)
#   
# X: Inputdaten der Form (X_1,...,X_n), wobei X_i \in [-1,-1]^d für i = 1,...,n
# sigma: Schwankung in den Werten (Noise) \in \{0.05,0.1\}
        
def gen_data_Y (X, sigma):
    
    n = np.size(X, 0)
    d = np.size(X, 1)
    
    m_X = np.zeros((n,1,)) 
    m_X[:] = np.nan
    
    S = np.random.standard_normal(size=(n,1)) 
    for t in range(0,n):
        m_X[t] = m_d(X[t], d)

    return m_X + sigma * iqr(m_X) * S