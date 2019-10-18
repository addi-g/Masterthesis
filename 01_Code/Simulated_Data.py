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
import pandas as pd
from scipy.stats import iqr
from random import choice

log = np.log
pi = np.pi
cos = np.cos
sin = np.sin
exp = np.exp
tan = np.tan

n = 100
d = 5

X = np.random.uniform(low=-1,high=1,size=(n,d))
S = np.random.standard_normal(size=(n,1))

# Regressionsfunktionen
#
# x: Ein Vektor x \in [-1,-1]^d
# d: Dimension des Vektors x

def m_d (x, d):
    
    if d == 1:
        return log(0.2 * x[0] + 0.9 * x[1]) + cos(pi / (log(0.5 * x[0] + 0.3 * x[1]))) + exp((1/50) * (0.7 * x[0] + 0.7 * x[1])) +  (tan(pi * (0.1 * x[0] + 0.3 * x[1]) ** 4))/((0.1 * x[0] + 0.3 * x[1]) ** 2)
    
    elif d == 2:
        return tan(sin(pi * (0.2 * x[0] + 0.5 * x[1] - 0.6 * x[2] + 0.2 * x[3]))) + (0.5 * (x[0] + x[1] + x[2] + x[3])) ** 3 + 1 / (((0.5 * x[0] + 0.3 * x[1] - 0.3 * x[2] + 0.25 * x[3]) ** 2) + 4)
    
    elif d == 3:
        return log(0.5 * (x[0] + 0.3 * x[1] + 0.6 * x[2] + x[3] - x[4]) ** 2) + sin(pi * (0.7 * x[0] + x[1] - 0.3 * x[2] - 0.4 * x[3] - 0.8 * x[4])) + cos(pi / (1 + sin(0.5 * (x[1] + 0.9 * x[2] - x[4]))))

# Generiert den Vektor Y_1,...,Y_n für den Datensatz (X_1,Y_1),...,(X_n,Y_n)
#
# X: Inputdaten der Form (X_1,...,X_n), wobei X_i \in [-1,-1]^d für i = 1,...,n
# sigma: Schwankung in den Werten (Noise) \in \{0.05,0.1\}
        
def gen_data_Y (X, sigma):

    n = X.size
    d = X[0].size
    
    m_X = np.zeros((n,1,)) 
    m_X[:] = np.nan

    for t in range(0,n):
        m_X[t] = m_d(X[t], d)
        
    Y = m_X + sigma * iqr(m_X[t]) * S
    
    return Y 

