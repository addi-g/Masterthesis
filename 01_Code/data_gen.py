#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 12:01:42 2019

@author: adrian

Generieren der Daten die wir für einen Vergleich von Regressionsschätzern benötigen
"""
# Wir wählen x gleichverteilt auf [-2,2]^d, wobei d die dimension des Inputs ist
# n is die Größe der Stichprobe

import numpy as np
from scipy.stats import iqr

# Regressionsfunktionen
#
# x: Ein Vektor x der Dimension d
# d: Dimension des Vektors x

def m_d (x, d):
    
    pi = np.pi
    cos = np.cos
    sin = np.sin
    exp = np.exp  
    
    if d == 1:
        return sin(0.2 * x[0] ** 2) + exp(0.5 * x[0]) + x[0] ** 3
                   
    elif d == 2:
        return np.sin(np.sqrt(x[0] ** 2 + x[1] ** 2))
    
    else:
        print("Your data has the wrong dimension!")
        
def error_limit (x, p, c, d):
    
        return c * (np.log(x) ** 3) * (x ** (-(2 * p)/(2 * p + d)))

# Generiert den Vektor Y_1,...,Y_n für den Datensatz (X_1,Y_1),...,(X_n,Y_n)
#   
# X: Inputdaten der Form (X_1,...,X_n), wobei X_i \in \R^d für i = 1,...,n
# sigma: Schwankung in den Werten (Noise) \in \{0.05,0.1\}
        
def gen_data_Y (X, sigma):
    
    n = np.size(X, 0)
    d = np.size(X, 1)
    
    m_X = np.zeros((n,1,)) 
    m_X[:] = np.nan
    
    S = np.random.standard_normal(size=(n,1)) 
    for t in range(0,n):
        m_X[t] = m_d(X[t], d)

    Y = m_X + sigma * iqr(m_X) * S
    return (m_X, Y)


    