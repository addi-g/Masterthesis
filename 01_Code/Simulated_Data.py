#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 12:01:42 2019

@author: adrian
Generate Simulated Data
"""
# Wir wählen X gleichverteilt auf [-1,1]^d, wobei d die dimension des Inputs ist
# n is die Größe der Stichprobe

import numpy as np
from scipy.stats import iqr

log = np.log
pi = np.pi
cos = np.cos
sin = np.sin
exp = np.exp
tan = np.tan

n = 100
d = 4

X = np.random.uniform(low=-1,high=1,size=(n,d))
S = np.random.standard_normal(size=n)


def m_1 (X):
    result = log(0.2 * X[0] + 0.9 * X[1]) + cos(pi/(log(0.5 * X[0] + 0.3 * X[1]))) + exp((1/50) * (0.7 * X[0] + 0.7 * X[1])) +  (tan(pi * (0.1 * X[0] + 0.3 * X[1])**4))/((0.1 * X[0] + 0.3 * X[1])**2)
    return result

def m_2 (X):
    return tan(sin(pi * (0.2 * X[0] + 0.5 * X[1] - 0.6 * X[2] + 0.2 * X[3]))) + (0.5 * (X[0] + X[1] + X[2] + X[3]))**3 + 1/((0.5 * X[0] + 0.3 * X[1] - 0.3 * X[2] + 0.25 * X[3])**2 + 4)

m_1_X = np.zeros(n) 

for i in range(0,n):
    m_1_X[i] = m_2(X[i])
    
"""
IQR bestimmten    
 q75, q25 = np.percentile(m_1_X, [75 ,25])
iqr = q75 - q25
"""