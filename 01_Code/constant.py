#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 15:26:19 2019

@author: adrian
Constant Estimator
"""
import numpy as np
from scipy import mean

# Gibt den Mittelwert der Funktionswerte einer Funktion als Schätzer zurück
#
# Y: Datensatz der Form (Y_1,...) wobei Y_i \in \R für i = 1,...

def constant_estimate(Y):
    
    split = int(0.8*np.size(Y,0))
    
    m = np.zeros((len(Y),1,)) 
    m[:] = mean(Y)
    return m[split:]