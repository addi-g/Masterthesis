#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 11:22:02 2019

@author: adrian

Implementation von Neuronalen-Netzen welche wir für die Konstruktion unseres 
Neuronale-Netze-Regressionschätzers benötigen
"""
import numpy as np

# Sigmoidfunktion
#
# x: x \in \R 

def sigmoid (x):
    
    return 1 / (1 + np.exp(-x))

# Neuronales Netz welches die Funktion f(x) = x approximiert
#
# x: reelle Zahl
# R: reelle Zahl >= 1
    
def f_id (x, R):
    
    return 4 * R * sigmoid(x / R) - 2 * R 

# Neuronales Netz welches die Funktion f(x, y) = x * y approximiert
#
# x: reelle Zahl
# y: reelle Zahl    
# R: reelle Zahl >= 1
    
def f_mult (x, y, R):
    
    return (((R ** 2) / 4) * (((1 + np.exp(-1)) ** 3) / (np.exp(-2) - np.exp(-1)))) \
    * (sigmoid(((2 * (x + y)) / R) + 1) - 2 * sigmoid(((x + y) / R) + 1) \
    - sigmoid(((2 * (x - y)) / R) + 1) + 2 * sigmoid(((x - y) / R) + 1)) 
    
# Neuronales Netz welches die Funktion f(x) = max(x,0) approximiert
#
# x: reelle Zahl   
# R: reelle Zahl >= 1
    
def f_relu (x, R):
    
    return f_mult(f_id(x, R), sigmoid(R * x),R)

# Neuronales Netz welches die Funktion f(x) = max(1 - (M/(2 * a)) * abs(x - y),0) approximiert
#
# x: reelle Zahl 
# y: fixe reelle Zahl
# R: reelle Zahl >= 1
# M: fixe natürliche Zahl
# a: fixe Zahl > 0
    
def f_hat (x, y, R, M, a):
    
    return f_relu((M / (2 * a)) * (x - y) + 1, R) - 2 * f_relu((M / (2 * a)) * (x - y), R) + 2 * f_relu((M / (2 * a)) * (x - y) - 1, R)
    