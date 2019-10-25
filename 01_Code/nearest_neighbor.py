#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 11:14:47 2019

@author: adrian
K-Nearest-Neighbors Algorithm
"""

# Generate sample data
import numpy as np
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV
import warnings 

warnings.simplefilter(action='ignore', category=FutureWarning)

# Implementierung des k-Nächste-Nachbarn-Algorithmus. Dieser bestimmt auch selber bei einer Liste von Anzahlen an Nachbarn die betrachtet werden 
# sollen welches die beste Wahl ist.
#
# X: Inputvektor für das Kalibirieren des Modells 
# Y: Inputvektor für das Kalibirieren des Modells (Zielvektor an den die Gewichte angepasst werden) 
# T: Inputvektor für den eine Vorhersage bestimmte werden soll

def nearest_neighbor_estimate (X_train,Y_train,X_test):
      
    params = {'n_neighbors':[2,3,4,5,6,7,8,9], 'weights': ['uniform', 'distance']}

    knn = neighbors.KNeighborsRegressor()
    
    knn_gridsearch_model = GridSearchCV(knn, params, cv=5)
    knn_gridsearch_model.fit(X_train,Y_train)
    
    return knn_gridsearch_model.predict(X_test)