#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 11:14:47 2019

@author: adrian
K-Nearest-Neighbors Algorithm
"""

# Generate sample data
import numpy as np
from sklear import neighbors
from sklearn.model_selection import GridSearchCV

np.random.seed(1)
n = 100
d = 2

X = np.random.uniform(low=-1,high=1,size=(n,d))

T = np.random.uniform(low=-1,high=1,size=(10*n,d))

y = 2 * X



#find k

#x = np.array([range(4,4 * math.floor(nl) + 1,4)])
# z = np.append(y,x) um bei params so wie im papaer die nachbarn auszuwählen

params = {'n_neighbors':[2,3,4,5,6,7,8,9], 'weights': ['uniform', 'distance']}

knn = neighbors.KNeighborsRegressor()

knn_gridsearch_model = GridSearchCV(knn, params, cv=5)
knn_gridsearch_model.fit(X,y)
y_ = knn_gridsearch_model.predict(X)

# Implementierung des k-Nächste-Nachbarn-Algorithmus. Dieser bestimmt auch selber bei einer Liste von Anzahlen an Nachbarn die betrachtet werden 
# sollen welches die beste Wahl ist.
#
# X: Inputvektor für das Kalibirieren des Modells 
# Y: Inputvektor für das Kalibirieren des Modells (Zielvektor an den die Gewichte angepasst werden) 
# T: Inputvektor für den eine Vorhersage bestimmte werden soll

def k_nearest_neighbor (X,Y,T,k):

