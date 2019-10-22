#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 14:23:15 2019

@author: adrian
"""
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import make_regression
from sklearn.preprocessing import MinMaxScaler


# generate regression dataset
X, y = make_regression(n_samples=100, n_features=2, noise=0.1, random_state=1)
scalarX, scalarY = MinMaxScaler(), MinMaxScaler()
scalarX.fit(X)
scalarY.fit(y.reshape(100,1))
X = scalarX.transform(X)
y = scalarY.transform(y.reshape(100,1))

Xnew, a = make_regression(n_samples=3, n_features=2, noise=0.1, random_state=1)
Xnew = scalarX.transform(Xnew)

# Fully-Connected Neuronales Netzt mit einer Verborgenen schicht welches die 
# Anzahl der Neuronen adaptiv, durch minimierung des L2 fehlers, aus der Menge \{5, 10, 25, 50, 75\} auswählt. 
#
# X: Eingabevektoren der Form (X_1,...,X_n) für das Neuronale Netz aus dem Datensatz (X_1,Y_1),...,(X_n,Y_n)
# Y: Eingabevektoren der Form (Y_1,...,Y_n) für das Neuronale Netz aus dem Datensatz (X_1,Y_1),...,(X_n,Y_n)

def fc_neural_1 (X,Y):
    
    split = int(0.8*np.size(X,0))
    
    X_train = X[:split,:]
    Y_train = Y[:split]
    X_test = X[split:,:]
    
    Ynew = np.empty((len(X_train), len([5,10,25,50,75]),))
    Ynew[:] = np.nan
  
    count = 0
    n_neurons = [5,10,25,50,75]

    d = np.size(X_train, 1)

    for j in n_neurons:
        model = Sequential()
        model.add(Dense(j, input_dim=d, activation='relu'))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mse', optimizer='adam')
        model.fit(X_train, Y_train, epochs=1000, verbose=0)
    
        Ynew[:,count] = model.predict(X_train)[:,0]
        count += 1
    
    Diff = Ynew[:] - Y_train[:]
    best_n_neurons = n_neurons[(1/len(X_train) *(Diff.sum(axis=0) ** 2)).argmax()]
    
    model = Sequential()
    model.add(Dense(best_n_neurons, input_dim=d, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, Y_train, epochs=1000, verbose=0) 
    
    return model.predict(X_test)

prediction_fc_neural_1 = fc_neural_1(X,y)
   
