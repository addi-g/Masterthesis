"""
Implementation eines fully connected neuronalen Netzes mit einer verborgenen Schicht.
"""
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# Fully connected neuronales Netz mit einer verborgenen Schicht welches die 
# Anzahl der Neuronen adaptiv, durch Minimierung des emp. L2-Fehlers, aus der Menge \{5, 10, 25, 50, 75\} waehlt. 
#
# X: Eingabevektor der Form (X_1,...,X_n) fuer das neuronale Netz aus dem Datensatz (X_1,Y_1),...,(X_n,Y_n)
# Y: Eingabevektor der Form (Y_1,...,Y_n) fuer das neuronale Netz aus dem Datensatz (X_1,Y_1),...,(X_n,Y_n)

def fc_neural_1_estimate (X_train,Y_train,X_test):
    
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
    best_n_neurons = n_neurons[(1/len(X_train) *(Diff.sum(axis=0) ** 2)).argmin()]
    
    model = Sequential()
    model.add(Dense(best_n_neurons, input_dim=d, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, Y_train, epochs=1000, verbose=1) 
    
    return model.predict(X_test)