"""
Implementation des konstanten Schätzers.
"""
import numpy as np
from scipy import mean

# Gibt den Mittelwert der Funktionswerte einer Funktion als Schätzung zurück
#
# Y: Datensatz der Form (Y_1,...) wobei Y_i \in \R für i = 1,...

def constant_estimate(Y):    
    m = np.zeros((len(Y),1,)) 
    m[:] = mean(Y)
    return m