"""
Implementation eines Nächste-Nachbarn-Schätzer
"""

from sklearn import neighbors
from sklearn.model_selection import GridSearchCV
import warnings 

warnings.simplefilter(action='ignore', category=FutureWarning)

# Implementierung des k-Nächste-Nachbarn-Schätzer. Dieser bestimmt auch selber bei einer Liste von Anzahlen an Nachbarn die betrachtet werden 
# sollen welches die beste Wahl ist. Dieser gibt die Schätzung für X_test aus.
#
# X_train: Inputvektor für das Training des Schätzers 
# Y_train: Inputvektor für das Training des Schätzers
# X_test: Inputvektor der geschätzt werden soll

def nearest_neighbor_estimate (X_train,Y_train,X_test):
      
    params = {'n_neighbors':[2,3,4,5,6,7,8,9], 'weights': ['uniform', 'distance']}

    knn = neighbors.KNeighborsRegressor()
    
    knn_gridsearch_model = GridSearchCV(knn, params, cv=5)
    knn_gridsearch_model.fit(X_train,Y_train)
    
    return knn_gridsearch_model.predict(X_test)