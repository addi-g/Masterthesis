"""
Parameterstudie fuer den eindimensionalen Fall
"""
import numpy as np
import matplotlib . pyplot as plt
from scipy.stats import iqr
import tikzplotlib
from new_neural_network import new_neural_network_estimate

# Regressionsfunktionen
#
# x: Ein Vektor x der Dimension d
# d: Dimension des Vektors x

def m (x,d):
    
    sin = np.sin
    pi = np.pi  
    
    return sin(pi/2 * x ** 2)

# Generiert den Vektor Y_1,...,Y_n fuer den Datensatz (X_1,Y_1),...,(X_n,Y_n)
#   
# X: Inputdaten der Form (X_1,...,X_n), wobei X_i \in \R^d fuer i = 1,...,n
# sigma: Streuungsfaktor \in \{0.05,0.1\}
        
def gen_data_Y (X, sigma):
    
    n = np.size(X, 0)
    d = np.size(X, 1)
    
    m_X = np.zeros((n,1,)) 
    m_X[:] = np.nan
    
    S = np.random.standard_normal(size=(n,1)) 
    for t in range(0,n):
        m_X[t] = m(X[t], d)

    Y = m_X + sigma * iqr(m_X) * S
    return (m_X, Y)  

n = 1000
n_train = int(n * 0.8)
n_test = int(n * 0.2)

# Parametersets
# Vergleich 1
#N, q, R, M = 2, 2, 10 ** 6, 2
#N, q, R, M = 4, 2, 10 ** 6, 2
#N, q, R, M = 8, 2, 10 ** 6, 2
#N, q, R, M = 16, 2, 10 ** 6, 2

# Vergleich 2
#N, q, R, M = 2, 2, 10 ** 6, 2
#N, q, R, M = 2, 2, 10 ** 6, 4
#N, q, R, M = 2, 2, 10 ** 6, 8
#N, q, R, M = 2, 2, 10 ** 6, 16

# Vergleich 3
#N, q, R, M = 16, 2, 10 ** 6, 2
#N, q, R, M = 16, 2, 10 ** 6, 4
#N, q, R, M = 16, 2, 10 ** 6, 8
#N, q, R, M = 16, 2, 10 ** 6, 16

# Vergleich 4
#N, q, R, M = 2, 2, 10 ** 6, 16
#N, q, R, M = 4, 2, 10 ** 6, 16
#N, q, R, M = 8, 2, 10 ** 6, 16
#N, q, R, M = 16, 2, 10 ** 6, 16

# Vergleich 5
#N, q, R, M = 4, 2, 10 ** 6, 9
N, q, R, M = 9, 2, 10 ** 6, 4

a = 3
d = 1
sigma = 0.05

np.random.seed(1)
X_train = np.random.uniform(low=-a,high=a,size=(int(n_train),d))
m_X_train, Y_train = gen_data_Y(X_train,sigma)
 
X_test = np.random.uniform(low=-a,high=a,size=(int(n_test),d))
    
Y_pred_new_nn = new_neural_network_estimate(X_train, Y_train, X_test, N, q, R, d, M, a,)
m_X_test, dummy = gen_data_Y(X_test,sigma)
    

gridpoints = np.arange(-a,a,0.01)
plt.plot(gridpoints,m(gridpoints,d),color='black')
plt.scatter(X_test, Y_pred_new_nn, color = 'red', alpha=1, marker= "x")
plt.title('N = '+str(N)+', '+'M = '+str(M))
tikzplotlib.save("mytikz_N"+str(N)+"_M"+str(M)+".tex")