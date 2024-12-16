from cvxopt import solvers, matrix
import numpy as np
np.random.seed(42)

def initialize_experiment():
    # Problem dimensions
    w_A1 = 300
    h_A1 = 1000
    w_A2 = 300
    h_A2 = 1000
    assert h_A1 == h_A2
    m, n = w_A1 + w_A2, h_A1  # A's shape
    
    # 1. Generate A1 and A2
    A1_mean = np.random.randint(1, 11)  
    A1_var = np.random.randint(1, 11)   
    A1 = np.random.normal(A1_mean, A1_var, size=(w_A1, h_A1))

    A2_mean = np.random.randint(1, 11)  
    A2_var = np.random.randint(1, 11)   
    A2 = np.random.normal(A2_mean, A2_var, size=(w_A2, h_A2))
    
    # 2. Generate b and g
    b1_mean = np.random.randint(-100, 101)  
    b1_var = np.random.randint(1, 101)      
    b1 = np.random.normal(b1_mean, b1_var, size=(w_A1, ))

    b2_mean = np.random.randint(-100, 101)  
    b2_var = np.random.randint(1, 101)      
    b2 = np.random.normal(b2_mean, b2_var, size=(w_A2, ))
    
    g_mean = np.random.randint(-100, 101)
    g_var = np.random.randint(1, 101)
    g = np.random.normal(g_mean, g_var, size=(n, ))
    
    # 3. Generate H = 0.1I + LL^T
    L = np.random.normal(1, np.sqrt(2), size=(n, n))
    H = 0.1 * np.eye(n) + L @ L.T
    
    return matrix(A1), matrix(b1), matrix(A2), matrix(b2), matrix(g), matrix(H)

A1, b1, A2, b2, g, H = initialize_experiment()

# H = matrix([[4., 1.], [1., 2.]])
# g = matrix([1., 1.])
# A2 = matrix([[-1., 0.], [0., -1.]])
# b2 = matrix([0., 0.])
# A1 = matrix([1., 1.], (1, 2))
# b1 = matrix([1.])
'''
Optimal solution found.
[ 2.50e-01]
[ 7.50e-01]
'''

# Cvxopt.solvers.qp(P, q, G, h, A, b)
'''
min (1/2)x^T P x + q^T x
s.t. Gx <= h
     Ax = b

<=> 
min (1/2)x^T H x + g^T x
s.t. A2x <= b2
     A1x = b1
'''


sol = solvers.qp(H, g, A2, b2, A1, b1)  
print(sol['x'])

