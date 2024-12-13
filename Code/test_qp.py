from cvxopt import solvers, matrix
P = matrix([[4.,1.], [1.,2.]])
q = matrix([1., 1.])
G = matrix([[-1., 0.], [0., -1.]])
h = matrix([0., 0.])
A = matrix([1.,1.]).T
b = matrix([1.])
print(b)

# # Cvxopt.solvers.qp(P, q, G, h, A, b)
# '''
# min (1/2)x^T P x + q^T x
# s.t. Gx <= h
#      Ax = b

# <=> 
# min (1/2)x^T H x + g^T x
# s.t. A2x <= b2
#      A1x = b1
# '''
# sol = solvers.qp(P,q,G,h)  
# print(sol['x'])