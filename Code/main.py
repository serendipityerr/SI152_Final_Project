from ADAL_bay import adal_solver, adal_solver_bay
from IRWA_bay import irwa_solver, irwa_solver_bay
from func import initialize_experiment, objective, experimentSetup
from cvxopt import solvers, matrix

w_A1 = 1
w_A2 = 1
h_A = 2
A1, b1, A2, b2, g, H = initialize_experiment(
    w_A1=w_A1, w_A2=w_A2, h_A=h_A, type="numpy"
)

A1_matrix = matrix(A1)
b1_matrix = matrix(-b1)
A2_matrix = matrix(A2)
b2_matrix = matrix(-b2)
g_matrix = matrix(g)
H_matrix = matrix(H)

cvxopt_sol = solvers.qp(H_matrix, g_matrix, A2_matrix, b2_matrix, A1_matrix, b1_matrix)
print("CVXOPT Result: ", cvxopt_sol["x"])
print("CVXOPT Objective: ", objective(g, H, cvxopt_sol["x"]))

x, p, u = adal_solver(A1, b1, A2, b2, g, H)
print("ADAL Result: ", x)
print("ADAL Objective: ", objective(g, H, x))

x, p, u, M1, M2 = adal_solver_bay(A1, b1, A2, b2, g, H)
print("ADAL Bay Result: ", x)
print("ADAL Bay Objective: ", objective(g, H, x))
try:
    x = irwa_solver(A1, b1, A2, b2, g, H)
    print("IRWA Result: ", x)
    print("IRWA Objective: ", objective(g, H, x))
except:
    print("IRWA Result: Failed")

x, M1, M2 = irwa_solver_bay(A1, b1, A2, b2, g, H)
print("IRWA Bay Result: ", x)
print("IRWA Bay Objective: ", objective(g, H, x))
