from ADAL_bay import adal_solver, adal_solver_bay
from IRWA_bay import irwa_solver, irwa_solver_bay
from func import initialize_experiment, objective, experimentSetup
from cvxopt import solvers, matrix
from time import time

w_A1 = 1
w_A2 = 1
h_A = w_A1 + w_A2
A1, b1, A2, b2, g, H = initialize_experiment(
    w_A1=w_A1, w_A2=w_A2, h_A=h_A, type="numpy"
)

A1_matrix = matrix(A1)
b1_matrix = matrix(-b1)
A2_matrix = matrix(A2)
b2_matrix = matrix(-b2)
g_matrix = matrix(g)
H_matrix = matrix(H)

t1 = time()
cvxopt_sol = solvers.qp(H_matrix, g_matrix, A2_matrix, b2_matrix, A1_matrix, b1_matrix)
t2 = time()
x_opt = cvxopt_sol["x"]
objective_opt = objective(g, H, x_opt)
time_opt = t2 - t1
print("CVXOPT Result: ", x_opt)
print("CVXOPT Objective: ", objective_opt)

t1 = time()
x_adal, p, u = adal_solver(A1, b1, A2, b2, g, H)
t2 = time()
tiem_adal = t2 - t1
objective_adal = objective(g, H, x_adal)
print("ADAL Result: ", x_adal)
print("ADAL Objective: ", objective_adal)

t1 = time()
x_adal_bay, p, u, M1_adal_bay, M2_adal_bay, iter_M_adal_bay, iter_objective_adal_bay = (
    adal_solver_bay(A1, b1, A2, b2, g, H)
)
t2 = time()
tiem_adal_bay = t2 - t1
objective_adal_bay = objective(g, H, x_adal_bay)
print("ADAL Bay Result: ", x_adal_bay, M1_adal_bay, M2_adal_bay)
print("ADAL Bay Objective: ", objective_adal_bay)

try:
    t1 = time()
    x_irwa = irwa_solver(A1, b1, A2, b2, g, H)
    t2 = time()
    tiem_irwa = t2 - t1
    objective_irwa = objective(g, H, x_irwa)
    print("IRWA Result: ", x_irwa)
    print("IRWA Objective: ", objective_irwa)
except:
    print("IRWA Result: Failed")

t1 = time()
x_irwa_bay, M1_irwa_bay, M2_irwa_bay, iter_M_irwa_bay, iter_objective_irwa_bay = (
    irwa_solver_bay(A1, b1, A2, b2, g, H)
)
t2 = time()
tiem_irwa_bay = t2 - t1
objective_irwa_bay = objective(g, H, x_irwa_bay)
print("IRWA Bay Result: ", x_irwa_bay, M1_irwa_bay, M2_irwa_bay)
print("IRWA Bay Objective: ", objective_irwa_bay)
