import numpy as np
from scipy.optimize import minimize
from cvxopt import solvers, matrix
import numpy as np
from IRWA import initialize_experiment
import numpy as np
from scipy.optimize import minimize
from skopt import gp_minimize
from skopt.space import Real
import matplotlib.pyplot as plt
from IRWA import initialize_experiment


def objective_value(g, H, x):
    return 0.5 * np.dot(np.dot(x.T, H), x) + np.dot(g, x)


def adal_solver(A1, A2, b1, b2, g, H, M1=0, M2=0):
    A = np.concatenate((A1, A2))
    b = np.concatenate((b1, b2))
    mu = 1
    sigma = 1e-5
    sigma_prime = 1e-5
    max_iter = 1000
    m = A1.shape[0] + A2.shape[0]
    n = H.shape[0]
    x = np.zeros(n)
    p = A @ x + b
    u = np.zeros(m)
    for k in range(max_iter):
        # print(k, x)

        # Step 1: Solve augmented Lagrangian subproblems
        # Solve for x
        def lagrangian_x(x):
            return (
                g.T @ x
                + 0.5 * x.T @ H @ x
                + M1 * np.sum(np.abs(A1 @ x + b1))
                + M2 * np.sum(np.maximum(0, A2 @ x + b2))
                + u.T @ (A @ x + b - p)
                + (mu / 2) * np.linalg.norm(A @ x + b - p) ** 2
            )

        x_next = minimize(lagrangian_x, x, method="BFGS").x

        # Solve for p
        def lagrangian_p(p):
            return (
                g.T @ x_next
                + 0.5 * x_next.T @ H @ x_next
                + M1 * np.sum(np.abs(A1 @ x_next + b1))
                + M2 * np.sum(np.maximum(0, A2 @ x_next + b2))
                + u.T @ (A @ x_next + b - p)
                + (mu / 2) * np.linalg.norm(A @ x_next + b - p) ** 2
            )

        p_next = minimize(lagrangian_p, p, method="BFGS").x

        # Step 2: Update dual variable u
        u_next = u + (1 / mu) * (A @ x_next + b - p_next)
        # Step 3: Check if the constraints are violated and adjust M1, M2

        if (
            np.linalg.norm(x_next - x) <= sigma
            and np.max(np.abs(A @ x_next + b - p_next)) <= sigma_prime
        ):
            break

        # Update variables for next iteration
        x, p, u = x_next, p_next, u_next
    return x, p, u


A1, b1, A2, b2, g, H = initialize_experiment("numpy")


def experimentSetup():
    # 生成实验数据
    global A1, b1, A2, b2, g, H

    return A1, A2, b1, b2, g, H


maxValue = 0


def objective(params):
    global maxValue
    M1, M2 = params
    A1, A2, b1, b2, g, H = experimentSetup()
    x, p, u = adal_solver(A1, A2, b1, b2, g, H, M1, M2)

    constraint_violation_1 = np.linalg.norm(A1 @ x + b1)
    constraint_violation_2 = np.sum(np.maximum(0, A2 @ x + b2))
    value = g.T @ x + 0.5 * x.T @ H @ x
    return value + constraint_violation_1 * 1000 + constraint_violation_2 * 1000


param_space = [
    Real(10, 100.0, name="M1"),  # M1的取值范围
    Real(10, 100.0, name="M2"),  # M2的取值范围
]
# 使用贝叶斯优化方法
result = gp_minimize(objective, param_space, n_calls=100, random_state=42)

# 输出最佳的 M1 和 M2 值
best_M1, best_M2 = result.x
print(f"Optimal M1: {best_M1}, Optimal M2: {best_M2}")

# 打印最终的目标函数值
print(f"Best objective value: {result.fun}")
A1, A2, b1, b2, g, H = experimentSetup()
x, p, u = adal_solver(A1, A2, b1, b2, g, H, best_M1, best_M2)
print(x, p, u)
A1_matrix = matrix(A1)
b1_matrix = matrix(-b1)
A2_matrix = matrix(A2)
b2_matrix = matrix(-b2)
g_matrix = matrix(g)
H_matrix = matrix(H)

cvxopt_sol = solvers.qp(H_matrix, g_matrix, A2_matrix, b2_matrix, A1_matrix, b1_matrix)
cvxopt_x = np.array(cvxopt_sol["x"]).flatten()

print("Result: ", cvxopt_x)
print("objective:", objective_value(g, H, cvxopt_x))
print("Result: ", x)
print("objective:", objective_value(g, H, x))

print("Norm: ", np.linalg.norm(cvxopt_x - x))
# 画出目标函数的收敛过程

np.random.seed(42)
