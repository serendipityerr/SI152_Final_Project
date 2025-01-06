import numpy as np
from scipy.optimize import minimize
from cvxopt import solvers, matrix
import numpy as np
import numpy as np
from scipy.optimize import minimize
from skopt import gp_minimize
from skopt.space import Real
import matplotlib.pyplot as plt
from func import initialize_experiment, objective, experimentSetup


def adal_solver(A1, b1, A2, b2, g, H, M1=1, M2=1):
    A = np.concatenate((A1, A2))
    b = np.concatenate((b1, b2))
    mu = 1
    sigma = 1e-5
    sigma_prime = 1e-5
    max_iter = 100
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


def adal_solver_bay(A1, b1, A2, b2, g, H):
    experimentPara = experimentSetup(A1, b1, A2, b2, g, H)
    iter_M = []
    iter_objective = []
    def objective_func(params):
        M1, M2 = params
        A1, b1, A2, b2, g, H = experimentPara.para()
        x, p, u = adal_solver(A1, b1, A2, b2, g, H, M1, M2)

        constraint_violation_1 = np.linalg.norm(A1 @ x + b1)
        constraint_violation_2 = np.sum(np.maximum(0, A2 @ x + b2))
        value = g.T @ x + 0.5 * x.T @ H @ x
        return value + constraint_violation_1 * 1000 + constraint_violation_2 * 1000
        if constraint_violation_1 > 0 or constraint_violation_2 > 0:
            return 0
        else:
            return value

    def save_para(res):
        M1, M2 = res.x
        iter_M.append([M1, M2])
        iter_objective.append(res.fun)

    result = gp_minimize(
        objective_func,
        experimentPara.bay_para(),
        n_calls=100,
        random_state=42,
        callback=[save_para],
    )
    best_M1, best_M2 = result.x
    x, p, u = adal_solver(A1, b1, A2, b2, g, H, best_M1, best_M2)
    return x, p, u, best_M1, best_M2, iter_M, iter_objective
