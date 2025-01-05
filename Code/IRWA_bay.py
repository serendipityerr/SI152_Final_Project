from cvxopt import solvers, matrix
import numpy as np
from func import initialize_experiment, objective, experimentSetup
from scipy.optimize import minimize
from skopt import gp_minimize


def irwa_solver(A1, b1, A2, b2, g, H, M1=1, M2=1):
    """
    IRWA: Iterative Reweighting Algorithm for QP problems.

    Args:
        A1, A2: Matrices for the equality and inequality constraints.
        b1, b2: Vectors for the equality and inequality constraints.
        g: Gradient vector.
        H: Hessian matrix.
        M1, M2: Penalty

    Returns:
        x: Solution to the quadratic programming subproblem.
    """
    # Initialization
    A = np.concatenate([A1, A2])
    b = np.concatenate([b1, b2])
    eta = 0.6
    gamma = 1 / 6
    M = 1e4
    sigma = 1e-5
    sigma_prime = 1e-5
    max_iter = 1e3
    m = A1.shape[0] + A2.shape[0]
    n = H.shape[0]
    x = np.zeros(n)
    k = 0
    epsilon = 2e3 * np.ones(m)
    while k < max_iter:
        W_diag = []
        v = []

        for i in range(m):
            if i < A1.shape[0]:
                wi = M1 / np.sqrt((np.dot(A[i], x) + b[i]) ** 2 + epsilon[i] ** 2)
                vi = b[i]
            else:
                max_term_w = max(np.dot(A[i], x) + b[i], 0)
                max_term_v = max(-np.dot(A[i], x), b[i])
                if np.sqrt(max_term_w**2 + epsilon[i] ** 2) == 0:
                    print(epsilon[i])
                    print(max_term_w)
                    print(M1)
                    raise ValueError("0")
                wi = M2 / np.sqrt(max_term_w**2 + epsilon[i] ** 2)
                vi = max_term_v
            W_diag.append(wi)
            v.append(vi)

        W = np.diag(np.array(list(W_diag)).squeeze())
        v = np.array(v)
        AT_W = np.dot(A.T, W)
        H_eff = H + np.dot(AT_W, A)
        g_eff = g + np.dot(AT_W, v)
        x_new = np.linalg.solve(H_eff, -g_eff)
        q = np.zeros(m)
        r = np.zeros(m)
        for i in range(m):
            q[i] = np.dot(A[i], x_new - x)
            r[i] = (1 - v[i]) * (np.dot(A[i], x) + b[i])
        if np.all(np.abs(q) <= M * (r**2 + epsilon**2) ** (0.5 + gamma)):
            epsilon_new = eta * epsilon
        else:
            epsilon_new = epsilon
        if (
            np.linalg.norm(x_new - x) <= sigma
            and np.linalg.norm(epsilon) <= sigma_prime
        ):
            break

        x = x_new
        epsilon = epsilon_new
        k += 1
    return x


def irwa_solver_bay(A1, b1, A2, b2, g, H):
    experimentPara = experimentSetup(A1, b1, A2, b2, g, H)

    def objective_func(params):
        M1, M2 = params
        A1, b1, A2, b2, g, H = experimentPara.para()
        try:
            x = irwa_solver(A1, b1, A2, b2, g, H, M1, M2)
        except:
            return 100000
        constraint_violation_1 = np.linalg.norm(A1 @ x + b1)
        constraint_violation_2 = np.sum(np.maximum(0, A2 @ x + b2))
        value = g.T @ x + 0.5 * x.T @ H @ x
        return value + constraint_violation_1 * 1000 + constraint_violation_2 * 1000
        if constraint_violation_1 > 0 or constraint_violation_2 > 0:
            return 0
        else:
            return value

    result = gp_minimize(
        objective_func, experimentPara.bay_para(), n_calls=100, random_state=42
    )
    best_M1, best_M2 = result.x
    x = irwa_solver(A1, b1, A2, b2, g, H, best_M1, best_M2)
    return x, best_M1, best_M2


# if __name__ == "__main__":
#     '''
#     H = np.array([[4., 1.], [1., 2.]])
#     g = np.array([1., 1.])
#     A2 = np.array([[-1., 0.], [0., -1.]])
#     b2 = np.array([0., 0.])
#     A1 = np.array([1., 1.])
#     b1 = np.array([1.])
#     '''
#     A1_numpy, b1_numpy, A2_numpy, b2_numpy, g_numpy, H_numpy = initialize_experiment(
#         "numpy"
#     )
#     A1_matrix = matrix(A1_numpy)
#     b1_matrix = matrix(-b1_numpy)
#     A2_matrix = matrix(A2_numpy)
#     b2_matrix = matrix(-b2_numpy)
#     g_matrix = matrix(g_numpy)
#     H_matrix = matrix(H_numpy)

#     irwa_x = IRWA_QP_solver(A1_numpy, A2_numpy, b1_numpy, b2_numpy, g_numpy, H_numpy)

#     print("Result: ", irwa_x)
#     print("Objective: ", objective(g_numpy, H_numpy, irwa_x))
