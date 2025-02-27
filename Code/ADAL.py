import numpy as np
from scipy.optimize import minimize
from cvxopt import solvers, matrix
from IRWA import initialize_experiment


def objective(g, H, x):
    return 0.5 * np.dot(np.dot(x.T, H), x) + np.dot(g, x)


def adal_solver(A1, A2, b1, b2, g, H):
    A = np.concatenate((A1, A2), axis=0)
    b = np.concatenate((b1, b2), axis=0)
    mu = 1
    sigma = 1e-5
    sigma_prime = 1e-5
    max_iter = 10000
    m = A1.shape[0] + A2.shape[0]
    n = H.shape[0]
    x = np.zeros(n)
    p = A @ x + b
    u = np.zeros(m)
    M1 = 0
    M2 = 0
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

        # Step 3: Check stopping criteria
        if (
            np.linalg.norm(x_next - x) <= sigma
            and np.max(np.abs(A @ x_next + b - p_next)) <= sigma_prime
        ):
            break

        # Update variables for next iteration
        x, p, u = x_next, p_next, u_next
        M1 += 1
        M2 += 1

    return x, p, u
A1_numpy, b1_numpy, A2_numpy, b2_numpy, g_numpy, H_numpy = initialize_experiment(
    "numpy"
)

x, p, u = adal_solver(A1_numpy, A2_numpy, b1_numpy, b2_numpy, g_numpy, H_numpy)

A1_matrix = matrix(A1_numpy)
b1_matrix = matrix(-b1_numpy)
A2_matrix = matrix(A2_numpy)
b2_matrix = matrix(-b2_numpy)
g_matrix = matrix(g_numpy)
H_matrix = matrix(H_numpy)


cvxopt_sol = solvers.qp(H_matrix, g_matrix, A2_matrix, b2_matrix, A1_matrix, b1_matrix)
cvxopt_x = np.array(cvxopt_sol["x"]).flatten()


print("Result: ", cvxopt_x)
print("objective:", objective(g_numpy, H_numpy, cvxopt_x))
print("Result: ", x)
print("objective:", objective(g_numpy, H_numpy, x))

print("Norm: ", np.linalg.norm(cvxopt_x - x))
