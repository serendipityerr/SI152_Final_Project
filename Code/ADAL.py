import numpy as np


def adal_solver(
    H, g, A, b, I1, I2, x0, u0, mu, sigma=1e-6, sigma_prime=1e-6, max_iter=1000
):
    """
    Alternating Direction Augmented Lagrangian (ADAL) Algorithm.

    Parameters:
    - H: Quadratic term in the objective (symmetric positive definite matrix).
    - g: Linear term in the objective (vector).
    - A: Constraint matrix.
    - b: Constraint vector.
    - I1: Indices of terms with |p_i| in the objective.
    - I2: Indices of terms with max{p_i, 0} in the objective.
    - x0: Initial guess for x.
    - u0: Initial guess for dual variables (u).
    - mu: Penalty parameter.
    - sigma: Tolerance for x convergence.
    - sigma_prime: Tolerance for constraint residuals.
    - max_iter: Maximum number of iterations.

    Returns:
    - x, p, u: Optimal solutions for variables x, p, and dual variables u.
    """
    # Initialize variables
    x = x0
    u = u0
    p = np.zeros(A.shape[0])  # Initialize p

    for k in range(max_iter):
        # Step 1: Solve subproblem for x
        q = g + A.T @ (u + mu * (p - b))
        x_new = np.linalg.solve(H + mu * A.T @ A, -q)

        # Step 1: Solve subproblem for p
        p_tilde = A @ x_new + b - u / mu
        p_new = np.copy(p_tilde)

        # Apply the proximal operator for p terms
        for i in range(len(p_tilde)):
            if i in I1:  # |p_i|
                p_new[i] = np.sign(p_tilde[i]) * max(abs(p_tilde[i]) - 1 / mu, 0)
            elif i in I2:  # max{p_i, 0}
                p_new[i] = max(p_tilde[i], 0)

        # Step 2: Update dual variables
        residual = A @ x_new + b - p_new
        u_new = u + mu * residual

        # Check stopping criteria
        if (
            np.linalg.norm(x_new - x) <= sigma
            and np.max(np.abs(residual)) <= sigma_prime
        ):
            break

        # Update variables for the next iteration
        x, p, u = x_new, p_new, u_new

    return x, p, u


# Example Usage
H = np.array([[4, 1], [1, 2]])  # Quadratic term
g = np.array([-1, -1])  # Linear term
A = np.array([[1, 1], [1, -1]])  # Constraint matrix
b = np.array([1, 0])  # Constraint vector
I1 = [0]  # Indices for |p_i|
I2 = [1]  # Indices for max{p_i, 0}
x0 = np.zeros(2)  # Initial x
u0 = np.zeros(2)  # Initial dual variables
mu = 10  # Penalty parameter

x_opt, p_opt, u_opt = adal_solver(H, g, A, b, I1, I2, x0, u0, mu)
print("Optimal x:", x_opt)
print("Optimal p:", p_opt)
print("Optimal u:", u_opt)
