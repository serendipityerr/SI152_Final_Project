import numpy as np
def QP_solver(A1,A2,b1,b2,g,H):
    ###TODO###
    x=None
    return x


def IRWA_QP_solver(A1, A2, b1, b2, g, H):
    """
    Solves a quadratic programming subproblem using IRWA.

    Args:
        A1, A2: Matrices for the equality and inequality constraints.
        b1, b2: Vectors for the equality and inequality constraints.
        g: Gradient vector.
        H: Hessian matrix.

    Returns:
        x: Solution to the quadratic programming subproblem.
    """
    # Parameters
    max_iter = 1  # Maximum iterations
    tol = 1e-6      # Convergence tolerance
    eta = 0.5       # Scaling parameter
    gamma = 1e-2    # Relaxation parameter
    M = 10          # Threshold parameter
    A = np.concatenate([A1, A2])
    print("A:", A, A.shape)

    # Initialization
    x = np.zeros(g.shape)  # Initial guess
    epsilon = np.ones(b1.shape) * 1e-2  # Initial relaxation vector

    for k in range(max_iter):
        # Step 1: Solve the reweighted subproblem
        # TODO: W, v redefine!!!
        W = np.diag([1.0 / (np.linalg.norm(A1[i] @ x - b1[i]) + epsilon[i]) for i in range(len(b1))])
        v2 = np.maximum(b2 - A2 @ x, 0)
        print("v2", v2, v2.shape)
        v = np.maximum(b2 - A2 @ x, 0)
        
        # TODO: H_eff, g_eff redefine!!!
        H_eff = H + A2.T @ W @ A2
        g_eff = g + A2.T @ v

        # Solve linear system H_eff x + g_eff = 0
        x_new = np.linalg.solve(H_eff, -g_eff)

        # Step 2: Update relaxation vector
        # TODO: q_i, r_i redefine!!! there should be a for iteration
        q_i = A1 @ x_new - b1
        r_i = np.maximum(0, A2 @ x_new + b2)

        epsilon = np.minimum(eta * epsilon, (q_i / r_i) ** (2 / (3 + gamma)))

        # Step 3: Convergence check
        if np.linalg.norm(x_new - x) < tol:
            break

        x = x_new

    return x

if __name__ == "__main__":
    A1 = np.array()