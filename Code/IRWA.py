import numpy as np
def QP_solver(A1,A2,b1,b2,g,H):
    ###TODO###
    x=None
    return x


def IRWA_QP_solver(A1, A2, b1, b2, g, H):
    """
    IRWA: Iterative Reweighting Algorithm for QP problems. 

    Args:
        A1, A2: Matrices for the equality and inequality constraints.
        b1, b2: Vectors for the equality and inequality constraints.
        g: Gradient vector.
        H: Hessian matrix.

    Returns:
        x: Solution to the quadratic programming subproblem.
    """

    # Initialization
    eta=0.9
    gamma=0.1
    M=10
    sigma=1e-6
    sigma_prime=1e-6
    max_iter=1000
    x = x0 # wait to be defined
    epsilon = epsilon0 # wait to be defined
    k = 0
    m = A1.shape[0] + A2.shape[0]
    I1 = range(A1.shape[0])  # Equality indices
    I2 = range(A1.shape[0], m)  # Inequality indices

    while k < max_iter:
        # Step 1: Solve the reweighted subproblem
        W_diag = []
        v = []

        for i in I1:
            wi = 1 / np.sqrt((np.dot(A1[i], x) + b1[i])**2 + epsilon[i]**2)
            W_diag.append(wi)
            v.append(b1[i])

        for i, j in zip(I2, range(A2.shape[0])):
            max_term = max(np.dot(A2[j], x) + b2[j], 0)
            wi = 1 / np.sqrt(max_term**2 + epsilon[i]**2)
            W_diag.append(wi)
            v.append(max(-np.dot(A2[j], x), b2[j]))

        W = np.diag(W_diag)
        v = np.array(v)

        H_eff = H + np.dot(np.dot(A1.T, W[:A1.shape[0], :A1.shape[0]]), A1) + \
                np.dot(np.dot(A2.T, W[A1.shape[0]:, A1.shape[0]:]), A2)
        g_eff = g + np.dot(np.dot(A1.T, W[:A1.shape[0], :A1.shape[0]]), v[:A1.shape[0]]) + \
                np.dot(np.dot(A2.T, W[A1.shape[0]:, A1.shape[0]:]), v[A1.shape[0]:])

        # Solve for x
        x_new = np.linalg.solve(H_eff, -g_eff)

        # Step 2: Update relaxation vector
        q = np.array([np.dot(A1[i], x_new - x) for i in I1] + 
                     [np.dot(A2[j], x_new - x) for j in range(A2.shape[0])])
        r = np.array([1 - v[i] * (np.dot(A1[i], x) + b1[i]) for i in I1] +
                     [1 - v[i + len(I1)] * (np.dot(A2[j], x) + b2[j]) for j in range(A2.shape[0])])

        if np.all(np.abs(q) <= M * np.sqrt(r**2 + epsilon**2) ** (1 + gamma)):
            epsilon = eta * epsilon
        else:
            epsilon = epsilon

        # Step 3: Check stopping criteria
        if np.linalg.norm(x_new - x) <= sigma and np.linalg.norm(epsilon) <= sigma_prime:
            break

        x = x_new
        k += 1

    return x

if __name__ == "__main__":
    A1 = np.array()