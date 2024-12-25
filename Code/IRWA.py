from cvxopt import solvers, matrix
import numpy as np
np.random.seed(42)


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
    A = np.concatenate([A1, A2])
    b = np.concatenate([b1, b2])
    print(A)
    # print(A[0].shape)
    print(b)
    # print(b)
    eta = 0.6
    gamma = 1 / 6
    M = 1e4
    sigma = 1e-5
    sigma_prime = 1e-5
    max_iter = 1e5
    m = A1.shape[0] + A2.shape[0]
    n = H.shape[0]
    print("m: ", m)
    print("n: ", n)
    M1 = 1
    M2 = 1
    while True:
        k = 0
        x = np.zeros(n) # wait to be defined
        epsilon = 2e3 * np.ones(m) # influence a lot
        while k < max_iter:
            # Step 1: Solve the reweighted subproblem
            # print("Step 1: Solve the reweighted subproblem")
            W_diag = []
            v = []

            for i in range(m):
                if i < A1.shape[0]:
                    wi = M1 / np.sqrt((np.dot(A[i], x) + b[i]) ** 2 + epsilon[i] ** 2)
                    vi = b[i]
                else:
                    max_term_w = max(np.dot(A[i], x) + b[i], 0)
                    max_term_v = max(-np.dot(A[i], x), b[i])
                    if np.sqrt(max_term_w ** 2 + epsilon[i] ** 2) == 0:
                        print(epsilon[i])
                        print(max_term_w)
                        print(M1)
                        raise ValueError("0")
                    wi = M2 / np.sqrt(max_term_w ** 2 + epsilon[i] ** 2)
                    vi = max_term_v
                W_diag.append(wi)
                v.append(vi)

            # print("W_diag: ", W_diag)
            # print("v: ", v)


            # for i in I1:
            #     wi = 1 / np.sqrt((np.dot(A1[i], x) + b1[i])**2 + epsilon[i]**2)
            #     W_diag.append(wi)
            #     v.append(b1[i])
            
            # # print("W_diag.shape: ", len(W_diag))
            # # print("v.shape: ", len(v))

            # for i, j in zip(I2, range(A2.shape[0])):
            #     max_term = max(np.dot(A2[j], x) + b2[j], 0)
            #     wi = 1 / np.sqrt(max_term**2 + epsilon[i]**2)
            #     W_diag.append(wi)
            #     v.append(max(-np.dot(A2[j], x), b2[j]))

            W = np.diag(np.array(list(W_diag)).squeeze())
            v = np.array(v)
            # print("W_diag.shape: ", W.shape)
            # print("v.shape: ", v.shape)
            
            AT_W = np.dot(A.T, W)
            H_eff = H + np.dot(AT_W, A)
            g_eff = g + np.dot(AT_W, v)
            

            # Solve for x
            x_new = np.linalg.solve(H_eff, -g_eff)

            # Step 2: Update relaxation vector
            # print("Step 2: Update relaxation vector")
            q = np.zeros(m)
            r = np.zeros(m)
            for i in range(m):
                q[i] = np.dot(A[i], x_new - x)
                r[i] = (1 - v[i]) * (np.dot(A[i], x) + b[i])

            # print("q: ", q)
            # print("r: ", r)
            # q = np.array([np.dot(A1[i], x_new - x) for i in I1] + 
            #              [np.dot(A2[j], x_new - x) for j in range(A2.shape[0])])
            # r = np.array([1 - v[i] * (np.dot(A1[i], x) + b1[i]) for i in I1] +
            #              [1 - v[i + len(I1)] * (np.dot(A2[j], x) + b2[j]) for j in range(A2.shape[0])])

            # for i in range(m):
            #     if np.abs(q[i]) > M * (r[i] ** 2 + epsilon[i] ** 2) ** (0.5 + gamma):
            #         epsilon = eta * epsilon
            #         break
            if np.all(np.abs(q) <= M * (r**2 + epsilon**2) ** (0.5 + gamma)):
                epsilon_new = eta * epsilon
                # for i in range(A1.shape[0], m):
                #     if np.dot(A[i], x) + b[i] <= -epsilon_new[i]:
                #         epsilon_new[i] = epsilon[i]
            else:
                epsilon_new = epsilon

            # Step 3: Check stopping criteria
            # print("Step 3: Check stopping criteria")
            if np.linalg.norm(x_new - x) <= sigma and np.linalg.norm(epsilon) <= sigma_prime:
                # print(np.linalg.norm(x_new - x))
                # print(np.linalg.norm(epsilon))
                # print(np.dot(A1, x_new))
                # print(np.dot(A2, x_new))
                break
            

            x = x_new
            epsilon = epsilon_new
            print("k: ", k)
            print("x: ", x)
            k += 1

        if np.all(np.abs(np.dot(A1, x) + b1) <= 1e-5) and np.all(np.dot(A2, x) <= -b2):
            break
        M1 += 10
        M2 += 10

    return x


def initialize_experiment(type):
    # Problem dimensions 300 1000
    w_A1 = 3
    h_A1 = 6
    w_A2 = 3
    h_A2 = 6
    assert h_A1 == h_A2
    m, n = w_A1 + w_A2, h_A1  # A's shape
    
    # 1. Generate A1 and A2
    A1_mean = np.random.randint(1, 11)  
    A1_var = np.random.randint(1, 11)   
    A1 = np.random.normal(A1_mean, A1_var, size=(w_A1, h_A1))

    A2_mean = np.random.randint(1, 11)  
    A2_var = np.random.randint(1, 11)   
    A2 = np.random.normal(A2_mean, A2_var, size=(w_A2, h_A2))
    
    # 2. Generate b and g
    b1_mean = np.random.randint(-100, 101)  
    b1_var = np.random.randint(1, 101)      
    b1 = np.random.normal(b1_mean, b1_var, size=(w_A1, ))

    b2_mean = np.random.randint(-100, 101)  
    b2_var = np.random.randint(1, 101)      
    b2 = np.random.normal(b2_mean, b2_var, size=(w_A2, ))
    
    g_mean = np.random.randint(-100, 101)
    g_var = np.random.randint(1, 101)
    g = np.random.normal(g_mean, g_var, size=(n, ))
    
    # 3. Generate H = 0.1I + LL^T
    L = np.random.normal(1, np.sqrt(2), size=(n, n))
    H = 0.1 * np.eye(n) + L @ L.T

    if type == "matrix":
        return matrix(A1), matrix(b1), matrix(A2), matrix(b2), matrix(g), matrix(H)
    elif type == "numpy":
        return A1, -b1, A2, -b2, g, H
    else:
        raise NotImplementedError("Not implemented type")
    


if __name__ == "__main__":
    '''
    H = np.array([[4., 1.], [1., 2.]])
    g = np.array([1., 1.])
    A2 = np.array([[-1., 0.], [0., -1.]])
    b2 = np.array([0., 0.])
    A1 = np.array([1., 1.])
    b1 = np.array([1.])
    '''
    A1_numpy, b1_numpy, A2_numpy, b2_numpy, g_numpy, H_numpy = initialize_experiment("numpy")
    A1_matrix, b1_matrix, A2_matrix, b2_matrix, g_matrix, H_matrix = initialize_experiment("matrix")

    # H = np.array([[4, 1], [1, 2]])
    # g = np.array([1, 1])
    # A2 = np.array([[-1, 0], [0, -1]])
    # b2 = np.array([0, 0])
    # A1 = np.array([[1], [1]]).T
    # print(A1.shape)
    # b1 = np.array([-1])

    cvxopt_sol = solvers.qp(H_matrix, g_matrix, A2_matrix, b2_matrix, A1_matrix, b1_matrix)  
    cvxopt_x = np.array(cvxopt_sol['x'])

    irwa_x = IRWA_QP_solver(A1_numpy, A2_numpy, b1_numpy, b2_numpy, g_numpy, H_numpy)
    # irwa_x = IRWA_QP_solver(A1, A2, b1, b2, g, H)

    print("Result: ", cvxopt_x)
    print("Result: ", irwa_x)
    print("Norm: ", np.linalg.norm(cvxopt_x - irwa_x))