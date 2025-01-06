from cvxopt import solvers, matrix
import numpy as np
from skopt.space import Real


def initialize_experiment(w_A1=1, w_A2=1, h_A=2, seed=42, type="numpy"):
    np.random.seed(seed)
    # Problem dimensions 300 1000
    h_A1 = h_A
    h_A2 = h_A1
    m = w_A1 + w_A2
    n = h_A1  # A's shape

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
    b1 = np.random.normal(b1_mean, b1_var, size=(w_A1,))

    b2_mean = np.random.randint(-100, 101)
    b2_var = np.random.randint(1, 101)
    b2 = np.random.normal(b2_mean, b2_var, size=(w_A2,))

    g_mean = np.random.randint(-100, 101)
    g_var = np.random.randint(1, 101)
    g = np.random.normal(g_mean, g_var, size=(n,))

    # 3. Generate H = 0.1I + LL^T
    L = np.random.normal(1, np.sqrt(2), size=(n, n))
    H = 0.01 * np.eye(n) + L @ L.T

    if type == "matrix":
        return matrix(A1), matrix(b1), matrix(A2), matrix(b2), matrix(g), matrix(H)
    elif type == "numpy":
        return A1, b1, A2, b2, g, H
    else:
        raise NotImplementedError("Not implemented type")


def objective(g, H, x):
    return 0.5 * np.dot(np.dot(x.T, H), x) + np.dot(g, x)


def constraint(A1, b1, A2, b2, x):
    constraint_violation_1 = np.linalg.norm(A1 @ x + b1)
    constraint_violation_2 = np.sum(np.maximum(0, A2 @ x + b2))
    return constraint_violation_1, constraint_violation_2


class experimentSetup:
    def __init__(self, A1, b1, A2, b2, g, H):
        # 生成实验数据
        self.A1 = A1
        self.A2 = A2
        self.b1 = b1
        self.b2 = b2
        self.g = g
        self.H = H

    def para(self):
        return self.A1, self.b1, self.A2, self.b2, self.g, self.H

    def bay_para(self):
        return [
            Real(1, 100.0, name="M1"),  # M1的取值范围
            Real(1, 100.0, name="M2"),  # M2的取值范围
        ]
