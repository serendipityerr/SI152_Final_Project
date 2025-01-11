# SI152-Final-Project

## Overview

This folder contains the implementation of methods and experiments for solving quadratic programming (QP) problems using the ADAL (Augmented Dual Augmented Lagrangian) and IRWA (Iterative Reweighting Algorithm) frameworks. The implementation includes both basic solvers and extensions using Bayesian optimization to make the solution feasible and improve the results. The accompanying `Final_Report.pdf` report provides detailed theoretical explanations, algorithm derivations, and experimental results.

---

## File Descriptions

### **1. `ADAL_bay.py`**
- Contains two key functions:
  - `adal_solver`: Implements the basic ADAL algorithm for solving QP problems by finding the optimal solution of the penalty function with a fixed penalty coefficient (equal to 1).
  - `adal_solver_bay`: Extends `adal_solver` by applying Bayesian optimization to dynamically tune the penalty coefficient for different problems. This produces feasible solutions for the original QP problem.

---

### **2. `IRWA_bay.py`**
- Contains functions implementing the IRWA framework:
  - `irwa_solver`: Implements the basic IRWA algorithm for solving QP problems by finding the optimal solution of the penalty function with a fixed penalty coefficient (equal to 1).
  - `irwa_solver_bay`: Extends `irwa_solver` by applying Bayesian optimization to dynamically tune the penalty coefficient. This produces feasible solutions for the original QP problem.


---

### **3. `func.py`**
- Includes utility functions for generating the initial matrices and vectors $A_1$, $b_1$, $A_2$, $b_2$, $H$, $g$ required for the QP problems.

---

### **4. `main.py`**
- This script serves as the entry point for running experiments and benchmarking different QP solvers.
- **Key Functionalities:**
  - Benchmarks and compares the following solvers:
    1. **CVXOPT:** A widely used QP solver for obtaining optimal solutions as benchmarks.
    2. **ADAL Solver (`adal_solver`):** Solves the penalized QP problem with a fixed penalty coefficient by ADAL.
    3. **ADAL Solver with Bayesian Optimization (`adal_solver_bay`):** Uses Bayesian optimization to tune the penalty coefficient for feasible solutions.
    4. **IRWA Solver (`irwa_solver`):** Solves the penalized QP problem with a fixed penalty coefficient by IRWA.
    5. **IRWA Solver with Bayesian Optimization (`irwa_solver_bay`):** Uses Bayesian optimization to tune the penalty coefficient for feasible solutions.
  - Measures execution time, iteration counts, and objectives for all solvers, comparing performance and accuracy.

---

### **5. `Final_Report.pdf`**
- The final project report. More details can be found in the pdf

---

## Usage

1. **Dependencies:**
   - Ensure Python is installed with the required libraries (e.g., `numpy`, `scipy`, `skopt` and `cvxopt`).

2. **Function-calling:**
    - Call the function `adal_solver` and `irwa_solver` with $6$ input matrices and vectors $A_1$, $b_1$, $A_2$, $b_2$, $H$, $g$ to get the basic solutions to the optimization problem of the penalty function with a fixed penalty coefficient (equal to 1).
    - Call the function `adal_solver_bay` and `irwa_solver_bay` with $6$ input matrices and vectors $A_1$, $b_1$, $A_2$, $b_2$, $H$, $g$ to get the feasible solutions to the origin QP problem.
---

For further details, please refer to the report `Final_Report.pdf`.