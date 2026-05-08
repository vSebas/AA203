#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

def main():
    data = np.genfromtxt('data_asteroid_regression.csv', delimiter=',', dtype=None)

    d = data[:,0]
    y = data[:,1]

    A = np.column_stack((d, d**2, d**3))

    # Solve the system of the form Ax = b --> solve(A,b)
    x_opt = np.linalg.solve(A.T@A, A.T@y)
    print(x_opt)

if __name__ == "__main__":
    main()