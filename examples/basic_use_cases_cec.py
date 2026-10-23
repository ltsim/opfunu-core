#!/usr/bin/env python
# Created by "Thieu" at 10:46, 30/06/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np

import opfunu

if __name__ == "__main__":
    # Get all the available separable functions accepting 2D
    separable_2d_cec = opfunu.get_cec_based_functions(ndim=2, separable=True)
    print(separable_2d_cec)

    # Import a specific function and evaluate a solution
    f12005 = opfunu.cec_based.F12005()
    print(f12005.evaluate(np.array([5, 4, 5])))  # get results
    print(f12005.bounds)
    print(f12005.lb)
    print(f12005.ub)

    # Pass custom bounds
    bounds = [[-10.0] * 15, [10.0] * 15]
    f12005 = opfunu.cec_based.F12005(bounds=bounds)
    print(f12005.lb)

    # Read the matrix from file
    f32005 = opfunu.cec_based.F32005(ndim=10)
    x = np.ones(10)
    print(f32005.evaluate(x))
    print(f32005.f_matrix)
    print(f32005.x_global)

    problem = opfunu.cec_based.F212005(ndim=10)
    x = np.ones(10)
    print(problem.evaluate(x))
    print(problem.x_global)
    print(problem.is_succeed(problem.x_global))

    # Get all the available rotated functions accepting 2D
    my_list = opfunu.get_cec_based_functions(ndim=2, rotated=True)
    print(my_list)

    # Get all noise functions
    my_list = opfunu.get_cec_based_functions(randomized_term=True)
    print(my_list)
