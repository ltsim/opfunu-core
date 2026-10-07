#!/usr/bin/env python
# Created by "Thieu" at 20:59, 29/06/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np

import opfunu

if __name__ == "__main__":
    # Get all the available functions accepting any dimension
    any_dim_functions = opfunu.get_name_based_functions(None)
    print(any_dim_functions)

    # Get all the available differentiable functions accepting 2D
    differentiable_2d_functions = opfunu.get_name_based_functions(
        ndim=2,  # dimension
        differentiable=True,
    )
    print(differentiable_2d_functions)

    # Import a specific function and evaluate a solution
    ackley03 = opfunu.name_based.Ackley03()
    print(ackley03.evaluate(np.array([5, 4])))  # get results

    # Access/change the parameters of parametric functions
    ackley02 = opfunu.name_based.Ackley02()
    print(ackley02.get_paras())

    # Get the global minimum for a specific dimension
    print(ackley02.f_global)
    print(ackley02.x_global)

    # Access the latex formulas
    print(ackley02.latex_formula)

    # Shifted function: bounds and global optimum move with the shift vector
    shifted = opfunu.name_based.Ackley01(ndim=3, shift=[2.0, -1.5, 3.0])
    print(shifted.x_global)  # base optimum + shift
    print(shifted.bounds)  # base bounds + shift
    print(shifted.is_succeed(shifted.x_global))  # True

    # Rotated function: optimum maps through M.T and bounds become the enclosing
    # axis-aligned box of the rotated box (compose with a shift as needed)
    theta = np.pi / 4
    rotate = np.eye(3)
    rotate[:2, :2] = [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    rotated = opfunu.name_based.Ackley01(ndim=3, rotate=rotate)
    print(rotated.x_global)  # M.T @ base optimum
    print(rotated.bounds)  # enclosing AABB of the rotated bounds
    print(rotated.is_succeed(rotated.x_global))  # True
    rotated.set_rotate(None)  # back to the unrotated problem
