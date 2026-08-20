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
