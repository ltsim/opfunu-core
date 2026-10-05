#!/usr/bin/env python
# Created by "Thieu" at 17:22, 22/07/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import typing

import numpy as np

from opfunu.benchmark.func import FuncBenchmark


class FreudensteinRoth(FuncBenchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.
    """

    name = "Freudenstein Roth Function"
    latex_formula = (
        r"f(x) = \left\{x_1 - 13 + \left[(5 - x_2) x_2"
        + r"- 2 \right] x_2 \right\}^2 + \left \{x_1 - 29 + \left[(x_2 + 1) x_2 - 14 \right] x_2 \right\}^2"
    )
    latex_formula_dimension = r"d = 2"
    latex_formula_bounds = r"x_i \in [-10, 10], \forall i \in \llbracket 1, d\rrbracket"
    latex_formula_global_optimum = r"f(5, 4) = 0"
    continuous = True
    linear = False
    convex = True
    unimodal = False
    separable = False

    differentiable = True
    scalable = False
    randomized_term = False
    parametric = False

    modality = False  # Number of ambiguous peaks, unknown # peaks

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
    ) -> None:
        def compute(x: np.ndarray, out: np.ndarray) -> None:
            f1 = (-13.0 + x[0] + ((5.0 - x[1]) * x[1] - 2.0) * x[1]) ** 2
            f2 = (-29.0 + x[0] + ((x[1] + 1.0) * x[1] - 14.0) * x[1]) ** 2
            out[0] = f1 + f2

        super().__init__(
            compute=compute,
            ndim=ndim,
            bounds=bounds,
            default_bounds=np.array([[-10.0, 10.0] for _ in range(2)]),
            f_global=0.0,
            x_global=np.array([5.0, 4.0]),
            dim_changeable=False,
            dim_default=2,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
        )
