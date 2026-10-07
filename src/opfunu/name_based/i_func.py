#!/usr/bin/env python
# Created by "Thieu" at 18:46, 22/07/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import typing

import numpy as np

from opfunu.benchmark.func import FuncBenchmark


class Infinity(FuncBenchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.
    """

    name = "Hansen Function"
    latex_formula = r"f(x) = \sum_{i=1}^{n} x_i^{6} \left [ \sin\left ( \frac{1}{x_i} \right ) + 2 \right ]"
    latex_formula_dimension = r"d \in N^+"
    latex_formula_bounds = r"x_i \in [-1, 1], \forall i \in \llbracket 1, d\rrbracket"
    latex_formula_global_optimum = r"f(0,..,0) = 0"
    continuous = True
    linear = False
    convex = True
    unimodal = False
    separable = True

    differentiable = True
    scalable = True
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
        shift: typing.Any = None,
        rotate: typing.Any = None,
        rotate_bounds: bool = True,
    ) -> None:
        def compute(x: np.ndarray, epsilon: float, out: np.ndarray) -> None:
            out[0] = np.sum(x**6.0 * (np.sin(1.0 / (x + epsilon)) + 2.0))

        super().__init__(
            compute=compute,
            ndim=ndim,
            bounds=bounds,
            default_bounds=np.array([[-1.0, 1.0] for _ in range(2)]),
            f_global=0.0,
            x_global=lambda nd: 1e-16 * np.zeros(nd),
            dim_changeable=True,
            dim_default=2,
            param_names=["epsilon"],
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            shift=shift,
            rotate=rotate,
            rotate_bounds=rotate_bounds,
        )
