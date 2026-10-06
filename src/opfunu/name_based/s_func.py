#!/usr/bin/env python
# Created by "Thieu" at 17:31, 30/07/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import typing

import numpy as np

from opfunu.benchmark.func import FuncBenchmark


class Salomon(FuncBenchmark):
    """
    .. [1]  Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization Problems
    Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.

    .. math::

        f_{\text{Salomon}}(x) = 1 - \\cos \\left (2 \\pi \\sqrt{\\sum_{i=1}^{n} x_i^2} \right) + 0.1 \\sqrt{\\sum_{i=1}^n x_i^2}


    Here, :math:`n` represents the number of dimensions and :math:`x_i \\in [-100, 100]` for :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x_i = 0` for :math:`i = 1, ..., n`
    """

    name = "Qing Function"
    latex_formula = r"f_{\text{Salomon}}(x) = 1 - \cos \left (2 \pi \sqrt{\sum_{i=1}^{n} x_i^2} \right) + 0.1 \sqrt{\sum_{i=1}^n x_i^2}"
    latex_formula_dimension = r"d = n"
    latex_formula_bounds = r"x_i \in [-10, 10, ..., 10]"
    latex_formula_global_optimum = r"f(0, 0, ...,0) = 1.0"
    continuous = True
    linear = False
    convex = True
    unimodal = False
    separable = False

    differentiable = True
    scalable = True
    randomized_term = False
    parametric = False

    modality = True  # Number of ambiguous peaks, unknown # peaks

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
        shift: typing.Any = None,
    ) -> None:
        def compute(x: np.ndarray, out: np.ndarray) -> None:
            u = np.sqrt(np.sum(x**2))
            out[0] = 1 - np.cos(2 * np.pi * u) + 0.1 * u

        super().__init__(
            compute=compute,
            ndim=ndim,
            bounds=bounds,
            default_bounds=np.array([[-100.0, 100.0] for _ in range(2)]),
            f_global=0.0,
            x_global=lambda nd: np.zeros(nd),
            dim_changeable=True,
            dim_default=2,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            shift=shift,
        )
