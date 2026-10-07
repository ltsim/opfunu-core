#!/usr/bin/env python
# Created by "Thieu" at 17:31, 30/07/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import typing

import numpy as np

from opfunu.benchmark.func import FuncBenchmark


class TestTubeHolder(FuncBenchmark):
    """
    .. [1]  Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization Problems
    Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.

    .. math::

        f_{\text{TestTubeHolder}}(x) = - 4 \\left | {e^{\\left|{\\cos \\left(\frac{1}{200} x_{1}^{2} +
        \frac{1}{200} x_{2}^{2}\right)} \right|}\\sin\\left(x_{1}\right) \\cos\\left(x_{2}\right)}\right|

    with :math:`x_i \\in [-10, 10]` for :math:`i = 1, 2`.

    *Global optimum*: :math:`f(x) = -10.872299901558` for :math:`x= [-\\pi/2, 0]`
    """

    name = "Qing Function"
    latex_formula = r"f_{\text{TestTubeHolder}}(x)="
    latex_formula_dimension = r"d = n"
    latex_formula_bounds = r"x_i \in [-10, 10, ..., 10]"
    latex_formula_global_optimum = r"f(0, 0, ...,0) = 1.0"
    continuous = True
    linear = False
    convex = True
    unimodal = False
    separable = True

    differentiable = True
    scalable = False
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
        rotate: typing.Any = None,
        rotate_bounds: bool = True,
    ) -> None:
        def compute(x: np.ndarray, out: np.ndarray) -> None:
            u = np.sin(x[0]) * np.cos(x[1])
            v = (x[0] ** 2 + x[1] ** 2) / 200
            out[0] = -4 * np.abs(u * np.exp(np.abs(np.cos(v))))

        super().__init__(
            compute=compute,
            ndim=ndim,
            bounds=bounds,
            default_bounds=np.array([[-10.0, 10.0] for _ in range(2)]),
            f_global=-10.872299901558,
            x_global=np.array([-np.pi / 2, 0.0]),
            dim_changeable=False,
            dim_default=2,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            shift=shift,
            rotate=rotate,
            rotate_bounds=rotate_bounds,
        )
