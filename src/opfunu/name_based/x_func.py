#!/usr/bin/env python
# Created by "Thieu" at 17:32, 30/07/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import typing

import numpy as np

from opfunu.benchmark.func import FuncBenchmark


class XinSheYang01(FuncBenchmark):
    """
    .. [1]  Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization Problems
    Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.

    .. math::

         f(x) = \\sum_{i=1}^{n} \\epsilon_i \\lvert x_i \rvert^i

    The variable :math:`\\epsilon_i, (i = 1, ..., n)` is a random variable uniformly distributed in :math:`[0, 1]`.

    Here, :math:`n` represents the number of dimensions and :math:`x_i \\in [-5, 5]` for :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x_i = 0` for :math:`i = 1, ..., n`
    """

    name = "Xin-She Yang 1 Function"
    latex_formula = r"f(x) = \sum_{i=1}^{n} \epsilon_i \lvert x_i \rvert^i"
    latex_formula_dimension = r"d = n"
    latex_formula_bounds = r"x_i \in [-10, 10, ..., 10]"
    latex_formula_global_optimum = r"f(0, 0, ...,0) = 1.0"
    continuous = True
    linear = False
    convex = True
    unimodal = False
    separable = True

    differentiable = False
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
        rotate: typing.Any = None,
        rotate_bounds: bool = True,
    ) -> None:
        def compute(x: np.ndarray, out: np.ndarray) -> None:
            i = np.arange(1.0, x.shape[0] + 1.0)
            out[0] = np.sum(np.random.random(x.shape[0]) * (np.abs(x) ** i))

        super().__init__(
            compute=compute,
            ndim=ndim,
            bounds=bounds,
            default_bounds=np.array([[-5.0, 5.0] for _ in range(2)]),
            f_global=0.0,
            x_global=lambda nd: np.zeros(nd),
            dim_changeable=True,
            dim_default=2,
            param_names=[],
            plain=True,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            shift=shift,
            rotate=rotate,
            rotate_bounds=rotate_bounds,
        )
