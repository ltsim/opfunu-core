#!/usr/bin/env python
# Created by "Thieu" at 17:30, 30/07/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import typing

import numpy as np

from opfunu.benchmark.func import FuncBenchmark


class JennrichSampson(FuncBenchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.
    """

    name = "Jennrich-Sampson Function"
    latex_formula = r"f_{\text{JennrichSampson}}(x) = \sum_{i=1}^{10} \left [2 + 2i - (e^{ix_1} + e^{ix_2}) \right ]^2"
    latex_formula_dimension = r"d = 2"
    latex_formula_bounds = r"x_i \in [-1, 1]"
    latex_formula_global_optimum = r"f(0.257825, 0.257825) = 124.3621824"
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
            idx = np.arange(1, 11)
            out[0] = np.sum((2 + 2 * idx - (np.exp(idx * x[0]) + np.exp(idx * x[1]))) ** 2)

        super().__init__(
            compute=compute,
            ndim=ndim,
            bounds=bounds,
            default_bounds=np.array([[-1.0, 1.0] for _ in range(2)]),
            f_global=124.36,
            x_global=np.array([0.257825, 0.257825]),
            dim_changeable=False,
            dim_default=2,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
        )


class Judge(FuncBenchmark):
    """
    .. [1] Gavana, A. Global Optimization Benchmarks and AMPGO retrieved 2015

    .. math::
        f_{\text{Judge}}(x) = \\sum_{i=1}^{20}
        \\left [ \\left (x_1 + A_i x_2 + B x_2^2 \right ) - C_i \right ]^2

    .. math::
        \begin{cases}
        C = [4.284, 4.149, 3.877, 0.533, 2.211, 2.389, 2.145,
        3.231, 1.998, 1.379, 2.106, 1.428, 1.011, 2.179, 2.858, 1.388, 1.651,
        1.593, 1.046, 2.152] \\
        A = [0.286, 0.973, 0.384, 0.276, 0.973, 0.543, 0.957, 0.948, 0.543,
             0.797, 0.936, 0.889, 0.006, 0.828, 0.399, 0.617, 0.939, 0.784,
             0.072, 0.889] \\
        B = [0.645, 0.585, 0.310, 0.058, 0.455, 0.779, 0.259, 0.202, 0.028,
             0.099, 0.142, 0.296, 0.175, 0.180, 0.842, 0.039, 0.103, 0.620,
             0.158, 0.704]
        \\end{cases}
    with :math:`x_i \\in [-10, 10]` for :math:`i = 1, 2`.
    *Global optimum*: :math:`f(x_i) = 16.0817307` for :math:`\\mathbf{x} = [0.86479, 1.2357]`.
    """

    name = "Judge Function"
    latex_formula = (
        r"f_{\text{Judge}}(x) = \sum_{i=1}^{20} \left [ \left (x_1 + A_i x_2 + B x_2^2 \right ) - C_i \right ]^2"
    )
    latex_formula_dimension = r"d \in N^+"
    latex_formula_bounds = r"x_i \in [-10, 10]"
    latex_formula_global_optimum = r"f(0.86479, 1.2357) = 16.0817307"
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
        def compute(x: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray, out: np.ndarray) -> None:
            out[0] = np.sum(((x[0] + x[1] * a + (x[1] ** 2.0) * b) - c) ** 2.0)

        self.c = np.asarray(
            [
                4.284,
                4.149,
                3.877,
                0.533,
                2.211,
                2.389,
                2.145,
                3.231,
                1.998,
                1.379,
                2.106,
                1.428,
                1.011,
                2.179,
                2.858,
                1.388,
                1.651,
                1.593,
                1.046,
                2.152,
            ]
        )
        self.a = np.asarray(
            [
                0.286,
                0.973,
                0.384,
                0.276,
                0.973,
                0.543,
                0.957,
                0.948,
                0.543,
                0.797,
                0.936,
                0.889,
                0.006,
                0.828,
                0.399,
                0.617,
                0.939,
                0.784,
                0.072,
                0.889,
            ]
        )
        self.b = np.asarray(
            [
                0.645,
                0.585,
                0.31,
                0.058,
                0.455,
                0.779,
                0.259,
                0.202,
                0.028,
                0.099,
                0.142,
                0.296,
                0.175,
                0.18,
                0.842,
                0.039,
                0.103,
                0.62,
                0.158,
                0.704,
            ]
        )

        super().__init__(
            compute=compute,
            ndim=ndim,
            bounds=bounds,
            default_bounds=np.array([[-10.0, 10.0] for _ in range(2)]),
            f_global=16.0817307,
            x_global=np.array([0.86479, 1.2357]),
            dim_changeable=False,
            dim_default=2,
            param_names=["a", "b", "c"],
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
        )
