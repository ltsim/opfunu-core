#!/usr/bin/env python
# Created by "Thieu" at 17:31, 30/07/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import typing

import numpy as np

from opfunu.benchmark.func import FuncBenchmark


class Qing(FuncBenchmark):
    """
    .. [1]  Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization Problems
    Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.

    .. math::

        f_{\text{Qing}}(x) = \\sum_{i=1}^{n} (x_i^2 - i)^2

    Here, :math:`n` represents the number of dimensions and :math:`x_i \\in [-500, 500]` for :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x_i = \\pm \\sqrt(i)` for :math:`i = 1, ..., n`
    """

    name = "Qing Function"
    latex_formula = r"f_{\text{Qing}}(x) = \sum_{i=1}^{n} (x_i^2 - i)^2"
    latex_formula_dimension = r"d = n"
    latex_formula_bounds = r"x_i \in [-10, 10, ..., 10]"
    latex_formula_global_optimum = r"f(0, 0, ...,0) = 1.0"
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
    ) -> None:
        def compute(x: np.ndarray, out: np.ndarray) -> None:
            i = np.arange(1, x.shape[0] + 1)
            out[0] = np.sum((x**2.0 - i) ** 2.0)

        super().__init__(
            compute=compute,
            ndim=ndim,
            bounds=bounds,
            default_bounds=np.array([[-500.0, 500.0] for _ in range(2)]),
            f_global=0.0,
            x_global=lambda nd: np.array([np.sqrt(_) for _ in range(1, nd + 1)]),
            dim_changeable=True,
            dim_default=2,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            shift=shift,
        )


class Quadratic(FuncBenchmark):
    """
    .. [1]  Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization Problems
    Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.

    .. math::

        f_{\text{Quadratic}}(x) = -3803.84 - 138.08x_1 - 232.92x_2 + 128.08x_1^2+ 203.64x_2^2 + 182.25x_1x_2

    Here, :math:`n` represents the number of dimensions and :math:`x_i \\in [-10, 10]` for :math:`i = 1, 2`.

    *Global optimum*: :math:`f(x) = -3873.72418` for :math:`x = [0.19388, 0.48513]`
    """

    name = "Quadratic Function"
    latex_formula = (
        r"f_{\text{Quadratic}}(x) = -3803.84 - 138.08x_1 - 232.92x_2 + 128.08x_1^2+ 203.64x_2^2 + 182.25x_1x_2"
    )
    latex_formula_dimension = r"d = n"
    latex_formula_bounds = r"x_i \in [-10, 10, ..., 10]"
    latex_formula_global_optimum = r"f(0, 0, ...,0) = 1.0"
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
        shift: typing.Any = None,
    ) -> None:
        def compute(x: np.ndarray, out: np.ndarray) -> None:
            out[0] = (
                -3803.84
                - 138.08 * x[0]
                - 232.92 * x[1]
                + 128.08 * x[0] ** 2.0
                + 203.64 * x[1] ** 2.0
                + 182.25 * x[0] * x[1]
            )

        super().__init__(
            compute=compute,
            ndim=ndim,
            bounds=bounds,
            default_bounds=np.array([[-10.0, 10.0] for _ in range(2)]),
            f_global=-3873.72418,
            x_global=np.array([0.19388, 0.48513]),
            dim_changeable=False,
            dim_default=2,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            shift=shift,
        )


class Quartic(FuncBenchmark):
    """
    .. [1]  Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization Problems
    Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.

    .. math::

        f_{\text{Quartic}}(x) =

    Here, :math:`n` represents the number of dimensions and :math:`x_i \\in [-10, 10]` for :math:`i = 1, 2`.

    *Global optimum*: :math:`f(x) = -3873.72418` for :math:`x = [0.19388, 0.48513]`
    """

    name = "Quartic Function"
    latex_formula = r"f_{\text{Quartic}}(x) = "
    latex_formula_dimension = r"d = n"
    latex_formula_bounds = r"x_i \in [-10, 10, ..., 10]"
    latex_formula_global_optimum = r"f(0, 0, ...,0) = 1.0"
    continuous = True
    linear = False
    convex = True
    unimodal = False
    separable = False

    differentiable = True
    scalable = False
    randomized_term = True
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
    ) -> None:
        def compute(x: np.ndarray, out: np.ndarray) -> None:
            i = np.arange(1, x.shape[0] + 1)
            out[0] = np.sum(i * x**4.0) + np.random.rand()

        super().__init__(
            compute=compute,
            ndim=ndim,
            bounds=bounds,
            default_bounds=np.array([[-1.28, 1.28] for _ in range(2)]),
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
        )


class Quintic(FuncBenchmark):
    """
    .. [1]  Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization Problems
    Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.

    .. math::

        f_{\text{Quintic}}(x) = \\sum_{i=1}^{n} \\left|{x_{i}^{5} - 3 x_{i}^{4}+ 4 x_{i}^{3} + 2 x_{i}^{2} - 10 x_{i} -4}\right|

    Here, :math:`n` represents the number of dimensions and :math:`x_i \\in [-10, 10]` for :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(x_i) = 0` for :math:`x_i = -1` for :math:`i = 1, ..., n`
    """

    name = "Quartic Function"
    latex_formula = r"f_{\text{Quintic}}(x) = \sum_{i=1}^{n} \left|{x_{i}^{5} - 3 x_{i}^{4}+ 4 x_{i}^{3} + 2 x_{i}^{2} - 10 x_{i} -4}\right|"
    latex_formula_dimension = r"d = n"
    latex_formula_bounds = r"x_i \in [-10, 10, ..., 10]"
    latex_formula_global_optimum = r"f(0, 0, ...,0) = 1.0"
    continuous = True
    linear = False
    convex = True
    unimodal = False
    separable = True

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
            out[0] = np.sum(np.abs(x**5 - 3 * x**4 + 4 * x**3 + 2 * x**2 - 10 * x - 4))

        super().__init__(
            compute=compute,
            ndim=ndim,
            bounds=bounds,
            default_bounds=np.array([[-10.0, 10.0] for _ in range(2)]),
            f_global=0.0,
            x_global=lambda nd: -1 * np.ones(nd),
            dim_changeable=True,
            dim_default=2,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            shift=shift,
        )
