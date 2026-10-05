#!/usr/bin/env python
# Created by "Thieu" at 17:26, 22/07/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import typing

import numpy as np

from opfunu.benchmark.func import FuncBenchmark


class Giunta(FuncBenchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.
    """

    name = "Giunta Function"
    latex_formula = (
        r"f(x) = 0.6 + \sum_{i=1}^{n} \left[\sin^{2}\left(1"
        + r"- \frac{16}{15} x_i\right) - \frac{1}{50} \sin\left(4 - \frac{64}{15} x_i\right) - \sin\left(1 - \frac{16}{15} x_i\right)\right]"
    )
    latex_formula_dimension = r"d = 2"
    latex_formula_bounds = r"x_i \in [-1, 1], \forall i \in \llbracket 1, d\rrbracket"
    latex_formula_global_optimum = r"f([0.4673200277395354, 0.4673200169591304]) = 0.06447042053690566"
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
            arg = 16 * x / 15.0 - 1
            out[0] = 0.6 + np.sum(np.sin(arg) + np.sin(arg) ** 2 + np.sin(4 * arg) / 50.0)

        super().__init__(
            compute=compute,
            ndim=ndim,
            bounds=bounds,
            default_bounds=np.array([[-1.0, 1.0] for _ in range(2)]),
            f_global=0.06447042053690566,
            x_global=np.array([0.4673200277395354, 0.4673200169591304]),
            dim_changeable=False,
            dim_default=2,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
        )


class GoldsteinPrice(FuncBenchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.
    """

    name = "Goldstein Price Function"
    latex_formula = (
        r"f(x) = \left[ 1 + (x_1 + x_2 + 1)^2 (19 - 14 x_1 + 3 x_1^2 - 14 x_2 + 6 x_1 x_2 + 3 x_2^2) "
        + r"\right] \left[ 30 + ( 2x_1 - 3 x_2)^2 (18 - 32 x_1 + 12 x_1^2 + 48 x_2 - 36 x_1 x_2 + 27 x_2^2) \right]"
    )
    latex_formula_dimension = r"d = 2"
    latex_formula_bounds = r"x_i \in [-2, 2], \forall i \in \llbracket 1, d\rrbracket"
    latex_formula_global_optimum = r"f([0, -1]) = 3"
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
            a = 1 + (x[0] + x[1] + 1) ** 2 * (
                19 - 14 * x[0] + 3 * x[0] ** 2 - 14 * x[1] + 6 * x[0] * x[1] + 3 * x[1] ** 2
            )
            b = 30 + (2 * x[0] - 3 * x[1]) ** 2 * (
                18 - 32 * x[0] + 12 * x[0] ** 2 + 48 * x[1] - 36 * x[0] * x[1] + 27 * x[1] ** 2
            )
            out[0] = a * b

        super().__init__(
            compute=compute,
            ndim=ndim,
            bounds=bounds,
            default_bounds=np.array([[-2.0, 2.0] for _ in range(2)]),
            f_global=3.0,
            x_global=np.array([0.0, -1.0]),
            dim_changeable=False,
            dim_default=2,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
        )


class Griewank(FuncBenchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.
    """

    name = "Griewank Function"
    latex_formula = r"f(x) = \frac{1}{4000}\sum_{i=1}^n x_i^2 - \prod_{i=1}^n\cos\left(\frac{x_i}{\sqrt{i}}\right) + 1"
    latex_formula_dimension = r"d \in N^+"
    latex_formula_bounds = r"x_i \in [-100, 100], \forall i \in \llbracket 1, d\rrbracket"
    latex_formula_global_optimum = r"f(0,...,0) = 0"
    continuous = True
    linear = False
    convex = True
    unimodal = False
    separable = False

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
    ) -> None:
        def compute(x: np.ndarray, out: np.ndarray) -> None:
            i = np.arange(1.0, np.size(x) + 1.0)
            out[0] = np.sum(x**2 / 4000) - np.prod(np.cos(x / np.sqrt(i))) + 1

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
        )


class Gulf(FuncBenchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.
    """

    name = "Gulf Research Problem"
    latex_formula = r"f(x) = \sum_{i=1}^99 \left( e^{-\frac{\lvert y_i - x_2 \rvert^{x_3}}{x_1}}  - t_i \right)"
    latex_formula_dimension = r"d = 3"
    latex_formula_bounds = r"x_i \in [0, 60], \forall i \in \llbracket 1, d\rrbracket"
    latex_formula_global_optimum = r"f(50, 25, 1.5) = 0"
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
            m = 99.0
            i = np.arange(1.0, m + 1)
            u = 25 + (-50 * np.log(i / 100.0)) ** (2 / 3.0)
            vec = np.exp(-((np.abs(u - x[1])) ** x[2] / x[0])) - i / 100.0
            out[0] = np.sum(vec**2)

        super().__init__(
            compute=compute,
            ndim=ndim,
            bounds=bounds,
            default_bounds=np.array([[0.0, 50.0] for _ in range(3)]),
            f_global=0.0,
            x_global=np.array([50.0, 25.0, 1.5]),
            dim_changeable=False,
            dim_default=3,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
        )


class Gear(FuncBenchmark):
    """
    .. [1] Gavana, A. Global Optimization Benchmarks and AMPGO retrieved 2015
    """

    name = "Gear Problem"
    latex_formula = (
        r"f(x) = \left \{ \frac{1.0}{6.931}"
        + r"- \frac{\lfloor x_1\rfloor \lfloor x_2 \rfloor } {\lfloor x_3 \rfloor \lfloor x_4 \rfloor } \right\}^2"
    )
    latex_formula_dimension = r"d = 4"
    latex_formula_bounds = r"x_i \in [12, 60], \forall i \in \llbracket 1, d\rrbracket"
    latex_formula_global_optimum = r"f(16, 19, 43, 49) = 2.7 \cdot 10^{-12}"
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
            out[0] = (1.0 / 6.931 - np.floor(x[0]) * np.floor(x[1]) / np.floor(x[2]) / np.floor(x[3])) ** 2

        super().__init__(
            compute=compute,
            ndim=ndim,
            bounds=bounds,
            default_bounds=np.array([[12.0, 60.0] for _ in range(4)]),
            f_global=2.7e-12,
            x_global=np.array([16, 19, 43, 49]),
            dim_changeable=False,
            dim_default=4,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
        )
