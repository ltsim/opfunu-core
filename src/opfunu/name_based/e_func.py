#!/usr/bin/env python
# Created by "Thieu" at 11:04, 21/07/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%


import typing

import numpy as np

from opfunu.benchmark.func import FuncBenchmark


class Easom(FuncBenchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.
    """

    name = "Easom Function"
    latex_formula = (
        r"f(x) = a - \frac{a}{e^{b \sqrt{\frac{\sum_{i=1}^{n}"
        + r"x_i^{2}}{n}}}} + e - e^{\frac{\sum_{i=1}^{n} \cos\left(c x_i\right)} {n}}"
    )
    latex_formula_dimension = r"d = 2"
    latex_formula_bounds = r"x_i \in [-100, 100], \forall i \in \llbracket 1, d\rrbracket"
    latex_formula_global_optimum = r"f(pi, pi) = -1"
    continuous = True
    linear = False
    convex = True
    unimodal = False
    separable = True

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
            a = (x[0] - np.pi) ** 2 + (x[1] - np.pi) ** 2
            out[0] = -np.cos(x[0]) * np.cos(x[1]) * np.exp(-a)

        super().__init__(
            compute=compute,
            ndim=ndim,
            bounds=bounds,
            default_bounds=np.array([[-100.0, 100.0] for _ in range(2)]),
            f_global=-1.0,
            x_global=lambda nd: np.pi * np.ones(nd),
            dim_changeable=False,
            dim_default=2,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
        )


class ElAttarVidyasagarDutta(FuncBenchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.
    """

    name = "El-Attar-Vidyasagar-Dutta Function"
    latex_formula = r"f(x) = (x_1^2 + x_2 - 10)^2 + (x_1 + x_2^2 - 7)^2 + (x_1^2 + x_2^3 - 1)^2"
    latex_formula_dimension = r"d = 2"
    latex_formula_bounds = r"x_i \in [-500, 500], \forall i \in \llbracket 1, d\rrbracket"
    latex_formula_global_optimum = r"f(3.40918683, -2.17143304) = 1.712780354"
    continuous = True
    linear = False
    convex = True
    unimodal = True
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
            out[0] = (x[0] ** 2 + x[1] - 10) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2 + (x[0] ** 2 + x[1] ** 3 - 1) ** 2

        super().__init__(
            compute=compute,
            ndim=ndim,
            bounds=bounds,
            default_bounds=np.array([[-500.0, 500.0] for _ in range(2)]),
            f_global=1.712780354,
            x_global=np.array([3.40918683, -2.17143304]),
            dim_changeable=False,
            dim_default=2,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
        )


class EggCrate(FuncBenchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.
    """

    name = "Egg Crate Function"
    latex_formula = r"f(x) = x_1^2 + x_2^2 + 25 \left[ \sin^2(x_1) + \sin^2(x_2) \right]"
    latex_formula_dimension = r"d = 2"
    latex_formula_bounds = r"x_i \in [-5, 5], \forall i \in \llbracket 1, d\rrbracket"
    latex_formula_global_optimum = r"f(0, 0) = 0"
    continuous = True
    linear = False
    convex = True
    unimodal = False
    separable = True

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
            out[0] = x[0] ** 2 + x[1] ** 2 + 25 * (np.sin(x[0]) ** 2 + np.sin(x[1]) ** 2)

        super().__init__(
            compute=compute,
            ndim=ndim,
            bounds=bounds,
            default_bounds=np.array([[-500.0, 500.0] for _ in range(2)]),
            f_global=0.0,
            x_global=np.array([0.0, 0.0]),
            dim_changeable=False,
            dim_default=2,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
        )


class EggHolder(FuncBenchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.
    """

    name = "Egg Holder Function"
    latex_formula = (
        r"f(x) = \sum_{1}^{n - 1}\left[-\left(x_{i + 1}"
        + r"+ 47 \right ) \sin\sqrt{\lvert x_{i+1} + x_i/2 + 47 \rvert} - x_i \sin\sqrt{\lvert x_i - (x_{i + 1} + 47)\rvert}\right ]"
    )
    latex_formula_dimension = r"d \in N^+"
    latex_formula_bounds = r"x_i \in [-512, 512], \forall i \in \llbracket 1, d\rrbracket"
    latex_formula_global_optimum = r"f(512, 404.2319) = -959.640662711"
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
            vec = -(x[1:] + 47) * np.sin(np.sqrt(np.abs(x[1:] + x[:-1] / 2.0 + 47))) - x[:-1] * np.sin(
                np.sqrt(np.abs(x[:-1] - (x[1:] + 47)))
            )
            out[0] = np.sum(vec)

        super().__init__(
            compute=compute,
            ndim=ndim,
            bounds=bounds,
            default_bounds=np.array([[-512.0, 512.0] for _ in range(2)]),
            f_global=-959.640662711,
            x_global=lambda nd: np.zeros(nd),
            dim_changeable=True,
            dim_default=2,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
        )


class Exponential(FuncBenchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.
    """

    name = "Exponential Function"
    latex_formula = r"f(x) = -e^{-0.5 \sum_{i=1}^n x_i^2}"
    latex_formula_dimension = r"d \in N^+"
    latex_formula_bounds = r"x_i \in [-1, 1], \forall i \in \llbracket 1, d\rrbracket"
    latex_formula_global_optimum = r"f(0,..,0) = -1"
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
            out[0] = -np.exp(-0.5 * np.sum(x**2.0))

        super().__init__(
            compute=compute,
            ndim=ndim,
            bounds=bounds,
            default_bounds=np.array([[-1.0, 1.0] for _ in range(2)]),
            f_global=-1,
            x_global=lambda nd: np.zeros(nd),
            dim_changeable=True,
            dim_default=2,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
        )


class Exp2(FuncBenchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.
    """

    name = "Exp 2 Function"
    latex_formula = r"f(x) = \sum_{i=0}^9 \left ( e^{-ix_1/10} - 5e^{-ix_2/10} - e^{-i/10} + 5e^{-i} \right )^2"
    latex_formula_dimension = r"d = 2"
    latex_formula_bounds = r"x_i \in [0, 20], \forall i \in \llbracket 1, d\rrbracket"
    latex_formula_global_optimum = r"f(1, 10) = 0"
    continuous = True
    linear = False
    convex = True
    unimodal = False
    separable = True

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
            i = np.arange(10.0)
            vec = (np.exp(-i * x[0] / 10.0) - 5 * np.exp(-i * x[1] / 10.0) - np.exp(-i / 10.0) + 5 * np.exp(-i)) ** 2
            out[0] = np.sum(vec)

        super().__init__(
            compute=compute,
            ndim=ndim,
            bounds=bounds,
            default_bounds=np.array([[0.0, 20.0] for _ in range(2)]),
            f_global=0.0,
            x_global=np.array([1.0, 10.0]),
            dim_changeable=False,
            dim_default=2,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
        )


class Eckerle4(FuncBenchmark):
    """
    [1] Eckerle, K., NIST (1979). Circular Interference Transmittance Study.
    [2] https://www.itl.nist.gov/div898/strd/nls/data/eckerle4.shtml
    """

    name = "Eckerle 4 Function"
    latex_formula = r"f(x) = "
    latex_formula_dimension = r"d = 3"
    latex_formula_bounds = r"0 <= x_1 <=20, 1 <= x_2 <= 20, 10 <= x_3 <= 600"
    latex_formula_global_optimum = r"f(1.5543827178, 4.0888321754, 4.5154121844e2) = 1.4635887487E-03"
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
        def compute(x: np.ndarray, b: np.ndarray, a: np.ndarray, out: np.ndarray) -> None:
            vec = x[0] / x[1] * np.exp(-((b - x[2]) ** 2) / (2 * x[1] ** 2))
            out[0] = np.sum((a - vec) ** 2)

        self.a = np.asarray(
            [
                0.0001575,
                0.0001699,
                0.000235,
                0.0003102,
                0.0004917,
                0.000871,
                0.0017418,
                0.00464,
                0.0065895,
                0.0097302,
                0.0149002,
                0.023731,
                0.0401683,
                0.0712559,
                0.1264458,
                0.2073413,
                0.2902366,
                0.3445623,
                0.3698049,
                0.3668534,
                0.3106727,
                0.2078154,
                0.1164354,
                0.0616764,
                0.03372,
                0.0194023,
                0.0117831,
                0.0074357,
                0.0022732,
                0.00088,
                0.0004579,
                0.0002345,
                0.0001586,
                0.0001143,
                7.1e-05,
            ]
        )
        self.b = np.asarray(
            [
                400.0,
                405.0,
                410.0,
                415.0,
                420.0,
                425.0,
                430.0,
                435.0,
                436.5,
                438.0,
                439.5,
                441.0,
                442.5,
                444.0,
                445.5,
                447.0,
                448.5,
                450.0,
                451.5,
                453.0,
                454.5,
                456.0,
                457.5,
                459.0,
                460.5,
                462.0,
                463.5,
                465.0,
                470.0,
                475.0,
                480.0,
                485.0,
                490.0,
                495.0,
                500.0,
            ]
        )

        super().__init__(
            compute=compute,
            ndim=ndim,
            bounds=bounds,
            default_bounds=np.array([[0.0, 20.0], [1.0, 20.0], [10.0, 600.0]]),
            f_global=0.0014635887487,
            x_global=np.array([1.5543827178, 4.0888321754, 451.54121844]),
            dim_changeable=False,
            dim_default=3,
            param_names=["b", "a"],
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
        )
