#!/usr/bin/env python
# Created by "Thieu" at 17:55, 22/07/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import typing

import numpy as np

from opfunu.benchmark.func import FuncBenchmark


class Hansen(FuncBenchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.
    """

    name = "Hansen Function"
    latex_formula = (
        r"f(x) = \left[ \sum_{i=0}^4(i+1)\cos(ix_1+i+1)\right ]\left[\sum_{j=0}^4(j+1)\cos[(j+2)x_2+j+1])\right ]"
    )
    latex_formula_dimension = r"d = 2"
    latex_formula_bounds = r"x_i \in [-10, 10], \forall i \in \llbracket 1, d\rrbracket"
    latex_formula_global_optimum = r"f(-7.58989583, -7.70831466) = -176.54179"
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
        shift: typing.Any = None,
        rotate: typing.Any = None,
        rotate_bounds: bool = True,
    ) -> None:
        def compute(x: np.ndarray, out: np.ndarray) -> None:
            i = np.arange(5.0)
            a = (i + 1) * np.cos(i * x[0] + i + 1)
            b = (i + 1) * np.cos((i + 2) * x[1] + i + 1)
            out[0] = np.sum(a) * np.sum(b)

        super().__init__(
            compute=compute,
            ndim=ndim,
            bounds=bounds,
            default_bounds=np.array([[-10.0, 10.0] for _ in range(2)]),
            f_global=-176.54179,
            x_global=np.array([-7.58989583, -7.70831466]),
            dim_changeable=False,
            dim_default=2,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            shift=shift,
            rotate=rotate,
            rotate_bounds=rotate_bounds,
        )


class Hartmann3(FuncBenchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.
    """

    name = "Hartman 3 Function"
    latex_formula = r"f(x) = -\sum\limits_{i=1}^{4} c_i e^{-\sum\limits_{j=1}^{n}a_{ij}(x_j - p_{ij})^2}"
    latex_formula_dimension = r"d = 3"
    latex_formula_bounds = r"x_i \in [0, 1], \forall i \in \llbracket 1, d\rrbracket"
    latex_formula_global_optimum = r"f([0.11461292,  0.55564907,  0.85254697]) = -3.8627821478"
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
        rotate: typing.Any = None,
        rotate_bounds: bool = True,
    ) -> None:
        def compute(x: np.ndarray, a: np.ndarray, p: np.ndarray, c: np.ndarray, out: np.ndarray) -> None:
            XX = np.atleast_2d(x)
            d = np.sum(a * (XX - p) ** 2, axis=1)
            out[0] = -np.sum(c * np.exp(-d))

        self.a = np.asarray([[3.0, 10.0, 30.0], [0.1, 10.0, 35.0], [3.0, 10.0, 30.0], [0.1, 10.0, 35.0]])
        self.p = np.asarray(
            [[0.3689, 0.117, 0.2673], [0.4699, 0.4387, 0.747], [0.1091, 0.8732, 0.5547], [0.03815, 0.5743, 0.8828]]
        )
        self.c = np.asarray([1.0, 1.2, 3.0, 3.2])

        super().__init__(
            compute=compute,
            ndim=ndim,
            bounds=bounds,
            default_bounds=np.array([[0.0, 1.0] for _ in range(3)]),
            f_global=-3.8627821478,
            x_global=np.array([0.11461292, 0.55564907, 0.85254697]),
            dim_changeable=False,
            dim_default=3,
            param_names=["a", "p", "c"],
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            shift=shift,
            rotate=rotate,
            rotate_bounds=rotate_bounds,
        )


class Hartmann6(FuncBenchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.
    """

    name = "Hartman 6 Function"
    latex_formula = r"f(x) = -\sum\limits_{i=1}^{4} c_i e^{-\sum\limits_{j=1}^{n}a_{ij}(x_j - p_{ij})^2}"
    latex_formula_dimension = r"d = 3"
    latex_formula_bounds = r"x_i \in [0, 1], \forall i \in \llbracket 1, d\rrbracket"
    latex_formula_global_optimum = (
        r"f([0.20168952, 0.15001069, 0.47687398, 0.27533243, 0.31165162, 0.65730054]) = -3.32236801141551"
    )
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
        rotate: typing.Any = None,
        rotate_bounds: bool = True,
    ) -> None:
        def compute(x: np.ndarray, a: np.ndarray, p: np.ndarray, c: np.ndarray, out: np.ndarray) -> None:
            XX = np.atleast_2d(x)
            d = np.sum(a * (XX - p) ** 2, axis=1)
            out[0] = -np.sum(c * np.exp(-d))

        self.a = np.asarray(
            [
                [10.0, 3.0, 17.0, 3.5, 1.7, 8.0],
                [0.05, 10.0, 17.0, 0.1, 8.0, 14.0],
                [3.0, 3.5, 1.7, 10.0, 17.0, 8.0],
                [17.0, 8.0, 0.05, 10.0, 0.1, 14.0],
            ]
        )
        self.p = np.asarray(
            [
                [0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
                [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
                [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.665],
                [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381],
            ]
        )
        self.c = np.asarray([1.0, 1.2, 3.0, 3.2])

        super().__init__(
            compute=compute,
            ndim=ndim,
            bounds=bounds,
            default_bounds=np.array([[0.0, 1.0] for _ in range(6)]),
            f_global=-3.32236801141551,
            x_global=np.array([0.20168952, 0.15001069, 0.47687398, 0.27533243, 0.31165162, 0.65730054]),
            dim_changeable=False,
            dim_default=6,
            param_names=["a", "p", "c"],
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            shift=shift,
            rotate=rotate,
            rotate_bounds=rotate_bounds,
        )


class HelicalValley(FuncBenchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.
    """

    name = "Helical Valley"
    latex_formula = r"f(x) = 100{[z-10\Psi(x_1,x_2)]^2 +(\sqrt{x_1^2+x_2^2}-1)^2}+x_3^2"
    latex_formula_dimension = r"d \in N^+"
    latex_formula_bounds = r"x_i \in [-10, 10], \forall i \in \llbracket 1, d\rrbracket"
    latex_formula_global_optimum = r"f([1.0, 0.0, 0.0]) = 0"
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
        rotate: typing.Any = None,
        rotate_bounds: bool = True,
    ) -> None:
        def compute(x: np.ndarray, out: np.ndarray) -> None:
            r = np.sqrt(x[0] ** 2 + x[1] ** 2)
            theta = 1 / (2.0 * np.pi) * np.arctan2(x[1], x[0])
            out[0] = x[2] ** 2 + 100 * ((x[2] - 10 * theta) ** 2 + (r - 1) ** 2)

        super().__init__(
            compute=compute,
            ndim=ndim,
            bounds=bounds,
            default_bounds=np.array([[-10.0, 10.0] for _ in range(3)]),
            f_global=0.0,
            x_global=np.array([1.0, 0.0, 0.0]),
            dim_changeable=False,
            dim_default=3,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            shift=shift,
            rotate=rotate,
            rotate_bounds=rotate_bounds,
        )


class Himmelblau(FuncBenchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.
    """

    name = "Himmelblau Function"
    latex_formula = r"f(x) = (x_1^2 + x_2 - 11)^2 + (x_1 + x_2^2 - 7)^2"
    latex_formula_dimension = r"d \in N^+"
    latex_formula_bounds = r"x_i \in [-5, 5], \forall i \in \llbracket 1, d\rrbracket"
    latex_formula_global_optimum = r"f([3, 2]) = 0"
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
        rotate: typing.Any = None,
        rotate_bounds: bool = True,
    ) -> None:
        def compute(x: np.ndarray, out: np.ndarray) -> None:
            out[0] = (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2

        super().__init__(
            compute=compute,
            ndim=ndim,
            bounds=bounds,
            default_bounds=np.array([[-5.0, 5.0] for _ in range(2)]),
            f_global=0.0,
            x_global=np.array([3.0, 2.0]),
            dim_changeable=False,
            dim_default=2,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            shift=shift,
            rotate=rotate,
            rotate_bounds=rotate_bounds,
        )


class Hosaki(FuncBenchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.
    """

    name = "Hosaki Function"
    latex_formula = (
        r"f(x) = \left ( 1 - 8 x_1 + 7 x_1^2 - \frac{7}{3} x_1^3 + \frac{1}{4} x_1^4 \right ) x_2^2 e^{-x_1}"
    )
    latex_formula_dimension = r"d = 2"
    latex_formula_bounds = r" 0 <= x_1 <= 5, 0 <= x2 <= 6"
    latex_formula_global_optimum = r"f(4, 2) = −2.3458"
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
        rotate: typing.Any = None,
        rotate_bounds: bool = True,
    ) -> None:
        def compute(x: np.ndarray, out: np.ndarray) -> None:
            val = 1 - 8 * x[0] + 7 * x[0] ** 2 - 7 / 3.0 * x[0] ** 3 + 0.25 * x[0] ** 4
            out[0] = val * x[1] ** 2 * np.exp(-x[1])

        super().__init__(
            compute=compute,
            ndim=ndim,
            bounds=bounds,
            default_bounds=np.array([[0.0, 5.0], [0.0, 6.0]]),
            f_global=-2.345811576101292,
            x_global=np.array([4.0, 2.0]),
            dim_changeable=False,
            dim_default=2,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            shift=shift,
            rotate=rotate,
            rotate_bounds=rotate_bounds,
        )


class HolderTable(FuncBenchmark):
    """
    .. [1] Gavana, A. Global Optimization Benchmarks and AMPGO retrieved 2015
    """

    name = "Hosaki Function"
    latex_formula = (
        r"f(x) = - \left|{e^{\left|{1"
        + r"- \frac{\sqrt{x_{1}^{2} + x_{2}^{2}}}{\pi} }\right|} \sin\left(x_{1}\right) \cos\left(x_{2}\right)}\right|"
    )
    latex_formula_dimension = r"d = 2"
    latex_formula_bounds = r" 0 <= x_1 <= 5, 0 <= x2 <= 6"
    latex_formula_global_optimum = r"f(\pm 9.664590028909654) = -19.20850256788675"
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
        rotate: typing.Any = None,
        rotate_bounds: bool = True,
    ) -> None:
        def compute(x: np.ndarray, out: np.ndarray) -> None:
            out[0] = -np.abs(np.sin(x[0]) * np.cos(x[1]) * np.exp(np.abs(1 - np.sqrt(x[0] ** 2 + x[1] ** 2) / np.pi)))

        super().__init__(
            compute=compute,
            ndim=ndim,
            bounds=bounds,
            default_bounds=np.array([[-10.0, 10.0] for _ in range(2)]),
            f_global=-19.20850256788675,
            x_global=np.array([8.055023472141116, 9.664590028909654]),
            dim_changeable=False,
            dim_default=2,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            shift=shift,
            rotate=rotate,
            rotate_bounds=rotate_bounds,
        )
        self.x_globals = np.array(
            [
                [8.055023472141116, 9.664590028909654],
                [-8.055023472141116, 9.664590028909654],
                [8.055023472141116, -9.664590028909654],
                [-8.055023472141116, -9.664590028909654],
            ]
        )
