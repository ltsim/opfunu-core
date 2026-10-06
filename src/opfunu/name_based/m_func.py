#!/usr/bin/env python
# Created by "Thieu" at 17:31, 30/07/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import math
import typing

import numpy as np

from opfunu.benchmark.func import FuncBenchmark


class Matyas(FuncBenchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.

    .. math::
        f_{\text{Matyas}}(x) = 0.26(x_1^2 + x_2^2) - 0.48 x_1 x_2

    Here :math:`x_i \\in [-10, 10]` for :math:`i = 1, 2`.
    *Global optimum*: :math:`f(x) = 0.0`for :math:`x = [0, 0]`
    """

    name = "Matyas Function"
    latex_formula = r"f_{\text{Matyas}}(x) = 0.26(x_1^2 + x_2^2) - 0.48 x_1 x_2"
    latex_formula_dimension = r"d = 2"
    latex_formula_bounds = r"x_i \in [-10, 10]"
    latex_formula_global_optimum = r"f(0, 0) = 0"
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
            out[0] = 0.26 * (x[0] ** 2 + x[1] ** 2) - 0.48 * x[0] * x[1]

        super().__init__(
            compute=compute,
            ndim=ndim,
            bounds=bounds,
            default_bounds=np.array([[-10.0, 10.0] for _ in range(2)]),
            f_global=0.0,
            x_global=np.array([0.0, 0.0]),
            dim_changeable=False,
            dim_default=2,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
        )


class McCormick(FuncBenchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.

    .. math::
        f(x) = - x_{1} + 2 x_{2} + \\left(x_{1} - x_{2}\right)^{2} + \\sin\\left(x_{1} + x_{2}\right) + 1

    Here :math:`x_1 \\in [-1.5, 4], x_2 \\in [-3, 4]` .
    *Global optimum*: :math:`f(x) = -1.913222954981037`for :math:`x = [-0.5471975602214493, -1.547197559268372]`
    """

    name = "McCormick Function"
    latex_formula = r"f(x) = - x_{1} + 2 x_{2} + \left(x_{1} - x_{2}\right)^{2} + \sin\left(x_{1} + x_{2}\right) + 1"
    latex_formula_dimension = r"d = 2"
    latex_formula_bounds = r"x_1 \in [-1.5, 4], x_2 \in [-3, 4]"
    latex_formula_global_optimum = r"f(-0.5471975602214493, -1.547197559268372) = -1.913222954981037"
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
            out[0] = np.sin(x[0] + x[1]) + (x[0] - x[1]) ** 2 - 1.5 * x[0] + 2.5 * x[1] + 1

        super().__init__(
            compute=compute,
            ndim=ndim,
            bounds=bounds,
            default_bounds=np.array([[-1.5, 4.0], [-3.0, 4.0]]),
            f_global=-1.913222954981037,
            x_global=np.array([-0.5471975602214493, -1.547197559268372]),
            dim_changeable=False,
            dim_default=2,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
        )


class Meyer(FuncBenchmark):
    """
    .. [1] https://www.itl.nist.gov/div898/strd/nls/data/mgh10.shtml
    """

    name = "Meyer Function"
    latex_formula = r"f(x)"
    latex_formula_dimension = r"d = 3"
    latex_formula_bounds = r"x_1 \in [0, 1], x_2 \in [100, 1000], x_3 \in [100, 500]"
    latex_formula_global_optimum = r"f(5.6096364710e-3, 6.1813463463e2, 3.4522363462e2) = 8.7945855171e1"
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
            vec = x[0] * np.exp(x[1] / (b + x[2]))
            out[0] = np.sum((a - vec) ** 2)

        self.a = np.asarray(
            [
                34780.0,
                28610.0,
                23650.0,
                19630.0,
                16370.0,
                13720.0,
                11540.0,
                9744.0,
                8261.0,
                7030.0,
                6005.0,
                5147.0,
                4427.0,
                3820.0,
                3307.0,
                2872.0,
            ]
        )
        self.b = np.asarray(
            [50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0, 125.0]
        )

        super().__init__(
            compute=compute,
            ndim=ndim,
            bounds=bounds,
            default_bounds=np.array([[0.0, 1.0], [100.0, 1000.0], [100.0, 500.0]]),
            f_global=87.945855171,
            x_global=np.array([0.005609636471, 618.13463463, 345.22363462]),
            dim_changeable=False,
            dim_default=3,
            param_names=["b", "a"],
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
        )


class Michalewicz(FuncBenchmark):
    """
    .. [1] Adorio, E. MVF - "Multivariate Test Functions Library in C for
    Unconstrained Global Optimization", 2005

    .. math::
        f(x) = - \\sum_{i=1}^{2} \\sin\\left(x_i\right) \\sin^{2 m}\\left(\frac{i x_i^{2}}{\\pi}\right)

    Here :math:`x_i \\in [0, \\pi]`.
    *Global optimum*: :math:`f(x) = -1.8013`for :math:`x = [0, 0]`
    """

    name = "McCormick Function"
    latex_formula = r"f(x) = - x_{1} + 2 x_{2} + \left(x_{1} - x_{2}\right)^{2} + \sin\left(x_{1} + x_{2}\right) + 1"
    latex_formula_dimension = r"d = 2"
    latex_formula_bounds = r"x_i \in [0, \pi]`"
    latex_formula_global_optimum = r"f(0, 0) = -1.8013"
    continuous = True
    linear = False
    convex = False
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
            m = 10.0
            idx = np.arange(1, x.shape[0] + 1)
            out[0] = -np.sum(np.sin(x) * np.sin(idx * x**2 / np.pi) ** (2 * m))

        super().__init__(
            compute=compute,
            ndim=ndim,
            bounds=bounds,
            default_bounds=np.array([[0.0, np.pi] for _ in range(2)]),
            f_global=-1.8013,
            x_global=np.array([0, 0]),
            dim_changeable=False,
            dim_default=2,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
        )


class MieleCantrell(FuncBenchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.

    .. math::
        f(x) = (e^{-x_1} - x_2)^4 + 100(x_2 - x_3)^6 + \tan^4(x_3 - x_4) + x_1^8

    Here :math:`x_i \\in [-1, 1] for i \\in [1, 4]`.
    *Global optimum*: :math:`f(x) = 0`for :math:`x = [0, 1, 1, 1]`
    """

    name = "Miele Cantrell Function"
    latex_formula = r"f(x) = (e^{-x_1} - x_2)^4 + 100(x_2 - x_3)^6 + \tan^4(x_3 - x_4) + x_1^8"
    latex_formula_dimension = r"d = 4"
    latex_formula_bounds = r"x_i \in [-1, 1] for i \in [1, 4]"
    latex_formula_global_optimum = r"f(0, 1, 1, 1) = 0"
    continuous = True
    linear = False
    convex = False
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
            out[0] = (np.exp(-x[0]) - x[1]) ** 4 + 100 * (x[1] - x[2]) ** 6 + np.tan(x[2] - x[3]) ** 4 + x[0] ** 8

        super().__init__(
            compute=compute,
            ndim=ndim,
            bounds=bounds,
            default_bounds=np.array([[-1.0, 1.0] for _ in range(4)]),
            f_global=0,
            x_global=np.array([0, 1, 1, 1]),
            dim_changeable=False,
            dim_default=4,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
        )


class Mishra01(FuncBenchmark):
    r"""
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.

    .. math::
        f(x) = (1 + x_n)^{x_n}
        x_n = n - \sum_{i=1}^{n-1} x_i

    Here :math:`x_i \in [0, 1] for i \in [1, n]`.
    *Global optimum*: :math:`f(x) = 2`for :math:`x_i = 1 for all i \in [1, n]`
    """

    name = "Mishra 1 Function"
    latex_formula = r"f(x) = (1 + x_n)^{x_n}; x_n = n - \sum_{i=1}^{n-1} x_i"
    latex_formula_dimension = r"d = n"
    latex_formula_bounds = r"x_i \in [0, 1] for i \in [1, n]"
    latex_formula_global_optimum = r"f(1) = 2"
    continuous = True
    linear = False
    convex = False
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
            xn = x.shape[0] - np.sum(x[0:-1])
            out[0] = (1 + xn) ** xn

        super().__init__(
            compute=compute,
            ndim=ndim,
            bounds=bounds,
            default_bounds=np.array([[0.0, 1.0] for _ in range(2)]),
            f_global=2.0,
            x_global=lambda nd: np.ones(nd),
            dim_changeable=True,
            dim_default=2,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
        )


class Mishra02(FuncBenchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.

    .. math::
        f(x) = (1 + x_n)^{x_n}
        x_n = n - \\sum_{i=1}^{n-1} \frac{(x_i + x_{i+1})}{2}

    Here :math:`x_i \\in [0, 1] for i \\in [1, n]`.
    *Global optimum*: :math:`f(x) = 2`for :math:`x_i = 1 for all i \\in [1, n]`
    """

    name = "Mishra 2 Function"
    latex_formula = r"f(x) = (1 + x_n)^{x_n}; x_n = n - \sum_{i=1}^{n-1} \frac{(x_i + x_{i+1})}{2}"
    latex_formula_dimension = r"d = n"
    latex_formula_bounds = r"x_i \in [0, 1] for i \in [1, n]"
    latex_formula_global_optimum = r"f(1) = 2"
    continuous = True
    linear = False
    convex = False
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
            xn = x.shape[0] - np.sum((x[:-1] + x[1:]) / 2.0)
            out[0] = (1 + xn) ** xn

        super().__init__(
            compute=compute,
            ndim=ndim,
            bounds=bounds,
            default_bounds=np.array([[0.0, 1.0 + 1e-09] for _ in range(2)]),
            f_global=2.0,
            x_global=lambda nd: np.ones(nd),
            dim_changeable=True,
            dim_default=2,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
        )


class Mishra03(FuncBenchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.

    .. math::
        f(x) = \\sqrt{\\lvert \\cos{\\sqrt{\\lvert x_1^2 + x_2^2 \rvert}} \rvert} + 0.01(x_1 + x_2)

    Here :math:`x_i \\in [0, 1] for i \\in [1, n]`.
    *Global optimum*: :math:`f(-9.99378322, -9.99918927) = -0.19990562`
    """

    name = "Mishra 3 Function"
    latex_formula = r"f(x) = \sqrt{\lvert \cos{\sqrt{\lvert x_1^2 + x_2^2 \rvert}} \rvert} + 0.01(x_1 + x_2)"
    latex_formula_dimension = r"d = 2"
    latex_formula_bounds = r"x_i \in [-10, 10] for i \in [1, 2]"
    latex_formula_global_optimum = r"f(-9.99378322, -9.99918927) = -0.19990562"
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
            out[0] = 0.01 * (x[0] + x[1]) + np.sqrt(np.abs(np.cos(np.sqrt(np.abs(x[0] ** 2 + x[1] ** 2)))))

        super().__init__(
            compute=compute,
            ndim=ndim,
            bounds=bounds,
            default_bounds=np.array([[-10.0, 10.0] for _ in range(2)]),
            f_global=-0.19990562,
            x_global=np.array([-9.99378322, -9.99918927]),
            dim_changeable=False,
            dim_default=2,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
        )


class Mishra04(FuncBenchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.

    .. math::
        f(x) = \\sqrt{\\lvert \\sin{\\sqrt{\\lvert x_1^2 + x_2^2 \rvert}} \rvert} + 0.01(x_1 + x_2)

    Here :math:`x_i \\in [-10, 10] for i \\in [1, n]`.
    *Global optimum*: :math:`f(-8.71499636, -9.0533148) = -0.17767`
    """

    name = "Mishra 4 Function"
    latex_formula = r"f(x) = \sqrt{\lvert \sin{\sqrt{\lvert x_1^2 + x_2^2 \rvert}} \rvert} + 0.01(x_1 + x_2)"
    latex_formula_dimension = r"d = 2"
    latex_formula_bounds = r"x_i \in [-10, 10] for i \in [1, 2]"
    latex_formula_global_optimum = r"f(-8.71499636, -9.0533148) = -0.17767"
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
            out[0] = 0.01 * (x[0] + x[1]) + np.sqrt(np.abs(np.sin(np.sqrt(np.abs(x[0] ** 2 + x[1] ** 2)))))

        super().__init__(
            compute=compute,
            ndim=ndim,
            bounds=bounds,
            default_bounds=np.array([[-10.0, 10.0] for _ in range(2)]),
            f_global=-0.17767,
            x_global=np.array([-8.71499636, -9.0533148]),
            dim_changeable=False,
            dim_default=2,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
        )


class Mishra05(FuncBenchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.

    .. math::
        f(x) = \\left [ \\sin^2 ((\\cos(x_1) + \\cos(x_2))^2) + \\cos^2 ((\\sin(x_1) + \\sin(x_2))^2) + x_1 \right ]^2 + 0.01(x_1 + x_2)

    Here :math:`x_i \\in [-10, 10] for i \\in [1, 2]`.
    *Global optimum*: :math:`f(-1.98682, -10) = -1.019829519930646`
    """

    name = "Mishra 5 Function"
    latex_formula = r"f(x) = \left [ \sin^2 ((\cos(x_1) + \cos(x_2))^2) + \cos^2 ((\sin(x_1) + \sin(x_2))^2) + x_1 \right ]^2 + 0.01(x_1 + x_2)"
    latex_formula_dimension = r"d = 2"
    latex_formula_bounds = r"x_i \in [-10, 10] for i \in [1, 2]"
    latex_formula_global_optimum = r"f(-1.98682, -10) = -1.019829519930646"
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
            out[0] = (
                0.01 * x[0]
                + 0.1 * x[1]
                + (
                    np.sin((np.cos(x[0]) + np.cos(x[1])) ** 2) ** 2
                    + np.cos((np.sin(x[0]) + np.sin(x[1])) ** 2) ** 2
                    + x[0]
                )
                ** 2
            )

        super().__init__(
            compute=compute,
            ndim=ndim,
            bounds=bounds,
            default_bounds=np.array([[-10.0, 10.0] for _ in range(2)]),
            f_global=-1.019829519930646,
            x_global=np.array([-1.98682, -10]),
            dim_changeable=False,
            dim_default=2,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
        )


class Mishra06(FuncBenchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.

    .. math::
        f(x) = -\\log{\\left [ \\sin^2 ((\\cos(x_1) + \\cos(x_2))^2) - \\cos^2 ((\\sin(x_1) + \\sin(x_2))^2) + x_1 \right ]^2} + 0.01 \\left[(x_1 -1)^2 + (x_2 - 1)^2 \right]

    Here :math:`x_i \\in [-10, 10] for i \\in [1, 2]`.
    *Global optimum*: :math:`f(2.88631, 1.82326) = -2.28395`
    """

    name = "Mishra 6 Function"
    latex_formula = r"f(x) = -\log{\left [ \sin^2 ((\cos(x_1) + \cos(x_2))^2) - \cos^2 ((\sin(x_1) + \sin(x_2))^2) + x_1 \right ]^2} + 0.01 \left[(x_1 -1)^2 + (x_2 - 1)^2 \right]"
    latex_formula_dimension = r"d = 2"
    latex_formula_bounds = r"x_i \in [-10, 10] for i \in [1, 2]"
    latex_formula_global_optimum = r"f(2.88631, 1.82326) = -2.28395"
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
            a = 0.1 * ((x[0] - 1) ** 2 + (x[1] - 1) ** 2)
            u = (np.cos(x[0]) + np.cos(x[1])) ** 2
            v = (np.sin(x[0]) + np.sin(x[1])) ** 2
            out[0] = a - np.log((np.sin(u) ** 2 - np.cos(v) ** 2 + x[0]) ** 2)

        super().__init__(
            compute=compute,
            ndim=ndim,
            bounds=bounds,
            default_bounds=np.array([[-10.0, 10.0] for _ in range(2)]),
            f_global=-2.28395,
            x_global=np.array([2.88631, 1.82326]),
            dim_changeable=False,
            dim_default=2,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
        )


class Mishra07(FuncBenchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.

    .. math::
        f(x) = \\left [\\prod_{i=1}^{n} x_i - n! \right]^2

    Here :math:`x_i \\in [-10, 10] for i \\in [1, n]`.
    *Global optimum*: :math:`f(\\sqrt{n}) = 0, `
    """

    name = "Mishra 7 Function"
    latex_formula = r"f(x) = \left [\prod_{i=1}^{n} x_i - n! \right]^2"
    latex_formula_dimension = r"d = n"
    latex_formula_bounds = r"x_i \in [-10, 10] \forall i \in [1, n]"
    latex_formula_global_optimum = r"f(\sqrt{n}) = 0"
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
        def compute(x: np.ndarray, fact: float, out: np.ndarray) -> None:
            out[0] = (np.prod(x) - fact) ** 2.0

        resolved_ndim = ndim if isinstance(ndim, int) and ndim > 1 else 2
        self.fact = float(math.factorial(resolved_ndim))

        super().__init__(
            compute=compute,
            ndim=ndim,
            bounds=bounds,
            default_bounds=np.array([[-10.0, 10.0] for _ in range(2)]),
            f_global=0,
            x_global=lambda nd: np.sqrt(nd) * np.ones(nd),
            dim_changeable=True,
            dim_default=2,
            param_names=["fact"],
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
        )


class Mishra08(FuncBenchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.

    .. math::
        f(x) = 0.001 \\left[\\lvert x_1^{10} - 20x_1^9 + 180x_1^8 - 960 x_1^7 + 3360x_1^6 - 8064x_1^5 + 13340x_1^4 - 15360x_1^3
       + 11520x_1^2 - 5120x_1 + 2624 \rvert \\lvert x_2^4 + 12x_2^3 + 54x_2^2 + 108x_2 + 81 \rvert \right]^2

    Here :math:`x_i \\in [-10, 10] for i \\in [1, 2]`.
    *Global optimum*: :math:`f(2, -3) = 0, `
    """

    name = "Mishra 8 Function"
    latex_formula = r"f(x) = 0.001 \left[\lvert x_1^{10} - 20x_1^9 + 180x_1^8 - 960 x_1^7 + 3360x_1^6 - 8064x_1^5 + 13340x_1^4 - 15360x_1^3 + 11520x_1^2 - 5120x_1 + 2624 \rvert \lvert x_2^4 + 12x_2^3 + 54x_2^2 + 108x_2 + 81 \rvert \right]^2"
    latex_formula_dimension = r"d = 2"
    latex_formula_bounds = r"x_i \in [-10, 10] \forall i \in [1, 2]"
    latex_formula_global_optimum = r"f(2, -3) = 0"
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
            val = np.abs(
                x[0] ** 10
                - 20 * x[0] ** 9
                + 180 * x[0] ** 8
                - 960 * x[0] ** 7
                + 3360 * x[0] ** 6
                - 8064 * x[0] ** 5
                + 13340 * x[0] ** 4
                - 15360 * x[0] ** 3
                + 11520 * x[0] ** 2
                - 5120 * x[0]
                + 2624
            )
            val += np.abs(x[1] ** 4 + 12 * x[1] ** 3 + 54 * x[1] ** 2 + 108 * x[1] + 81)
            out[0] = 0.001 * val**2

        super().__init__(
            compute=compute,
            ndim=ndim,
            bounds=bounds,
            default_bounds=np.array([[-10.0, 10.0] for _ in range(2)]),
            f_global=0.0,
            x_global=np.array([2.0, -3.0]),
            dim_changeable=False,
            dim_default=2,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
        )


class Mishra09(FuncBenchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.

    .. math::
        f(x) = \\left[ ab^2c + abc^2 + b^2 + (x_1 + x_2 - x_3)^2 \right]^2

    Where, in this exercise:

    .. math::
        \begin{cases} a = 2x_1^3 + 5x_1x_2 + 4x_3 - 2x_1^2x_3 - 18 \\
        b = x_1 + x_2^3 + x_1x_2^2 + x_1x_3^2 - 22 \\
        c = 8x_1^2 + 2x_2x_3 + 2x_2^2 + 3x_2^3 - 52 \\end{cases}


    Here :math:`x_i \\in [-10, 10] for i \\in [1, 2, 3]`.
    *Global optimum*: :math:`f(1, 2, 3) = 0, `
    """

    name = "Mishra 9 Function"
    latex_formula = r"\left[ ab^2c + abc^2 + b^2 + (x_1 + x_2 - x_3)^2 \right]^2"
    latex_formula_dimension = r"d = 2"
    latex_formula_bounds = r"x_i \in [-10, 10] \forall i \in [1, 2, 3]"
    latex_formula_global_optimum = r"f(1, 2, 3) = 0"
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
            a = 2 * x[0] ** 3 + 5 * x[0] * x[1] + 4 * x[2] - 2 * x[0] ** 2 * x[2] - 18
            b = x[0] + x[1] ** 3 + x[0] * x[1] ** 2 + x[0] * x[2] ** 2 - 22.0
            c = 8 * x[0] ** 2 + 2 * x[1] * x[2] + 2 * x[1] ** 2 + 3 * x[1] ** 3 - 52
            out[0] = (a * c * b**2 + a * b * c**2 + b**2 + (x[0] + x[1] - x[2]) ** 2) ** 2

        super().__init__(
            compute=compute,
            ndim=ndim,
            bounds=bounds,
            default_bounds=np.array([[-10.0, 10.0] for _ in range(3)]),
            f_global=0.0,
            x_global=np.array([1.0, 2.0, 3.0]),
            dim_changeable=False,
            dim_default=3,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
        )


class Mishra10(FuncBenchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.

    .. math::
        f(x) = \\left[ \\lfloor x_1 \\perp x_2 \rfloor - \\lfloor x_1 \rfloor - \\lfloor x_2 \rfloor \right]^2

    Here :math:`x_i \\in [-10, 10] for i \\in [1, 2]`.
    *Global optimum*: :math:`f(2, 2) = 0, `
    """

    name = "Mishra 10 Function"
    latex_formula = r"\left[ \lfloor x_1 \perp x_2 \rfloor - \lfloor x_1 \rfloor - \lfloor x_2 \rfloor \right]^2"
    latex_formula_dimension = r"d = 2"
    latex_formula_bounds = r"x_i \in [-10, 10] \forall i \in [1, 2]"
    latex_formula_global_optimum = r"f(2, 2) = 0"
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
            x_int = x.astype(np.int64)
            out[0] = ((x_int[0] + x_int[1]) - (x_int[0] * x_int[1])) ** 2.0

        super().__init__(
            compute=compute,
            ndim=ndim,
            bounds=bounds,
            default_bounds=np.array([[-10.0, 10.0] for _ in range(2)]),
            f_global=0.0,
            x_global=np.array([2.0, 2.0]),
            dim_changeable=False,
            dim_default=2,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
        )


class Mishra11(FuncBenchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.

    .. math::
        f(x) = \\left [ \frac{1}{n} \\sum_{i=1}^{n} \\lvert x_i \rvert - \\left(\\prod_{i=1}^{n} \\lvert x_i \rvert \right )^{\frac{1}{n}} \right]^2

    Here :math:`x_i \\in [-10, 10] for i \\in [1, 2]`.
    *Global optimum*: :math:`f(0) = 0, `
    """

    name = "Mishra 11 Function"
    latex_formula = r"\left [ \frac{1}{n} \sum_{i=1}^{n} \lvert x_i \rvert - \left(\prod_{i=1}^{n} \lvert x_i \rvert \right )^{\frac{1}{n}} \right]^2"
    latex_formula_dimension = r"d = n"
    latex_formula_bounds = r"x_i \in [-10, 10] \forall i \in [1, n]"
    latex_formula_global_optimum = r"f(0) = 0"
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
            out[0] = ((1.0 / x.shape[0]) * np.sum(np.abs(x)) - (np.prod(np.abs(x))) ** 1.0 / x.shape[0]) ** 2.0

        super().__init__(
            compute=compute,
            ndim=ndim,
            bounds=bounds,
            default_bounds=np.array([[-10.0, 10.0] for _ in range(2)]),
            f_global=0.0,
            x_global=lambda nd: np.zeros(nd, dtype=float),
            dim_changeable=True,
            dim_default=2,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
        )


class MultiModal(FuncBenchmark):
    """
    .. [1] Gavana, A. Global Optimization Benchmarks and AMPGO retrieved 2015

    .. math::
        f(x) = \\left( \\sum_{i=1}^n \\lvert x_i \rvert \right) \\left( \\prod_{i=1}^n \\lvert x_i \rvert \right)

    Here :math:`x_i \\in [-10, 10] for i \\in [1, n]`.
    *Global optimum*: :math:`f(0) = 0, `
    """

    name = "Mishra 11 Function"
    latex_formula = r"\left( \sum_{i=1}^n \lvert x_i \rvert \right) \left( \prod_{i=1}^n \lvert x_i \rvert \right)"
    latex_formula_dimension = r"d = n"
    latex_formula_bounds = r"x_i \in [-10, 10] \forall i \in [1, n]"
    latex_formula_global_optimum = r"f(0) = 0"
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
            out[0] = np.sum(np.abs(x)) * np.prod(np.abs(x))

        super().__init__(
            compute=compute,
            ndim=ndim,
            bounds=bounds,
            default_bounds=np.array([[-10.0, 10.0] for _ in range(2)]),
            f_global=0.0,
            x_global=lambda nd: np.zeros(nd, dtype=float),
            dim_changeable=True,
            dim_default=2,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
        )
