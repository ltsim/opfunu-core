#!/usr/bin/env python
# Created by "Thieu" at 17:32, 30/07/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import typing

import numpy as np

from opfunu.benchmark.func import FuncBenchmark


class Zacharov(FuncBenchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.

    .. math::

        f_{\text{Zacharov}}(x) = \\sum_{i=1}^{n} x_i^2 + \\left ( \frac{1}{2}\\sum_{i=1}^{n} i x_i \right )^2
                                 + \\left ( \frac{1}{2} \\sum_{i=1}^{n} i x_i \right )^4

    Here :math:`x_i \\in [-5, 10]` for :math:`i = 1, ..., n`.
    *Global optimum*: :math:`f(x) = 0.0`for :math:`x = [0, 0,,,0]`
    """

    name = "Zacharov Function"
    latex_formula = "\\sum_{i=1}^{n} x_i^2 + \\left ( \frac{1}{2}\\sum_{i=1}^{n} i x_i \right )^2+ \\left ( \frac{1}{2} \\sum_{i=1}^{n} i x_i \right )^4"
    latex_formula_dimension = r"d = n"
    latex_formula_bounds = r"x_i \in [-5, 10]"
    latex_formula_global_optimum = r"f(0, 0,...,0) = 0"
    continuous = False
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
            u = np.sum(x**2)
            v = np.sum(np.arange(1, x.shape[0] + 1) * x)
            out[0] = u + (0.5 * v) ** 2 + (0.5 * v) ** 4

        super().__init__(
            compute=compute,
            ndim=ndim,
            bounds=bounds,
            default_bounds=np.array([[-5.0, 10.0] for _ in range(2)]),
            f_global=0.0,
            x_global=lambda nd: np.zeros(nd),
            dim_changeable=True,
            dim_default=2,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            shift=shift,
        )


class ZeroSum(FuncBenchmark):
    """
    .. [1] Gavana, A. Global Optimization Benchmarks and AMPGO retrieved 2015

    .. math::

        f_{\text{ZeroSum}}(x) = \begin{cases} 0 & \textrm{if} \\sum_{i=1}^n x_i = 0 \\
                                1 + \\left(10000 \\left |\\sum_{i=1}^n x_i\right| \right)^{0.5} & \textrm{otherwise} \\end{cases}

    Here :math:`x_i \\in [-10, 10]` for :math:`i = 1, ..., n`.
    *Global optimum*: :math:`f(x) = 0.0`for :math:`\\sum_{i=1}^n x_i = 0`
    """

    name = "ZeroSum Function"
    latex_formula = "\begin{cases} 0 & \textrm{if} \\sum_{i=1}^n x_i = 0 \\ 1 + \\left(10000 \\left |\\sum_{i=1}^n x_i\right| \right)^{0.5} & \textrm{otherwise} \\end{cases}"
    latex_formula_dimension = r"d = n"
    latex_formula_bounds = r"x_i \in [-10, 10]"
    latex_formula_global_optimum = r"f(x_best) = 0"
    continuous = False
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
            if np.abs(np.sum(x)) < 3e-16:
                out[0] = 0.0
            else:
                out[0] = 1.0 + (10000.0 * np.abs(np.sum(x))) ** 0.5

        super().__init__(
            compute=compute,
            ndim=ndim,
            bounds=bounds,
            default_bounds=np.array([[-10.0, 10.0] for _ in range(2)]),
            f_global=0.0,
            x_global=lambda nd: np.zeros(nd),
            dim_changeable=True,
            dim_default=2,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            shift=shift,
        )


class Zettl(FuncBenchmark):
    """
    .. [1] Gavana, A. Global Optimization Benchmarks and AMPGO retrieved 2015

    .. math::

        f_{\text{Zettl}}(x) = \frac{1}{4} x_{1} + \\left(x_{1}^{2} - 2 x_{1}
                             + x_{2}^{2}\right)^{2}

    Here :math:`x_i \\in [-1, 5]` for :math:`i = 1, 2`.
    *Global optimum*: :math:`f(x) = -0.0037912`for :math:`x = [-0.029896, 0.0]`
    """

    name = "Zettl Function"
    latex_formula = "\\sum_{i=1}^{n} x_i^2 + \\left ( \frac{1}{2}\\sum_{i=1}^{n} i x_i \right )^2+ \\left ( \frac{1}{2} \\sum_{i=1}^{n} i x_i \right )^4"
    latex_formula_dimension = r"d = 2"
    latex_formula_bounds = r"x_i \in [-0.029896, 0.0]"
    latex_formula_global_optimum = r"f(x_best) = -0.0037912"
    continuous = False
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
            out[0] = (x[0] ** 2 + x[1] ** 2 - 2 * x[0]) ** 2 + 0.25 * x[0]

        super().__init__(
            compute=compute,
            ndim=ndim,
            bounds=bounds,
            default_bounds=np.array([[-1.0, 5.0] for _ in range(2)]),
            f_global=-0.0037912,
            x_global=np.array([-0.029896, 0.0]),
            dim_changeable=False,
            dim_default=2,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            shift=shift,
        )


class Zimmerman(FuncBenchmark):
    """
    .. [1] Gavana, A. Global Optimization Benchmarks and AMPGO retrieved 2015

    .. math::

        f_{\text{Zimmerman}}(x) = \\max \\left[Zh1(x), Zp(Zh2(x))\textrm{sgn}(Zh2(x)), Zp(Zh3(x))\textrm{sgn}(Zh3(x)),
                                  Zp(-x_1)\textrm{sgn}(x_1),Zp(-x_2)\textrm{sgn}(x_2) \right]

    .. math::

        \begin{cases}
        Zh1(x) = 9 - x_1 - x_2 \\
        Zh2(x) = (x_1 - 3)^2 + (x_2 - 2)^2 \\
        Zh3(x) = x_1x_2 - 14 \\
        Zp(t) = 100(1 + t)
        \\end{cases}

    Where :math:`x` is a vector and :math:`t` is a scalar.
    Here, :math:`x_i \\in [0, 100]` for :math:`i = 1, 2`.
    *Global optimum*: :math:`f(x) = 0` for :math:`x = [7, 2]`
    """

    name = "Zimmerman Function"
    latex_formula = (
        "\\max \\left[Zh1(x), Zp(Zh2(x))\textrm{sgn}(Zh2(x)), Zp(Zh3(x))\textrm{sgn}(Zh3(x)), "
        "Zp(-x_1)\textrm{sgn}(x_1),Zp(-x_2)\textrm{sgn}(x_2) \right]"
    )
    latex_formula_dimension = r"d = 2"
    latex_formula_bounds = r"x_i \in [0, 100]"
    latex_formula_global_optimum = r"f([7, 2]) = 0."
    continuous = False
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
            def Zh1(x: typing.Any) -> typing.Any:
                return 9.0 - x[0] - x[1]

            def Zh2(x: typing.Any) -> typing.Any:
                return (x[0] - 3.0) ** 2.0 + (x[1] - 2.0) ** 2.0 - 16.0

            def Zh3(x: typing.Any) -> typing.Any:
                return x[0] * x[1] - 14.0

            def Zp(x: typing.Any) -> typing.Any:
                return 100.0 * (1.0 + x)

            out[0] = max(
                Zh1(x),
                Zp(Zh2(x)) * np.sign(Zh2(x)),
                Zp(Zh3(x)) * np.sign(Zh3(x)),
                Zp(-x[0]) * np.sign(x[0]),
                Zp(-x[1]) * np.sign(x[1]),
            )

        super().__init__(
            compute=compute,
            ndim=ndim,
            bounds=bounds,
            default_bounds=np.array([[0, 100] for _ in range(2)]),
            f_global=0.0,
            x_global=np.array([7.0, 2.0]),
            dim_changeable=False,
            dim_default=2,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            shift=shift,
        )


class Zirilli(FuncBenchmark):
    """
    .. [1] Gavana, A. Global Optimization Benchmarks and AMPGO retrieved 2015

    .. math::

         f_{\text{Zirilli}}(x) = 0.25x_1^4 - 0.5x_1^2 + 0.1x_1 + 0.5x_2^2

    .. math::

        \begin{cases}
        Zh1(x) = 9 - x_1 - x_2 \\
        Zh2(x) = (x_1 - 3)^2 + (x_2 - 2)^2 \\
        Zh3(x) = x_1x_2 - 14 \\
        Zp(t) = 100(1 + t)
        \\end{cases}

    Where :math:`x` is a vector and :math:`t` is a scalar.
    Here, :math:`x_i \\in [-10, 10]` for :math:`i = 1, 2`.
    *Global optimum*: :math:`f(x) = -0.3523` for :math:`x = [-1.0465, 0]`
    """

    name = "Zirilli Function"
    latex_formula = "f_{\text{Zirilli}}(x) = 0.25x_1^4 - 0.5x_1^2 + 0.1x_1 + 0.5x_2^2"
    latex_formula_dimension = r"d = 2"
    latex_formula_bounds = r"x_i \in [-10, 10]"
    latex_formula_global_optimum = r"f([-1.0465, 0]) = -0.3523"
    continuous = False
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
        shift: typing.Any = None,
    ) -> None:
        def compute(x: np.ndarray, out: np.ndarray) -> None:
            out[0] = 0.25 * x[0] ** 4 - 0.5 * x[0] ** 2 + 0.1 * x[0] + 0.5 * x[1] ** 2

        super().__init__(
            compute=compute,
            ndim=ndim,
            bounds=bounds,
            default_bounds=np.array([[-10, 10] for _ in range(2)]),
            f_global=-0.3523,
            x_global=np.array([-1.0465, 0]),
            dim_changeable=False,
            dim_default=2,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            shift=shift,
        )
