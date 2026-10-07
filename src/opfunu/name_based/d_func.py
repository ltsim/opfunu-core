#!/usr/bin/env python
# Created by "Thieu" at 09:33, 21/07/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import typing

import numpy as np

from opfunu.benchmark.func import FuncBenchmark


class Damavandi(FuncBenchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.
    """

    name = "Damavandi Function"
    latex_formula = (
        r"f(x) = \left[ 1 - \lvert{\frac{\sin[\pi (x_1 - 2)]\sin[\pi (x2 - 2)]}"
        r"{\pi^2 (x_1 - 2)(x_2 - 2)}}\rvert^5 \right] \left[2 + (x_1 - 7)^2 + 2(x_2 - 7)^2 \right]"
    )
    latex_formula_dimension = r"d = 2"
    latex_formula_bounds = r"x_i \in [0, 14], \forall i \in \llbracket 1, d\rrbracket"
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
        shift: typing.Any = None,
        rotate: typing.Any = None,
        rotate_bounds: bool = True,
    ) -> None:
        def compute(x: np.ndarray, epsilon: float, out: np.ndarray) -> None:
            num = np.sin(np.pi * (x[0] - 2.0)) * np.sin(np.pi * (x[1] - 2.0))
            den = (np.pi**2) * (x[0] - 2.0) * (x[1] - 2.0) + epsilon
            factor1 = 1.0 - (np.abs(num / den)) ** 5.0
            factor2 = 2 + (x[0] - 7.0) ** 2.0 + 2 * (x[1] - 7.0) ** 2.0
            out[0] = factor1 * factor2

        super().__init__(
            compute=compute,
            ndim=ndim,
            bounds=bounds,
            default_bounds=np.array([[0.0, 14.0] for _ in range(2)]),
            f_global=0.0,
            x_global=lambda nd: 2 * np.ones(nd),
            dim_changeable=False,
            dim_default=2,
            param_names=["epsilon"],
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            shift=shift,
            rotate=rotate,
            rotate_bounds=rotate_bounds,
        )


class Deb01(FuncBenchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.
    """

    name = "Deb 1 Function"
    latex_formula = r"f(x) = - \frac{1}{N} \sum_{i=1}^n \sin^6(5 \pi x_i)"
    latex_formula_dimension = r"d = 2"
    latex_formula_bounds = r"x_i \in [-1, 1], \forall i \in \llbracket 1, d\rrbracket"
    latex_formula_global_optimum = r"f(0.3, -0.3) = -1"
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
            out[0] = -(1.0 / x.shape[0]) * np.sum(np.sin(5 * np.pi * x) ** 6.0)

        super().__init__(
            compute=compute,
            ndim=ndim,
            bounds=bounds,
            default_bounds=np.array([[-1, -1.0] for _ in range(2)]),
            f_global=-1.0,
            x_global=lambda nd: np.zeros(nd),
            dim_changeable=True,
            dim_default=2,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            shift=shift,
            rotate=rotate,
            rotate_bounds=rotate_bounds,
        )


class Deb03(FuncBenchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.
    """

    name = "Deb 3 Function"
    latex_formula = r"f(x) = - \frac{1}{N} \sum_{i=1}^n \sin^6 \left[ 5 \pi\left ( x_i^{3/4} - 0.05 \right) \right ]"
    latex_formula_dimension = r"d = 2"
    latex_formula_bounds = r"x_i \in [-1, 1], \forall i \in \llbracket 1, d\rrbracket"
    latex_formula_global_optimum = r"f(0.3, -0.3) = -1"
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
            out[0] = -(1.0 / x.shape[0]) * np.sum(np.sin(5 * np.pi * (x**0.75 - 0.05)) ** 6.0)

        super().__init__(
            compute=compute,
            ndim=ndim,
            bounds=bounds,
            default_bounds=np.array([[-1.0, 1.0] for _ in range(2)]),
            f_global=-1.0,
            x_global=lambda nd: np.zeros(nd),
            dim_changeable=True,
            dim_default=2,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            shift=shift,
            rotate=rotate,
            rotate_bounds=rotate_bounds,
        )


class Decanomial(FuncBenchmark):
    """
    .. [1] Gavana, A. Global Optimization Benchmarks and AMPGO retrieved 2015
    """

    name = "Decanomial Function"
    latex_formula = (
        r"f(x) = 0.001 \left(\lvert{x_{2}^{4} + 12 x_{2}^{3}"
        + r"+ 54 x_{2}^{2} + 108 x_{2} + 81.0}\rvert + \lvert{x_{1}^{10} - 20 x_{1}^{9} + 180 x_{1}^{8} - 960 x_{1}^{7} + 3360 x_{1}^{6}"
        + r"- 8064 x_{1}^{5} + 13340 x_{1}^{4} - 15360 x_{1}^{3} + 11520 x_{1}^{2} - 5120 x_{1} + 2624.0}\rvert\right)^{2}"
    )
    latex_formula_dimension = r"d = 2"
    latex_formula_bounds = r"x_i \in [-10, 10], \forall i \in \llbracket 1, d\rrbracket"
    latex_formula_global_optimum = r"f(2, -3) = 0"
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
            val = x[1] ** 4 + 12 * x[1] ** 3 + 54 * x[1] ** 2 + 108 * x[1] + 81.0
            val2 = x[0] ** 10.0 - 20 * x[0] ** 9 + 180 * x[0] ** 8 - 960 * x[0] ** 7
            val2 += 3360 * x[0] ** 6 - 8064 * x[0] ** 5 + 13340 * x[0] ** 4
            val2 += -15360 * x[0] ** 3 + 11520 * x[0] ** 2 - 5120 * x[0] + 2624
            out[0] = 0.001 * (np.abs(val) + np.abs(val2)) ** 2.0

        super().__init__(
            compute=compute,
            ndim=ndim,
            bounds=bounds,
            default_bounds=np.array([[-10.0, 10.0] for _ in range(2)]),
            f_global=0.0,
            x_global=np.array([2, -3]),
            dim_changeable=False,
            dim_default=2,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            shift=shift,
            rotate=rotate,
            rotate_bounds=rotate_bounds,
        )


class Deceptive(FuncBenchmark):
    """
    .. [1] Gavana, A. Global Optimization Benchmarks and AMPGO retrieved 2015
    """

    name = "Deceptive Function"
    latex_formula = r"f(x) = - \left [\frac{1}{n} \sum_{i=1}^{n} g_i(x_i) \right ]^{\beta}"
    latex_formula_dimension = r"d \in N^+"
    latex_formula_bounds = r"x_i \in [0, 1], \forall i \in \llbracket 1, d\rrbracket"
    latex_formula_global_optimum = r"f(alpha_i) = -1"
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
        shift: typing.Any = None,
        rotate: typing.Any = None,
        rotate_bounds: bool = True,
    ) -> None:
        def compute(x: np.ndarray, out: np.ndarray) -> None:
            alpha = np.arange(1.0, x.shape[0] + 1.0) / (x.shape[0] + 1.0)
            beta = 2.0
            g = np.zeros((x.shape[0],))
            for i in range(x.shape[0]):
                if x[i] <= 0.0:
                    g[i] = x[i]
                elif x[i] < 0.8 * alpha[i]:
                    g[i] = -x[i] / alpha[i] + 0.8
                elif x[i] < alpha[i]:
                    g[i] = 5.0 * x[i] / alpha[i] - 4.0
                elif x[i] < (1.0 + 4 * alpha[i]) / 5.0:
                    g[i] = 5.0 * (x[i] - alpha[i]) / (alpha[i] - 1.0) + 1.0
                elif x[i] <= 1.0:
                    g[i] = (x[i] - 1.0) / (1.0 - alpha[i]) + 4.0 / 5.0
                else:
                    g[i] = x[i] - 1.0
            out[0] = -(((1.0 / x.shape[0]) * np.sum(g)) ** beta)

        super().__init__(
            compute=compute,
            ndim=ndim,
            bounds=bounds,
            default_bounds=np.array([[0.0, 1.0] for _ in range(2)]),
            f_global=-1.0,
            x_global=lambda nd: np.arange(1.0, nd + 1.0) / (nd + 1.0),
            dim_changeable=True,
            dim_default=2,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            shift=shift,
            rotate=rotate,
            rotate_bounds=rotate_bounds,
        )


class DeckkersAarts(FuncBenchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.
    """

    name = "Deckkers-Aarts Function"
    latex_formula = r"f(x) = 10^5x_1^2 + x_2^2 - (x_1^2 + x_2^2)^2 + 10^{-5}(x_1^2 + x_2^2)^4"
    latex_formula_dimension = r"d = 2"
    latex_formula_bounds = r"x_i \in [-20, 20], \forall i \in \llbracket 1, d\rrbracket"
    latex_formula_global_optimum = r"f(0, \pm 14.9451209) = -24776.518242168"
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
        shift: typing.Any = None,
        rotate: typing.Any = None,
        rotate_bounds: bool = True,
    ) -> None:
        def compute(x: np.ndarray, out: np.ndarray) -> None:
            out[0] = (
                1.0e5 * x[0] ** 2 + x[1] ** 2 - (x[0] ** 2 + x[1] ** 2) ** 2 + 1.0e-5 * (x[0] ** 2 + x[1] ** 2) ** 4
            )

        super().__init__(
            compute=compute,
            ndim=ndim,
            bounds=bounds,
            default_bounds=np.array([[-20.0, 20.0] for _ in range(2)]),
            f_global=-24776.5183421814,
            x_global=np.array([0, 14.9451209]),
            dim_changeable=False,
            dim_default=2,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            shift=shift,
            rotate=rotate,
            rotate_bounds=rotate_bounds,
        )
        self.x_globals = np.array([[0, 14.9451209], [0, -14.9451209]])


class DeflectedCorrugatedSpring(FuncBenchmark):
    """
    .. [1] Gavana, A. Global Optimization Benchmarks and AMPGO retrieved 2015
    """

    name = "Deflected Corrugated Spring Function"
    latex_formula = r"f(x) = 0.1\sum_{i=1}^n \left[ (x_i - \alpha)^2 - \cos \left( K \sqrt {\sum_{i=1}^n (x_i - \alpha)^2}\right ) \right ]"
    latex_formula_dimension = r"d = 2"
    latex_formula_bounds = r"x_i \in [-0, 2\alpha], \forall i \in \llbracket 1, d\rrbracket"
    latex_formula_global_optimum = r"f(x_i) = f(alpha_i) = -1"
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
        alpha: float = 5.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
        shift: typing.Any = None,
        rotate: typing.Any = None,
        rotate_bounds: bool = True,
    ) -> None:
        def compute(x: np.ndarray, alpha: typing.Any, out: np.ndarray) -> None:
            K = 5.0
            out[0] = -np.cos(K * np.sqrt(np.sum((x - alpha) ** 2))) + 0.1 * np.sum((x - alpha) ** 2)

        self.alpha = alpha

        super().__init__(
            compute=compute,
            ndim=ndim,
            bounds=bounds,
            default_bounds=np.array([[0.0, 2.0 * self.alpha] for _ in range(2)]),
            f_global=-1.0,
            x_global=lambda nd: self.alpha * np.ones(nd),
            dim_changeable=True,
            dim_default=2,
            param_names=["alpha"],
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            shift=shift,
            rotate=rotate,
            rotate_bounds=rotate_bounds,
        )


class DeVilliersGlasser01(FuncBenchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.
    """

    name = "DeVilliers-Glasser 1 Function"
    latex_formula = r"f(x) = \sum_{i=1}^{24} \left[ x_1x_2^{t_i}\sin(x_3t_i + x_4) - y_i \right ]^2"
    latex_formula_dimension = r"d = 4"
    latex_formula_bounds = r"x_i \in [-500, 500], \forall i \in \llbracket 1, d\rrbracket"
    latex_formula_global_optimum = r"f(60.137, 1.371, 3.112, 1.761) = 0"
    continuous = True
    linear = False
    convex = True
    unimodal = False
    separable = False

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
            t = 0.1 * np.arange(24)
            y = 60.137 * (1.371**t) * np.sin(3.112 * t + 1.761)
            out[0] = sum((x[0] * (x[1] ** t) * np.sin(x[2] * t + x[3]) - y) ** 2.0)

        super().__init__(
            compute=compute,
            ndim=ndim,
            bounds=bounds,
            default_bounds=np.array([[-500.0, 500.0] for _ in range(4)]),
            f_global=0.0,
            x_global=np.array([60.137, 1.371, 3.112, 1.761]),
            dim_changeable=False,
            dim_default=4,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            shift=shift,
            rotate=rotate,
            rotate_bounds=rotate_bounds,
        )


class DeVilliersGlasser02(FuncBenchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.
    """

    name = "DeVilliers-Glasser 2 Function"
    latex_formula = (
        r"f(x) = \sum_{i=1}^{24} \left[ x_1x_2^{t_i}"
        + r"\tanh \left [x_3t_i + \sin(x_4t_i) \right] \cos(t_ie^{x_5}) - y_i \right ]^2"
    )
    latex_formula_dimension = r"d = 5"
    latex_formula_bounds = r"x_i \in [-500, 500], \forall i \in \llbracket 1, d\rrbracket"
    latex_formula_global_optimum = r"f(53.81, 1.27, 3.012, 2.13, 0.507) = 0"
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
            t = 0.1 * np.arange(16)
            y = 53.81 * 1.27**t * np.tanh(3.012 * t + np.sin(2.13 * t)) * np.cos(np.exp(0.507) * t)
            out[0] = np.sum(
                (x[0] * (x[1] ** t) * np.tanh(x[2] * t + np.sin(x[3] * t)) * np.cos(t * np.exp(x[4])) - y) ** 2.0
            )

        super().__init__(
            compute=compute,
            ndim=ndim,
            bounds=bounds,
            default_bounds=np.array([[-500.0, 500.0] for _ in range(5)]),
            f_global=0.0,
            x_global=np.array([53.81, 1.27, 3.012, 2.13, 0.507]),
            dim_changeable=False,
            dim_default=5,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            shift=shift,
            rotate=rotate,
            rotate_bounds=rotate_bounds,
        )


class DixonPrice(FuncBenchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.
    """

    name = "Dixon & Price Function"
    latex_formula = r"f(x) = (x_i - 1)^2 + \sum_{i=2}^n i(2x_i^2 - x_{i-1})^2"
    latex_formula_dimension = r"d \in N^+"
    latex_formula_bounds = r"x_i \in [-10, 10], \forall i \in \llbracket 1, d\rrbracket"
    latex_formula_global_optimum = r"f(2^{- \frac{(2^i - 2)}{2^i}}) = 0"
    continuous = True
    linear = False
    convex = True
    unimodal = True
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
        shift: typing.Any = None,
        rotate: typing.Any = None,
        rotate_bounds: bool = True,
    ) -> None:
        def compute(x: np.ndarray, out: np.ndarray) -> None:
            i = np.arange(2, x.shape[0] + 1)
            s = i * (2.0 * x[1:] ** 2.0 - x[:-1]) ** 2.0
            out[0] = np.sum(s) + (x[0] - 1.0) ** 2.0

        super().__init__(
            compute=compute,
            ndim=ndim,
            bounds=bounds,
            default_bounds=np.array([[-10.0, 10.0] for _ in range(2)]),
            f_global=0.0,
            x_global=lambda nd: np.array([2.0 ** (-(2.0**i - 2.0) / 2.0**i) for i in range(1, nd + 1)]),
            dim_changeable=True,
            dim_default=2,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            shift=shift,
            rotate=rotate,
            rotate_bounds=rotate_bounds,
        )


class Dolan(FuncBenchmark):
    """
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization
    Problems Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.
    """

    name = "Dolan Function"
    latex_formula = (
        r"f(x) = \lvert (x_1 + 1.7 x_2)\sin(x_1) - 1.5 x_3 - 0.1 x_4\cos(x_5 + x_5 - x_1) + 0.2 x_5^2 - x_2 - 1 \rvert"
    )
    latex_formula_dimension = r"d = 5"
    latex_formula_bounds = r"x_i \in [-100, 100], \forall i \in \llbracket 1, d\rrbracket"
    latex_formula_global_optimum = r"f(8.39045925, 4.81424707, 7.34574133, 68.88246895, 3.85470806) = 0"
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
            out[0] = np.abs(
                (x[0] + 1.7 * x[1]) * np.sin(x[0])
                - 1.5 * x[2]
                - 0.1 * x[3] * np.cos(x[3] + x[4] - x[0])
                + 0.2 * x[4] ** 2
                - x[1]
                - 1
            )

        super().__init__(
            compute=compute,
            ndim=ndim,
            bounds=bounds,
            default_bounds=np.array([[-100.0, 100.0] for _ in range(5)]),
            f_global=0.0,
            x_global=np.array([-74.10522498, 44.33511286, 6.21069214, 18.42772233, -16.5839403]),
            dim_changeable=False,
            dim_default=5,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            shift=shift,
            rotate=rotate,
            rotate_bounds=rotate_bounds,
        )


class DropWave(FuncBenchmark):
    """
    .. [1] Gavana, A. Global Optimization Benchmarks and AMPGO retrieved 2015
    """

    name = "DropWave Function"
    latex_formula = (
        r"f(x) = - \frac{1 + \cos\left(12 \sqrt{\sum_{i=1}^{n} x_i^{2}}\right)}{2 + 0.5 \sum_{i=1}^{n} x_i^{2}}"
    )
    latex_formula_dimension = r"d = 2"
    latex_formula_bounds = r"x_i \in [-5.12, 5.12], \forall i \in \llbracket 1, d\rrbracket"
    latex_formula_global_optimum = r"f(0, 0) = -1"
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
            norm_x = np.sum(x**2)
            out[0] = -(1 + np.cos(12 * np.sqrt(norm_x))) / (0.5 * norm_x + 2)

        super().__init__(
            compute=compute,
            ndim=ndim,
            bounds=bounds,
            default_bounds=np.array([[-5.12, 5.12] for _ in range(2)]),
            f_global=-1.0,
            x_global=np.array([0.0, 0.0]),
            dim_changeable=True,
            dim_default=2,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            shift=shift,
            rotate=rotate,
            rotate_bounds=rotate_bounds,
        )
