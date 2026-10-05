#!/usr/bin/env python
# Created by "Thieu" at 22:19, 01/07/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import typing

import numpy as np

from opfunu.benchmark.cec import CecBenchmark
from opfunu.utils import operator


class F12008(CecBenchmark):
    """
    .. [1] Tang, K., Yáo, X., Suganthan, P. N., MacNish, C., Chen, Y. P., Chen, C. M., & Yang, Z. (2007). Benchmark functions
    for the CEC’2008 special session and competition on large scale global optimization.
    Nature inspired computation and applications laboratory, USTC, China, 24, 1-18.
    """

    name = "F1: Shifted Sphere Function"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = bias = -450.0"
    continuous = True
    linear = False
    convex = True
    unimodal = True
    separable = True

    differentiable = True
    scalable = True
    randomized_term = False
    parametric = True
    shifted = True
    rotated = False

    modality = False  # Number of ambiguous peaks, unknown # peaks

    # n_basins = 1
    # n_valleys = 1

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "sphere_shift_func_data",
        f_bias: float = -450.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
    ) -> None:
        def compute(x: np.ndarray, f_shift: typing.Any, f_bias: typing.Any, out: np.ndarray) -> None:
            out[0] = operator.sphere_func(x - f_shift) + f_bias

        super().__init__(
            ndim=ndim,
            bounds=bounds,
            f_shift=f_shift,
            f_bias=f_bias,
            dim_default=500,
            dim_max=1000,
            data_name="data_2008",
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            compute=compute,
            param_names=["f_shift", "f_bias"],
        )


class F22008(CecBenchmark):
    """
    .. [1] Tang, K., Yáo, X., Suganthan, P. N., MacNish, C., Chen, Y. P., Chen, C. M., & Yang, Z. (2007). Benchmark functions
    for the CEC’2008 special session and competition on large scale global optimization.
    Nature inspired computation and applications laboratory, USTC, China, 24, 1-18.
    """

    name = "F2: Schwefel’s Problem 2.21"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = bias = -450.0"
    continuous = False
    linear = True
    convex = True
    unimodal = True
    separable = False

    differentiable = False
    scalable = True
    randomized_term = False
    parametric = True
    shifted = True
    rotated = False

    modality = False  # Number of ambiguous peaks, unknown # peaks

    # n_basins = 1
    # n_valleys = 1

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "schwefel_shift_func_data",
        f_bias: float = -450.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
    ) -> None:
        def compute(x: np.ndarray, f_shift: typing.Any, f_bias: typing.Any, out: np.ndarray) -> None:
            out[0] = np.max(np.abs(x - f_shift)) + f_bias

        super().__init__(
            ndim=ndim,
            bounds=bounds,
            f_shift=f_shift,
            f_bias=f_bias,
            dim_default=500,
            dim_max=1000,
            data_name="data_2008",
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            compute=compute,
            param_names=["f_shift", "f_bias"],
        )


class F32008(CecBenchmark):
    """
    .. [1] Tang, K., Yáo, X., Suganthan, P. N., MacNish, C., Chen, Y. P., Chen, C. M., & Yang, Z. (2007). Benchmark functions
    for the CEC’2008 special session and competition on large scale global optimization.
    Nature inspired computation and applications laboratory, USTC, China, 24, 1-18.
    """

    name = "F3: Shifted Rosenbrock’s Function"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = bias = -450.0"
    continuous = True
    linear = False
    convex = True
    unimodal = False
    separable = False

    differentiable = True
    scalable = True
    randomized_term = False
    parametric = True
    shifted = True
    rotated = False

    modality = False  # Number of ambiguous peaks, unknown # peaks

    # n_basins = 1
    # n_valleys = 1

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "rosenbrock_shift_func_data",
        f_bias: float = -390.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
    ) -> None:
        def compute(x: np.ndarray, f_shift: typing.Any, f_bias: typing.Any, out: np.ndarray) -> None:
            out[0] = operator.rosenbrock_func(x - f_shift, shift=1.0) + f_bias

        super().__init__(
            ndim=ndim,
            bounds=bounds,
            f_shift=f_shift,
            f_bias=f_bias,
            dim_default=500,
            dim_max=1000,
            data_name="data_2008",
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            compute=compute,
            param_names=["f_shift", "f_bias"],
        )


class F42008(CecBenchmark):
    """
    .. [1] Tang, K., Yáo, X., Suganthan, P. N., MacNish, C., Chen, Y. P., Chen, C. M., & Yang, Z. (2007). Benchmark functions
    for the CEC’2008 special session and competition on large scale global optimization.
    Nature inspired computation and applications laboratory, USTC, China, 24, 1-18.
    """

    name = "F4: Shifted Rastrigin’s Function"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = bias = -450.0"
    continuous = True
    linear = False
    convex = True
    unimodal = False
    separable = True

    differentiable = True
    scalable = True
    randomized_term = False
    parametric = True
    shifted = True
    rotated = False

    modality = True  # Number of ambiguous peaks, unknown # peaks

    # n_basins = 1
    # n_valleys = 1

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "rastrigin_shift_func_data",
        f_bias: float = -330.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
    ) -> None:
        def compute(x: np.ndarray, f_shift: typing.Any, f_bias: typing.Any, out: np.ndarray) -> None:
            z = x - f_shift
            out[0] = operator.rastrigin_func(z) + f_bias

        super().__init__(
            ndim=ndim,
            bounds=bounds,
            f_shift=f_shift,
            f_bias=f_bias,
            default_bounds=[[-5.0, 5.0]],
            dim_default=500,
            dim_max=1000,
            data_name="data_2008",
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            compute=compute,
            param_names=["f_shift", "f_bias"],
        )


class F52008(CecBenchmark):
    """
    .. [1] Tang, K., Yáo, X., Suganthan, P. N., MacNish, C., Chen, Y. P., Chen, C. M., & Yang, Z. (2007). Benchmark functions
    for the CEC’2008 special session and competition on large scale global optimization.
    Nature inspired computation and applications laboratory, USTC, China, 24, 1-18.
    """

    name = "F5: Shifted Griewank’s Function"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = bias = -450.0"
    continuous = True
    linear = False
    convex = True
    unimodal = False
    separable = False

    differentiable = True
    scalable = True
    randomized_term = False
    parametric = True
    shifted = True
    rotated = False

    modality = False  # Number of ambiguous peaks, unknown # peaks

    # n_basins = 1
    # n_valleys = 1

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "griewank_shift_func_data",
        f_bias: float = -180.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
    ) -> None:
        def compute(x: np.ndarray, f_shift: typing.Any, f_bias: typing.Any, out: np.ndarray) -> None:
            z = x - f_shift
            out[0] = operator.griewank_func(z) + f_bias

        super().__init__(
            ndim=ndim,
            bounds=bounds,
            f_shift=f_shift,
            f_bias=f_bias,
            default_bounds=[[-600.0, 600.0]],
            dim_default=500,
            dim_max=1000,
            data_name="data_2008",
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            compute=compute,
            param_names=["f_shift", "f_bias"],
        )


class F62008(CecBenchmark):
    """
    .. [1] Tang, K., Yáo, X., Suganthan, P. N., MacNish, C., Chen, Y. P., Chen, C. M., & Yang, Z. (2007). Benchmark functions
    for the CEC’2008 special session and competition on large scale global optimization.
    Nature inspired computation and applications laboratory, USTC, China, 24, 1-18.
    """

    name = "F6: Shifted Ackley’s Function"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = bias = -450.0"
    continuous = True
    linear = False
    convex = True
    unimodal = False
    separable = True

    differentiable = True
    scalable = True
    randomized_term = False
    parametric = True
    shifted = True
    rotated = False

    modality = False  # Number of ambiguous peaks, unknown # peaks

    # n_basins = 1
    # n_valleys = 1

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "ackley_shift_func_data",
        f_bias: float = -140.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
    ) -> None:
        def compute(x: np.ndarray, f_shift: typing.Any, f_bias: typing.Any, out: np.ndarray) -> None:
            z = x - f_shift
            out[0] = operator.ackley_func(z) + f_bias

        super().__init__(
            ndim=ndim,
            bounds=bounds,
            f_shift=f_shift,
            f_bias=f_bias,
            default_bounds=[[-32.0, 32.0]],
            dim_default=500,
            dim_max=1000,
            data_name="data_2008",
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            compute=compute,
            param_names=["f_shift", "f_bias"],
        )


class F72008(CecBenchmark):
    """
    .. [1] Tang, K., Yáo, X., Suganthan, P. N., MacNish, C., Chen, Y. P., Chen, C. M., & Yang, Z. (2007). Benchmark functions
    for the CEC’2008 special session and competition on large scale global optimization.
    Nature inspired computation and applications laboratory, USTC, China, 24, 1-18.
    """

    name = "F7: FastFractal “DoubleDip” Function"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = unknown, F_1(x^*) = unknown"
    continuous = True
    linear = False
    convex = False
    unimodal = False
    separable = False

    differentiable = False
    scalable = True
    randomized_term = True
    parametric = True
    shifted = True
    rotated = False

    modality = True  # Number of ambiguous peaks, unknown # peaks

    # n_basins = 1
    # n_valleys = 1

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "rastrigin_shift_func_data",
        f_bias: float = 0.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
    ) -> None:
        def compute(x: np.ndarray, f_bias: typing.Any, out: np.ndarray) -> None:
            ndim = len(x)
            results = [operator.fractal_1d_func(x[idx] + operator.twist_func(x[idx + 1])) for idx in range(0, ndim - 1)]
            out[0] = np.sum(results) + operator.fractal_1d_func(x[-1] + operator.twist_func(x[0])) + f_bias

        super().__init__(
            ndim=ndim,
            bounds=bounds,
            f_shift=f_shift,
            f_bias=f_bias,
            default_bounds=[[-1.0, 1.0]],
            dim_default=500,
            dim_max=1000,
            data_name="data_2008",
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
        )
        self.f_shift = self.f_shift / np.max(self.f_shift)
        self.f_global = -1e32
        self.x_global = self.f_shift
        self._bind_kernel(compute, ["f_bias"], plain=True, paras={"f_shift": self.f_shift, "f_bias": self.f_bias})
