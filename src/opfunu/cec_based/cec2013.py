#!/usr/bin/env python
# Created by "Thieu" at 18:15, 02/07/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import typing

import numpy as np

from opfunu.benchmark.cec import CecBenchmark
from opfunu.utils import operator


class F12013(CecBenchmark):
    """
    .. [1] Liang, J. J., Qu, B. Y., Suganthan, P. N., & Hernández-Díaz, A. G. (2013). Problem definitions and evaluation criteria
    for the CEC 2013 special session on real-parameter optimization. Computational Intelligence Laboratory, Zhengzhou University,
    Zhengzhou, China and Nanyang Technological University, Singapore, Technical Report, 201212(34), 281-295..
    """

    name = "F1: Sphere Function"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = bias = -1400.0"

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
        f_shift: typing.Any = "shift_data",
        f_bias: float = -1400.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
    ) -> None:
        def compute(x: np.ndarray, f_shift: typing.Any, f_bias: typing.Any, out: np.ndarray) -> None:
            out[0] = operator.sphere_func(x - f_shift) + f_bias

        super().__init__(parallel=parallel, fastmath=fastmath, dtype=dtype)
        self.dim_changeable = True
        self.dim_default = 30
        self.dim_max = 100
        self.dim_supported = [2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        self.check_ndim_and_bounds(
            ndim, self.dim_max, bounds, np.array([[-100.0, 100.0] for _ in range(self.dim_default)])
        )
        self.make_support_data_path("data_2013")
        self.f_shift = self.check_shift_matrix(f_shift, selected_idx=0)
        self.f_bias = f_bias
        self.f_global = f_bias
        self.x_global = self.f_shift
        self._bind_kernel(compute, ["f_shift", "f_bias"], paras={"f_shift": self.f_shift, "f_bias": self.f_bias})


class F22013(CecBenchmark):
    """
    .. [1] Liang, J. J., Qu, B. Y., Suganthan, P. N., & Hernández-Díaz, A. G. (2013). Problem definitions and evaluation criteria
    for the CEC 2013 special session on real-parameter optimization. Computational Intelligence Laboratory, Zhengzhou University,
    Zhengzhou, China and Nanyang Technological University, Singapore, Technical Report, 201212(34), 281-295..
    """

    name = "F2: Rotated High Conditioned Elliptic Function"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = bias = -1300.0"

    continuous = True
    linear = False
    convex = True
    unimodal = True
    separable = False

    differentiable = True
    scalable = True
    randomized_term = False
    parametric = True
    shifted = True
    rotated = True

    modality = False  # Number of ambiguous peaks, unknown # peaks
    # n_basins = 1
    # n_valleys = 1

    characteristics = ["Quadratic ill-conditioned", "Smooth local irregularities"]

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "shift_data",
        f_matrix: typing.Any = "M_D",
        f_bias: float = -1300.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
    ) -> None:
        def compute(
            x: np.ndarray, f_matrix: typing.Any, f_shift: typing.Any, f_bias: typing.Any, out: np.ndarray
        ) -> None:
            z = operator.tosz_func(operator.dot_mv(f_matrix, x - f_shift))
            out[0] = operator.elliptic_func(z) + f_bias

        super().__init__(parallel=parallel, fastmath=fastmath, dtype=dtype)
        self.dim_changeable = True
        self.dim_default = 30
        self.dim_max = 100
        self.dim_supported = [2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        self.check_ndim_and_bounds(
            ndim, self.dim_max, bounds, np.array([[-100.0, 100.0] for _ in range(self.dim_default)])
        )
        self.make_support_data_path("data_2013")
        self.f_shift = self.check_shift_matrix(f_shift, selected_idx=0)
        self.f_matrix = self.check_matrix_data(f_matrix)[: self.ndim, : self.ndim]
        self.f_bias = f_bias
        self.f_global = f_bias
        self.x_global = self.f_shift
        self._bind_kernel(
            compute,
            ["f_matrix", "f_shift", "f_bias"],
            paras={"f_shift": self.f_shift, "f_bias": self.f_bias, "f_matrix": self.f_matrix},
        )


class F32013(CecBenchmark):
    """
    .. [1] Liang, J. J., Qu, B. Y., Suganthan, P. N., & Hernández-Díaz, A. G. (2013). Problem definitions and evaluation criteria
    for the CEC 2013 special session on real-parameter optimization. Computational Intelligence Laboratory, Zhengzhou University,
    Zhengzhou, China and Nanyang Technological University, Singapore, Technical Report, 201212(34), 281-295..
    """

    name = "F3: Rotated Bent Cigar Function"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = bias = -1200.0"

    continuous = True
    linear = False
    convex = True
    unimodal = True
    separable = False

    differentiable = True
    scalable = True
    randomized_term = False
    parametric = True
    shifted = True
    rotated = True

    modality = False  # Number of ambiguous peaks, unknown # peaks
    # n_basins = 1
    # n_valleys = 1

    characteristics = ["Smooth but narrow ridge"]

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "shift_data",
        f_matrix: typing.Any = "M_D",
        f_bias: float = -1200.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
    ) -> None:
        def compute(
            x: np.ndarray, f_matrix: typing.Any, f_shift: typing.Any, f_bias: typing.Any, out: np.ndarray
        ) -> None:
            M1 = f_matrix[: x.shape[0], :]
            M2 = f_matrix[x.shape[0] : 2 * x.shape[0], :]
            z = operator.tasy_func(operator.dot_mv(M1, x - f_shift), beta=0.5)
            out[0] = operator.bent_cigar_func(operator.dot_mv(M2, z)) + f_bias

        super().__init__(parallel=parallel, fastmath=fastmath, dtype=dtype)
        self.dim_changeable = True
        self.dim_default = 30
        self.dim_max = 100
        self.dim_supported = [2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        self.check_ndim_and_bounds(
            ndim, self.dim_max, bounds, np.array([[-100.0, 100.0] for _ in range(self.dim_default)])
        )
        self.make_support_data_path("data_2013")
        self.f_shift = self.check_shift_matrix(f_shift, selected_idx=0)
        self.f_matrix = self.check_matrix_data(f_matrix)[: 2 * self.ndim, : self.ndim]
        self.f_bias = f_bias
        self.f_global = f_bias
        self.x_global = self.f_shift
        self._bind_kernel(
            compute,
            ["f_matrix", "f_shift", "f_bias"],
            paras={"f_shift": self.f_shift, "f_bias": self.f_bias, "f_matrix": self.f_matrix},
        )


class F42013(F22013):
    """
    .. [1] Liang, J. J., Qu, B. Y., Suganthan, P. N., & Hernández-Díaz, A. G. (2013). Problem definitions and evaluation criteria
    for the CEC 2013 special session on real-parameter optimization. Computational Intelligence Laboratory, Zhengzhou University,
    Zhengzhou, China and Nanyang Technological University, Singapore, Technical Report, 201212(34), 281-295..
    """

    name = "F4: Rotated Discus Function"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = bias = -1100.0"

    characteristics = ["Asymmetrical", "Smooth local irregularities", "With one sensitive direction"]

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "shift_data",
        f_matrix: typing.Any = "M_D",
        f_bias: float = -1100.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
    ) -> None:
        def compute(
            x: np.ndarray, f_matrix: typing.Any, f_shift: typing.Any, f_bias: typing.Any, out: np.ndarray
        ) -> None:
            z = operator.tosz_func(operator.dot_mv(f_matrix, x - f_shift))
            out[0] = operator.discus_func(z) + f_bias

        super().__init__(
            ndim,
            bounds,
            f_shift,
            f_matrix,
            f_bias,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
        )
        self._bind_kernel(compute, ["f_matrix", "f_shift", "f_bias"])


class F52013(F12013):
    """
    .. [1] Liang, J. J., Qu, B. Y., Suganthan, P. N., & Hernández-Díaz, A. G. (2013). Problem definitions and evaluation criteria
    for the CEC 2013 special session on real-parameter optimization. Computational Intelligence Laboratory, Zhengzhou University,
    Zhengzhou, China and Nanyang Technological University, Singapore, Technical Report, 201212(34), 281-295..
    """

    name = "F5: Different Powers Function"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = bias = -1000.0"

    continuous = False
    differentiable = False

    characteristics = ["Sensitivities of the zi-variables are different"]

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "shift_data",
        f_bias: float = -1000.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
    ) -> None:
        def compute(x: np.ndarray, f_shift: typing.Any, f_bias: typing.Any, out: np.ndarray) -> None:
            out[0] = operator.different_powers_func(x - f_shift) + f_bias

        super().__init__(
            ndim,
            bounds,
            f_shift,
            f_bias,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
        )
        self._bind_kernel(compute, ["f_shift", "f_bias"])


class F62013(F22013):
    """
    .. [1] Liang, J. J., Qu, B. Y., Suganthan, P. N., & Hernández-Díaz, A. G. (2013). Problem definitions and evaluation criteria
    for the CEC 2013 special session on real-parameter optimization. Computational Intelligence Laboratory, Zhengzhou University,
    Zhengzhou, China and Nanyang Technological University, Singapore, Technical Report, 201212(34), 281-295..
    """

    name = "F6: Rotated Rosenbrock’s Function"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = bias = -900.0"

    unimodal = False
    characteristics = ["Having a very narrow valley from local optimum to global optimum"]

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "shift_data",
        f_matrix: typing.Any = "M_D",
        f_bias: float = -900.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
    ) -> None:
        def compute(
            x: np.ndarray, f_matrix: typing.Any, f_shift: typing.Any, f_bias: typing.Any, out: np.ndarray
        ) -> None:
            z = operator.dot_mv(f_matrix, 2.048 * (x - f_shift) / 100) + 1
            out[0] = operator.rosenbrock_func(z) + f_bias

        super().__init__(
            ndim,
            bounds,
            f_shift,
            f_matrix,
            f_bias,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
        )
        self._bind_kernel(compute, ["f_matrix", "f_shift", "f_bias"])


class F72013(F32013):
    """
    .. [1] Liang, J. J., Qu, B. Y., Suganthan, P. N., & Hernández-Díaz, A. G. (2013). Problem definitions and evaluation criteria
    for the CEC 2013 special session on real-parameter optimization. Computational Intelligence Laboratory, Zhengzhou University,
    Zhengzhou, China and Nanyang Technological University, Singapore, Technical Report, 201212(34), 281-295..
    """

    name = "F7: Rotated Schaffers F7 Function"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = bias = -800.0"

    unimodal = False
    modality = True  # Number of ambiguous peaks, unknown # peaks
    characteristics = ["Asymmetrical", "Local optima’s number is huge"]

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "shift_data",
        f_matrix: typing.Any = "M_D",
        f_bias: float = -800.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
    ) -> None:
        def compute(
            x: np.ndarray, f_matrix: typing.Any, f_shift: typing.Any, f_bias: typing.Any, out: np.ndarray
        ) -> None:
            M1 = f_matrix[: x.shape[0], :]
            M2 = f_matrix[x.shape[0] : 2 * x.shape[0], :]
            alpha = operator.generate_diagonal_matrix(x.shape[0], alpha=10)
            temp = operator.tasy_func(operator.dot_mv(M1, x - f_shift), beta=0.5)
            y = operator.dot_mv(operator.dot_mm(alpha, M2), temp)
            result = 0.0
            for idx in range(0, x.shape[0] - 1):
                z = np.sqrt(y[idx] ** 2 + y[idx + 1] ** 2)
                result += np.sqrt(z) * (1 + np.sin(50 * z**0.2) ** 2)
            out[0] = result**2 + f_bias

        super().__init__(
            ndim,
            bounds,
            f_shift,
            f_matrix,
            f_bias,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
        )
        self._bind_kernel(compute, ["f_matrix", "f_shift", "f_bias"])


class F82013(F32013):
    """
    .. [1] Liang, J. J., Qu, B. Y., Suganthan, P. N., & Hernández-Díaz, A. G. (2013). Problem definitions and evaluation criteria
    for the CEC 2013 special session on real-parameter optimization. Computational Intelligence Laboratory, Zhengzhou University,
    Zhengzhou, China and Nanyang Technological University, Singapore, Technical Report, 201212(34), 281-295..
    """

    name = "F8: Rotated Ackley’s Function"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = bias = -700.0"

    unimodal = False
    characteristics = ["Asymmetrical"]

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "shift_data",
        f_matrix: typing.Any = "M_D",
        f_bias: float = -700.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
    ) -> None:
        def compute(
            x: np.ndarray, f_matrix: typing.Any, f_shift: typing.Any, f_bias: typing.Any, out: np.ndarray
        ) -> None:
            M1 = f_matrix[: x.shape[0], :]
            M2 = f_matrix[x.shape[0] : 2 * x.shape[0], :]
            alpha = operator.generate_diagonal_matrix(x.shape[0], alpha=10)
            temp = operator.tasy_func(operator.dot_mv(M1, x - f_shift), beta=0.5)
            z = operator.dot_mv(operator.dot_mm(alpha, M2), temp)
            out[0] = operator.ackley_func(z) + f_bias

        super().__init__(
            ndim,
            bounds,
            f_shift,
            f_matrix,
            f_bias,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
        )
        self._bind_kernel(compute, ["f_matrix", "f_shift", "f_bias"])


class F92013(F32013):
    """
    .. [1] Liang, J. J., Qu, B. Y., Suganthan, P. N., & Hernández-Díaz, A. G. (2013). Problem definitions and evaluation criteria
    for the CEC 2013 special session on real-parameter optimization. Computational Intelligence Laboratory, Zhengzhou University,
    Zhengzhou, China and Nanyang Technological University, Singapore, Technical Report, 201212(34), 281-295..
    """

    name = "F9: Rotated Weierstrass Function"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = bias = -600.0"

    unimodal = False
    modality = True  # Number of ambiguous peaks, unknown # peaks
    characteristics = ["Asymmetrical", "Continuous but differentiable only on a set of points"]

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "shift_data",
        f_matrix: typing.Any = "M_D",
        f_bias: float = -600.0,
        a: float = 0.5,
        b: float = 3.0,
        k_max: int = 20,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
    ) -> None:
        def compute(
            x: np.ndarray, f_matrix: typing.Any, f_shift: typing.Any, f_bias: typing.Any, out: np.ndarray
        ) -> None:
            M1 = f_matrix[: x.shape[0], :]
            M2 = f_matrix[x.shape[0] : 2 * x.shape[0], :]
            alpha = operator.generate_diagonal_matrix(x.shape[0], alpha=10)
            temp = operator.tasy_func(operator.dot_mv(M1, 0.5 * (x - f_shift) / 100), beta=0.5)
            z = operator.dot_mv(operator.dot_mm(alpha, M2), temp)
            out[0] = operator.weierstrass_norm_func(z, a=0.5, b=3.0, k_max=20) + f_bias

        super().__init__(ndim, bounds, f_shift, f_matrix, f_bias, parallel=parallel, fastmath=fastmath, dtype=dtype)
        self.a = a
        self.b = b
        self.k_max = k_max
        self._bind_kernel(
            compute,
            ["f_matrix", "f_shift", "f_bias"],
            paras={
                "f_shift": self.f_shift,
                "f_bias": self.f_bias,
                "f_matrix": self.f_matrix,
                "a": self.a,
                "b": self.b,
                "k_max": self.k_max,
            },
        )


class F102013(F22013):
    """
    .. [1] Liang, J. J., Qu, B. Y., Suganthan, P. N., & Hernández-Díaz, A. G. (2013). Problem definitions and evaluation criteria
    for the CEC 2013 special session on real-parameter optimization. Computational Intelligence Laboratory, Zhengzhou University,
    Zhengzhou, China and Nanyang Technological University, Singapore, Technical Report, 201212(34), 281-295..
    """

    name = "F10: Rotated Griewank’s Function"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = bias = -500.0"

    unimodal = False
    characteristics = []

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "shift_data",
        f_matrix: typing.Any = "M_D",
        f_bias: float = -500.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
    ) -> None:
        def compute(
            x: np.ndarray, f_matrix: typing.Any, f_shift: typing.Any, f_bias: typing.Any, out: np.ndarray
        ) -> None:
            alpha = operator.generate_diagonal_matrix(x.shape[0], alpha=100)
            temp = operator.dot_mm(alpha, f_matrix)
            z = operator.dot_mv(temp, 600.0 * (x - f_shift) / 100)
            out[0] = operator.griewank_func(z) + f_bias

        super().__init__(
            ndim,
            bounds,
            f_shift,
            f_matrix,
            f_bias,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
        )
        self._bind_kernel(compute, ["f_matrix", "f_shift", "f_bias"])


class F112013(F12013):
    """
    .. [1] Liang, J. J., Qu, B. Y., Suganthan, P. N., & Hernández-Díaz, A. G. (2013). Problem definitions and evaluation criteria
    for the CEC 2013 special session on real-parameter optimization. Computational Intelligence Laboratory, Zhengzhou University,
    Zhengzhou, China and Nanyang Technological University, Singapore, Technical Report, 201212(34), 281-295..
    """

    name = "F11: Rastrigin’s Function"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = bias = -400.0"

    convex = False
    unimodal = False
    modality = True  # Number of ambiguous peaks, unknown # peaks
    # n_basins = 1
    # n_valleys = 1
    characteristics = ["Asymmetrical", "Local optima’s number is huge"]

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "shift_data",
        f_bias: float = -400.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
    ) -> None:
        def compute(x: np.ndarray, f_shift: typing.Any, f_bias: typing.Any, out: np.ndarray) -> None:
            alpha = operator.generate_diagonal_matrix(x.shape[0], alpha=10)
            temp = operator.tosz_func(5.12 * (x - f_shift) / 100)
            z = operator.dot_mv(alpha, operator.tasy_func(temp, beta=0.2))
            out[0] = operator.rastrigin_func(z) + f_bias

        super().__init__(
            ndim,
            bounds,
            f_shift,
            f_bias,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
        )
        self._bind_kernel(compute, ["f_shift", "f_bias"])


class F122013(F32013):
    """
    .. [1] Liang, J. J., Qu, B. Y., Suganthan, P. N., & Hernández-Díaz, A. G. (2013). Problem definitions and evaluation criteria
    for the CEC 2013 special session on real-parameter optimization. Computational Intelligence Laboratory, Zhengzhou University,
    Zhengzhou, China and Nanyang Technological University, Singapore, Technical Report, 201212(34), 281-295..
    """

    name = "F12: Rotated Rastrigin’s Function"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = bias = -300.0"

    convex = False
    unimodal = False
    modality = True  # Number of ambiguous peaks, unknown # peaks
    # n_basins = 1
    # n_valleys = 1

    characteristics = ["Asymmetrical", "Local optima’s number is huge"]

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "shift_data",
        f_matrix: typing.Any = "M_D",
        f_bias: float = -300.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
    ) -> None:
        def compute(
            x: np.ndarray, f_matrix: typing.Any, f_shift: typing.Any, f_bias: typing.Any, out: np.ndarray
        ) -> None:
            M1 = f_matrix[: x.shape[0], :]
            M2 = f_matrix[x.shape[0] : 2 * x.shape[0], :]
            alpha = operator.generate_diagonal_matrix(x.shape[0], alpha=10)
            temp1 = operator.dot_mm(operator.dot_mm(M1, alpha), M2)
            temp2 = operator.tosz_func(operator.dot_mv(M1, 5.12 * (x - f_shift) / 100))
            z = operator.dot_mv(temp1, operator.tasy_func(temp2, beta=0.2))
            out[0] = operator.rastrigin_func(z) + f_bias

        super().__init__(
            ndim,
            bounds,
            f_shift,
            f_matrix,
            f_bias,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
        )
        self._bind_kernel(compute, ["f_matrix", "f_shift", "f_bias"])


class F132013(F32013):
    """
    .. [1] Liang, J. J., Qu, B. Y., Suganthan, P. N., & Hernández-Díaz, A. G. (2013). Problem definitions and evaluation criteria
    for the CEC 2013 special session on real-parameter optimization. Computational Intelligence Laboratory, Zhengzhou University,
    Zhengzhou, China and Nanyang Technological University, Singapore, Technical Report, 201212(34), 281-295..
    """

    name = "F13: Non-continuous Rotated Rastrigin’s Function"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = bias = -200.0"

    continuous = False
    linear = False
    convex = False
    unimodal = False
    separable = False
    modality = True  # Number of ambiguous peaks, unknown # peaks
    # n_basins = 1
    # n_valleys = 1

    characteristics = ["Asymmetrical", "Local optima’s number is huge"]

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "shift_data",
        f_matrix: typing.Any = "M_D",
        f_bias: float = -200.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
    ) -> None:
        def compute(
            x: np.ndarray, f_matrix: typing.Any, f_shift: typing.Any, f_bias: typing.Any, out: np.ndarray
        ) -> None:
            M1 = f_matrix[: x.shape[0], :]
            M2 = f_matrix[x.shape[0] : 2 * x.shape[0], :]
            alpha = operator.generate_diagonal_matrix(x.shape[0], alpha=10)
            x_star = operator.dot_mv(M1, 5.12 * (x - f_shift) / 100)
            y = operator.rounder(x_star, np.abs(x_star))
            temp1 = operator.tasy_func(operator.tosz_func(y), beta=0.2)
            temp2 = operator.dot_mm(operator.dot_mm(M1, alpha), M2)
            z = operator.dot_mv(temp2, temp1)
            out[0] = operator.rastrigin_func(z) + f_bias

        super().__init__(
            ndim,
            bounds,
            f_shift,
            f_matrix,
            f_bias,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
        )
        self._bind_kernel(compute, ["f_matrix", "f_shift", "f_bias"])


class F142013(F12013):
    """
    .. [1] Liang, J. J., Qu, B. Y., Suganthan, P. N., & Hernández-Díaz, A. G. (2013). Problem definitions and evaluation criteria
    for the CEC 2013 special session on real-parameter optimization. Computational Intelligence Laboratory, Zhengzhou University,
    Zhengzhou, China and Nanyang Technological University, Singapore, Technical Report, 201212(34), 281-295..
    """

    name = "F14: Schwefel’s Function"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = bias = -100.0"

    continuous = True
    convex = False
    unimodal = False
    separable = False
    differentiable = False
    rotated = True

    modality = True  # Number of ambiguous peaks, unknown # peaks
    # n_basins = 1
    # n_valleys = 1

    characteristics = [
        "Asymmetrical",
        "Local optima’s number is huge",
        "Second better local optimum is far from the global optimum",
    ]

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "shift_data",
        f_bias: float = -100.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
    ) -> None:
        def compute(x: np.ndarray, f_shift: typing.Any, f_bias: typing.Any, out: np.ndarray) -> None:
            alpha = operator.generate_diagonal_matrix(x.shape[0], alpha=10)
            z = operator.dot_mv(alpha, 1000 * (x - f_shift) / 100)
            out[0] = operator.modified_schwefel_func(z) + f_bias

        super().__init__(
            ndim,
            bounds,
            f_shift,
            f_bias,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
        )
        self._bind_kernel(compute, ["f_shift", "f_bias"])


class F152013(F22013):
    """
    .. [1] Liang, J. J., Qu, B. Y., Suganthan, P. N., & Hernández-Díaz, A. G. (2013). Problem definitions and evaluation criteria
    for the CEC 2013 special session on real-parameter optimization. Computational Intelligence Laboratory, Zhengzhou University,
    Zhengzhou, China and Nanyang Technological University, Singapore, Technical Report, 201212(34), 281-295..
    """

    name = "F15: Rotated Schwefel’s Function"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = bias = 100.0"

    continuous = True
    convex = False
    unimodal = False
    modality = True  # Number of ambiguous peaks, unknown # peaks
    # n_basins = 1
    # n_valleys = 1

    characteristics = [
        "Asymmetrical",
        "Local optima’s number is huge",
        "The second better local optimum is far from the global optimum",
    ]

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "shift_data",
        f_matrix: typing.Any = "M_D",
        f_bias: float = 100.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
    ) -> None:
        def compute(
            x: np.ndarray, f_matrix: typing.Any, f_shift: typing.Any, f_bias: typing.Any, out: np.ndarray
        ) -> None:
            alpha = operator.generate_diagonal_matrix(x.shape[0], alpha=10)
            z = operator.dot_mv(operator.dot_mm(alpha, f_matrix), 1000 * (x - f_shift) / 100)
            out[0] = operator.modified_schwefel_func(z) + f_bias

        super().__init__(
            ndim,
            bounds,
            f_shift,
            f_matrix,
            f_bias,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
        )
        self._bind_kernel(compute, ["f_matrix", "f_shift", "f_bias"])


class F162013(F32013):
    """
    .. [1] Liang, J. J., Qu, B. Y., Suganthan, P. N., & Hernández-Díaz, A. G. (2013). Problem definitions and evaluation criteria
    for the CEC 2013 special session on real-parameter optimization. Computational Intelligence Laboratory, Zhengzhou University,
    Zhengzhou, China and Nanyang Technological University, Singapore, Technical Report, 201212(34), 281-295..
    """

    name = "F16: Rotated Katsuura Function"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = bias = 200.0"

    convex = False
    unimodal = False
    differentiable = False
    modality = True  # Number of ambiguous peaks, unknown # peaks
    # n_basins = 1
    # n_valleys = 1

    characteristics = ["Asymmetrical", "Continuous everywhere yet differentiable nowhere"]

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "shift_data",
        f_matrix: typing.Any = "M_D",
        f_bias: float = 200.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
    ) -> None:
        def compute(
            x: np.ndarray, f_matrix: typing.Any, f_shift: typing.Any, f_bias: typing.Any, out: np.ndarray
        ) -> None:
            M1 = f_matrix[: x.shape[0], :]
            M2 = f_matrix[x.shape[0] : 2 * x.shape[0], :]
            alpha = operator.generate_diagonal_matrix(x.shape[0], alpha=100)
            temp = operator.dot_mv(M1, 5.0 * (x - f_shift) / 100)
            z = operator.dot_mv(operator.dot_mm(M2, alpha), temp)
            out[0] = operator.katsuura_func(z) + f_bias

        super().__init__(
            ndim,
            bounds,
            f_shift,
            f_matrix,
            f_bias,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
        )
        self._bind_kernel(compute, ["f_matrix", "f_shift", "f_bias"])


class F172013(F12013):
    """
    .. [1] Liang, J. J., Qu, B. Y., Suganthan, P. N., & Hernández-Díaz, A. G. (2013). Problem definitions and evaluation criteria
    for the CEC 2013 special session on real-parameter optimization. Computational Intelligence Laboratory, Zhengzhou University,
    Zhengzhou, China and Nanyang Technological University, Singapore, Technical Report, 201212(34), 281-295..
    """

    name = "F17: Lunacek bi-Rastrigin Function"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = bias = 300.0"

    convex = False
    unimodal = False
    separable = False
    differentiable = False
    modality = True  # Number of ambiguous peaks, unknown # peaks

    # n_basins = 1
    # n_valleys = 1

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "shift_data",
        f_bias: float = 300.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
    ) -> None:
        def compute(x: np.ndarray, f_shift: typing.Any, f_bias: typing.Any, out: np.ndarray) -> None:
            miu0 = 2.5
            y = 10.0 * (x - f_shift) / 100
            x_hat = 2 * np.sign(f_shift) * y
            out[0] = operator.lunacek_bi_rastrigin_func(x_hat, miu0=miu0, d=1, shift=miu0) + f_bias

        super().__init__(
            ndim,
            bounds,
            f_shift,
            f_bias,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
        )
        self._bind_kernel(compute, ["f_shift", "f_bias"])


class F182013(F32013):
    """
    .. [1] Liang, J. J., Qu, B. Y., Suganthan, P. N., & Hernández-Díaz, A. G. (2013). Problem definitions and evaluation criteria
    for the CEC 2013 special session on real-parameter optimization. Computational Intelligence Laboratory, Zhengzhou University,
    Zhengzhou, China and Nanyang Technological University, Singapore, Technical Report, 201212(34), 281-295..
    """

    name = "F18: Rotated Lunacek bi-Rastrigin Function"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = bias = 400.0"

    continuous = True
    convex = False
    unimodal = False
    differentiable = False
    modality = False  # Number of ambiguous peaks, unknown # peaks
    # n_basins = 1
    # n_valleys = 1

    characteristics = ["Asymmetrical", "Continuous everywhere yet differentiable nowhere"]

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "shift_data",
        f_matrix: typing.Any = "M_D",
        f_bias: float = 400.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
    ) -> None:
        def compute(x: np.ndarray, f_shift: typing.Any, f_bias: typing.Any, out: np.ndarray) -> None:
            miu0 = 2.5
            y = 10.0 * (x - f_shift) / 100
            x_hat = 2 * np.sign(y) * y
            out[0] = operator.lunacek_bi_rastrigin_func(x_hat, miu0=miu0, d=1, shift=miu0) + f_bias

        super().__init__(
            ndim,
            bounds,
            f_shift,
            f_matrix,
            f_bias,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
        )
        self._bind_kernel(compute, ["f_shift", "f_bias"])


class F192013(F22013):
    """
    .. [1] Liang, J. J., Qu, B. Y., Suganthan, P. N., & Hernández-Díaz, A. G. (2013). Problem definitions and evaluation criteria
    for the CEC 2013 special session on real-parameter optimization. Computational Intelligence Laboratory, Zhengzhou University,
    Zhengzhou, China and Nanyang Technological University, Singapore, Technical Report, 201212(34), 281-295..
    """

    name = "F19: Rotated Expanded Griewank’s plus Rosenbrock’s Function"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = bias = 500.0"

    unimodal = False
    modality = True  # Number of ambiguous peaks, unknown # peaks
    # n_basins = 1
    # n_valleys = 1
    characteristics = []

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "shift_data",
        f_matrix: typing.Any = "M_D",
        f_bias: float = 500.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
    ) -> None:
        def compute(
            x: np.ndarray, f_matrix: typing.Any, f_shift: typing.Any, f_bias: typing.Any, out: np.ndarray
        ) -> None:
            z = operator.dot_mv(f_matrix, 5 * (x - f_shift) / 100)
            out[0] = operator.grie_rosen_cec_func(z) + f_bias

        super().__init__(
            ndim,
            bounds,
            f_shift,
            f_matrix,
            f_bias,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
        )
        self._bind_kernel(compute, ["f_matrix", "f_shift", "f_bias"])


class F202013(F32013):
    """
    .. [1] Liang, J. J., Qu, B. Y., Suganthan, P. N., & Hernández-Díaz, A. G. (2013). Problem definitions and evaluation criteria
    for the CEC 2013 special session on real-parameter optimization. Computational Intelligence Laboratory, Zhengzhou University,
    Zhengzhou, China and Nanyang Technological University, Singapore, Technical Report, 201212(34), 281-295..
    """

    name = "F20: Rotated Expanded Scaffer’s F6 Function"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = bias = 600.0"

    convex = False
    unimodal = False
    modality = True  # Number of ambiguous peaks, unknown # peaks
    # n_basins = 1
    # n_valleys = 1
    characteristics = ["Asymmetrical"]

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "shift_data",
        f_matrix: typing.Any = "M_D",
        f_bias: float = 600.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
    ) -> None:
        def compute(
            x: np.ndarray, f_matrix: typing.Any, f_shift: typing.Any, f_bias: typing.Any, out: np.ndarray
        ) -> None:
            M1 = f_matrix[: x.shape[0], :]
            M2 = f_matrix[x.shape[0] : 2 * x.shape[0], :]
            z = operator.dot_mv(M2, operator.tasy_func(operator.dot_mv(M1, x - f_shift), beta=0.5))
            out[0] = operator.expanded_scaffer_f6_func(z) + f_bias

        super().__init__(
            ndim,
            bounds,
            f_shift,
            f_matrix,
            f_bias,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
        )
        self._bind_kernel(compute, ["f_matrix", "f_shift", "f_bias"])


class F212013(CecBenchmark):
    """
    .. [1] Liang, J. J., Qu, B. Y., Suganthan, P. N., & Hernández-Díaz, A. G. (2013). Problem definitions and evaluation criteria
    for the CEC 2013 special session on real-parameter optimization. Computational Intelligence Laboratory, Zhengzhou University,
    Zhengzhou, China and Nanyang Technological University, Singapore, Technical Report, 201212(34), 281-295..
    """

    name = "F21: Composition Function 1"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = bias = 700.0"

    continuous = True
    linear = False
    convex = False
    unimodal = False
    separable = False

    differentiable = True
    scalable = True
    randomized_term = False
    parametric = True
    shifted = True
    rotated = True

    modality = True  # Number of ambiguous peaks, unknown # peaks
    # n_basins = 1
    # n_valleys = 1
    characteristics = ["Asymmetrical", "Different properties around different local optima"]

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "shift_data",
        f_matrix: typing.Any = "M_D",
        f_bias: float = 700.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
    ) -> None:
        def compute(
            x: np.ndarray,
            lamdas: typing.Any,
            g0: typing.Any,
            bias: typing.Any,
            f_shift: typing.Any,
            xichmas: typing.Any,
            g1: typing.Any,
            f_matrix: typing.Any,
            g2: typing.Any,
            g3: typing.Any,
            g4: typing.Any,
            f_bias: typing.Any,
            out: np.ndarray,
        ) -> None:
            g0 = lamdas[0] * g0.evaluate(x) + bias[0]
            w0 = operator.calculate_weight(x - f_shift[0], xichmas[0])

            # g2: Rotated Different Powers Function f5’
            g1 = lamdas[1] * g1.evaluate(x) + bias[1]
            w1 = operator.calculate_weight(x - f_shift[1], xichmas[1])

            # s1 = operator.dot_mv(f_matrix[: x.shape[0], :], x - f_shift[1])
            # g1 = lamdas[1] * operator.different_powers_func(s1) + bias[1]
            # w1 = operator.calculate_weight(x - f_shift[1], xichmas[1])

            # g3 Rotated Bent Cigar Function f3’
            g2 = lamdas[2] * g2.evaluate(x) + bias[2]
            w2 = operator.calculate_weight(x - f_shift[2], xichmas[2])

            # g4: Rotated Discus Function f4’
            g3 = lamdas[3] * g3.evaluate(x) + bias[3]
            w3 = operator.calculate_weight(x - f_shift[3], xichmas[3])

            # g5: Sphere Function f1
            g4 = lamdas[4] * g4.evaluate(x) + bias[4]
            w4 = operator.calculate_weight(x - f_shift[4], xichmas[4])

            ws = np.array([w0, w1, w2, w3, w4])
            ws = ws / np.sum(ws)
            gs = np.array([g0, g1, g2, g3, g4])
            out[0] = operator.dot_vv(ws, gs) + f_bias

        super().__init__(parallel=parallel, fastmath=fastmath, dtype=dtype)
        self.dim_changeable = True
        self.dim_default = 30
        self.dim_max = 100
        self.dim_supported = [2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        self.check_ndim_and_bounds(
            ndim, self.dim_max, bounds, np.array([[-100.0, 100.0] for _ in range(self.dim_default)])
        )
        self.make_support_data_path("data_2013")
        self.f_shift = self.check_shift_matrix(f_shift)[:, : self.ndim]
        self.f_matrix = self.check_matrix_data(f_matrix)[:, : self.ndim]
        self.f_bias = f_bias
        self.f_global = f_bias
        self.x_global = self.f_shift[0]
        self.n_funcs = 5
        self.xichmas = [10, 20, 30, 40, 50]
        self.lamdas = [1.0, 1e-6, 1e-26, 1e-6, 0.1]
        self.bias = [0, 100, 200, 300, 400]
        self.g0 = F62013(self.ndim, f_shift=self.f_shift[0], f_matrix=self.f_matrix[: self.ndim, : self.ndim], f_bias=0)
        self.g1 = F52013(self.ndim, None, self.f_shift[1], f_bias=0)
        self.g2 = F32013(
            self.ndim, None, self.f_shift[2], f_matrix=self.f_matrix[: 2 * self.ndim, : self.ndim], f_bias=0
        )
        self.g3 = F42013(self.ndim, None, self.f_shift[3], self.f_matrix[: self.ndim, : self.ndim], 0)
        self.g4 = F12013(self.ndim, None, self.f_shift[4], f_bias=0)
        self._bind_kernel(
            compute,
            ["lamdas", "g0", "bias", "f_shift", "xichmas", "g1", "f_matrix", "g2", "g3", "g4", "f_bias"],
            paras={"f_shift": self.f_shift, "f_bias": self.f_bias, "f_matrix": self.f_matrix},
            plain=True,
        )


class F222013(CecBenchmark):
    """
    .. [1] Liang, J. J., Qu, B. Y., Suganthan, P. N., & Hernández-Díaz, A. G. (2013). Problem definitions and evaluation criteria
    for the CEC 2013 special session on real-parameter optimization. Computational Intelligence Laboratory, Zhengzhou University,
    Zhengzhou, China and Nanyang Technological University, Singapore, Technical Report, 201212(34), 281-295..
    """

    name = "F22: Composition Function 2"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = bias = 800.0"

    continuous = True
    linear = False
    convex = False
    unimodal = False
    separable = True

    differentiable = True
    scalable = True
    randomized_term = False
    parametric = True
    shifted = True
    rotated = True

    modality = True  # Number of ambiguous peaks, unknown # peaks
    # n_basins = 1
    # n_valleys = 1
    characteristics = ["Asymmetrical", "Different properties around different local optima"]

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "shift_data",
        f_bias: float = 800.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
    ) -> None:
        def compute(
            x: np.ndarray,
            lamdas: typing.Any,
            g0: typing.Any,
            bias: typing.Any,
            f_shift: typing.Any,
            xichmas: typing.Any,
            g1: typing.Any,
            g2: typing.Any,
            f_bias: typing.Any,
            out: np.ndarray,
        ) -> None:
            g0 = lamdas[0] * g0.evaluate(x) + bias[0]
            w0 = operator.calculate_weight(x - f_shift[0], xichmas[0])

            g1 = lamdas[1] * g1.evaluate(x) + bias[1]
            w1 = operator.calculate_weight(x - f_shift[1], xichmas[1])

            g2 = lamdas[2] * g2.evaluate(x) + bias[2]
            w2 = operator.calculate_weight(x - f_shift[2], xichmas[2])

            ws = np.array([w0, w1, w2])
            ws = ws / np.sum(ws)
            gs = np.array([g0, g1, g2])
            out[0] = operator.dot_vv(ws, gs) + f_bias

        super().__init__(parallel=parallel, fastmath=fastmath, dtype=dtype)
        self.dim_changeable = True
        self.dim_default = 30
        self.dim_max = 100
        self.dim_supported = [2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        self.check_ndim_and_bounds(
            ndim, self.dim_max, bounds, np.array([[-100.0, 100.0] for _ in range(self.dim_default)])
        )
        self.make_support_data_path("data_2013")
        self.f_shift = self.check_shift_matrix(f_shift)[:, : self.ndim]
        self.f_bias = f_bias
        self.f_global = f_bias
        self.x_global = self.f_shift[0]
        self.n_funcs = 3
        self.xichmas = [20, 20, 20]
        self.lamdas = [1.0, 1.0, 1.0]
        self.bias = [0, 100, 200]
        self.g0 = F142013(self.ndim, None, self.f_shift[0], f_bias=0)
        self.g1 = F142013(self.ndim, None, self.f_shift[1], f_bias=0)
        self.g2 = F142013(self.ndim, None, self.f_shift[2], f_bias=0)
        self._bind_kernel(
            compute,
            ["lamdas", "g0", "bias", "f_shift", "xichmas", "g1", "g2", "f_bias"],
            paras={"f_shift": self.f_shift, "f_bias": self.f_bias},
            plain=True,
        )


class F232013(CecBenchmark):
    """
    .. [1] Liang, J. J., Qu, B. Y., Suganthan, P. N., & Hernández-Díaz, A. G. (2013). Problem definitions and evaluation criteria
    for the CEC 2013 special session on real-parameter optimization. Computational Intelligence Laboratory, Zhengzhou University,
    Zhengzhou, China and Nanyang Technological University, Singapore, Technical Report, 201212(34), 281-295..
    """

    name = "F23: Composition Function 3"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = bias = 900.0"

    continuous = True
    linear = False
    convex = False
    unimodal = False
    separable = False

    differentiable = True
    scalable = True
    randomized_term = False
    parametric = True
    shifted = True
    rotated = True

    modality = True  # Number of ambiguous peaks, unknown # peaks
    # n_basins = 1
    # n_valleys = 1
    characteristics = ["Asymmetrical", "Different properties around different local optima"]

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "shift_data",
        f_matrix: typing.Any = "M_D",
        f_bias: float = 900.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
    ) -> None:
        def compute(
            x: np.ndarray,
            lamdas: typing.Any,
            g0: typing.Any,
            bias: typing.Any,
            f_shift: typing.Any,
            xichmas: typing.Any,
            g1: typing.Any,
            g2: typing.Any,
            f_bias: typing.Any,
            out: np.ndarray,
        ) -> None:
            g0 = lamdas[0] * g0.evaluate(x) + bias[0]
            w0 = operator.calculate_weight(x - f_shift[0], xichmas[0])

            g1 = lamdas[1] * g1.evaluate(x) + bias[1]
            w1 = operator.calculate_weight(x - f_shift[1], xichmas[1])

            g2 = lamdas[2] * g2.evaluate(x) + bias[2]
            w2 = operator.calculate_weight(x - f_shift[2], xichmas[2])

            ws = np.array([w0, w1, w2])
            ws = ws / np.sum(ws)
            gs = np.array([g0, g1, g2])
            out[0] = operator.dot_vv(ws, gs) + f_bias

        super().__init__(parallel=parallel, fastmath=fastmath, dtype=dtype)
        self.dim_changeable = True
        self.dim_default = 30
        self.dim_max = 100
        self.dim_supported = [2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        self.check_ndim_and_bounds(
            ndim, self.dim_max, bounds, np.array([[-100.0, 100.0] for _ in range(self.dim_default)])
        )
        self.make_support_data_path("data_2013")
        self.f_shift = self.check_shift_matrix(f_shift)[:, : self.ndim]
        self.f_matrix = self.check_matrix_data(f_matrix)[:, : self.ndim]
        self.f_bias = f_bias
        self.f_global = f_bias
        self.x_global = self.f_shift[0]
        self.n_funcs = 3
        self.xichmas = [20, 20, 20]
        self.lamdas = [1.0, 1.0, 1.0]
        self.bias = [0, 100, 200]
        self.g0 = F152013(self.ndim, None, self.f_shift[0], self.f_matrix[: self.ndim, : self.ndim], f_bias=0)
        self.g1 = F152013(self.ndim, None, self.f_shift[1], self.f_matrix[: self.ndim, : self.ndim], f_bias=0)
        self.g2 = F152013(self.ndim, None, self.f_shift[2], self.f_matrix[: self.ndim, : self.ndim], f_bias=0)
        self._bind_kernel(
            compute,
            ["lamdas", "g0", "bias", "f_shift", "xichmas", "g1", "g2", "f_bias"],
            paras={"f_shift": self.f_shift, "f_bias": self.f_bias},
            plain=True,
        )


class F242013(CecBenchmark):
    """
    .. [1] Liang, J. J., Qu, B. Y., Suganthan, P. N., & Hernández-Díaz, A. G. (2013). Problem definitions and evaluation criteria
    for the CEC 2013 special session on real-parameter optimization. Computational Intelligence Laboratory, Zhengzhou University,
    Zhengzhou, China and Nanyang Technological University, Singapore, Technical Report, 201212(34), 281-295..
    """

    name = "F24: Composition Function 4"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = bias = 1000.0"

    continuous = True
    linear = False
    convex = False
    unimodal = False
    separable = False

    differentiable = True
    scalable = True
    randomized_term = False
    parametric = True
    shifted = True
    rotated = True

    modality = True  # Number of ambiguous peaks, unknown # peaks
    # n_basins = 1
    # n_valleys = 1
    characteristics = ["Asymmetrical", "Different properties around different local optima"]

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "shift_data",
        f_matrix: typing.Any = "M_D",
        f_bias: float = 1000.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
    ) -> None:
        def compute(
            x: np.ndarray,
            lamdas: typing.Any,
            g0: typing.Any,
            bias: typing.Any,
            f_shift: typing.Any,
            xichmas: typing.Any,
            g1: typing.Any,
            g2: typing.Any,
            f_bias: typing.Any,
            out: np.ndarray,
        ) -> None:
            g0 = lamdas[0] * g0.evaluate(x) + bias[0]
            w0 = operator.calculate_weight(x - f_shift[0], xichmas[0])

            # g2: Rotated Rastrigin’s Function f12’
            g1 = lamdas[1] * g1.evaluate(x) + bias[1]
            w1 = operator.calculate_weight(x - f_shift[1], xichmas[1])

            # g3: Rotated Weierstrass Function f9’
            g2 = lamdas[2] * g2.evaluate(x) + bias[2]
            w2 = operator.calculate_weight(x - f_shift[2], xichmas[2])

            ws = np.array([w0, w1, w2])
            ws = ws / np.sum(ws)
            gs = np.array([g0, g1, g2])
            out[0] = operator.dot_vv(ws, gs) + f_bias

        super().__init__(parallel=parallel, fastmath=fastmath, dtype=dtype)
        self.dim_changeable = True
        self.dim_default = 30
        self.dim_max = 100
        self.dim_supported = [2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        self.check_ndim_and_bounds(
            ndim, self.dim_max, bounds, np.array([[-100.0, 100.0] for _ in range(self.dim_default)])
        )
        self.make_support_data_path("data_2013")
        self.f_shift = self.check_shift_matrix(f_shift)[:, : self.ndim]
        self.f_matrix = self.check_matrix_data(f_matrix)[:, : self.ndim]
        self.f_bias = f_bias
        self.f_global = f_bias
        self.x_global = self.f_shift[0]
        self.n_funcs = 3
        self.xichmas = [20, 20, 20]
        self.lamdas = [0.25, 1.0, 2.5]
        self.bias = [0, 100, 200]
        self.g0 = F152013(self.ndim, None, self.f_shift[0], self.f_matrix[: self.ndim, : self.ndim], f_bias=0)
        self.g1 = F122013(self.ndim, None, self.f_shift[1], self.f_matrix[: 2 * self.ndim, : self.ndim], f_bias=0)
        self.g2 = F92013(self.ndim, None, self.f_shift[2], self.f_matrix[: 2 * self.ndim, : self.ndim], f_bias=0)
        self._bind_kernel(
            compute,
            ["lamdas", "g0", "bias", "f_shift", "xichmas", "g1", "g2", "f_bias"],
            paras={"f_shift": self.f_shift, "f_bias": self.f_bias},
            plain=True,
        )


class F252013(F242013):
    """
    .. [1] Liang, J. J., Qu, B. Y., Suganthan, P. N., & Hernández-Díaz, A. G. (2013). Problem definitions and evaluation criteria
    for the CEC 2013 special session on real-parameter optimization. Computational Intelligence Laboratory, Zhengzhou University,
    Zhengzhou, China and Nanyang Technological University, Singapore, Technical Report, 201212(34), 281-295..
    """

    name = "F25: Composition Function 5"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = bias = 1100.0"

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "shift_data",
        f_matrix: typing.Any = "M_D",
        f_bias: float = 1100.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
    ) -> None:
        super().__init__(ndim, bounds, f_shift, f_matrix, f_bias, parallel=parallel, fastmath=fastmath, dtype=dtype)
        self.xichmas = [10, 30, 50]


class F262013(CecBenchmark):
    """
    .. [1] Liang, J. J., Qu, B. Y., Suganthan, P. N., & Hernández-Díaz, A. G. (2013). Problem definitions and evaluation criteria
    for the CEC 2013 special session on real-parameter optimization. Computational Intelligence Laboratory, Zhengzhou University,
    Zhengzhou, China and Nanyang Technological University, Singapore, Technical Report, 201212(34), 281-295..
    """

    name = "F26: Composition Function 6"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = bias = 1200.0"

    continuous = True
    linear = False
    convex = False
    unimodal = False
    separable = False

    differentiable = True
    scalable = True
    randomized_term = False
    parametric = True
    shifted = True
    rotated = True

    modality = True  # Number of ambiguous peaks, unknown # peaks
    # n_basins = 1
    # n_valleys = 1
    characteristics = ["Asymmetrical", "Different properties around different local optima"]

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "shift_data",
        f_matrix: typing.Any = "M_D",
        f_bias: float = 1200.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
    ) -> None:
        def compute(
            x: np.ndarray,
            lamdas: typing.Any,
            g0: typing.Any,
            bias: typing.Any,
            f_shift: typing.Any,
            xichmas: typing.Any,
            g1: typing.Any,
            g2: typing.Any,
            g3: typing.Any,
            g4: typing.Any,
            f_bias: typing.Any,
            out: np.ndarray,
        ) -> None:
            g0 = lamdas[0] * g0.evaluate(x) + bias[0]
            w0 = operator.calculate_weight(x - f_shift[0], xichmas[0])

            # g2: Rotated Rastrigin’s Function f12
            g1 = lamdas[1] * g1.evaluate(x) + bias[1]
            w1 = operator.calculate_weight(x - f_shift[1], xichmas[1])

            # g3: Rotated High Conditioned Elliptic Function f2’
            g2 = lamdas[2] * g2.evaluate(x) + bias[2]
            w2 = operator.calculate_weight(x - f_shift[2], xichmas[2])

            # g4: Rotated Weierstrass Function f9’
            g3 = lamdas[3] * g3.evaluate(x) + bias[3]
            w3 = operator.calculate_weight(x - f_shift[3], xichmas[3])

            # g5: Rotated Griewank’s Function f10’
            g4 = lamdas[4] * g4.evaluate(x) + bias[4]
            w4 = operator.calculate_weight(x - f_shift[4], xichmas[4])

            ws = np.array([w0, w1, w2, w3, w4])
            ws = ws / np.sum(ws)
            gs = np.array([g0, g1, g2, g3, g4])
            out[0] = operator.dot_vv(ws, gs) + f_bias

        super().__init__(parallel=parallel, fastmath=fastmath, dtype=dtype)
        self.dim_changeable = True
        self.dim_default = 30
        self.dim_max = 100
        self.dim_supported = [2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        self.check_ndim_and_bounds(
            ndim, self.dim_max, bounds, np.array([[-100.0, 100.0] for _ in range(self.dim_default)])
        )
        self.make_support_data_path("data_2013")
        self.f_shift = self.check_shift_matrix(f_shift)[:, : self.ndim]
        self.f_matrix = self.check_matrix_data(f_matrix)[:, : self.ndim]
        self.f_bias = f_bias
        self.f_global = f_bias
        self.x_global = self.f_shift[0]
        self.n_funcs = 5
        self.xichmas = [10, 10, 10, 10, 10]
        self.lamdas = [0.25, 1.0, 1e-7, 2.5, 10.0]
        self.bias = [0, 100, 200, 300, 400]
        self.g0 = F152013(self.ndim, None, self.f_shift[0], self.f_matrix[: self.ndim, : self.ndim], f_bias=0)
        self.g1 = F122013(self.ndim, None, self.f_shift[1], self.f_matrix[: 2 * self.ndim, : self.ndim], f_bias=0)
        self.g2 = F22013(self.ndim, None, self.f_shift[2], self.f_matrix[: self.ndim, : self.ndim], f_bias=0)
        self.g3 = F92013(self.ndim, None, self.f_shift[3], self.f_matrix[: 2 * self.ndim, : self.ndim], f_bias=0)
        self.g4 = F102013(self.ndim, None, self.f_shift[4], self.f_matrix[: self.ndim, : self.ndim], f_bias=0)
        self._bind_kernel(
            compute,
            ["lamdas", "g0", "bias", "f_shift", "xichmas", "g1", "g2", "g3", "g4", "f_bias"],
            paras={"f_shift": self.f_shift, "f_bias": self.f_bias, "f_matrix": self.f_matrix},
            plain=True,
        )


class F272013(CecBenchmark):
    """
    .. [1] Liang, J. J., Qu, B. Y., Suganthan, P. N., & Hernández-Díaz, A. G. (2013). Problem definitions and evaluation criteria
    for the CEC 2013 special session on real-parameter optimization. Computational Intelligence Laboratory, Zhengzhou University,
    Zhengzhou, China and Nanyang Technological University, Singapore, Technical Report, 201212(34), 281-295..
    """

    name = "F27: Composition Function 7"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = bias = 1300.0"

    continuous = True
    linear = False
    convex = False
    unimodal = False
    separable = False

    differentiable = True
    scalable = True
    randomized_term = False
    parametric = True
    shifted = True
    rotated = True

    modality = True  # Number of ambiguous peaks, unknown # peaks
    # n_basins = 1
    # n_valleys = 1
    characteristics = ["Asymmetrical", "Different properties around different local optima"]

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "shift_data",
        f_matrix: typing.Any = "M_D",
        f_bias: float = 1300.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
    ) -> None:
        def compute(
            x: np.ndarray,
            lamdas: typing.Any,
            g0: typing.Any,
            bias: typing.Any,
            f_shift: typing.Any,
            xichmas: typing.Any,
            g1: typing.Any,
            g2: typing.Any,
            g3: typing.Any,
            g4: typing.Any,
            f_bias: typing.Any,
            out: np.ndarray,
        ) -> None:
            g0 = lamdas[0] * g0.evaluate(x) + bias[0]
            w0 = operator.calculate_weight(x - f_shift[0], xichmas[0])

            # g2: Rotated Rastrigin’s Function f12’
            g1 = lamdas[1] * g1.evaluate(x) + bias[1]
            w1 = operator.calculate_weight(x - f_shift[1], xichmas[1])

            # g3: Rotated Schwefel's Function f15’
            g2 = lamdas[2] * g2.evaluate(x) + bias[2]
            w2 = operator.calculate_weight(x - f_shift[2], xichmas[2])

            # g4: Rotated Weierstrass Function f9’
            g3 = lamdas[3] * g3.evaluate(x) + bias[3]
            w3 = operator.calculate_weight(x - f_shift[3], xichmas[3])

            # g5: Sphere Function f1’
            g4 = lamdas[4] * g4.evaluate(x) + bias[4]
            w4 = operator.calculate_weight(x - f_shift[4], xichmas[4])

            ws = np.array([w0, w1, w2, w3, w4])
            ws = ws / np.sum(ws)
            gs = np.array([g0, g1, g2, g3, g4])
            out[0] = operator.dot_vv(ws, gs) + f_bias

        super().__init__(parallel=parallel, fastmath=fastmath, dtype=dtype)
        self.dim_changeable = True
        self.dim_default = 30
        self.dim_max = 100
        self.dim_supported = [2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        self.check_ndim_and_bounds(
            ndim, self.dim_max, bounds, np.array([[-100.0, 100.0] for _ in range(self.dim_default)])
        )
        self.make_support_data_path("data_2013")
        self.f_shift = self.check_shift_matrix(f_shift)[:, : self.ndim]
        self.f_matrix = self.check_matrix_data(f_matrix)[:, : self.ndim]
        self.f_bias = f_bias
        self.f_global = f_bias
        self.x_global = self.f_shift[0]
        self.n_funcs = 5
        self.xichmas = [10, 10, 10, 20, 20]
        self.lamdas = [100, 10, 2.5, 25, 0.1]
        self.bias = [0, 100, 200, 300, 400]
        self.g0 = F102013(self.ndim, None, self.f_shift[0], self.f_matrix[: self.ndim, : self.ndim], f_bias=0)
        self.g1 = F122013(self.ndim, None, self.f_shift[1], self.f_matrix[: 2 * self.ndim, : self.ndim], f_bias=0)
        self.g2 = F152013(self.ndim, None, self.f_shift[2], self.f_matrix[: self.ndim, : self.ndim], f_bias=0)
        self.g3 = F92013(self.ndim, None, self.f_shift[3], self.f_matrix[: 2 * self.ndim, : self.ndim], f_bias=0)
        self.g4 = F12013(self.ndim, None, self.f_shift[4], f_bias=0)
        self._bind_kernel(
            compute,
            ["lamdas", "g0", "bias", "f_shift", "xichmas", "g1", "g2", "g3", "g4", "f_bias"],
            paras={"f_shift": self.f_shift, "f_bias": self.f_bias, "f_matrix": self.f_matrix},
            plain=True,
        )


class F282013(CecBenchmark):
    """
    .. [1] Liang, J. J., Qu, B. Y., Suganthan, P. N., & Hernández-Díaz, A. G. (2013). Problem definitions and evaluation criteria
    for the CEC 2013 special session on real-parameter optimization. Computational Intelligence Laboratory, Zhengzhou University,
    Zhengzhou, China and Nanyang Technological University, Singapore, Technical Report, 201212(34), 281-295..
    """

    name = "F28: Composition Function 8"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = bias = 1400.0"

    continuous = True
    linear = False
    convex = False
    unimodal = False
    separable = False

    differentiable = True
    scalable = True
    randomized_term = False
    parametric = True
    shifted = True
    rotated = True

    modality = True  # Number of ambiguous peaks, unknown # peaks
    # n_basins = 1
    # n_valleys = 1
    characteristics = ["Asymmetrical", "Different properties around different local optima"]

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "shift_data",
        f_matrix: typing.Any = "M_D",
        f_bias: float = 1400.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
    ) -> None:
        def compute(
            x: np.ndarray,
            lamdas: typing.Any,
            g0: typing.Any,
            bias: typing.Any,
            f_shift: typing.Any,
            xichmas: typing.Any,
            g1: typing.Any,
            g2: typing.Any,
            g3: typing.Any,
            g4: typing.Any,
            f_bias: typing.Any,
            out: np.ndarray,
        ) -> None:
            g0 = lamdas[0] * g0.evaluate(x) + bias[0]
            w0 = operator.calculate_weight(x - f_shift[0], xichmas[0])

            # g2: Rotated Schaffers F7 Function f7’
            g1 = lamdas[1] * g1.evaluate(x) + bias[1]
            w1 = operator.calculate_weight(x - f_shift[1], xichmas[1])

            # g3: Rotated Schwefel's Function f15'
            g2 = lamdas[2] * g2.evaluate(x) + bias[2]
            w2 = operator.calculate_weight(x - f_shift[2], xichmas[2])

            # g4: Rotated Expanded Scaffer’s F6 Function f20’
            g3 = lamdas[3] * g3.evaluate(x) + bias[3]
            w3 = operator.calculate_weight(x - f_shift[3], xichmas[3])

            # g5: Sphere Function f1’
            g4 = lamdas[4] * g4.evaluate(x) + bias[4]
            w4 = operator.calculate_weight(x - f_shift[4], xichmas[4])

            ws = np.array([w0, w1, w2, w3, w4])
            ws = ws / np.sum(ws)
            gs = np.array([g0, g1, g2, g3, g4])
            out[0] = operator.dot_vv(ws, gs) + f_bias

        super().__init__(parallel=parallel, fastmath=fastmath, dtype=dtype)
        self.dim_changeable = True
        self.dim_default = 30
        self.dim_max = 100
        self.dim_supported = [2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        self.check_ndim_and_bounds(
            ndim, self.dim_max, bounds, np.array([[-100.0, 100.0] for _ in range(self.dim_default)])
        )
        self.make_support_data_path("data_2013")
        self.f_shift = self.check_shift_matrix(f_shift)[:, : self.ndim]
        self.f_matrix = self.check_matrix_data(f_matrix)[:, : self.ndim]
        self.f_bias = f_bias
        self.f_global = f_bias
        self.x_global = self.f_shift[0]
        self.n_funcs = 5
        self.xichmas = [10, 20, 30, 40, 50]
        self.lamdas = [2.5, 2.5e-6, 2.5, 5e-4, 0.1]
        self.bias = [0, 100, 200, 300, 400]
        self.g0 = F192013(self.ndim, None, self.f_shift[0], self.f_matrix[: self.ndim, : self.ndim], f_bias=0)
        self.g1 = F72013(
            self.ndim, None, self.f_shift[1], self.f_matrix[2 * self.ndim : 4 * self.ndim, : self.ndim], f_bias=0
        )
        self.g2 = F152013(self.ndim, None, self.f_shift[2], self.f_matrix[: self.ndim, : self.ndim], f_bias=0)
        self.g3 = F202013(self.ndim, None, self.f_shift[3], self.f_matrix[: 2 * self.ndim, : self.ndim], f_bias=0)
        self.g4 = F12013(self.ndim, None, self.f_shift[4], f_bias=0)
        self._bind_kernel(
            compute,
            ["lamdas", "g0", "bias", "f_shift", "xichmas", "g1", "g2", "g3", "g4", "f_bias"],
            paras={"f_shift": self.f_shift, "f_bias": self.f_bias, "f_matrix": self.f_matrix},
            plain=True,
        )
