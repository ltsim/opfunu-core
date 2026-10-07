#!/usr/bin/env python
# Created by "Thieu" at 06:36, 30/06/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import typing

import numpy as np

from opfunu.benchmark.cec import CecBenchmark
from opfunu.utils import operator
from opfunu.utils.numba_compat import njit


class F12005(CecBenchmark):
    """
    .. [1] Suganthan, P.N., Hansen, N., Liang, J.J., Deb, K., Chen, Y.P., Auger, A. and Tiwari, S., 2005.
    Problem definitions and evaluation criteria for the CEC 2005 special session on real-parameter optimization.
    KanGAL report, 2005005(2005), p.2005.
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

    modality = True  # Number of ambiguous peaks, unknown # peaks

    # n_basins = 1
    # n_valleys = 1

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "data_sphere",
        f_bias: float = -450.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
        shift: typing.Any = None,
        rotate: typing.Any = None,
        rotate_bounds: bool = True,
    ) -> None:
        def compute(x: np.ndarray, f_shift: typing.Any, f_bias: typing.Any, out: np.ndarray) -> None:
            out[0] = np.sum((x - f_shift) ** 2) + f_bias

        super().__init__(
            ndim=ndim,
            bounds=bounds,
            f_shift=f_shift,
            f_bias=f_bias,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            compute=compute,
            param_names=["f_shift", "f_bias"],
            shift=shift,
            rotate=rotate,
            rotate_bounds=rotate_bounds,
        )


class F22005(CecBenchmark):
    """
    .. [1] Suganthan, P.N., Hansen, N., Liang, J.J., Deb, K., Chen, Y.P., Auger, A. and Tiwari, S., 2005.
    Problem definitions and evaluation criteria for the CEC 2005 special session on real-parameter optimization.
    KanGAL report, 2005005(2005), p.2005.
    """

    name = "F2: Shifted Schwefel’s Problem 1.2"
    latex_formula = r"F_2(x) = \sum_{i=1}^D (\sum_{j=1}^i z_j)^2 + bias, z=x-o, \\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_2(x^*) = bias = -450.0"
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
    rotated = False

    modality = True  # Number of ambiguous peaks, unknown # peaks

    # n_basins = 1
    # n_valleys = 1

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "data_schwefel_102",
        f_bias: float = -450.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
        shift: typing.Any = None,
        rotate: typing.Any = None,
        rotate_bounds: bool = True,
    ) -> None:
        def compute(x: np.ndarray, f_shift: typing.Any, f_bias: typing.Any, out: np.ndarray) -> None:
            ndim = x.shape[0]
            total = 0.0
            for idx in range(0, ndim):
                partial = 0.0
                for jdx in range(0, idx):
                    partial += x[jdx] - f_shift[jdx]
                total += partial * partial
            out[0] = total + f_bias

        super().__init__(
            ndim=ndim,
            bounds=bounds,
            f_shift=f_shift,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            f_bias=f_bias,
            compute=compute,
            param_names=["f_shift", "f_bias"],
            shift=shift,
            rotate=rotate,
            rotate_bounds=rotate_bounds,
        )


class F32005(CecBenchmark):
    """
    .. [1] Suganthan, P.N., Hansen, N., Liang, J.J., Deb, K., Chen, Y.P., Auger, A. and Tiwari, S., 2005.
    Problem definitions and evaluation criteria for the CEC 2005 special session on real-parameter optimization.
    KanGAL report, 2005005(2005), p.2005.
    """

    name = "F3: Shifted Rotated High Conditioned Elliptic Function"
    latex_formula = (
        r"F_3(x) = \sum_{i=1}^D (10^6)^{\frac{i-1}{D-1}} z_i^2 + bias; \\ z=(x-o).M; x=[x_1, ..., x_D], "
        + r"\\o=[o_1, ..., o_D]: \text{the shifted global optimum}\\ M: \text{orthogonal matrix}"
    )
    latex_formula_dimension = r"D \in [10, 30, 50]"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_3(x^*) = bias = -450.0"
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
    rotated = False

    modality = True  # Number of ambiguous peaks, unknown # peaks

    # n_basins = 1
    # n_valleys = 1

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "data_high_cond_elliptic_rot",
        f_matrix: typing.Any = "elliptic_M_D",
        f_bias: float = -450.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
        shift: typing.Any = None,
        rotate: typing.Any = None,
        rotate_bounds: bool = True,
    ) -> None:
        def compute(
            x: np.ndarray, f_shift: typing.Any, f_matrix: typing.Any, f_bias: typing.Any, out: np.ndarray
        ) -> None:
            z = operator.dot_vm((x - f_shift), f_matrix)
            out[0] = operator.elliptic_func(z) + f_bias

        super().__init__(
            ndim=ndim,
            f_shift=f_shift,
            f_matrix=f_matrix,
            data_name="data_2005",
            dim_supported=[10, 30, 50],
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            f_bias=f_bias,
            compute=compute,
            param_names=["f_shift", "f_matrix", "f_bias"],
            shift=shift,
            rotate=rotate,
            rotate_bounds=rotate_bounds,
        )


class F42005(CecBenchmark):
    """
    .. [1] Suganthan, P.N., Hansen, N., Liang, J.J., Deb, K., Chen, Y.P., Auger, A. and Tiwari, S., 2005.
    Problem definitions and evaluation criteria for the CEC 2005 special session on real-parameter optimization.
    KanGAL report, 2005005(2005), p.2005.
    """

    name = "F4: Shifted Schwefel’s Problem 1.2 with Noise in Fitness"
    latex_formula = (
        r"F_4(x) = \Big(\sum_{i=1}^D (\sum_{j=1}^i)^2\Big)*\Big(1 + 0.4|N(0, 1)|\Big)+ bias;"
        + r"\\ z=(x-o).M; x=[x_1, ..., x_D], \\o=[o_1, ..., o_D]: \text{the shifted global optimum}\\ N(0,1): \text{gaussian noise}"
    )
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_4(x^*) = bias = -450.0"
    continuous = True
    linear = False
    convex = True
    unimodal = True
    separable = False

    differentiable = True
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
        f_shift: typing.Any = "data_schwefel_102",
        f_bias: float = -450.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
        shift: typing.Any = None,
        rotate: typing.Any = None,
        rotate_bounds: bool = True,
    ) -> None:
        def compute(x: np.ndarray, f_shift: typing.Any, f_bias: typing.Any, out: np.ndarray) -> None:
            ndim = len(x)
            results = [np.sum(x[:idx] - f_shift[:idx]) ** 2 for idx in range(0, ndim)]
            out[0] = np.sum(results) * (1 + 0.4 * np.abs(np.random.normal(0, 1))) + f_bias

        super().__init__(
            ndim=ndim,
            bounds=bounds,
            f_shift=f_shift,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            f_bias=f_bias,
            compute=compute,
            param_names=["f_shift", "f_bias"],
            plain=True,
            shift=shift,
            rotate=rotate,
            rotate_bounds=rotate_bounds,
        )

        # Numba-incompatible body: plain-Python kernel


class F52005(CecBenchmark):
    """
    .. [1] Suganthan, P.N., Hansen, N., Liang, J.J., Deb, K., Chen, Y.P., Auger, A. and Tiwari, S., 2005.
    Problem definitions and evaluation criteria for the CEC 2005 special session on real-parameter optimization.
    KanGAL report, 2005005(2005), p.2005.
    """

    name = "F5: Schwefel’s Problem 2.6 with Global Optimum on Bounds"
    latex_formula = (
        r"F_5(x) = max{\Big| A_ix - B_i \Big|} + bias; i=1,...,D; x=[x_1, ..., x_D];"
        + r"\\A: \text{is D*D matrix}, a_{ij}: \text{are integer random numbers in range [-500, 500]};"
        + r"\\det(A) \neq 0; A_i: \text{is the } i^{th} \text{ row of A.}"
        + r"\\B_i = A_i * o, o=[o_1, ..., o_D]: \text{the shifted global optimum}"
        + r"\\ \text{After load the data file, set } o_i=-100, \text{ for } i=1,2,...[D/4], \text{and }o_i=100 \text{ for } i=[3D/4,...,D]"
    )
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_5(x^*) = bias = -310.0"
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

    modality = True  # Number of ambiguous peaks, unknown # peaks

    # n_basins = 1
    # n_valleys = 1

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "data_schwefel_206",
        f_bias: float = -310.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
        shift: typing.Any = None,
        rotate: typing.Any = None,
        rotate_bounds: bool = True,
    ) -> None:
        def compute(
            x: np.ndarray, f_matrix: typing.Any, f_shift: typing.Any, f_bias: typing.Any, out: np.ndarray
        ) -> None:
            ndim = x.shape[0]
            best = 0.0
            for idx in range(0, ndim):
                diff = 0.0
                for jdx in range(0, ndim):
                    diff += f_matrix[idx, jdx] * (x[jdx] - f_shift[jdx])
                if diff < 0.0:
                    diff = -diff
                if diff > best:
                    best = diff
            out[0] = best + f_bias

        super().__init__(
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            dim_changeable=True,
            dim_default=30,
            dim_max=100,
            f_bias=f_bias,
            shift=shift,
            rotate=rotate,
            rotate_bounds=rotate_bounds,
        )
        self.check_ndim_and_bounds(
            ndim, self.dim_max, bounds, np.array([[-100.0, 100.0] for _ in range(self.dim_default)])
        )
        self.make_support_data_path("data_2005")
        shift_data, matrix_data = self.load_shift_and_matrix_data(f_shift)
        self.f_shift = shift_data[: self.ndim]
        self.f_matrix = matrix_data[: self.ndim, : self.ndim]
        self._x_global = np.asarray(self.f_shift)
        self.f_shift[: int(0.25 * self.ndim) + 1] = -100
        self.f_shift[int(0.75 * self.ndim) :] = 100
        self._bind_kernel(
            compute,
            ["f_matrix", "f_shift", "f_bias"],
            paras={"f_shift": self.f_shift, "f_bias": self.f_bias, "f_matrix": self.f_matrix},
        )


class F62005(CecBenchmark):
    """
    .. [1] Suganthan, P.N., Hansen, N., Liang, J.J., Deb, K., Chen, Y.P., Auger, A. and Tiwari, S., 2005.
    Problem definitions and evaluation criteria for the CEC 2005 special session on real-parameter optimization.
    KanGAL report, 2005005(2005), p.2005.
    """

    name = "F6: Shifted Rosenbrock’s Function"
    latex_formula = (
        r"F_6(x) = \sum_{i=1}^D \Big(100(z_i^2 - z_{i+1})^2 + (z_i-1)^2 \Big) + bias; z=x-o+1;"
        + "\\x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    )
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_6(x^*) = bias = 390.0"
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
        f_shift: typing.Any = "data_rosenbrock",
        f_bias: float = 390.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
        shift: typing.Any = None,
        rotate: typing.Any = None,
        rotate_bounds: bool = True,
    ) -> None:
        def compute(x: np.ndarray, f_shift: typing.Any, f_bias: typing.Any, out: np.ndarray) -> None:
            out[0] = operator.rosenbrock_func(x - f_shift, shift=1.0) + f_bias

        super().__init__(
            ndim=ndim,
            bounds=bounds,
            f_shift=f_shift,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            f_bias=f_bias,
            compute=compute,
            param_names=["f_shift", "f_bias"],
            shift=shift,
            rotate=rotate,
            rotate_bounds=rotate_bounds,
        )


class F72005(CecBenchmark):
    """
    .. [1] Suganthan, P.N., Hansen, N., Liang, J.J., Deb, K., Chen, Y.P., Auger, A. and Tiwari, S., 2005.
    Problem definitions and evaluation criteria for the CEC 2005 special session on real-parameter optimization.
    KanGAL report, 2005005(2005), p.2005.
    """

    name = "F7: Shifted Rotated Griewank’s Function without Bounds"
    latex_formula = (
        r"F_6(x) = \sum_{i=1}^D \Big(100(z_i^2 - z_{i+1})^2 + (z_i-1)^2 \Big) + bias; z=x-o+1;"
        + "\\x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    )
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_6(x^*) = bias = 390.0"
    continuous = True
    linear = False
    convex = True
    unimodal = False
    separable = False

    differentiable = True
    scalable = True
    randomized_term = True
    parametric = True
    shifted = True
    rotated = True

    modality = False  # Number of ambiguous peaks, unknown # peaks

    # n_basins = 1
    # n_valleys = 1

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "data_griewank",
        f_matrix: typing.Any = "griewank_M_D",
        f_bias: float = -180.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
        shift: typing.Any = None,
        rotate: typing.Any = None,
        rotate_bounds: bool = True,
    ) -> None:
        def compute(
            x: np.ndarray, f_shift: typing.Any, f_matrix: typing.Any, f_bias: typing.Any, out: np.ndarray
        ) -> None:
            z = operator.dot_vm((x - f_shift), f_matrix)
            out[0] = operator.griewank_func(z) + f_bias

        super().__init__(
            ndim=ndim,
            bounds=bounds,
            f_shift=f_shift,
            f_matrix=f_matrix,
            dim_supported=[10, 30, 50],
            f_bias=f_bias,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            compute=compute,
            param_names=["f_shift", "f_matrix", "f_bias"],
            shift=shift,
            rotate=rotate,
            rotate_bounds=rotate_bounds,
        )


class F82005(CecBenchmark):
    """
    .. [1] Suganthan, P.N., Hansen, N., Liang, J.J., Deb, K., Chen, Y.P., Auger, A. and Tiwari, S., 2005.
    Problem definitions and evaluation criteria for the CEC 2005 special session on real-parameter optimization.
    KanGAL report, 2005005(2005), p.2005.
    """

    name = "F8: Shifted Rotated Ackley’s Function with Global Optimum on Bounds"
    latex_formula = r"F_8(x) = -20 \exp \left( -0.2 \sqrt{\frac{1}{D} \sum_{i=1}^D z_i^2} \right) - \exp \left( \frac{1}{D} \sum_{i=1}^D \cos(2\pi z_i) \right) + 20 + e + bias, \\ z=(x-o) \times M, x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}; M: \text{rotation matrix}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-32.0, 32.0], \forall i \in [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_8(x^*) = bias = -140.0"
    continuous = True
    linear = False
    convex = False
    unimodal = False
    separable = False

    differentiable = True
    scalable = True
    randomized_term = True
    parametric = True
    shifted = True
    rotated = True

    modality = True  # Number of ambiguous peaks, unknown # peaks

    # n_basins = 1
    # n_valleys = 1

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "data_ackley",
        f_matrix: typing.Any = "ackley_M_D",
        f_bias: float = -140.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
        shift: typing.Any = None,
        rotate: typing.Any = None,
        rotate_bounds: bool = True,
    ) -> None:
        def compute(
            x: np.ndarray, f_shift: typing.Any, f_matrix: typing.Any, f_bias: typing.Any, out: np.ndarray
        ) -> None:
            z = operator.dot_vm((x - f_shift), f_matrix)
            out[0] = operator.ackley_func(z) + f_bias

        super().__init__(
            ndim=ndim,
            bounds=bounds,
            f_shift=f_shift,
            f_matrix=f_matrix,
            dim_supported=[10, 30, 50],
            f_bias=f_bias,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            shift=shift,
            rotate=rotate,
            rotate_bounds=rotate_bounds,
        )

        a = np.arange(0, self.ndim)
        self.f_shift[a % 2 == 0] = -32
        self.f_shift[a % 2 == 1] = np.random.uniform(-32.0, 32.0, int(self.ndim / 2))
        self._bind_kernel(
            compute,
            ["f_shift", "f_matrix", "f_bias"],
            paras={"f_shift": self.f_shift, "f_matrix": self.f_matrix, "f_bias": self.f_bias},
        )


class F92005(CecBenchmark):
    """
    .. [1] Suganthan, P.N., Hansen, N., Liang, J.J., Deb, K., Chen, Y.P., Auger, A. and Tiwari, S., 2005.
    Problem definitions and evaluation criteria for the CEC 2005 special session on real-parameter optimization.
    KanGAL report, 2005005(2005), p.2005.
    """

    name = "F9: Shifted Rastrigin’s Function"
    latex_formula = (
        r"F_5(x) = max{\Big| A_ix - B_i \Big|} + bias; i=1,...,D; x=[x_1, ..., x_D];"
        + r"\\A: \text{is D*D matrix}, a_{ij}: \text{are integer random numbers in range [-500, 500]};"
        + r"\\det(A) \neq 0; A_i: \text{is the } i^{th} \text{ row of A.}"
        + r"\\B_i = A_i * o, o=[o_1, ..., o_D]: \text{the shifted global optimum}"
        + r"\\ \text{After load the data file, set } o_i=-100, \text{ for } i=1,2,...[D/4], \text{and }o_i=100 \text{ for } i=[3D/4,...,D]"
    )
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_5(x^*) = bias = -310.0"
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
    rotated = False

    modality = True  # Number of ambiguous peaks, unknown # peaks

    # n_basins = 1
    # n_valleys = 1

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "data_rastrigin",
        f_bias: float = -330.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
        shift: typing.Any = None,
        rotate: typing.Any = None,
        rotate_bounds: bool = True,
    ) -> None:
        def compute(x: np.ndarray, f_shift: typing.Any, f_bias: typing.Any, out: np.ndarray) -> None:
            z = x - f_shift
            out[0] = operator.rastrigin_func(z) + f_bias

        super().__init__(
            ndim=ndim,
            bounds=bounds,
            f_shift=f_shift,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            f_bias=f_bias,
            compute=compute,
            param_names=["f_shift", "f_bias"],
            shift=shift,
            rotate=rotate,
            rotate_bounds=rotate_bounds,
        )


class F102005(CecBenchmark):
    """
    .. [1] Suganthan, P.N., Hansen, N., Liang, J.J., Deb, K., Chen, Y.P., Auger, A. and Tiwari, S., 2005.
    Problem definitions and evaluation criteria for the CEC 2005 special session on real-parameter optimization.
    KanGAL report, 2005005(2005), p.2005.
    """

    name = "F10: Shifted Rotated Rastrigin’s Function"
    latex_formula = (
        r"F_6(x) = \sum_{i=1}^D \Big(100(z_i^2 - z_{i+1})^2 + (z_i-1)^2 \Big) + bias; z=x-o+1;"
        + "\\x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    )
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_6(x^*) = bias = 390.0"
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

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "data_rastrigin",
        f_matrix: typing.Any = "rastrigin_M_D",
        f_bias: float = -330.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
        shift: typing.Any = None,
        rotate: typing.Any = None,
        rotate_bounds: bool = True,
    ) -> None:
        def compute(
            x: np.ndarray, f_shift: typing.Any, f_matrix: typing.Any, f_bias: typing.Any, out: np.ndarray
        ) -> None:
            z = operator.dot_vm((x - f_shift), f_matrix)
            out[0] = operator.rastrigin_func(z) + f_bias

        super().__init__(
            ndim=ndim,
            bounds=bounds,
            f_shift=f_shift,
            dim_supported=[10, 30, 50],
            f_matrix=f_matrix,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            f_bias=f_bias,
            compute=compute,
            param_names=["f_shift", "f_matrix", "f_bias"],
            shift=shift,
            rotate=rotate,
            rotate_bounds=rotate_bounds,
        )


class F112005(CecBenchmark):
    """
    .. [1] Suganthan, P.N., Hansen, N., Liang, J.J., Deb, K., Chen, Y.P., Auger, A. and Tiwari, S., 2005.
    Problem definitions and evaluation criteria for the CEC 2005 special session on real-parameter optimization.
    KanGAL report, 2005005(2005), p.2005.
    """

    name = "F11: Shifted Rotated Weierstrass Function"
    latex_formula = (
        r"F_6(x) = \sum_{i=1}^D \Big(100(z_i^2 - z_{i+1})^2 + (z_i-1)^2 \Big) + bias; z=x-o+1;"
        + "\\x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    )
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_6(x^*) = bias = 390.0"
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

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "data_weierstrass",
        f_matrix: typing.Any = "weierstrass_M_D",
        f_bias: float = 90.0,
        a: float = 0.5,
        b: int = 3,
        k_max: int = 20,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
        shift: typing.Any = None,
        rotate: typing.Any = None,
        rotate_bounds: bool = True,
    ) -> None:
        def compute(
            x: np.ndarray,
            f_shift: typing.Any,
            f_matrix: typing.Any,
            a: typing.Any,
            b: typing.Any,
            k_max: typing.Any,
            f_bias: typing.Any,
            out: np.ndarray,
        ) -> None:
            z = operator.dot_vm((x - f_shift), f_matrix)
            out[0] = operator.weierstrass_norm_func(z, a, b, k_max) + f_bias

        super().__init__(
            ndim=ndim,
            bounds=bounds,
            f_shift=f_shift,
            dim_supported=[10, 30, 50],
            f_matrix=f_matrix,
            f_bias=f_bias,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            shift=shift,
            rotate=rotate,
            rotate_bounds=rotate_bounds,
        )

        self.a = a
        self.b = b
        self.k_max = k_max
        self._bind_kernel(
            compute,
            ["f_shift", "f_matrix", "a", "b", "k_max", "f_bias"],
            paras={
                "f_shift": self.f_shift,
                "f_matrix": self.f_matrix,
                "f_bias": self.f_bias,
                "a": self.a,
                "b": self.b,
                "k_max": self.k_max,
            },
        )


class F122005(CecBenchmark):
    """
    .. [1] Suganthan, P.N., Hansen, N., Liang, J.J., Deb, K., Chen, Y.P., Auger, A. and Tiwari, S., 2005.
    Problem definitions and evaluation criteria for the CEC 2005 special session on real-parameter optimization.
    KanGAL report, 2005005(2005), p.2005.
    """

    name = "F12: Schwefel’s Problem 2.13"
    latex_formula = (
        r"F_6(x) = \sum_{i=1}^D \Big(100(z_i^2 - z_{i+1})^2 + (z_i-1)^2 \Big) + bias; z=x-o+1;"
        + "\\x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    )
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_6(x^*) = bias = 390.0"
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
    rotated = False

    modality = True  # Number of ambiguous peaks, unknown # peaks

    # n_basins = 1
    # n_valleys = 1

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "data_schwefel_213",
        f_bias: float = -460.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
        shift: typing.Any = None,
        rotate: typing.Any = None,
        rotate_bounds: bool = True,
    ) -> None:
        def compute(
            x: np.ndarray,
            f_matrix_a: typing.Any,
            f_shift: typing.Any,
            f_matrix_b: typing.Any,
            f_bias: typing.Any,
            out: np.ndarray,
        ) -> None:
            ndim = len(x)
            result = 0.0

            for idx in range(0, ndim):
                t1 = np.sum(f_matrix_a[idx] * np.sin(f_shift) + f_matrix_b[idx] * np.cos(f_shift))
                t2 = np.sum(f_matrix_a[idx] * np.sin(x) + f_matrix_b[idx] * np.cos(x))
                result += (t1 - t2) ** 2

            out[0] = result + f_bias

        super().__init__(
            ndim=ndim,
            bounds=bounds,
            default_bounds=[[-np.pi, np.pi]],
            f_shift=f_shift,
            load_two_matrix=True,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            f_bias=f_bias,
            compute=compute,
            param_names=["f_matrix_a", "f_shift", "f_matrix_b", "f_bias"],
            shift=shift,
            rotate=rotate,
            rotate_bounds=rotate_bounds,
        )


class F132005(CecBenchmark):
    """
    .. [1] Suganthan, P.N., Hansen, N., Liang, J.J., Deb, K., Chen, Y.P., Auger, A. and Tiwari, S., 2005.
    Problem definitions and evaluation criteria for the CEC 2005 special session on real-parameter optimization.
    KanGAL report, 2005005(2005), p.2005.
    """

    name = "F13: Shifted Expanded Griewank’s plus Rosenbrock’s Function (F8F2)"
    latex_formula = (
        r"F_5(x) = max{\Big| A_ix - B_i \Big|} + bias; i=1,...,D; x=[x_1, ..., x_D];"
        + r"\\A: \text{is D*D matrix}, a_{ij}: \text{are integer random numbers in range [-500, 500]};"
        + r"\\det(A) \neq 0; A_i: \text{is the } i^{th} \text{ row of A.}"
        + r"\\B_i = A_i * o, o=[o_1, ..., o_D]: \text{the shifted global optimum}"
        + r"\\ \text{After load the data file, set } o_i=-100, \text{ for } i=1,2,...[D/4], \text{and }o_i=100 \text{ for } i=[3D/4,...,D]"
    )
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_5(x^*) = bias = -310.0"
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
    rotated = False

    modality = True  # Number of ambiguous peaks, unknown # peaks

    # n_basins = 1
    # n_valleys = 1

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "data_EF8F2",
        f_bias: float = -130.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
        shift: typing.Any = None,
        rotate: typing.Any = None,
        rotate_bounds: bool = True,
    ) -> None:
        def compute(x: np.ndarray, f_shift: typing.Any, f_bias: typing.Any, out: np.ndarray) -> None:
            out[0] = operator.grie_rosen_cec_func(x - f_shift) + f_bias

        super().__init__(
            ndim=ndim,
            bounds=bounds,
            f_shift=f_shift,
            default_bounds=[[-3.0, 1.0]],
            f_bias=f_bias,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            shift=shift,
            rotate=rotate,
            rotate_bounds=rotate_bounds,
        )

        self.f8__ = operator.griewank_func
        self.f2__ = operator.rosenbrock_func
        self._bind_kernel(compute, ["f_shift", "f_bias"], paras={"f_shift": self.f_shift, "f_bias": self.f_bias})


class F142005(CecBenchmark):
    """
    .. [1] Suganthan, P.N., Hansen, N., Liang, J.J., Deb, K., Chen, Y.P., Auger, A. and Tiwari, S., 2005.
    Problem definitions and evaluation criteria for the CEC 2005 special session on real-parameter optimization.
    KanGAL report, 2005005(2005), p.2005.
    """

    name = "F14: Shifted Rotated Expanded Scaffer’s F6 Function"
    latex_formula = (
        r"F_6(x) = \sum_{i=1}^D \Big(100(z_i^2 - z_{i+1})^2 + (z_i-1)^2 \Big) + bias; z=x-o+1;"
        + "\\x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    )
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_6(x^*) = bias = 390.0"
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
    rotated = False

    modality = True  # Number of ambiguous peaks, unknown # peaks

    # n_basins = 1
    # n_valleys = 1

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "data_E_ScafferF6",
        f_matrix: typing.Any = "E_ScafferF6_M_D",
        f_bias: float = -300.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
        shift: typing.Any = None,
        rotate: typing.Any = None,
        rotate_bounds: bool = True,
    ) -> None:
        def compute(
            x: np.ndarray, f_shift: typing.Any, f_matrix: typing.Any, f_bias: typing.Any, out: np.ndarray
        ) -> None:
            z = operator.dot_vm((x - f_shift), f_matrix)
            out[0] = operator.expanded_schaffer_f6_func(z) + f_bias

        super().__init__(
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            dim_changeable=True,
            dim_default=30,
            dim_max=100,
            dim_supported=[10, 30, 50],
            f_bias=f_bias,
            shift=shift,
            rotate=rotate,
            rotate_bounds=rotate_bounds,
        )
        self.check_ndim_and_bounds(
            ndim, self.dim_max, bounds, np.array([[-100.0, 100.0] for _ in range(self.dim_default)])
        )
        self.make_support_data_path("data_2005")
        self.f_shift = self.check_shift_data(f_shift)[: self.ndim]
        self.f_matrix = self.check_matrix_data(f_matrix)
        self._x_global = np.asarray(self.f_shift)
        self._bind_kernel(
            compute,
            ["f_shift", "f_matrix", "f_bias"],
            paras={"f_shift": self.f_shift, "f_matrix": self.f_matrix, "f_bias": self.f_bias},
        )


@njit(fastmath=True)
def _F152005_fi__(x: np.ndarray, idx: int) -> float:
    if idx == 0 or idx == 1:
        return operator.rastrigin_func(x)
    elif idx == 2 or idx == 3:
        return operator.weierstrass_norm_func(x)
    elif idx == 4 or idx == 5:
        return operator.griewank_func(x)
    elif idx == 6 or idx == 7:
        return operator.ackley_func(x)
    else:
        return operator.sphere_func(x)


class F152005(CecBenchmark):
    """
    .. [1] Suganthan, P.N., Hansen, N., Liang, J.J., Deb, K., Chen, Y.P., Auger, A. and Tiwari, S., 2005.
    Problem definitions and evaluation criteria for the CEC 2005 special session on real-parameter optimization.
    KanGAL report, 2005005(2005), p.2005.
    """

    name = "F15: Hybrid Composition Function"
    latex_formula = (
        r"F_6(x) = \sum_{i=1}^D \Big(100(z_i^2 - z_{i+1})^2 + (z_i-1)^2 \Big) + bias; z=x-o+1;"
        + "\\x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    )
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_6(x^*) = bias = 390.0"
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
    rotated = False

    modality = True  # Number of ambiguous peaks, unknown # peaks

    # n_basins = 1
    # n_valleys = 1

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "data_hybrid_func1",
        f_bias: float = 120.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
        shift: typing.Any = None,
        rotate: typing.Any = None,
        rotate_bounds: bool = True,
    ) -> None:
        def compute(
            x: np.ndarray,
            n_funcs: int,
            f_shift: typing.Any,
            xichmas: np.ndarray,
            lamdas: np.ndarray,
            M: np.ndarray,
            y: typing.Any,
            C: int,
            bias: np.ndarray,
            f_bias: typing.Any,
            out: np.ndarray,
        ) -> None:
            ndim = len(x)
            weights = np.ones(n_funcs)
            fits = np.ones(n_funcs)
            for idx in range(0, n_funcs):
                w_i = np.exp(-np.sum((x - f_shift[idx]) ** 2) / (2 * ndim * xichmas[idx] ** 2))
                z = operator.dot_vm((x - f_shift[idx]) / lamdas[idx], M)
                fit_i = _F152005_fi__(z, idx)
                f_max_i = _F152005_fi__(operator.dot_vm((y / lamdas[idx]), M), idx)
                fit_i = C * fit_i / f_max_i
                weights[idx] = w_i
                fits[idx] = fit_i

            maxw = np.max(weights)
            weights = np.where(weights != maxw, weights * (1 - maxw**10), weights)
            weights = weights / np.sum(weights)
            out[0] = np.sum(operator.dot_vv(weights, (fits + bias))) + f_bias

        super().__init__(
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            dim_changeable=True,
            dim_default=30,
            dim_max=100,
            f_bias=f_bias,
            shift=shift,
            rotate=rotate,
            rotate_bounds=rotate_bounds,
        )
        self.check_ndim_and_bounds(ndim, self.dim_max, bounds, np.array([[-5.0, 5.0] for _ in range(self.dim_default)]))
        self.make_support_data_path("data_2005")
        self.f_shift = self.load_matrix_data(f_shift)[:, : self.ndim]  # This shift as matrix for M functions
        self.lamdas = np.array([1, 1, 10, 10, 5.0 / 60, 5.0 / 60, 5.0 / 32, 5.0 / 32, 5.0 / 100, 5.0 / 100])
        self.bias = np.array([0, 100, 200, 300, 400, 500, 600, 700, 800, 900])  # ==> f_shift[0] is the global optimum
        self.n_funcs = 10
        self.xichmas = np.ones(self.n_funcs)
        self.C = 2000
        self.M = np.identity(self.ndim)
        self.y = 5 * np.ones(self.ndim)
        self._x_global = np.asarray(self.f_shift[0])
        self._bind_kernel(
            compute,
            ["n_funcs", "f_shift", "xichmas", "lamdas", "M", "y", "C", "bias", "f_bias"],
            paras={
                "f_shift": self.f_shift,
                "f_bias": self.f_bias,
                "lambda": self.lamdas,
                "bias": self.bias,
                "n_funcs": self.n_funcs,
                "C": self.C,
                "M": self.M,
                "y": self.y,
            },
        )

    def fi__(self, x: typing.Any, idx: int) -> typing.Any:
        if idx == 0 or idx == 1:
            return operator.rastrigin_func(x)
        elif idx == 2 or idx == 3:
            return operator.weierstrass_norm_func(x)
        elif idx == 4 or idx == 5:
            return operator.griewank_func(x)
        elif idx == 6 or idx == 7:
            return operator.ackley_func(x)
        else:
            return operator.sphere_func(x)


@njit(fastmath=True)
def _F162005_fi__(x: np.ndarray, idx: int) -> float:
    if idx == 0 or idx == 1:
        return operator.rastrigin_func(x)
    elif idx == 2 or idx == 3:
        return operator.weierstrass_norm_func(x)
    elif idx == 4 or idx == 5:
        return operator.griewank_func(x)
    elif idx == 6 or idx == 7:
        return operator.ackley_func(x)
    else:
        return operator.sphere_func(x)


class F162005(CecBenchmark):
    """
    .. [1] Suganthan, P.N., Hansen, N., Liang, J.J., Deb, K., Chen, Y.P., Auger, A. and Tiwari, S., 2005.
    Problem definitions and evaluation criteria for the CEC 2005 special session on real-parameter optimization.
    KanGAL report, 2005005(2005), p.2005.
    """

    name = "F16: Rotated Version of Hybrid Composition Function F15"
    latex_formula = (
        r"F_6(x) = \sum_{i=1}^D \Big(100(z_i^2 - z_{i+1})^2 + (z_i-1)^2 \Big) + bias; z=x-o+1;"
        + "\\x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    )
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_6(x^*) = bias = 390.0"
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

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "data_hybrid_func1",
        f_matrix: typing.Any = "hybrid_func1_M_D",
        f_bias: float = 120.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
        shift: typing.Any = None,
        rotate: typing.Any = None,
        rotate_bounds: bool = True,
    ) -> None:
        def compute(
            x: np.ndarray,
            n_funcs: int,
            f_shift: typing.Any,
            xichmas: np.ndarray,
            lamdas: np.ndarray,
            M: typing.Any,
            y: typing.Any,
            C: int,
            bias: np.ndarray,
            f_bias: typing.Any,
            out: np.ndarray,
        ) -> None:
            ndim = len(x)
            weights = np.ones(n_funcs)
            fits = np.ones(n_funcs)
            for idx in range(0, n_funcs):
                w_i = np.exp(-np.sum((x - f_shift[idx]) ** 2) / (2 * ndim * xichmas[idx] ** 2))
                z = operator.dot_vm((x - f_shift[idx]) / lamdas[idx], M[idx * ndim : (idx + 1) * ndim, :])
                fit_i = _F162005_fi__(z, idx)
                f_max_i = _F162005_fi__(operator.dot_vm((y / lamdas[idx]), M[idx * ndim : (idx + 1) * ndim, :]), idx)
                fit_i = C * fit_i / f_max_i
                weights[idx] = w_i
                fits[idx] = fit_i

            maxw = np.max(weights)
            weights = np.where(weights != maxw, weights * (1 - maxw**10), weights)
            weights = weights / np.sum(weights)
            out[0] = np.sum(operator.dot_vv(weights, (fits + bias))) + f_bias

        super().__init__(
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            dim_changeable=True,
            dim_default=30,
            dim_max=100,
            dim_supported=[10, 30, 50],
            f_bias=f_bias,
            shift=shift,
            rotate=rotate,
            rotate_bounds=rotate_bounds,
        )
        self.check_ndim_and_bounds(ndim, self.dim_max, bounds, np.array([[-5.0, 5.0] for _ in range(self.dim_default)]))
        self.make_support_data_path("data_2005")
        self.f_shift = self.load_matrix_data(f_shift)[:, : self.ndim]  # This shift as matrix for M functions
        self.M = self.check_matrix_data(f_matrix)
        self.lamdas = np.array([1, 1, 10, 10, 5.0 / 60, 5.0 / 60, 5.0 / 32, 5.0 / 32, 5.0 / 100, 5.0 / 100])
        self.bias = np.array([0, 100, 200, 300, 400, 500, 600, 700, 800, 900])  # ==> f_shift[0] is the global optimum
        self.n_funcs = 10
        self.xichmas = np.ones(self.n_funcs)
        self.C = 2000
        self.y = 5 * np.ones(self.ndim)
        self._x_global = np.asarray(self.f_shift[0])
        self._bind_kernel(
            compute,
            ["n_funcs", "f_shift", "xichmas", "lamdas", "M", "y", "C", "bias", "f_bias"],
            paras={
                "f_shift": self.f_shift,
                "f_bias": self.f_bias,
                "lambda": self.lamdas,
                "bias": self.bias,
                "n_funcs": self.n_funcs,
                "C": self.C,
                "M": self.M,
                "y": self.y,
            },
        )

    def fi__(self, x: typing.Any, idx: int) -> typing.Any:
        if idx == 0 or idx == 1:
            return operator.rastrigin_func(x)
        elif idx == 2 or idx == 3:
            return operator.weierstrass_norm_func(x)
        elif idx == 4 or idx == 5:
            return operator.griewank_func(x)
        elif idx == 6 or idx == 7:
            return operator.ackley_func(x)
        else:
            return operator.sphere_func(x)


class F172005(F162005):
    """
    .. [1] Suganthan, P.N., Hansen, N., Liang, J.J., Deb, K., Chen, Y.P., Auger, A. and Tiwari, S., 2005.
    Problem definitions and evaluation criteria for the CEC 2005 special session on real-parameter optimization.
    KanGAL report, 2005005(2005), p.2005.
    """

    name = "F17: F16 with Noise in Fitness"
    latex_formula = (
        r"F_6(x) = \sum_{i=1}^D \Big(100(z_i^2 - z_{i+1})^2 + (z_i-1)^2 \Big) + bias; z=x-o+1;"
        + "\\x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    )
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_6(x^*) = bias = 390.0"
    randomized_term = True

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "data_hybrid_func1",
        f_matrix: typing.Any = "hybrid_func1_M_D",
        f_bias: float = 120.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
        shift: typing.Any = None,
        rotate: typing.Any = None,
        rotate_bounds: bool = True,
    ) -> None:
        def compute(
            x: np.ndarray,
            n_funcs: typing.Any,
            f_shift: typing.Any,
            xichmas: typing.Any,
            lamdas: typing.Any,
            M: typing.Any,
            fi__: typing.Any,
            y: typing.Any,
            C: typing.Any,
            bias: typing.Any,
            f_bias: typing.Any,
            out: np.ndarray,
        ) -> None:
            ndim = len(x)
            weights = np.ones(n_funcs)
            fits = np.ones(n_funcs)
            for idx in range(0, n_funcs):
                w_i = np.exp(-np.sum((x - f_shift[idx]) ** 2) / (2 * ndim * xichmas[idx] ** 2))
                z = operator.dot_vm((x - f_shift[idx]) / lamdas[idx], M[idx * ndim : (idx + 1) * ndim, :])
                fit_i = fi__(z, idx)
                f_max_i = fi__(operator.dot_vm((y / lamdas[idx]), M[idx * ndim : (idx + 1) * ndim, :]), idx)
                fit_i = C * fit_i / f_max_i
                weights[idx] = w_i
                fits[idx] = fit_i

            maxw = np.max(weights)
            weights = np.where(weights != maxw, weights * (1 - maxw**10), weights)
            weights = weights / np.sum(weights)
            out[0] = (
                np.sum(operator.dot_vv(weights, (fits + bias))) * (1 + 0.2 * np.abs(np.random.normal(0, 1))) + f_bias
            )

        super().__init__(
            ndim,
            bounds,
            f_shift,
            f_matrix,
            f_bias,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            shift=shift,
            rotate=rotate,
            rotate_bounds=rotate_bounds,
        )
        # Numba-incompatible body: plain-Python kernel
        self._bind_kernel(
            compute, ["n_funcs", "f_shift", "xichmas", "lamdas", "M", "fi__", "y", "C", "bias", "f_bias"], plain=True
        )


@njit(fastmath=True)
def _F182005_fi__(x: np.ndarray, idx: int) -> float:
    if idx == 0 or idx == 1:
        return operator.ackley_func(x)
    elif idx == 2 or idx == 3:
        return operator.rastrigin_func(x)
    elif idx == 4 or idx == 5:
        return operator.sphere_func(x)
    elif idx == 6 or idx == 7:
        return operator.weierstrass_norm_func(x)
    else:
        return operator.griewank_func(x)


class F182005(CecBenchmark):
    """
    .. [1] Suganthan, P.N., Hansen, N., Liang, J.J., Deb, K., Chen, Y.P., Auger, A. and Tiwari, S., 2005.
    Problem definitions and evaluation criteria for the CEC 2005 special session on real-parameter optimization.
    KanGAL report, 2005005(2005), p.2005.
    """

    name = "F18: Rotated Hybrid Composition Function"
    latex_formula = (
        r"F_6(x) = \sum_{i=1}^D \Big(100(z_i^2 - z_{i+1})^2 + (z_i-1)^2 \Big) + bias; z=x-o+1;"
        + "\\x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    )
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_6(x^*) = bias = 390.0"
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

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "data_hybrid_func2",
        f_matrix: typing.Any = "hybrid_func2_M_D",
        f_bias: float = 10.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
        shift: typing.Any = None,
        rotate: typing.Any = None,
        rotate_bounds: bool = True,
    ) -> None:
        def compute(
            x: np.ndarray,
            n_funcs: int,
            f_shift: typing.Any,
            xichmas: np.ndarray,
            lamdas: np.ndarray,
            M: typing.Any,
            y: typing.Any,
            C: int,
            bias: np.ndarray,
            f_bias: typing.Any,
            out: np.ndarray,
        ) -> None:
            ndim = len(x)
            weights = np.ones(n_funcs)
            fits = np.ones(n_funcs)
            for idx in range(0, n_funcs):
                w_i = np.exp(-np.sum((x - f_shift[idx]) ** 2) / (2 * ndim * xichmas[idx] ** 2))
                z = operator.dot_vm((x - f_shift[idx]) / lamdas[idx], M[idx * ndim : (idx + 1) * ndim, :])
                fit_i = _F182005_fi__(z, idx)
                f_max_i = _F182005_fi__(operator.dot_vm((y / lamdas[idx]), M[idx * ndim : (idx + 1) * ndim, :]), idx)
                fit_i = C * fit_i / f_max_i
                weights[idx] = w_i
                fits[idx] = fit_i

            maxw = np.max(weights)
            weights = np.where(weights != maxw, weights * (1 - maxw**10), weights)
            weights = weights / np.sum(weights)
            out[0] = np.sum(operator.dot_vv(weights, (fits + bias))) + f_bias

        super().__init__(
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            dim_changeable=True,
            dim_default=30,
            dim_max=100,
            dim_supported=[10, 30, 50],
            f_bias=f_bias,
            shift=shift,
            rotate=rotate,
            rotate_bounds=rotate_bounds,
        )
        self.check_ndim_and_bounds(ndim, self.dim_max, bounds, np.array([[-5.0, 5.0] for _ in range(self.dim_default)]))
        self.make_support_data_path("data_2005")
        self.f_shift = self.load_matrix_data(f_shift)[:, : self.ndim]  # This shift as matrix for M functions
        self.M = self.check_matrix_data(f_matrix)
        self.lamdas = np.array(
            [2 * 5.0 / 32, 5.0 / 32, 2 * 1, 1, 2 * 5.0 / 100, 5.0 / 100, 2 * 10, 10, 2 * 5.0 / 60, 5.0 / 60]
        )
        self.bias = np.array([0, 100, 200, 300, 400, 500, 600, 700, 800, 900])  # ==> f_shift[0] is the global optimum
        self.n_funcs = 10
        self.xichmas = np.array([1, 2, 1.5, 1.5, 1, 1, 1.5, 1.5, 2, 2])
        self.C = 2000
        self.y = 5 * np.ones(self.ndim)
        self._x_global = np.asarray(self.f_shift[0])
        self._bind_kernel(
            compute,
            ["n_funcs", "f_shift", "xichmas", "lamdas", "M", "y", "C", "bias", "f_bias"],
            paras={
                "f_shift": self.f_shift,
                "f_bias": self.f_bias,
                "lambda": self.lamdas,
                "bias": self.bias,
                "n_funcs": self.n_funcs,
                "C": self.C,
                "M": self.M,
                "y": self.y,
            },
        )

    def fi__(self, x: typing.Any, idx: int) -> typing.Any:
        if idx == 0 or idx == 1:
            return operator.ackley_func(x)
        elif idx == 2 or idx == 3:
            return operator.rastrigin_func(x)
        elif idx == 4 or idx == 5:
            return operator.sphere_func(x)
        elif idx == 6 or idx == 7:
            return operator.weierstrass_norm_func(x)
        else:
            return operator.griewank_func(x)


class F192005(F182005):
    """
    .. [1] Suganthan, P.N., Hansen, N., Liang, J.J., Deb, K., Chen, Y.P., Auger, A. and Tiwari, S., 2005.
    Problem definitions and evaluation criteria for the CEC 2005 special session on real-parameter optimization.
    KanGAL report, 2005005(2005), p.2005.
    """

    name = "F19: Rotated Hybrid Composition Function with narrow basin global optimum"
    latex_formula = (
        r"F_6(x) = \sum_{i=1}^D \Big(100(z_i^2 - z_{i+1})^2 + (z_i-1)^2 \Big) + bias; z=x-o+1;"
        + "\\x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    )
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_6(x^*) = bias = 390.0"

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "data_hybrid_func2",
        f_matrix: typing.Any = "hybrid_func2_M_D",
        f_bias: float = 10.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
        shift: typing.Any = None,
        rotate: typing.Any = None,
        rotate_bounds: bool = True,
    ) -> None:
        super().__init__(
            ndim,
            bounds,
            f_shift,
            f_matrix,
            f_bias,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            shift=shift,
            rotate=rotate,
            rotate_bounds=rotate_bounds,
        )
        self.lamdas = np.array(
            [0.1 * 5 / 32, 5.0 / 32, 2 * 1, 1, 2 * 5.0 / 100, 5.0 / 100, 2.0 * 10, 10, 2 * 5.0 / 60, 5.0 / 60]
        )
        self.xichmas = np.array([0.1, 2, 1.5, 1.5, 1, 1, 1.5, 1.5, 2, 2])
        self._paras = {
            "f_shift": self.f_shift,
            "f_bias": self.f_bias,
            "lambda": self.lamdas,
            "bias": self.bias,
            "n_funcs": self.n_funcs,
            "C": self.C,
            "M": self.M,
            "y": self.y,
        }


class F202005(F182005):
    """
    .. [1] Suganthan, P.N., Hansen, N., Liang, J.J., Deb, K., Chen, Y.P., Auger, A. and Tiwari, S., 2005.
    Problem definitions and evaluation criteria for the CEC 2005 special session on real-parameter optimization.
    KanGAL report, 2005005(2005), p.2005.
    """

    name = "F20: Rotated Hybrid Composition Function with Global Optimum on the Bounds"
    latex_formula = (
        r"F_6(x) = \sum_{i=1}^D \Big(100(z_i^2 - z_{i+1})^2 + (z_i-1)^2 \Big) + bias; z=x-o+1;"
        + "\\x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    )
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_6(x^*) = bias = 390.0"

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "data_hybrid_func2",
        f_matrix: typing.Any = "hybrid_func2_M_D",
        f_bias: float = 10.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
        shift: typing.Any = None,
        rotate: typing.Any = None,
        rotate_bounds: bool = True,
    ) -> None:
        super().__init__(
            ndim,
            bounds,
            f_shift,
            f_matrix,
            f_bias,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            shift=shift,
            rotate=rotate,
            rotate_bounds=rotate_bounds,
        )
        self.f_shift[0, 1::2] = 5
        self._x_global = np.asarray(self.f_shift[0])
        self._paras = {
            "f_shift": self.f_shift,
            "f_bias": self.f_bias,
            "lambda": self.lamdas,
            "bias": self.bias,
            "n_funcs": self.n_funcs,
            "C": self.C,
            "M": self.M,
            "y": self.y,
        }


@njit(fastmath=True)
def _F212005_fi__(x: np.ndarray, idx: int) -> float:
    if idx == 0 or idx == 1:
        return operator.rotated_expanded_schaffer_func(x)
    elif idx == 2 or idx == 3:
        return operator.rastrigin_func(x)
    elif idx == 4 or idx == 5:
        return operator.grie_rosen_cec_func(x)
    elif idx == 6 or idx == 7:
        return operator.weierstrass_norm_func(x)
    else:
        return operator.griewank_func(x)


class F212005(CecBenchmark):
    """
    .. [1] Suganthan, P.N., Hansen, N., Liang, J.J., Deb, K., Chen, Y.P., Auger, A. and Tiwari, S., 2005.
    Problem definitions and evaluation criteria for the CEC 2005 special session on real-parameter optimization.
    KanGAL report, 2005005(2005), p.2005.
    """

    name = "F21: Rotated Hybrid Composition Function"
    latex_formula = (
        r"F_6(x) = \sum_{i=1}^D \Big(100(z_i^2 - z_{i+1})^2 + (z_i-1)^2 \Big) + bias; z=x-o+1;"
        + "\\x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    )
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_6(x^*) = bias = 390.0"
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

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "data_hybrid_func3",
        f_matrix: typing.Any = "hybrid_func3_M_D",
        f_bias: float = 360.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
        shift: typing.Any = None,
        rotate: typing.Any = None,
        rotate_bounds: bool = True,
    ) -> None:
        def compute(
            x: np.ndarray,
            n_funcs: int,
            f_shift: typing.Any,
            xichmas: np.ndarray,
            lamdas: np.ndarray,
            M: typing.Any,
            y: typing.Any,
            C: int,
            bias: np.ndarray,
            f_bias: typing.Any,
            out: np.ndarray,
        ) -> None:
            ndim = len(x)
            weights = np.ones(n_funcs)
            fits = np.ones(n_funcs)

            for idx in range(0, n_funcs):
                w_i = np.exp(-np.sum((x - f_shift[idx]) ** 2) / (2 * ndim * xichmas[idx] ** 2))
                z = operator.dot_vm((x - f_shift[idx]) / lamdas[idx], M[idx * ndim : (idx + 1) * ndim, :])
                fit_i = _F212005_fi__(z, idx)
                f_max_i = _F212005_fi__(operator.dot_vm((y / lamdas[idx]), M[idx * ndim : (idx + 1) * ndim, :]), idx)
                fit_i = C * fit_i / f_max_i
                weights[idx] = w_i
                fits[idx] = fit_i

            maxw = np.max(weights)
            weights = np.where(weights != maxw, weights * (1 - maxw**10), weights)
            weights = weights / np.sum(weights)
            out[0] = np.sum(operator.dot_vv(weights, (fits + bias))) + f_bias

        super().__init__(
            ndim=ndim,
            bounds=bounds,
            default_bounds=[[-5.0, 5.0]],
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            shift=shift,
            rotate=rotate,
            rotate_bounds=rotate_bounds,
        )
        self.f_shift = self.load_matrix_data(f_shift)[:, : self.ndim]
        self._x_global = np.asarray(self.f_shift[0])
        self.M = self.check_matrix_data(f_matrix)
        self.lamdas = np.array(
            [
                5.0 * 5.0 / 100.0,
                5.0 / 100.0,
                5.0 * 1.0,
                1.0,
                5.0 * 1.0,
                1.0,
                5.0 * 10.0,
                10.0,
                5.0 * 5.0 / 200.0,
                5.0 / 200.0,
            ]
        )
        self.bias = np.array([0, 100, 200, 300, 400, 500, 600, 700, 800, 900])  # ==> f_shift[0] is the global optimum
        self.n_funcs = 10
        self.xichmas = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0])
        self.C = 2000
        self.y = 5 * np.ones(self.ndim)
        self._x_global = np.asarray(self.f_shift[0])
        self._bind_kernel(
            compute,
            ["n_funcs", "f_shift", "xichmas", "lamdas", "M", "y", "C", "bias", "f_bias"],
            paras={
                "f_shift": self.f_shift,
                "f_bias": self.f_bias,
                "lambda": self.lamdas,
                "bias": self.bias,
                "n_funcs": self.n_funcs,
                "C": self.C,
                "M": self.M,
                "y": self.y,
            },
        )

    def fi__(self, x: typing.Any, idx: int) -> typing.Any:
        if idx == 0 or idx == 1:
            return operator.rotated_expanded_schaffer_func(x)
        elif idx == 2 or idx == 3:
            return operator.rastrigin_func(x)
        elif idx == 4 or idx == 5:
            return operator.grie_rosen_cec_func(x)
        elif idx == 6 or idx == 7:
            return operator.weierstrass_norm_func(x)
        else:
            return operator.griewank_func(x)


class F222005(F212005):
    """
    .. [1] Suganthan, P.N., Hansen, N., Liang, J.J., Deb, K., Chen, Y.P., Auger, A. and Tiwari, S., 2005.
    Problem definitions and evaluation criteria for the CEC 2005 special session on real-parameter optimization.
    KanGAL report, 2005005(2005), p.2005.
    """

    name = "F22: Rotated Hybrid Composition Function with High Condition Number Matrix"
    latex_formula = (
        r"F_6(x) = \sum_{i=1}^D \Big(100(z_i^2 - z_{i+1})^2 + (z_i-1)^2 \Big) + bias; z=x-o+1;"
        + "\\x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    )
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_6(x^*) = bias = 390.0"

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "data_hybrid_func3",
        f_matrix: typing.Any = "hybrid_func3_HM_D",
        f_bias: float = 360.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
        shift: typing.Any = None,
        rotate: typing.Any = None,
        rotate_bounds: bool = True,
    ) -> None:
        super().__init__(
            ndim,
            bounds,
            f_shift,
            f_matrix,
            f_bias,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            shift=shift,
            rotate=rotate,
            rotate_bounds=rotate_bounds,
        )


class F232005(F212005):
    """
    .. [1] Suganthan, P.N., Hansen, N., Liang, J.J., Deb, K., Chen, Y.P., Auger, A. and Tiwari, S., 2005.
    Problem definitions and evaluation criteria for the CEC 2005 special session on real-parameter optimization.
    KanGAL report, 2005005(2005), p.2005.
    """

    name = "F21: Rotated Hybrid Composition Function"
    latex_formula = (
        r"F_6(x) = \sum_{i=1}^D \Big(100(z_i^2 - z_{i+1})^2 + (z_i-1)^2 \Big) + bias; z=x-o+1;"
        + "\\x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    )
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_6(x^*) = bias = 390.0"
    continuous = False
    differentiable = False

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "data_hybrid_func3",
        f_matrix: typing.Any = "hybrid_func3_M_D",
        f_bias: float = 360.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
        shift: typing.Any = None,
        rotate: typing.Any = None,
        rotate_bounds: bool = True,
    ) -> None:
        def compute(
            x: np.ndarray,
            f_shift: typing.Any,
            n_funcs: typing.Any,
            xichmas: typing.Any,
            lamdas: typing.Any,
            M: typing.Any,
            fi__: typing.Any,
            y: typing.Any,
            C: typing.Any,
            bias: typing.Any,
            f_bias: typing.Any,
            out: np.ndarray,
        ) -> None:
            ndim = len(x)
            x = operator.rounder(x, np.abs(x - f_shift[0]))
            weights = np.ones(n_funcs)
            fits = np.ones(n_funcs)
            for idx in range(0, n_funcs):
                w_i = np.exp(-np.sum((x - f_shift[idx]) ** 2) / (2 * ndim * xichmas[idx] ** 2))
                z = operator.dot_vm((x - f_shift[idx]) / lamdas[idx], M[idx * ndim : (idx + 1) * ndim, :])
                fit_i = fi__(z, idx)
                f_max_i = fi__(operator.dot_vm((y / lamdas[idx]), M[idx * ndim : (idx + 1) * ndim, :]), idx)
                fit_i = C * fit_i / f_max_i
                weights[idx] = w_i
                fits[idx] = fit_i

            maxw = np.max(weights)
            weights = np.where(weights != maxw, weights * (1 - maxw**10), weights)
            weights = weights / np.sum(weights)
            out[0] = np.sum(operator.dot_vv(weights, (fits + bias))) + f_bias

        super().__init__(
            ndim,
            bounds,
            f_shift,
            f_matrix,
            f_bias,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            shift=shift,
            rotate=rotate,
            rotate_bounds=rotate_bounds,
        )
        self._bind_kernel(
            compute, ["f_shift", "n_funcs", "xichmas", "lamdas", "M", "fi__", "y", "C", "bias", "f_bias"], plain=True
        )


@njit(fastmath=True)
def _F242005_fi__(x: np.ndarray, idx: int) -> float:
    if idx == 0:
        return operator.weierstrass_norm_func(x)
    elif idx == 1:
        return operator.rotated_expanded_schaffer_func(x)
    elif idx == 2:
        return operator.grie_rosen_cec_func(x)
    elif idx == 3:
        return operator.ackley_func(x)
    elif idx == 4:
        return operator.rastrigin_func(x)
    elif idx == 5:
        return operator.griewank_func(x)
    elif idx == 6:
        return operator.non_continuous_expanded_scaffer_func(x)
    elif idx == 7:
        return operator.non_continuous_rastrigin_func(x)
    elif idx == 8:
        return operator.elliptic_func(x)
    else:
        return operator.sphere_func(x)


class F242005(CecBenchmark):
    """
    .. [1] Suganthan, P.N., Hansen, N., Liang, J.J., Deb, K., Chen, Y.P., Auger, A. and Tiwari, S., 2005.
    Problem definitions and evaluation criteria for the CEC 2005 special session on real-parameter optimization.
    KanGAL report, 2005005(2005), p.2005.
    """

    name = "F24: Rotated Hybrid Composition Function"
    latex_formula = (
        r"F_6(x) = \sum_{i=1}^D \Big(100(z_i^2 - z_{i+1})^2 + (z_i-1)^2 \Big) + bias; z=x-o+1;"
        + "\\x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    )
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_6(x^*) = bias = 390.0"
    continuous = False
    linear = False
    convex = False
    unimodal = False
    separable = False

    differentiable = False
    scalable = True
    randomized_term = True
    parametric = True
    shifted = True
    rotated = True

    modality = True  # Number of ambiguous peaks, unknown # peaks

    # n_basins = 1
    # n_valleys = 1

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "data_hybrid_func4",
        f_matrix: typing.Any = "hybrid_func4_M_D",
        f_bias: float = 260.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
        shift: typing.Any = None,
        rotate: typing.Any = None,
        rotate_bounds: bool = True,
    ) -> None:
        def compute(
            x: np.ndarray,
            n_funcs: int,
            f_shift: typing.Any,
            xichmas: typing.Any,
            lamdas: np.ndarray,
            M: typing.Any,
            y: typing.Any,
            C: int,
            bias: np.ndarray,
            f_bias: typing.Any,
            out: np.ndarray,
        ) -> None:
            ndim = len(x)
            weights = np.ones(n_funcs)
            fits = np.ones(n_funcs)
            for idx in range(0, n_funcs):
                w_i = np.exp(-np.sum((x - f_shift[idx]) ** 2) / (2 * ndim * xichmas[idx] ** 2))
                z = operator.dot_vm((x - f_shift[idx]) / lamdas[idx], M[idx * ndim : (idx + 1) * ndim, :])
                fit_i = _F242005_fi__(z, idx)
                f_max_i = _F242005_fi__(operator.dot_vm((y / lamdas[idx]), M[idx * ndim : (idx + 1) * ndim, :]), idx)
                fit_i = C * fit_i / f_max_i
                weights[idx] = w_i
                fits[idx] = fit_i

            maxw = np.max(weights)
            weights = np.where(weights != maxw, weights * (1 - maxw**10), weights)
            weights = weights / np.sum(weights)
            out[0] = np.sum(operator.dot_vv(weights, (fits + bias))) + f_bias

        super().__init__(
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            dim_changeable=True,
            dim_default=30,
            dim_max=100,
            dim_supported=[10, 30, 50],
            f_bias=f_bias,
            shift=shift,
            rotate=rotate,
            rotate_bounds=rotate_bounds,
        )
        self.check_ndim_and_bounds(ndim, self.dim_max, bounds, np.array([[-5.0, 5.0] for _ in range(self.dim_default)]))
        self.make_support_data_path("data_2005")
        self.f_shift = self.load_matrix_data(f_shift)[:, : self.ndim]  # This shift as matrix for M functions
        self.M = self.check_matrix_data(f_matrix)
        self.lamdas = np.array(
            [10.0, 5.0 / 20.0, 1.0, 5.0 / 32.0, 1.0, 5.0 / 100.0, 5.0 / 50.0, 1.0, 5.0 / 100.0, 5.0 / 100.0]
        )
        self.bias = np.array([0, 100, 200, 300, 400, 500, 600, 700, 800, 900])  # ==> f_shift[0] is the global optimum
        self.n_funcs = 10
        self.xichmas = 2 * np.ones(self.ndim)
        self.C = 2000
        self.y = 5 * np.ones(self.ndim)
        self._x_global = np.asarray(self.f_shift[0])
        self._bind_kernel(
            compute,
            ["n_funcs", "f_shift", "xichmas", "lamdas", "M", "y", "C", "bias", "f_bias"],
            paras={
                "f_shift": self.f_shift,
                "f_bias": self.f_bias,
                "lambda": self.lamdas,
                "bias": self.bias,
                "n_funcs": self.n_funcs,
                "C": self.C,
                "M": self.M,
                "y": self.y,
            },
        )

    def fi__(self, x: typing.Any, idx: int) -> typing.Any:
        if idx == 0:
            return operator.weierstrass_norm_func(x)
        elif idx == 1:
            return operator.rotated_expanded_schaffer_func(x)
        elif idx == 2:
            return operator.grie_rosen_cec_func(x)
        elif idx == 3:
            return operator.ackley_func(x)
        elif idx == 4:
            return operator.rastrigin_func(x)
        elif idx == 5:
            return operator.griewank_func(x)
        elif idx == 6:
            return operator.non_continuous_expanded_scaffer_func(x)
        elif idx == 7:
            return operator.non_continuous_rastrigin_func(x)
        elif idx == 8:
            return operator.elliptic_func(x)
        else:
            return operator.sphere_func(x)


class F252005(F242005):
    """
    .. [1] Suganthan, P.N., Hansen, N., Liang, J.J., Deb, K., Chen, Y.P., Auger, A. and Tiwari, S., 2005.
    Problem definitions and evaluation criteria for the CEC 2005 special session on real-parameter optimization.
    KanGAL report, 2005005(2005), p.2005.
    """

    name = "F25: Rotated Hybrid Composition Function without bounds"
    latex_formula = (
        r"F_6(x) = \sum_{i=1}^D \Big(100(z_i^2 - z_{i+1})^2 + (z_i-1)^2 \Big) + bias; z=x-o+1;"
        + "\\x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    )
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_6(x^*) = bias = 390.0"

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "data_hybrid_func4",
        f_matrix: typing.Any = "hybrid_func4_M_D",
        f_bias: float = 260.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
        shift: typing.Any = None,
        rotate: typing.Any = None,
        rotate_bounds: bool = True,
    ) -> None:
        super().__init__(
            ndim,
            bounds,
            f_shift,
            f_matrix,
            f_bias,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            shift=shift,
            rotate=rotate,
            rotate_bounds=rotate_bounds,
        )
        self.check_ndim_and_bounds(ndim, self.dim_max, bounds, np.array([[2.0, 5.0] for _ in range(self.dim_default)]))
