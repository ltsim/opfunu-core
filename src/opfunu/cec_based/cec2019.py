#!/usr/bin/env python
# Created by "Thieu" at 16:32, 12/07/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import typing

import numpy as np

from opfunu.benchmark.cec import CecBenchmark
from opfunu.utils import operator


class F12019(CecBenchmark):
    """
    .. [1] The 100-Digit Challenge: Problem Definitions and Evaluation Criteria for the 100-Digit
    Challenge Special Session and Competition on Single Objective Numerical Optimization
    """

    name = "F1: Storn’s Chebyshev Polynomial Fitting Problem"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = bias = 1.0"

    continuous = False
    linear = False
    convex = False
    unimodal = False
    separable = False

    differentiable = False
    scalable = False
    randomized_term = False
    parametric = True
    shifted = True
    rotated = True

    modality = True  # Number of ambiguous peaks, unknown # peaks
    # n_basins = 1
    # n_valleys = 1

    characteristics = ["Multimodal with one global minimum", "Very highly conditioned", "fully parameter-dependent"]

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "shift_data_1",
        f_bias: float = 1.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
        shift: typing.Any = None,
        rotate: typing.Any = None,
        rotate_bounds: bool = True,
    ) -> None:
        def compute(x: np.ndarray, f_bias: typing.Any, out: np.ndarray) -> None:
            out[0] = operator.chebyshev_func(x) + f_bias

        super().__init__(
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            dim_changeable=False,
            dim_default=9,
            dim_max=9,
            f_bias=f_bias,
            shift=shift,
            rotate=rotate,
            rotate_bounds=rotate_bounds,
        )
        self.check_ndim_and_bounds(
            ndim, self.dim_max, bounds, np.array([[-8192.0, 8192.0] for _ in range(self.dim_default)])
        )
        self.make_support_data_path("data_2019")
        self.f_shift = self.check_shift_data(f_shift)[: self.ndim]
        self._x_global = np.asarray(np.zeros(self.ndim))
        self._bind_kernel(compute, ["f_bias"], paras={"f_shift": self.f_shift, "f_bias": self.f_bias})


class F22019(CecBenchmark):
    """
    .. [1] The 100-Digit Challenge: Problem Definitions and Evaluation Criteria for the 100-Digit
    Challenge Special Session and Competition on Single Objective Numerical Optimization
    """

    name = "F2: Inverse Hilbert Matrix Problem"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = bias = 1.0"

    continuous = False
    linear = False
    convex = False
    unimodal = False
    separable = False

    differentiable = False
    scalable = False
    randomized_term = False
    parametric = True
    shifted = True
    rotated = True

    modality = True  # Number of ambiguous peaks, unknown # peaks
    # n_basins = 1
    # n_valleys = 1

    characteristics = ["Multimodal with one global minimum", "Very highly conditioned", "fully parameter-dependent"]

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "shift_data_2",
        f_bias: float = 1.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
        shift: typing.Any = None,
        rotate: typing.Any = None,
        rotate_bounds: bool = True,
    ) -> None:
        def compute(x: np.ndarray, f_bias: typing.Any, out: np.ndarray) -> None:
            out[0] = operator.inverse_hilbert_func(x) + f_bias

        super().__init__(
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            dim_changeable=False,
            dim_default=16,
            dim_max=16,
            shift=shift,
            rotate=rotate,
            rotate_bounds=rotate_bounds,
        )
        self.check_ndim_and_bounds(
            ndim, self.dim_max, bounds, np.array([[-16384.0, 16384.0] for _ in range(self.dim_default)])
        )
        self.make_support_data_path("data_2019")
        self.f_shift = self.check_shift_data(f_shift)[: self.ndim]
        self.f_bias = f_bias
        # the f_global and x_global was obtained by executing the cec2019 c code
        self._f_global = float(int(np.sqrt(self.ndim)) + f_bias)
        self._x_global = np.asarray(np.zeros(self.ndim))
        self._bind_kernel(compute, ["f_bias"], paras={"f_shift": self.f_shift, "f_bias": self.f_bias})


class F32019(CecBenchmark):
    """
    .. [1] The 100-Digit Challenge: Problem Definitions and Evaluation Criteria for the 100-Digit
    Challenge Special Session and Competition on Single Objective Numerical Optimization

    **Note: The CEC 2019 implementation and this implementation results match when x* = [0,...,0] and
    """

    name = "F3: Lennard-Jones Minimum Energy Cluster Problem"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = bias = 1.0"

    continuous = False
    linear = False
    convex = False
    unimodal = False
    separable = False

    differentiable = False
    scalable = False
    randomized_term = False
    parametric = True
    shifted = True
    rotated = True

    modality = True  # Number of ambiguous peaks, unknown # peaks
    # n_basins = 1
    # n_valleys = 1

    characteristics = ["Multimodal with one global minimum", "Very highly conditioned", "fully parameter-dependent"]

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "shift_data_3",
        f_bias: float = 1.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
        shift: typing.Any = None,
        rotate: typing.Any = None,
        rotate_bounds: bool = True,
    ) -> None:
        def compute(x: np.ndarray, f_bias: typing.Any, out: np.ndarray) -> None:
            out[0] = operator.lennard_jones_func(x) + f_bias

        super().__init__(
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            dim_changeable=False,
            dim_default=18,
            dim_max=18,
            shift=shift,
            rotate=rotate,
            rotate_bounds=rotate_bounds,
        )
        self.check_ndim_and_bounds(ndim, self.dim_max, bounds, np.array([[-4.0, 4.0] for _ in range(self.dim_default)]))
        self.make_support_data_path("data_2019")
        self.f_shift = self.check_shift_data(f_shift)[: self.ndim]
        self.f_bias = f_bias
        # f_global calculated by verifying the cec2019 C value for f(x*) where x*==f_shift
        self._f_global = float(12.712062001703194 + self.f_bias)
        self._x_global = np.asarray(self.f_shift)
        self._bind_kernel(compute, ["f_bias"], paras={"f_shift": self.f_shift, "f_bias": self.f_bias})


class F42019(CecBenchmark):
    """
    .. [1] The 100-Digit Challenge: Problem Definitions and Evaluation Criteria for the 100-Digit
    Challenge Special Session and Competition on Single Objective Numerical Optimization
    """

    name = "F4: Shifted and Rotated Rastrigin’s Function"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = bias = 1.0"

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

    characteristics = ["Local optima’s number is huge", "The penultimate optimum is far from the global optimum"]

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "shift_data_4",
        f_matrix: typing.Any = "M_1_D",
        f_bias: float = 1.0,
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
            z = operator.dot_mv(f_matrix, x - f_shift)
            out[0] = operator.rastrigin_func(z) + f_bias

        super().__init__(
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            dim_changeable=True,
            dim_default=10,
            dim_max=10,
            dim_supported=[2, 10],
            f_bias=f_bias,
            shift=shift,
            rotate=rotate,
            rotate_bounds=rotate_bounds,
        )
        self.check_ndim_and_bounds(
            ndim, self.dim_max, bounds, np.array([[-100.0, 100.0] for _ in range(self.dim_default)])
        )
        self.make_support_data_path("data_2019")
        self.f_shift = self.check_shift_data(f_shift)[: self.ndim]
        self.f_matrix = self.check_matrix_data(f_matrix, needed_dim=True)
        self._x_global = np.asarray(self.f_shift)
        self._bind_kernel(
            compute,
            ["f_matrix", "f_shift", "f_bias"],
            paras={"f_shift": self.f_shift, "f_bias": self.f_bias, "f_matrix": self.f_matrix},
        )


class F52019(F42019):
    """
    .. [1] The 100-Digit Challenge: Problem Definitions and Evaluation Criteria for the 100-Digit
    Challenge Special Session and Competition on Single Objective Numerical Optimization
    """

    name = "F5: Shifted and Rotated Griewank’s Function"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = bias = 1.0"

    convex = True
    modality = False  # Number of ambiguous peaks, unknown # peaks

    characteristics = []

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "shift_data_5",
        f_matrix: typing.Any = "M_5_D",
        f_bias: float = 1.0,
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
            z = operator.dot_mv(f_matrix, x - f_shift)
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
            shift=shift,
            rotate=rotate,
            rotate_bounds=rotate_bounds,
        )
        self._bind_kernel(compute, ["f_matrix", "f_shift", "f_bias"])


class F62019(F42019):
    """
    .. [1] The 100-Digit Challenge: Problem Definitions and Evaluation Criteria for the 100-Digit
    Challenge Special Session and Competition on Single Objective Numerical Optimization
    """

    name = "F6: Shifted and Rotated Weierstrass Function"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = bias = 1.0"

    convex = False
    modality = True  # Number of ambiguous peaks, unknown # peaks

    characteristics = ["Local optima’s number is huge"]

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "shift_data_6",
        f_matrix: typing.Any = "M_6_D",
        f_bias: float = 1.0,
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
            z = operator.dot_mv(f_matrix, x - f_shift)
            out[0] = operator.weierstrass_norm_func(z) + f_bias

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
        self._bind_kernel(compute, ["f_matrix", "f_shift", "f_bias"])


class F72019(F42019):
    """
    .. [1] The 100-Digit Challenge: Problem Definitions and Evaluation Criteria for the 100-Digit
    Challenge Special Session and Competition on Single Objective Numerical Optimization
    """

    name = "F7: Shifted and Rotated Schwefel’s Function"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = bias = 1.0"

    convex = False
    modality = True  # Number of ambiguous peaks, unknown # peaks

    characteristics = ["Local optima’s number is huge"]

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "shift_data_7",
        f_matrix: typing.Any = "M_7_D",
        f_bias: float = 1.0,
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
            z = operator.dot_mv(f_matrix, x - f_shift)
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
            shift=shift,
            rotate=rotate,
            rotate_bounds=rotate_bounds,
        )
        self._bind_kernel(compute, ["f_matrix", "f_shift", "f_bias"])


class F82019(F42019):
    """
    .. [1] The 100-Digit Challenge: Problem Definitions and Evaluation Criteria for the 100-Digit
    Challenge Special Session and Competition on Single Objective Numerical Optimization
    """

    name = "F8: Shifted and Rotated Expanded Schaffer’s F6 Function"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = bias = 1.0"

    convex = False
    modality = True  # Number of ambiguous peaks, unknown # peaks

    characteristics = ["Local optima’s number is huge"]

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "shift_data_8",
        f_matrix: typing.Any = "M_8_D",
        f_bias: float = 1.0,
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
            z = operator.dot_mv(f_matrix, 0.005 * (x - f_shift))
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
            shift=shift,
            rotate=rotate,
            rotate_bounds=rotate_bounds,
        )
        self._bind_kernel(compute, ["f_matrix", "f_shift", "f_bias"])


class F92019(F42019):
    """
    .. [1] The 100-Digit Challenge: Problem Definitions and Evaluation Criteria for the 100-Digit
    Challenge Special Session and Competition on Single Objective Numerical Optimization
    """

    name = "F9: Shifted and Rotated Happy Cat Function"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = bias = 1.0"

    convex = False
    modality = False  # Number of ambiguous peaks, unknown # peaks

    characteristics = []

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "shift_data_9",
        f_matrix: typing.Any = "M_9_D",
        f_bias: float = 1.0,
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
            z = operator.dot_mv(f_matrix, x - f_shift)
            out[0] = operator.happy_cat_func(z, shift=-1.0) + f_bias

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
        self._bind_kernel(compute, ["f_matrix", "f_shift", "f_bias"])


class F102019(F42019):
    """
    .. [1] The 100-Digit Challenge: Problem Definitions and Evaluation Criteria for the 100-Digit
    Challenge Special Session and Competition on Single Objective Numerical Optimization
    """

    name = "F10: Shifted and Rotated Ackley Function"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = bias = 1.0"

    convex = False
    modality = False  # Number of ambiguous peaks, unknown # peaks

    characteristics = []

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "shift_data_10",
        f_matrix: typing.Any = "M_10_D",
        f_bias: float = 1.0,
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
            z = operator.dot_mv(f_matrix, x - f_shift)
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
            shift=shift,
            rotate=rotate,
            rotate_bounds=rotate_bounds,
        )
        self._bind_kernel(compute, ["f_matrix", "f_shift", "f_bias"])
