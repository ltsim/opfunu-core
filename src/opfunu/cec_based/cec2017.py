#!/usr/bin/env python
# Created by "Thieu" at 16:39, 08/07/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%


import typing

import numpy as np

from opfunu.benchmark.cec import CecBenchmark
from opfunu.utils import operator


class F12017(CecBenchmark):
    """
    .. [1] Problem Definitions and Evaluation Criteria for the CEC 2017
    Special Session and Competition on Single Objective Real-Parameter Numerical Optimization
    """

    name = "F1: Shifted and Rotated Bent Cigar"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = bias = 100.0"

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
        f_shift: typing.Any = "shift_data_1",
        f_matrix: typing.Any = "M_1_D",
        f_bias: float = 100.0,
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
            out[0] = operator.bent_cigar_func(z) + f_bias

        super().__init__(
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            dim_changeable=True,
            dim_default=30,
            dim_max=100,
            dim_supported=[2, 10, 20, 30, 50, 100],
            f_bias=f_bias,
            shift=shift,
            rotate=rotate,
            rotate_bounds=rotate_bounds,
        )
        self.check_ndim_and_bounds(
            ndim, self.dim_max, bounds, np.array([[-100.0, 100.0] for _ in range(self.dim_default)])
        )
        self.make_support_data_path("data_2017")
        self.f_shift = self.check_shift_data(f_shift)[: self.ndim]
        self.f_matrix = self.check_matrix_data(f_matrix, needed_dim=True)
        self._x_global = np.asarray(self.f_shift)
        self._bind_kernel(
            compute,
            ["f_matrix", "f_shift", "f_bias"],
            paras={"f_shift": self.f_shift, "f_bias": self.f_bias, "f_matrix": self.f_matrix},
        )


class F22017(F12017):
    """
    .. [1] Problem Definitions and Evaluation Criteria for the CEC 2017
    Special Session and Competition on Single Objective Real-Parameter Numerical Optimization
    """

    name = "F2: Shifted and Rotated Zakharov Function"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = bias = 200.0"

    characteristics = []

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "shift_data_2",
        f_matrix: typing.Any = "M_2_D",
        f_bias: float = 200.0,
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
            out[0] = operator.zakharov_func(z) + f_bias

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


class F32017(F12017):
    """
    .. [1] Problem Definitions and Evaluation Criteria for the CEC 2017
    Special Session and Competition on Single Objective Real-Parameter Numerical Optimization
    """

    name = "F3: Shifted and Rotated Rosenbrock’s Function"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = bias = 300.0"

    unimodal = False
    modality = True  # Number of ambiguous peaks, unknown # peaks
    # n_basins = 1
    # n_valleys = 1

    characteristics = ["Local optima’s number is huge"]

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "shift_data_3",
        f_matrix: typing.Any = "M_3_D",
        f_bias: float = 300.0,
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
            z = operator.dot_mv(f_matrix, 2.048 * (x - f_shift) / 100)
            out[0] = operator.rosenbrock_func(z, shift=1.0) + f_bias

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


class F42017(F12017):
    """
    .. [1] Problem Definitions and Evaluation Criteria for the CEC 2017
    Special Session and Competition on Single Objective Real-Parameter Numerical Optimization
    """

    name = "F4: Shifted and Rotated Rastrigin’s Function"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = bias = 400.0"

    unimodal = False
    modality = True  # Number of ambiguous peaks, unknown # peaks
    # n_basins = 1
    # n_valleys = 1

    characteristics = [
        "Local optima’s number is huge",
        "The second better local optimum is far from the global optimum",
    ]

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "shift_data_4",
        f_matrix: typing.Any = "M_4_D",
        f_bias: float = 400.0,
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


class F52017(F12017):
    """
    .. [1] Problem Definitions and Evaluation Criteria for the CEC 2017
    Special Session and Competition on Single Objective Real-Parameter Numerical Optimization
    """

    name = "F5: Shifted and Rotated Schaffer’s F7 Function"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = bias = 500.0"

    unimodal = False
    modality = True  # Number of ambiguous peaks, unknown # peaks
    # n_basins = 1
    # n_valleys = 1

    characteristics = ["Asymmetrical", "Local optima’s number is huge"]

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "shift_data_5",
        f_matrix: typing.Any = "M_5_D",
        f_bias: float = 500.0,
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
            z = operator.dot_mv(f_matrix, 0.5 * (x - f_shift) / 100)
            out[0] = operator.schaffer_f7_func(z) + f_bias

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


class F62017(F12017):
    """
    .. [1] Problem Definitions and Evaluation Criteria for the CEC 2017
    Special Session and Competition on Single Objective Real-Parameter Numerical Optimization
    """

    name = "F6: Shifted and Rotated Lunacek Bi-Rastrigin’s Function"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = bias = 600.0"

    unimodal = False
    modality = True  # Number of ambiguous peaks, unknown # peaks
    # n_basins = 1
    # n_valleys = 1

    characteristics = ["Asymmetrical", "Continuous everywhere yet differentiable nowhere"]

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "shift_data_6",
        f_matrix: typing.Any = "M_6_D",
        f_bias: float = 600.0,
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
            z = operator.dot_mv(f_matrix, 600.0 * (x - f_shift) / 100)
            out[0] = operator.lunacek_bi_rastrigin_func(z, shift=2.5) + f_bias

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


class F72017(F12017):
    """
    .. [1] Problem Definitions and Evaluation Criteria for the CEC 2017
    Special Session and Competition on Single Objective Real-Parameter Numerical Optimization
    """

    name = "F7: Shifted and Rotated Non-Continuous Rastrigin’s Function"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = bias = 700.0"

    unimodal = False
    modality = True  # Number of ambiguous peaks, unknown # peaks
    # n_basins = 1
    # n_valleys = 1

    characteristics = ["Asymmetrical", "Local optima’s number is huge"]

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "shift_data_7",
        f_matrix: typing.Any = "M_7_D",
        f_bias: float = 700.0,
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
            z = operator.dot_mv(f_matrix, 5.12 * (x - f_shift) / 100)
            out[0] = operator.non_continuous_rastrigin_func(z) + f_bias

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


class F82017(F12017):
    """
    .. [1] Problem Definitions and Evaluation Criteria for the CEC 2017
    Special Session and Competition on Single Objective Real-Parameter Numerical Optimization
    """

    name = "F8: Shifted and Rotated Levy Function"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = bias = 800.0"

    unimodal = False
    modality = True  # Number of ambiguous peaks, unknown # peaks
    characteristics = ["Local optima’s number is huge"]

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "shift_data_8",
        f_matrix: typing.Any = "M_8_D",
        f_bias: float = 800.0,
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
            z = operator.dot_mv(f_matrix, 5.12 * (x - f_shift) / 100)
            out[0] = operator.levy_func(z, shift=1.0) + f_bias

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


class F92017(F12017):
    """
    .. [1] Problem Definitions and Evaluation Criteria for the CEC 2017
    Special Session and Competition on Single Objective Real-Parameter Numerical Optimization
    """

    name = "F9: Shifted and Rotated Schwefel’s Function"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = bias = 900.0"

    unimodal = False
    modality = True  # Number of ambiguous peaks, unknown # peaks
    characteristics = [
        "Local optima’s number is huge",
        "The second better local optimum is far from the global optimum",
    ]

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "shift_data_9",
        f_matrix: typing.Any = "M_9_D",
        f_bias: float = 900.0,
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
            z = operator.dot_mv(f_matrix, 1000.0 * (x - f_shift) / 100)
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


class F102017(CecBenchmark):
    """
    .. [1] Problem Definitions and Evaluation Criteria for the CEC 2017
    Special Session and Competition on Single Objective Real-Parameter Numerical Optimization
    """

    name = "F10: Hybrid Function 1"
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

    characteristics = []

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "shift_data_10",
        f_matrix: typing.Any = "M_10_D",
        f_shuffle: typing.Any = "shuffle_data_10_D",
        f_bias: float = 1000.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
        shift: typing.Any = None,
        rotate: typing.Any = None,
        rotate_bounds: bool = True,
    ) -> None:
        def compute(
            x: np.ndarray,
            f_matrix: typing.Any,
            f_shift: typing.Any,
            idx1: typing.Any,
            idx2: typing.Any,
            idx3: typing.Any,
            f_bias: typing.Any,
            out: np.ndarray,
        ) -> None:
            mz = operator.dot_mv(f_matrix, x - f_shift)
            out[0] = (
                operator.zakharov_func(mz[idx1])
                + operator.rosenbrock_func(mz[idx2], shift=1.0)
                + operator.rastrigin_func(mz[idx3])
                + f_bias
            )

        super().__init__(
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            dim_changeable=True,
            dim_default=30,
            dim_max=100,
            dim_supported=[10, 30, 50, 100],
            f_bias=f_bias,
            shift=shift,
            rotate=rotate,
            rotate_bounds=rotate_bounds,
        )
        self.check_ndim_and_bounds(
            ndim, self.dim_max, bounds, np.array([[-100.0, 100.0] for _ in range(self.dim_default)])
        )
        self.make_support_data_path("data_2017")
        self.f_shift = self.check_shift_data(f_shift)[: self.ndim]
        self.f_matrix = self.check_matrix_data(f_matrix, needed_dim=True)
        self.f_shuffle = self.check_shuffle_data(f_shuffle, needed_dim=True)
        self.f_shuffle = (self.f_shuffle - 1).astype(int)
        self._x_global = np.asarray(self.f_shift)
        self.n_funcs = 3
        self.p = np.array([0.2, 0.4, 0.4])
        self.n1 = int(np.ceil(self.p[0] * self.ndim))
        self.n2 = int(np.ceil(self.p[1] * self.ndim)) + self.n1
        self.idx1, self.idx2, self.idx3 = (
            self.f_shuffle[: self.n1],
            self.f_shuffle[self.n1 : self.n2],
            self.f_shuffle[self.n2 : self.ndim],
        )
        self._bind_kernel(
            compute,
            ["f_matrix", "f_shift", "idx1", "idx2", "idx3", "f_bias"],
            paras={
                "f_shift": self.f_shift,
                "f_bias": self.f_bias,
                "f_matrix": self.f_matrix,
                "f_shuffle": self.f_shuffle,
            },
        )


class F112017(F102017):
    """
    .. [1] Problem Definitions and Evaluation Criteria for the CEC 2017
    Special Session and Competition on Single Objective Real-Parameter Numerical Optimization
    """

    name = "F11: Hybrid Function 2"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = bias = 1100.0"

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "shift_data_11",
        f_matrix: typing.Any = "M_11_D",
        f_shuffle: typing.Any = "shuffle_data_11_D",
        f_bias: float = 1100.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
        shift: typing.Any = None,
        rotate: typing.Any = None,
        rotate_bounds: bool = True,
    ) -> None:
        def compute(
            x: np.ndarray,
            f_matrix: typing.Any,
            f_shift: typing.Any,
            idx1: typing.Any,
            idx2: typing.Any,
            idx3: typing.Any,
            f_bias: typing.Any,
            out: np.ndarray,
        ) -> None:
            mz = operator.dot_mv(f_matrix, x - f_shift)
            out[0] = (
                operator.elliptic_func(mz[idx1])
                + operator.modified_schwefel_func(mz[idx2])
                + operator.bent_cigar_func(mz[idx3])
                + f_bias
            )

        super().__init__(
            ndim,
            bounds,
            f_shift,
            f_matrix,
            f_shuffle,
            f_bias,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            shift=shift,
            rotate=rotate,
            rotate_bounds=rotate_bounds,
        )
        self.n_funcs = 3
        self.p = np.array([0.3, 0.3, 0.4])
        self.n1 = int(np.ceil(self.p[0] * self.ndim))
        self.n2 = int(np.ceil(self.p[1] * self.ndim)) + self.n1
        self.idx1, self.idx2, self.idx3 = (
            self.f_shuffle[: self.n1],
            self.f_shuffle[self.n1 : self.n2],
            self.f_shuffle[self.n2 : self.ndim],
        )
        self.g1 = operator.elliptic_func
        self.g2 = operator.modified_schwefel_func
        self.g3 = operator.bent_cigar_func
        self._bind_kernel(
            compute,
            ["f_matrix", "f_shift", "idx1", "idx2", "idx3", "f_bias"],
            paras={
                "f_shift": self.f_shift,
                "f_bias": self.f_bias,
                "f_matrix": self.f_matrix,
                "f_shuffle": self.f_shuffle,
            },
        )


class F122017(F102017):
    """
    .. [1] Problem Definitions and Evaluation Criteria for the CEC 2017
    Special Session and Competition on Single Objective Real-Parameter Numerical Optimization
    """

    name = "F12: Hybrid Function 3"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = bias = 1200.0"

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "shift_data_12",
        f_matrix: typing.Any = "M_12_D",
        f_shuffle: typing.Any = "shuffle_data_12_D",
        f_bias: float = 1200.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
        shift: typing.Any = None,
        rotate: typing.Any = None,
        rotate_bounds: bool = True,
    ) -> None:
        def compute(
            x: np.ndarray,
            f_matrix: typing.Any,
            f_shift: typing.Any,
            idx1: typing.Any,
            idx2: typing.Any,
            idx3: typing.Any,
            f_bias: typing.Any,
            out: np.ndarray,
        ) -> None:
            mz = operator.dot_mv(f_matrix, x - f_shift)
            miu0 = 2.5
            out[0] = (
                operator.bent_cigar_func(mz[idx1])
                + operator.rosenbrock_func(mz[idx2], shift=1.0)
                + operator.lunacek_bi_rastrigin_func(mz[idx3], miu0, 1.0, shift=miu0)
                + f_bias
            )

        super().__init__(
            ndim,
            bounds,
            f_shift,
            f_matrix,
            f_shuffle,
            f_bias,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            shift=shift,
            rotate=rotate,
            rotate_bounds=rotate_bounds,
        )
        self.n_funcs = 3
        self.p = np.array([0.3, 0.3, 0.4])
        self.n1 = int(np.ceil(self.p[0] * self.ndim))
        self.n2 = int(np.ceil(self.p[1] * self.ndim)) + self.n1
        self.idx1, self.idx2, self.idx3 = (
            self.f_shuffle[: self.n1],
            self.f_shuffle[self.n1 : self.n2],
            self.f_shuffle[self.n2 : self.ndim],
        )
        self._bind_kernel(
            compute,
            ["f_matrix", "f_shift", "idx1", "idx2", "idx3", "f_bias"],
            paras={
                "f_shift": self.f_shift,
                "f_bias": self.f_bias,
                "f_matrix": self.f_matrix,
                "f_shuffle": self.f_shuffle,
            },
        )


class F132017(F102017):
    """
    .. [1] Problem Definitions and Evaluation Criteria for the CEC 2017
    Special Session and Competition on Single Objective Real-Parameter Numerical Optimization
    """

    name = "F13: Hybrid Function 4"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = bias = 1300.0"

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "shift_data_13",
        f_matrix: typing.Any = "M_13_D",
        f_shuffle: typing.Any = "shuffle_data_13_D",
        f_bias: float = 1300.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
        shift: typing.Any = None,
        rotate: typing.Any = None,
        rotate_bounds: bool = True,
    ) -> None:
        def compute(
            x: np.ndarray,
            f_matrix: typing.Any,
            f_shift: typing.Any,
            idx1: typing.Any,
            idx2: typing.Any,
            idx3: typing.Any,
            idx4: typing.Any,
            f_bias: typing.Any,
            out: np.ndarray,
        ) -> None:
            mz = operator.dot_mv(f_matrix, x - f_shift)
            out[0] = (
                operator.elliptic_func(mz[idx1])
                + operator.ackley_func(mz[idx2])
                + operator.schaffer_f7_func(mz[idx3])
                + operator.rastrigin_func(mz[idx4])
                + f_bias
            )

        super().__init__(
            ndim,
            bounds,
            f_shift,
            f_matrix,
            f_shuffle,
            f_bias,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            shift=shift,
            rotate=rotate,
            rotate_bounds=rotate_bounds,
        )
        self.n_funcs = 4
        self.p = np.array([0.2, 0.2, 0.2, 0.4])
        self.n1 = int(np.ceil(self.p[0] * self.ndim))
        self.n2 = int(np.ceil(self.p[1] * self.ndim)) + self.n1
        self.n3 = int(np.ceil(self.p[2] * self.ndim)) + self.n2
        self.idx1, self.idx2 = self.f_shuffle[: self.n1], self.f_shuffle[self.n1 : self.n2]
        self.idx3, self.idx4 = self.f_shuffle[self.n2 : self.n3], self.f_shuffle[self.n3 : self.ndim]
        self.g1 = operator.elliptic_func
        self.g2 = operator.ackley_func
        self.g3 = operator.schaffer_f7_func
        self.g4 = operator.rastrigin_func
        self._bind_kernel(
            compute,
            ["f_matrix", "f_shift", "idx1", "idx2", "idx3", "idx4", "f_bias"],
            paras={
                "f_shift": self.f_shift,
                "f_bias": self.f_bias,
                "f_matrix": self.f_matrix,
                "f_shuffle": self.f_shuffle,
            },
        )


class F142017(F102017):
    """
    .. [1] Problem Definitions and Evaluation Criteria for the CEC 2017
    Special Session and Competition on Single Objective Real-Parameter Numerical Optimization
    """

    name = "F14: Hybrid Function 5"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = bias = 1400.0"

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "shift_data_14",
        f_matrix: typing.Any = "M_14_D",
        f_shuffle: typing.Any = "shuffle_data_14_D",
        f_bias: float = 1400.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
        shift: typing.Any = None,
        rotate: typing.Any = None,
        rotate_bounds: bool = True,
    ) -> None:
        def compute(
            x: np.ndarray,
            f_matrix: typing.Any,
            f_shift: typing.Any,
            idx1: typing.Any,
            idx2: typing.Any,
            idx3: typing.Any,
            idx4: typing.Any,
            f_bias: typing.Any,
            out: np.ndarray,
        ) -> None:
            mz = operator.dot_mv(f_matrix, x - f_shift)
            out[0] = (
                operator.bent_cigar_func(mz[idx1])
                + operator.hgbat_func(mz[idx2], shift=-1.0)
                + operator.rastrigin_func(mz[idx3])
                + operator.rosenbrock_func(mz[idx4], shift=1.0)
                + f_bias
            )

        super().__init__(
            ndim,
            bounds,
            f_shift,
            f_matrix,
            f_shuffle,
            f_bias,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            shift=shift,
            rotate=rotate,
            rotate_bounds=rotate_bounds,
        )
        self.n_funcs = 4
        self.p = np.array([0.2, 0.2, 0.3, 0.3])
        self.n1 = int(np.ceil(self.p[0] * self.ndim))
        self.n2 = int(np.ceil(self.p[1] * self.ndim)) + self.n1
        self.n3 = int(np.ceil(self.p[2] * self.ndim)) + self.n2
        self.idx1, self.idx2 = self.f_shuffle[: self.n1], self.f_shuffle[self.n1 : self.n2]
        self.idx3, self.idx4 = self.f_shuffle[self.n2 : self.n3], self.f_shuffle[self.n3 : self.ndim]
        self._bind_kernel(
            compute,
            ["f_matrix", "f_shift", "idx1", "idx2", "idx3", "idx4", "f_bias"],
            paras={
                "f_shift": self.f_shift,
                "f_bias": self.f_bias,
                "f_matrix": self.f_matrix,
                "f_shuffle": self.f_shuffle,
            },
        )


class F152017(F102017):
    """
    .. [1] Problem Definitions and Evaluation Criteria for the CEC 2017
    Special Session and Competition on Single Objective Real-Parameter Numerical Optimization
    """

    name = "F15: Hybrid Function 6"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = bias = 1500.0"

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "shift_data_15",
        f_matrix: typing.Any = "M_15_D",
        f_shuffle: typing.Any = "shuffle_data_15_D",
        f_bias: float = 1500.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
        shift: typing.Any = None,
        rotate: typing.Any = None,
        rotate_bounds: bool = True,
    ) -> None:
        def compute(
            x: np.ndarray,
            f_matrix: typing.Any,
            f_shift: typing.Any,
            idx1: typing.Any,
            idx2: typing.Any,
            idx3: typing.Any,
            idx4: typing.Any,
            f_bias: typing.Any,
            out: np.ndarray,
        ) -> None:
            mz = operator.dot_mv(f_matrix, x - f_shift)
            out[0] = (
                operator.expanded_scaffer_f6_func(mz[idx1])
                + operator.hgbat_func(mz[idx2], shift=-1.0)
                + operator.rosenbrock_func(mz[idx3], shift=1.0)
                + operator.modified_schwefel_func(mz[idx4])
                + f_bias
            )

        super().__init__(
            ndim,
            bounds,
            f_shift,
            f_matrix,
            f_shuffle,
            f_bias,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            shift=shift,
            rotate=rotate,
            rotate_bounds=rotate_bounds,
        )
        self.n_funcs = 4
        self.p = np.array([0.2, 0.2, 0.3, 0.3])
        self.n1 = int(np.ceil(self.p[0] * self.ndim))
        self.n2 = int(np.ceil(self.p[1] * self.ndim)) + self.n1
        self.n3 = int(np.ceil(self.p[2] * self.ndim)) + self.n2
        self.idx1, self.idx2 = self.f_shuffle[: self.n1], self.f_shuffle[self.n1 : self.n2]
        self.idx3, self.idx4 = self.f_shuffle[self.n2 : self.n3], self.f_shuffle[self.n3 : self.ndim]
        self._bind_kernel(
            compute,
            ["f_matrix", "f_shift", "idx1", "idx2", "idx3", "idx4", "f_bias"],
            paras={
                "f_shift": self.f_shift,
                "f_bias": self.f_bias,
                "f_matrix": self.f_matrix,
                "f_shuffle": self.f_shuffle,
            },
        )


class F162017(F102017):
    """
    .. [1] Problem Definitions and Evaluation Criteria for the CEC 2017
    Special Session and Competition on Single Objective Real-Parameter Numerical Optimization
    """

    name = "F16: Hybrid Function 7"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = bias = 1600.0"

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "shift_data_16",
        f_matrix: typing.Any = "M_16_D",
        f_shuffle: typing.Any = "shuffle_data_16_D",
        f_bias: float = 1600.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
        shift: typing.Any = None,
        rotate: typing.Any = None,
        rotate_bounds: bool = True,
    ) -> None:
        def compute(
            x: np.ndarray,
            f_matrix: typing.Any,
            f_shift: typing.Any,
            idx1: typing.Any,
            idx2: typing.Any,
            idx3: typing.Any,
            idx4: typing.Any,
            idx5: typing.Any,
            f_bias: typing.Any,
            out: np.ndarray,
        ) -> None:
            mz = operator.dot_mv(f_matrix, x - f_shift)
            out[0] = (
                operator.katsuura_func(mz[idx1])
                + operator.ackley_func(mz[idx2])
                + operator.expanded_griewank_rosenbrock_func(mz[idx3])
                + operator.modified_schwefel_func(mz[idx4])
                + operator.rastrigin_func(mz[idx5])
                + f_bias
            )

        super().__init__(
            ndim,
            bounds,
            f_shift,
            f_matrix,
            f_shuffle,
            f_bias,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            shift=shift,
            rotate=rotate,
            rotate_bounds=rotate_bounds,
        )
        self.n_funcs = 5
        self.p = np.array([0.1, 0.2, 0.2, 0.2, 0.3])
        self.n1 = int(np.ceil(self.p[0] * self.ndim))
        self.n2 = int(np.ceil(self.p[1] * self.ndim)) + self.n1
        self.n3 = int(np.ceil(self.p[2] * self.ndim)) + self.n2
        self.n4 = int(np.ceil(self.p[3] * self.ndim)) + self.n3
        self.idx1, self.idx2 = self.f_shuffle[: self.n1], self.f_shuffle[self.n1 : self.n2]
        self.idx3, self.idx4 = self.f_shuffle[self.n2 : self.n3], self.f_shuffle[self.n3 : self.n4]
        self.idx5 = self.f_shuffle[self.n4 : self.ndim]
        self.g1 = operator.katsuura_func
        self.g2 = operator.ackley_func
        self.g3 = operator.expanded_griewank_rosenbrock_func
        self.g4 = operator.modified_schwefel_func
        self.g5 = operator.rastrigin_func
        self._bind_kernel(
            compute,
            ["f_matrix", "f_shift", "idx1", "idx2", "idx3", "idx4", "idx5", "f_bias"],
            paras={
                "f_shift": self.f_shift,
                "f_bias": self.f_bias,
                "f_matrix": self.f_matrix,
                "f_shuffle": self.f_shuffle,
            },
        )


class F172017(F102017):
    """
    .. [1] Problem Definitions and Evaluation Criteria for the CEC 2017
    Special Session and Competition on Single Objective Real-Parameter Numerical Optimization
    """

    name = "F17: Hybrid Function 8"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = bias = 1700.0"

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "shift_data_17",
        f_matrix: typing.Any = "M_17_D",
        f_shuffle: typing.Any = "shuffle_data_17_D",
        f_bias: float = 1700.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
        shift: typing.Any = None,
        rotate: typing.Any = None,
        rotate_bounds: bool = True,
    ) -> None:
        def compute(
            x: np.ndarray,
            f_matrix: typing.Any,
            f_shift: typing.Any,
            idx1: typing.Any,
            idx2: typing.Any,
            idx3: typing.Any,
            idx4: typing.Any,
            idx5: typing.Any,
            f_bias: typing.Any,
            out: np.ndarray,
        ) -> None:
            mz = operator.dot_mv(f_matrix, x - f_shift)
            out[0] = (
                operator.elliptic_func(mz[idx1])
                + operator.ackley_func(mz[idx2])
                + operator.rastrigin_func(mz[idx3])
                + operator.hgbat_func(mz[idx4], shift=-1.0)
                + operator.discus_func(mz[idx5])
                + f_bias
            )

        super().__init__(
            ndim,
            bounds,
            f_shift,
            f_matrix,
            f_shuffle,
            f_bias,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            shift=shift,
            rotate=rotate,
            rotate_bounds=rotate_bounds,
        )
        self.n_funcs = 5
        self.p = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        self.n1 = int(np.ceil(self.p[0] * self.ndim))
        self.n2 = int(np.ceil(self.p[1] * self.ndim)) + self.n1
        self.n3 = int(np.ceil(self.p[2] * self.ndim)) + self.n2
        self.n4 = int(np.ceil(self.p[3] * self.ndim)) + self.n3
        self.idx1, self.idx2 = self.f_shuffle[: self.n1], self.f_shuffle[self.n1 : self.n2]
        self.idx3, self.idx4 = self.f_shuffle[self.n2 : self.n3], self.f_shuffle[self.n3 : self.n4]
        self.idx5 = self.f_shuffle[self.n4 : self.ndim]
        self._bind_kernel(
            compute,
            ["f_matrix", "f_shift", "idx1", "idx2", "idx3", "idx4", "idx5", "f_bias"],
            paras={
                "f_shift": self.f_shift,
                "f_bias": self.f_bias,
                "f_matrix": self.f_matrix,
                "f_shuffle": self.f_shuffle,
            },
        )


class F182017(F102017):
    """
    .. [1] Problem Definitions and Evaluation Criteria for the CEC 2017
    Special Session and Competition on Single Objective Real-Parameter Numerical Optimization
    """

    name = "F18: Hybrid Function 9"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = bias = 1800.0"

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "shift_data_18",
        f_matrix: typing.Any = "M_18_D",
        f_shuffle: typing.Any = "shuffle_data_18_D",
        f_bias: float = 1800.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
        shift: typing.Any = None,
        rotate: typing.Any = None,
        rotate_bounds: bool = True,
    ) -> None:
        def compute(
            x: np.ndarray,
            f_matrix: typing.Any,
            f_shift: typing.Any,
            idx1: typing.Any,
            idx2: typing.Any,
            idx3: typing.Any,
            idx4: typing.Any,
            idx5: typing.Any,
            f_bias: typing.Any,
            out: np.ndarray,
        ) -> None:
            mz = operator.dot_mv(f_matrix, x - f_shift)
            out[0] = (
                operator.bent_cigar_func(mz[idx1])
                + operator.rastrigin_func(mz[idx2])
                + operator.expanded_griewank_rosenbrock_func(mz[idx3])
                + operator.weierstrass_norm_func(mz[idx4])
                + operator.expanded_scaffer_f6_func(mz[idx5])
                + f_bias
            )

        super().__init__(
            ndim,
            bounds,
            f_shift,
            f_matrix,
            f_shuffle,
            f_bias,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            shift=shift,
            rotate=rotate,
            rotate_bounds=rotate_bounds,
        )
        self.n_funcs = 5
        self.p = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        self.n1 = int(np.ceil(self.p[0] * self.ndim))
        self.n2 = int(np.ceil(self.p[1] * self.ndim)) + self.n1
        self.n3 = int(np.ceil(self.p[2] * self.ndim)) + self.n2
        self.n4 = int(np.ceil(self.p[3] * self.ndim)) + self.n3
        self.idx1, self.idx2 = self.f_shuffle[: self.n1], self.f_shuffle[self.n1 : self.n2]
        self.idx3, self.idx4 = self.f_shuffle[self.n2 : self.n3], self.f_shuffle[self.n3 : self.n4]
        self.idx5 = self.f_shuffle[self.n4 : self.ndim]
        self.g1 = operator.bent_cigar_func
        self.g2 = operator.rastrigin_func
        self.g3 = operator.expanded_griewank_rosenbrock_func
        self.g4 = operator.weierstrass_norm_func
        self.g5 = operator.expanded_scaffer_f6_func
        self._bind_kernel(
            compute,
            ["f_matrix", "f_shift", "idx1", "idx2", "idx3", "idx4", "idx5", "f_bias"],
            paras={
                "f_shift": self.f_shift,
                "f_bias": self.f_bias,
                "f_matrix": self.f_matrix,
                "f_shuffle": self.f_shuffle,
            },
        )


class F192017(F102017):
    """
    .. [1] Problem Definitions and Evaluation Criteria for the CEC 2017
    Special Session and Competition on Single Objective Real-Parameter Numerical Optimization
    """

    name = "F19: Hybrid Function 10"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = bias = 1900.0"

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "shift_data_19",
        f_matrix: typing.Any = "M_19_D",
        f_shuffle: typing.Any = "shuffle_data_19_D",
        f_bias: float = 1900.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
        shift: typing.Any = None,
        rotate: typing.Any = None,
        rotate_bounds: bool = True,
    ) -> None:
        def compute(
            x: np.ndarray,
            f_matrix: typing.Any,
            f_shift: typing.Any,
            idx1: typing.Any,
            idx2: typing.Any,
            idx3: typing.Any,
            idx4: typing.Any,
            idx5: typing.Any,
            idx6: typing.Any,
            f_bias: typing.Any,
            out: np.ndarray,
        ) -> None:
            mz = operator.dot_mv(f_matrix, x - f_shift)
            out[0] = (
                operator.happy_cat_func(mz[idx1], shift=-1.0)
                + operator.katsuura_func(mz[idx2])
                + operator.ackley_func(mz[idx3])
                + operator.rastrigin_func(mz[idx4])
                + operator.modified_schwefel_func(mz[idx5])
                + operator.schaffer_f7_func(mz[idx6])
                + f_bias
            )

        super().__init__(
            ndim,
            bounds,
            f_shift,
            f_matrix,
            f_shuffle,
            f_bias,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            shift=shift,
            rotate=rotate,
            rotate_bounds=rotate_bounds,
        )
        self.n_funcs = 6
        self.p = np.array([0.1, 0.1, 0.2, 0.2, 0.2, 0.2])
        self.n1 = int(np.ceil(self.p[0] * self.ndim))
        self.n2 = int(np.ceil(self.p[1] * self.ndim)) + self.n1
        self.n3 = int(np.ceil(self.p[2] * self.ndim)) + self.n2
        self.n4 = int(np.ceil(self.p[3] * self.ndim)) + self.n3
        self.n5 = int(np.ceil(self.p[4] * self.ndim)) + self.n4
        self.idx1, self.idx2 = self.f_shuffle[: self.n1], self.f_shuffle[self.n1 : self.n2]
        self.idx3, self.idx4 = self.f_shuffle[self.n2 : self.n3], self.f_shuffle[self.n3 : self.n4]
        self.idx5, self.idx6 = self.f_shuffle[self.n4 : self.n5], self.f_shuffle[self.n5 : self.ndim]
        self._bind_kernel(
            compute,
            ["f_matrix", "f_shift", "idx1", "idx2", "idx3", "idx4", "idx5", "idx6", "f_bias"],
            paras={
                "f_shift": self.f_shift,
                "f_bias": self.f_bias,
                "f_matrix": self.f_matrix,
                "f_shuffle": self.f_shuffle,
            },
        )


class F202017(CecBenchmark):
    """
    .. [1] Problem Definitions and Evaluation Criteria for the CEC 2017
    Special Session and Competition on Single Objective Real-Parameter Numerical Optimization
    """

    name = "F20: Composition Function 1"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = bias = 2000.0"

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

    modality = False  # Number of ambiguous peaks, unknown # peaks
    # n_basins = 1
    # n_valleys = 1
    characteristics = ["Asymmetrical", "Different properties around different local optima"]

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "shift_data_21",
        f_matrix: typing.Any = "M_21_D",
        f_bias: float = 2000.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
        shift: typing.Any = None,
        rotate: typing.Any = None,
        rotate_bounds: bool = True,
    ) -> None:
        def compute(
            x: np.ndarray,
            f_matrix: typing.Any,
            f_shift: typing.Any,
            lamdas: typing.Any,
            bias: typing.Any,
            xichmas: typing.Any,
            f_bias: typing.Any,
            out: np.ndarray,
        ) -> None:
            z0 = operator.dot_mv(f_matrix[: x.shape[0], :], 2.048 * (x - f_shift[0]) / 100) + 1
            g0 = lamdas[0] * operator.rosenbrock_func(z0) + bias[0]
            w0 = operator.calculate_weight(x - f_shift[0], xichmas[0])

            # 2. High Conditioned Elliptic Function F11’
            z1 = operator.dot_mv(f_matrix[x.shape[0] : 2 * x.shape[0], :], x - f_shift[1])
            g1 = lamdas[1] * operator.elliptic_func(z1) + bias[1]
            w1 = operator.calculate_weight(x - f_shift[1], xichmas[1])

            # 3. Rastrigin’s Function F4’
            z2 = operator.dot_mv(f_matrix[2 * x.shape[0] : 3 * x.shape[0], :], x - f_shift[2])
            g2 = lamdas[2] * operator.rastrigin_func(z2) + bias[2]
            w2 = operator.calculate_weight(x - f_shift[2], xichmas[2])

            ws = np.array([w0, w1, w2])
            ws = ws / np.sum(ws)
            gs = np.array([g0, g1, g2])
            out[0] = operator.dot_vv(ws, gs) + f_bias

        super().__init__(
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            dim_changeable=True,
            dim_default=30,
            dim_max=100,
            dim_supported=[2, 10, 20, 30, 50, 100],
            f_bias=f_bias,
            shift=shift,
            rotate=rotate,
            rotate_bounds=rotate_bounds,
        )
        self.check_ndim_and_bounds(
            ndim, self.dim_max, bounds, np.array([[-100.0, 100.0] for _ in range(self.dim_default)])
        )
        self.make_support_data_path("data_2017")
        self.f_shift = self.check_shift_matrix(f_shift)[:, : self.ndim]
        self.f_matrix = self.check_matrix_data(f_matrix)[:, : self.ndim]
        self._x_global = np.asarray(self.f_shift[0])
        self.n_funcs = 3
        self.xichmas = [10, 20, 30]
        self.lamdas = [1.0, 1e-6, 1.0]
        self.bias = [0, 100, 200]
        self.g0 = operator.rosenbrock_func
        self.g1 = operator.elliptic_func
        self.g2 = operator.rastrigin_func
        self._bind_kernel(
            compute,
            ["f_matrix", "f_shift", "lamdas", "bias", "xichmas", "f_bias"],
            paras={"f_shift": self.f_shift, "f_bias": self.f_bias, "f_matrix": self.f_matrix},
        )


class F212017(F202017):
    """
    .. [1] Problem Definitions and Evaluation Criteria for the CEC 2017
    Special Session and Competition on Single Objective Real-Parameter Numerical Optimization
    """

    name = "F21: Composition Function 2"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = bias = 2100.0"

    modality = True  # Number of ambiguous peaks, unknown # peaks
    # n_basins = 1
    # n_valleys = 1
    characteristics = ["Asymmetrical", "Different properties around different local optima"]

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "shift_data_22",
        f_matrix: typing.Any = "M_22_D",
        f_bias: float = 2100.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
        shift: typing.Any = None,
        rotate: typing.Any = None,
        rotate_bounds: bool = True,
    ) -> None:
        def compute(
            x: np.ndarray,
            f_matrix: typing.Any,
            f_shift: typing.Any,
            lamdas: typing.Any,
            bias: typing.Any,
            xichmas: typing.Any,
            f_bias: typing.Any,
            out: np.ndarray,
        ) -> None:
            z0 = operator.dot_mv(f_matrix[: x.shape[0], :], x - f_shift[0])
            g0 = lamdas[0] * operator.rastrigin_func(z0) + bias[0]
            w0 = operator.calculate_weight(x - f_shift[0], xichmas[0])

            # 2. Griewank’s Function F15’
            z1 = operator.dot_mv(f_matrix[x.shape[0] : 2 * x.shape[0], :], x - f_shift[1])
            g1 = lamdas[1] * operator.griewank_func(z1) + bias[1]
            w1 = operator.calculate_weight(x - f_shift[1], xichmas[1])

            # 3. Modifed Schwefel's Function F10’
            # z2 = operator.dot_mv(f_matrix[2*x.shape[0]:3*x.shape[0], :], x - f_shift[2])
            z2 = 1000 * (x - f_shift[2]) / 100
            g2 = lamdas[2] * operator.modified_schwefel_func(z2) + bias[2]
            w2 = operator.calculate_weight(x - f_shift[2], xichmas[2])

            ws = np.array([w0, w1, w2])
            ws = ws / np.sum(ws)
            gs = np.array([g0, g1, g2])
            out[0] = operator.dot_vv(ws, gs) + f_bias

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
        self.n_funcs = 3
        self.xichmas = [10, 20, 30]
        self.lamdas = [1.0, 10.0, 1.0]
        self.bias = [0, 100, 200]
        self.g0 = operator.rastrigin_func
        self.g1 = operator.griewank_func
        self.g2 = operator.modified_schwefel_func
        self._bind_kernel(
            compute,
            ["f_matrix", "f_shift", "lamdas", "bias", "xichmas", "f_bias"],
            paras={"f_shift": self.f_shift, "f_bias": self.f_bias, "f_matrix": self.f_matrix},
        )


class F222017(F202017):
    """
    .. [1] Problem Definitions and Evaluation Criteria for the CEC 2017
    Special Session and Competition on Single Objective Real-Parameter Numerical Optimization
    """

    name = "F22: Composition Function 3"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = bias = 2200.0"

    modality = True  # Number of ambiguous peaks, unknown # peaks
    # n_basins = 1
    # n_valleys = 1
    characteristics = ["Asymmetrical", "Different properties around different local optima"]

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "shift_data_23",
        f_matrix: typing.Any = "M_23_D",
        f_bias: float = 2200.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
        shift: typing.Any = None,
        rotate: typing.Any = None,
        rotate_bounds: bool = True,
    ) -> None:
        def compute(
            x: np.ndarray,
            f_matrix: typing.Any,
            f_shift: typing.Any,
            lamdas: typing.Any,
            bias: typing.Any,
            xichmas: typing.Any,
            f_bias: typing.Any,
            out: np.ndarray,
        ) -> None:
            z0 = operator.dot_mv(f_matrix[: x.shape[0], :], 2.048 * (x - f_shift[0]) / 100) + 1
            g0 = lamdas[0] * operator.rosenbrock_func(z0) + bias[0]
            w0 = operator.calculate_weight(x - f_shift[0], xichmas[0])

            # 2. Ackley’s Function F13’
            z1 = operator.dot_mv(f_matrix[x.shape[0] : 2 * x.shape[0], :], x - f_shift[1])
            g1 = lamdas[1] * operator.ackley_func(z1) + bias[1]
            w1 = operator.calculate_weight(x - f_shift[1], xichmas[1])

            # 3. Modified Schwefel's Function F10’
            z2 = operator.dot_mv(f_matrix[2 * x.shape[0] : 3 * x.shape[0], :], x - f_shift[2])
            g2 = lamdas[2] * operator.modified_schwefel_func(z2) + bias[2]
            w2 = operator.calculate_weight(x - f_shift[2], xichmas[2])

            # 4. Rastrigin’s Function F5’
            z3 = operator.dot_mv(f_matrix[3 * x.shape[0] : 4 * x.shape[0], :], x - f_shift[3])
            g3 = lamdas[3] * operator.rastrigin_func(z3) + bias[3]
            w3 = operator.calculate_weight(x - f_shift[3], xichmas[3])

            ws = np.array([w0, w1, w2, w3])
            ws = ws / np.sum(ws)
            gs = np.array([g0, g1, g2, g3])
            out[0] = operator.dot_vv(ws, gs) + f_bias

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
        self.n_funcs = 4
        self.xichmas = [10, 20, 30, 40]
        self.lamdas = [1.0, 10.0, 1.0, 1.0]
        self.bias = [0, 100, 200, 300]
        self.g0 = operator.rosenbrock_func
        self.g1 = operator.ackley_func
        self.g2 = operator.modified_schwefel_func
        self.g3 = operator.rastrigin_func
        self._bind_kernel(
            compute,
            ["f_matrix", "f_shift", "lamdas", "bias", "xichmas", "f_bias"],
            paras={"f_shift": self.f_shift, "f_bias": self.f_bias, "f_matrix": self.f_matrix},
        )


class F232017(F202017):
    """
    .. [1] Problem Definitions and Evaluation Criteria for the CEC 2017
    Special Session and Competition on Single Objective Real-Parameter Numerical Optimization
    """

    name = "F23: Composition Function 4"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = bias = 2300.0"

    modality = True  # Number of ambiguous peaks, unknown # peaks
    # n_basins = 1
    # n_valleys = 1
    characteristics = ["Asymmetrical", "Different properties around different local optima"]

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "shift_data_24",
        f_matrix: typing.Any = "M_24_D",
        f_bias: float = 2300.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
        shift: typing.Any = None,
        rotate: typing.Any = None,
        rotate_bounds: bool = True,
    ) -> None:
        def compute(
            x: np.ndarray,
            f_matrix: typing.Any,
            f_shift: typing.Any,
            lamdas: typing.Any,
            bias: typing.Any,
            xichmas: typing.Any,
            f_bias: typing.Any,
            out: np.ndarray,
        ) -> None:
            z0 = operator.dot_mv(f_matrix[: x.shape[0], :], x - f_shift[0])
            g0 = lamdas[0] * operator.ackley_func(z0) + bias[0]
            w0 = operator.calculate_weight(x - f_shift[0], xichmas[0])

            # 2. High Conditioned Elliptic Function F11’
            z1 = operator.dot_mv(f_matrix[x.shape[0] : 2 * x.shape[0], :], x - f_shift[1])
            g1 = lamdas[1] * operator.elliptic_func(z1) + bias[1]
            w1 = operator.calculate_weight(x - f_shift[1], xichmas[1])

            # 3. Girewank Function F15’
            z2 = operator.dot_mv(f_matrix[2 * x.shape[0] : 3 * x.shape[0], :], x - f_shift[2])
            g2 = lamdas[2] * operator.griewank_func(z2) + bias[2]
            w2 = operator.calculate_weight(x - f_shift[2], xichmas[2])

            # 4. Rastrigin’s Function F5’
            z3 = operator.dot_mv(f_matrix[3 * x.shape[0] : 4 * x.shape[0], :], x - f_shift[3])
            g3 = lamdas[3] * operator.rastrigin_func(z3) + bias[3]
            w3 = operator.calculate_weight(x - f_shift[3], xichmas[3])

            ws = np.array([w0, w1, w2, w3])
            ws = ws / np.sum(ws)
            gs = np.array([g0, g1, g2, g3])
            out[0] = operator.dot_vv(ws, gs) + f_bias

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
        self.n_funcs = 4
        self.xichmas = [10, 20, 30, 40]
        self.lamdas = [10.0, 1e-6, 10, 1.0]
        self.bias = [0, 100, 200, 300]
        self.g0 = operator.ackley_func
        self.g1 = operator.elliptic_func
        self.g2 = operator.griewank_func
        self.g3 = operator.rastrigin_func
        self._bind_kernel(
            compute,
            ["f_matrix", "f_shift", "lamdas", "bias", "xichmas", "f_bias"],
            paras={"f_shift": self.f_shift, "f_bias": self.f_bias, "f_matrix": self.f_matrix},
        )


class F242017(F202017):
    """
    .. [1] Problem Definitions and Evaluation Criteria for the CEC 2017
    Special Session and Competition on Single Objective Real-Parameter Numerical Optimization
    """

    name = "F24: Composition Function 5"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = bias = 2400.0"

    modality = True  # Number of ambiguous peaks, unknown # peaks
    # n_basins = 1
    # n_valleys = 1
    characteristics = ["Asymmetrical", "Different properties around different local optima"]

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "shift_data_25",
        f_matrix: typing.Any = "M_25_D",
        f_bias: float = 2400.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
        shift: typing.Any = None,
        rotate: typing.Any = None,
        rotate_bounds: bool = True,
    ) -> None:
        def compute(
            x: np.ndarray,
            f_matrix: typing.Any,
            f_shift: typing.Any,
            lamdas: typing.Any,
            bias: typing.Any,
            xichmas: typing.Any,
            f_bias: typing.Any,
            out: np.ndarray,
        ) -> None:
            z0 = operator.dot_mv(f_matrix[: x.shape[0], :], x - f_shift[0])
            g0 = lamdas[0] * operator.rastrigin_func(z0) + bias[0]
            w0 = operator.calculate_weight(x - f_shift[0], xichmas[0])

            # 2. Happycat Function F17’
            z1 = operator.dot_mv(f_matrix[x.shape[0] : 2 * x.shape[0], :], x - f_shift[0])
            g1 = lamdas[1] * operator.happy_cat_func(z1) + bias[1]
            w1 = operator.calculate_weight(x - f_shift[1], xichmas[1])

            # 3. Ackley Function F13’
            z2 = operator.dot_mv(f_matrix[2 * x.shape[0] : 3 * x.shape[0], :], x - f_shift[0])
            g2 = lamdas[2] * operator.ackley_func(z2) + bias[2]
            w2 = operator.calculate_weight(x - f_shift[2], xichmas[2])

            # 4. Discus Function F12’
            z3 = operator.dot_mv(f_matrix[3 * x.shape[0] : 4 * x.shape[0], :], x - f_shift[0])
            g3 = lamdas[3] * operator.discus_func(z3) + bias[3]
            w3 = operator.calculate_weight(x - f_shift[3], xichmas[3])

            # 5. Rosenbrock’s Function F4’
            z4 = operator.dot_mv(f_matrix[4 * x.shape[0] : 5 * x.shape[0], :], 2.048 * (x - f_shift[0]) / 100) + 1
            g4 = lamdas[4] * operator.rosenbrock_func(z4) + bias[4]
            w4 = operator.calculate_weight(x - f_shift[4], xichmas[4])

            ws = np.array([w0, w1, w2, w3, w4])
            ws = ws / np.sum(ws)
            gs = np.array([g0, g1, g2, g3, g4])
            out[0] = operator.dot_vv(ws, gs) + f_bias

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
        self.n_funcs = 5
        self.xichmas = [10, 20, 30, 40, 50]
        self.lamdas = [10.0, 1.0, 10.0, 1e-6, 1.0]
        self.bias = [0, 100, 200, 300, 400]
        self.g0 = operator.rastrigin_func
        self.g1 = operator.happy_cat_func
        self.g2 = operator.ackley_func
        self.g3 = operator.discus_func
        self.g4 = operator.rosenbrock_func
        self._bind_kernel(
            compute,
            ["f_matrix", "f_shift", "lamdas", "bias", "xichmas", "f_bias"],
            paras={"f_shift": self.f_shift, "f_bias": self.f_bias, "f_matrix": self.f_matrix},
        )


class F252017(F202017):
    """
    .. [1] Problem Definitions and Evaluation Criteria for the CEC 2017
    Special Session and Competition on Single Objective Real-Parameter Numerical Optimization
    """

    name = "F25: Composition Function 6"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = bias = 2500.0"

    modality = True  # Number of ambiguous peaks, unknown # peaks
    # n_basins = 1
    # n_valleys = 1
    characteristics = ["Asymmetrical", "Different properties around different local optima"]

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "shift_data_26",
        f_matrix: typing.Any = "M_26_D",
        f_bias: float = 2500.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
        shift: typing.Any = None,
        rotate: typing.Any = None,
        rotate_bounds: bool = True,
    ) -> None:
        def compute(
            x: np.ndarray,
            f_matrix: typing.Any,
            f_shift: typing.Any,
            lamdas: typing.Any,
            bias: typing.Any,
            xichmas: typing.Any,
            f_bias: typing.Any,
            out: np.ndarray,
        ) -> None:
            z0 = operator.dot_mv(f_matrix[: x.shape[0], :], x - f_shift[0]) + 1
            g0 = lamdas[0] * operator.expanded_scaffer_f6_func(z0) + bias[0]
            w0 = operator.calculate_weight(x - f_shift[0], xichmas[0])

            # 2. Modified Schwefel's Function F10’
            # z1 = operator.dot_mv(f_matrix[x.shape[0]:2*x.shape[0], :], x - f_shift[1])
            z1 = operator.dot_mv(f_matrix[x.shape[0] : 2 * x.shape[0], :], 1000 * (x - f_shift[0]) / 100)
            g1 = lamdas[1] * operator.modified_schwefel_func(z1) + bias[1]
            w1 = operator.calculate_weight(x - f_shift[1], xichmas[1])

            # 3. Griewank’s Function F15’
            z2 = operator.dot_mv(f_matrix[2 * x.shape[0] : 3 * x.shape[0], :], 600 * (x - f_shift[0]) / 100)
            g2 = lamdas[2] * operator.griewank_func(z2) + bias[2]
            w2 = operator.calculate_weight(x - f_shift[2], xichmas[2])

            # 4. Rosenbrock’s Function F4’
            z3 = operator.dot_mv(f_matrix[3 * x.shape[0] : 4 * x.shape[0], :], 2.048 * (x - f_shift[0]) / 100) + 1
            g3 = lamdas[3] * operator.rosenbrock_func(z3) + bias[3]
            w3 = operator.calculate_weight(x - f_shift[3], xichmas[3])

            # 5. Rastrigin’s Function F5’
            z4 = operator.dot_mv(f_matrix[4 * x.shape[0] : 5 * x.shape[0], :], x - f_shift[0])
            g4 = lamdas[4] * operator.rastrigin_func(z4) + bias[4]
            w4 = operator.calculate_weight(x - f_shift[4], xichmas[4])

            ws = np.array([w0, w1, w2, w3, w4])
            ws = ws / np.sum(ws)
            gs = np.array([g0, g1, g2, g3, g4])
            out[0] = operator.dot_vv(ws, gs) + f_bias

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
        self.n_funcs = 5
        self.xichmas = [10, 20, 20, 30, 40]
        self.lamdas = [1e-26, 10, 1e-6, 10, 5e-4]
        self.bias = [0, 100, 200, 300, 400]
        self.g0 = operator.expanded_scaffer_f6_func
        self.g1 = operator.modified_schwefel_func
        self.g2 = operator.griewank_func
        self.g3 = operator.rosenbrock_func
        self.g4 = operator.rastrigin_func
        self._bind_kernel(
            compute,
            ["f_matrix", "f_shift", "lamdas", "bias", "xichmas", "f_bias"],
            paras={"f_shift": self.f_shift, "f_bias": self.f_bias, "f_matrix": self.f_matrix},
        )


class F262017(F202017):
    """
    .. [1] Problem Definitions and Evaluation Criteria for the CEC 2017
    Special Session and Competition on Single Objective Real-Parameter Numerical Optimization
    """

    name = "F26: Composition Function 7"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = bias = 2600.0"

    modality = True  # Number of ambiguous peaks, unknown # peaks
    # n_basins = 1
    # n_valleys = 1
    characteristics = ["Asymmetrical", "Different properties around different local optima"]

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "shift_data_27",
        f_matrix: typing.Any = "M_27_D",
        f_bias: float = 2600.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
        shift: typing.Any = None,
        rotate: typing.Any = None,
        rotate_bounds: bool = True,
    ) -> None:
        def compute(
            x: np.ndarray,
            f_matrix: typing.Any,
            f_shift: typing.Any,
            lamdas: typing.Any,
            bias: typing.Any,
            xichmas: typing.Any,
            f_bias: typing.Any,
            out: np.ndarray,
        ) -> None:
            z0 = operator.dot_mv(f_matrix[: x.shape[0], :], 5 * (x - f_shift[0]) / 100)
            g0 = lamdas[0] * operator.hgbat_func(z0, shift=-1.0) + bias[0]
            w0 = operator.calculate_weight(x - f_shift[0], xichmas[0])

            # 2. Rastrigin’s Function F5’
            z1 = operator.dot_mv(f_matrix[x.shape[0] : 2 * x.shape[0], :], 5.12 * (x - f_shift[0]) / 100)
            g1 = lamdas[1] * operator.rastrigin_func(z1) + bias[1]
            w1 = operator.calculate_weight(x - f_shift[1], xichmas[1])

            # 3. Modified Schwefel's Function F10’
            z2 = operator.dot_mv(f_matrix[2 * x.shape[0] : 3 * x.shape[0], :], 1000 * (x - f_shift[0]) / 100)
            g2 = lamdas[2] * operator.modified_schwefel_func(z2) + bias[2]
            w2 = operator.calculate_weight(x - f_shift[2], xichmas[2])

            # 4. Bent-Cigar Function F11’
            z3 = operator.dot_mv(f_matrix[3 * x.shape[0] : 4 * x.shape[0], :], x - f_shift[0])
            g3 = lamdas[3] * operator.bent_cigar_func(z3) + bias[3]
            w3 = operator.calculate_weight(x - f_shift[3], xichmas[3])

            # 5. High Conditioned Elliptic Function F11’
            z4 = operator.dot_mv(f_matrix[4 * x.shape[0] : 5 * x.shape[0], :], x - f_shift[0])
            g4 = lamdas[4] * operator.elliptic_func(z4) + bias[4]
            w4 = operator.calculate_weight(x - f_shift[4], xichmas[4])

            # 6. Expanded Scaffer’s F6 Function F6’
            z5 = operator.dot_mv(f_matrix[5 * x.shape[0] : 6 * x.shape[0], :], x - f_shift[0]) + 1
            g5 = lamdas[5] * operator.expanded_scaffer_f6_func(z5) + bias[5]
            w5 = operator.calculate_weight(x - f_shift[5], xichmas[5])

            ws = np.array([w0, w1, w2, w3, w4, w5])
            ws = ws / np.sum(ws)
            gs = np.array([g0, g1, g2, g3, g4, g5])
            out[0] = operator.dot_vv(ws, gs) + f_bias

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
        self.n_funcs = 6
        self.xichmas = [
            10,
            20,
            30,
            40,
            50,
            60,
        ]
        self.lamdas = [10, 10, 2.5, 1e-26, 1e-6, 5e-4]
        self.bias = [0, 100, 200, 300, 400, 500]
        self._bind_kernel(
            compute,
            ["f_matrix", "f_shift", "lamdas", "bias", "xichmas", "f_bias"],
            paras={"f_shift": self.f_shift, "f_bias": self.f_bias, "f_matrix": self.f_matrix},
        )


class F272017(F202017):
    """
    .. [1] Problem Definitions and Evaluation Criteria for the CEC 2017
    Special Session and Competition on Single Objective Real-Parameter Numerical Optimization
    """

    name = "F27: Composition Function 8"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = bias = 2700.0"

    characteristics = ["Asymmetrical", "Different properties around different local optima"]

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "shift_data_28",
        f_matrix: typing.Any = "M_28_D",
        f_bias: float = 2700.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
        shift: typing.Any = None,
        rotate: typing.Any = None,
        rotate_bounds: bool = True,
    ) -> None:
        def compute(
            x: np.ndarray,
            f_matrix: typing.Any,
            f_shift: typing.Any,
            lamdas: typing.Any,
            bias: typing.Any,
            xichmas: typing.Any,
            f_bias: typing.Any,
            out: np.ndarray,
        ) -> None:
            z0 = operator.dot_mv(f_matrix[: x.shape[0], :], x - f_shift[0])
            g0 = lamdas[0] * operator.ackley_func(z0) + bias[0]
            w0 = operator.calculate_weight(x - f_shift[0], xichmas[0])

            # 2. Griewank’s Function F15’
            z1 = operator.dot_mv(f_matrix[x.shape[0] : 2 * x.shape[0], :], 600 * (x - f_shift[0]) / 100)
            g1 = lamdas[1] * operator.griewank_func(z1) + bias[1]
            w1 = operator.calculate_weight(x - f_shift[1], xichmas[1])

            # 3. Discus Function F12’
            z2 = operator.dot_mv(f_matrix[2 * x.shape[0] : 3 * x.shape[0], :], x - f_shift[0])
            g2 = lamdas[2] * operator.discus_func(z2) + bias[2]
            w2 = operator.calculate_weight(x - f_shift[2], xichmas[2])

            # 4. Rosenbrock’s Function F4’
            z3 = operator.dot_mv(f_matrix[3 * x.shape[0] : 4 * x.shape[0], :], 2.048 * (x - f_shift[0]) / 100) + 1
            g3 = lamdas[3] * operator.rosenbrock_func(z3) + bias[3]
            w3 = operator.calculate_weight(x - f_shift[3], xichmas[3])

            # 5. HappyCat Function F17’
            z4 = operator.dot_mv(f_matrix[4 * x.shape[0] : 5 * x.shape[0], :], 5 * (x - f_shift[0]) / 100)
            g4 = lamdas[4] * operator.happy_cat_func(z4) + bias[4]
            w4 = operator.calculate_weight(x - f_shift[4], xichmas[4])

            # 6. Expanded Scaffer’s F6 Function F6’
            z5 = operator.dot_mv(f_matrix[5 * x.shape[0] : 6 * x.shape[0], :], x - f_shift[0]) + 1
            g5 = lamdas[5] * operator.expanded_scaffer_f6_func(z5) + bias[5]
            w5 = operator.calculate_weight(x - f_shift[5], xichmas[5])

            ws = np.array([w0, w1, w2, w3, w4, w5])
            ws = ws / np.sum(ws)
            gs = np.array([g0, g1, g2, g3, g4, g5])
            out[0] = operator.dot_vv(ws, gs) + f_bias

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
        self.n_funcs = 6
        self.xichmas = [
            10,
            20,
            30,
            40,
            50,
            60,
        ]
        self.lamdas = [10, 10, 1e-6, 1, 1, 5e-4]
        self.bias = [0, 100, 200, 300, 400, 500]
        self.g0 = operator.ackley_func
        self.g1 = operator.griewank_func
        self.g2 = operator.discus_func
        self.g3 = operator.rosenbrock_func
        self.g4 = operator.happy_cat_func
        self.g5 = operator.expanded_scaffer_f6_func
        self._bind_kernel(
            compute,
            ["f_matrix", "f_shift", "lamdas", "bias", "xichmas", "f_bias"],
            paras={"f_shift": self.f_shift, "f_bias": self.f_bias, "f_matrix": self.f_matrix},
        )


class F282017(F202017):
    """
    .. [1] Problem Definitions and Evaluation Criteria for the CEC 2017
    Special Session and Competition on Single Objective Real-Parameter Numerical Optimization
    """

    name = "F28: Composition Function 9"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = bias = 2800.0"

    characteristics = [
        "Asymmetrical",
        "Different properties around different local optima",
        "Different properties for different variables subcomponents",
    ]

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "shift_data_29",
        f_matrix: typing.Any = "M_29_D",
        f_shuffle: typing.Any = "shuffle_data_29_D",
        f_bias: float = 2800.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
        shift: typing.Any = None,
        rotate: typing.Any = None,
        rotate_bounds: bool = True,
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

            # 2. Hybrid Function 6 F6’
            g1 = lamdas[1] * g1.evaluate(x) + bias[1]
            w1 = operator.calculate_weight(x - f_shift[1], xichmas[1])

            # 3. Hybrid Function 7 F7’
            g2 = lamdas[2] * g2.evaluate(x) + bias[2]
            w2 = operator.calculate_weight(x - f_shift[2], xichmas[2])

            ws = np.array([w0, w1, w2])
            ws = ws / np.sum(ws)
            gs = np.array([g0, g1, g2])
            out[0] = operator.dot_vv(ws, gs) + f_bias

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
        self.dim_supported = [10, 30, 50, 100]
        self.f_shuffle = self.check_shuffle_data(f_shuffle, needed_dim=True).reshape((10, -1))
        self.n_funcs = 3
        self.xichmas = [10, 30, 50]
        self.lamdas = [1.0, 1.0, 1.0]
        self.bias = [0, 100, 200]
        self.g0 = F142017(self.ndim, None, self.f_shift[0], self.f_matrix[: self.ndim, :], self.f_shuffle[0], 0)
        self.g1 = F152017(
            self.ndim, None, self.f_shift[0], self.f_matrix[self.ndim : 2 * self.ndim, :], self.f_shuffle[1], 0
        )
        self.g2 = F162017(
            self.ndim, None, self.f_shift[0], self.f_matrix[2 * self.ndim : 3 * self.ndim, :], self.f_shuffle[2], 0
        )
        self._bind_kernel(
            compute,
            ["lamdas", "g0", "bias", "f_shift", "xichmas", "g1", "g2", "f_bias"],
            paras={"f_shift": self.f_shift, "f_bias": self.f_bias, "f_matrix": self.f_matrix},
            plain=True,
        )


class F292017(F202017):
    """
    .. [1] Problem Definitions and Evaluation Criteria for the CEC 2017
    Special Session and Competition on Single Objective Real-Parameter Numerical Optimization
    """

    name = "F29: Composition Function 10"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = bias = 2900.0"

    characteristics = [
        "Asymmetrical",
        "Different properties around different local optima",
        "Different properties for different variables subcomponents",
    ]

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "shift_data_30",
        f_matrix: typing.Any = "M_30_D",
        f_shuffle: typing.Any = "shuffle_data_30_D",
        f_bias: float = 2900.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
        shift: typing.Any = None,
        rotate: typing.Any = None,
        rotate_bounds: bool = True,
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

            # 2. Hybrid Function 8 F8’
            g1 = lamdas[1] * g1.evaluate(x) + bias[1]
            w1 = operator.calculate_weight(x - f_shift[1], xichmas[1])

            # 3. Hybrid Function 9 F9’
            g2 = lamdas[2] * g2.evaluate(x) + bias[2]
            w2 = operator.calculate_weight(x - f_shift[2], xichmas[2])

            ws = np.array([w0, w1, w2])
            ws = ws / np.sum(ws)
            gs = np.array([g0, g1, g2])
            out[0] = operator.dot_vv(ws, gs) + f_bias

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
        self.dim_supported = [10, 30, 50, 100]
        self.f_shuffle = self.check_shuffle_data(f_shuffle, needed_dim=True).reshape((10, -1))
        self.n_funcs = 3
        self.xichmas = [10, 30, 50]
        self.lamdas = [1.0, 1.0, 1.0]
        self.bias = [0, 100, 200]
        self.g0 = F142017(self.ndim, None, self.f_shift[0], self.f_matrix[: self.ndim, :], self.f_shuffle[0], 0)
        self.g1 = F172017(
            self.ndim, None, self.f_shift[0], self.f_matrix[self.ndim : 2 * self.ndim, :], self.f_shuffle[1], 0
        )
        self.g2 = F182017(
            self.ndim, None, self.f_shift[0], self.f_matrix[2 * self.ndim : 3 * self.ndim, :], self.f_shuffle[2], 0
        )
        self._bind_kernel(
            compute,
            ["lamdas", "g0", "bias", "f_shift", "xichmas", "g1", "g2", "f_bias"],
            paras={"f_shift": self.f_shift, "f_bias": self.f_bias, "f_matrix": self.f_matrix},
            plain=True,
        )
