#!/usr/bin/env python
# Created by "Thieu" at 09:25, 13/07/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import typing

import numpy as np

from opfunu.benchmark.cec import CecBenchmark
from opfunu.utils import operator


class F12021(CecBenchmark):
    """
    .. [1] Problem Definitions and Evaluation Criteria for the CEC 2021
    Special Session and Competition on Single Objective Bound Constrained Numerical Optimization
    """

    name = "F1: Shifted and Rotated Bent Cigar Function (F1 CEC-2017)"
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
            dim_default=10,
            dim_max=20,
            dim_supported=[2, 10, 20],
            f_bias=f_bias,
            shift=shift,
        )
        self.check_ndim_and_bounds(
            ndim, self.dim_max, bounds, np.array([[-100.0, 100.0] for _ in range(self.dim_default)])
        )
        self.make_support_data_path("data_2021")
        self.f_shift = self.check_shift_data(f_shift)[: self.ndim]
        self.f_matrix = self.check_matrix_data(f_matrix, needed_dim=True)
        self._x_global = np.asarray(self.f_shift)
        self._bind_kernel(
            compute,
            ["f_matrix", "f_shift", "f_bias"],
            paras={"f_shift": self.f_shift, "f_bias": self.f_bias, "f_matrix": self.f_matrix},
        )


class F22021(CecBenchmark):
    """
    .. [1] Problem Definitions and Evaluation Criteria for the CEC 2021
    Special Session and Competition on Single Objective Bound Constrained Numerical Optimization
    """

    name = "F2: Shifted and Rotated Schwefel’s Function (F11 CEC-2014)"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = bias = 1100.0"

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

    characteristics = ["Local optima’s number is huge", "The penultimate local optimum is far from the global optimum."]

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "shift_data_2",
        f_matrix: typing.Any = "M_2_D",
        f_bias: float = 1100.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
        shift: typing.Any = None,
    ) -> None:
        def compute(
            x: np.ndarray, f_matrix: typing.Any, f_shift: typing.Any, f_bias: typing.Any, out: np.ndarray
        ) -> None:
            z = operator.dot_mv(f_matrix, 1000.0 * (x - f_shift) / 100)
            out[0] = operator.modified_schwefel_func(z) + f_bias

        super().__init__(
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            dim_changeable=True,
            dim_default=10,
            dim_max=20,
            dim_supported=[2, 10, 20],
            f_bias=f_bias,
            shift=shift,
        )
        self.check_ndim_and_bounds(
            ndim, self.dim_max, bounds, np.array([[-100.0, 100.0] for _ in range(self.dim_default)])
        )
        self.make_support_data_path("data_2021")
        self.f_shift = self.check_shift_data(f_shift)[: self.ndim]
        self.f_matrix = self.check_matrix_data(f_matrix, needed_dim=True)
        self._x_global = np.asarray(self.f_shift)
        self._bind_kernel(
            compute,
            ["f_matrix", "f_shift", "f_bias"],
            paras={"f_shift": self.f_shift, "f_bias": self.f_bias, "f_matrix": self.f_matrix},
        )


class F32021(CecBenchmark):
    """
    .. [1] Problem Definitions and Evaluation Criteria for the CEC 2021
    Special Session and Competition on Single Objective Bound Constrained Numerical Optimization
    """

    name = "F3: Shifted and Rotated Lunacek bi-Rastrigin Function (F7 CEC-2017)"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = bias = 700.0"

    continuous = True
    linear = False
    convex = False
    unimodal = False
    separable = False

    differentiable = False
    scalable = True
    randomized_term = False
    parametric = True
    shifted = True
    rotated = True

    modality = False  # Number of ambiguous peaks, unknown # peaks
    # n_basins = 1
    # n_valleys = 1

    characteristics = ["Asymmetrical", "Continuous everywhere yet differentiable nowhere"]

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "shift_data_3",
        f_matrix: typing.Any = "M_3_D",
        f_bias: float = 700.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
        shift: typing.Any = None,
    ) -> None:
        def compute(
            x: np.ndarray, f_matrix: typing.Any, f_shift: typing.Any, f_bias: typing.Any, out: np.ndarray
        ) -> None:
            z = operator.dot_mv(f_matrix, 600.0 * (x - f_shift) / 100)
            out[0] = operator.lunacek_bi_rastrigin_func(z, shift=2.5) + f_bias

        super().__init__(
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            dim_changeable=True,
            dim_default=10,
            dim_max=20,
            dim_supported=[2, 10, 20],
            f_bias=f_bias,
            shift=shift,
        )
        self.check_ndim_and_bounds(
            ndim, self.dim_max, bounds, np.array([[-100.0, 100.0] for _ in range(self.dim_default)])
        )
        self.make_support_data_path("data_2021")
        self.f_shift = self.check_shift_data(f_shift)[: self.ndim]
        self.f_matrix = self.check_matrix_data(f_matrix, needed_dim=True)
        self._x_global = np.asarray(self.f_shift)
        self._bind_kernel(
            compute,
            ["f_matrix", "f_shift", "f_bias"],
            paras={"f_shift": self.f_shift, "f_bias": self.f_bias, "f_matrix": self.f_matrix},
        )


class F42021(CecBenchmark):
    """
    .. [1] Problem Definitions and Evaluation Criteria for the CEC 2021
    Special Session and Competition on Single Objective Bound Constrained Numerical Optimization
    """

    name = "F4: Expanded Rosenbrock’s plus Griewangk’s Function (F15 CEC-2014)"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = bias = 1900.0"

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

    characteristics = ["Optimal point locates in flat area"]

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "shift_data_4",
        f_matrix: typing.Any = "M_4_D",
        f_bias: float = 1900.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
        shift: typing.Any = None,
    ) -> None:
        def compute(
            x: np.ndarray, f_matrix: typing.Any, f_shift: typing.Any, f_bias: typing.Any, out: np.ndarray
        ) -> None:
            z = operator.dot_mv(f_matrix, 5.0 * (x - f_shift) / 100)
            out[0] = operator.expanded_griewank_rosenbrock_func(z) + f_bias

        super().__init__(
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            dim_changeable=True,
            dim_default=10,
            dim_max=20,
            dim_supported=[2, 10, 20],
            f_bias=f_bias,
            shift=shift,
        )
        self.check_ndim_and_bounds(
            ndim, self.dim_max, bounds, np.array([[-100.0, 100.0] for _ in range(self.dim_default)])
        )
        self.make_support_data_path("data_2021")
        self.f_shift = self.check_shift_data(f_shift)[: self.ndim]
        self.f_matrix = self.check_matrix_data(f_matrix, needed_dim=True)
        self._x_global = np.asarray(self.f_shift)
        self._bind_kernel(
            compute,
            ["f_matrix", "f_shift", "f_bias"],
            paras={"f_shift": self.f_shift, "f_bias": self.f_bias, "f_matrix": self.f_matrix},
        )


class F52021(CecBenchmark):
    """
    .. [1] Problem Definitions and Evaluation Criteria for the CEC 2021
    Special Session and Competition on Single Objective Bound Constrained Numerical Optimization
    """

    name = "F17: Hybrid Function 1 (F17 CEC-2014)"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = bias = 1700.0"

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

    characteristics = ["Different properties for different variables subcomponents"]

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "shift_data_5",
        f_matrix: typing.Any = "M_5_D",
        f_shuffle: typing.Any = "shuffle_data_5_D",
        f_bias: float = 1700.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
        shift: typing.Any = None,
    ) -> None:
        def compute(
            x: np.ndarray,
            f_shift: typing.Any,
            idx1: typing.Any,
            idx2: typing.Any,
            idx3: typing.Any,
            f_matrix: typing.Any,
            n1: int,
            n2: typing.Any,
            f_bias: typing.Any,
            out: np.ndarray,
        ) -> None:
            z = x - f_shift
            z1 = np.concatenate((z[idx1], z[idx2], z[idx3]))
            mz = operator.dot_mv(f_matrix, z1)
            out[0] = (
                operator.modified_schwefel_func(mz[:n1])
                + operator.rastrigin_func(mz[n1:n2])
                + operator.elliptic_func(mz[n2:])
                + f_bias
            )

        super().__init__(
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            dim_changeable=True,
            dim_default=10,
            dim_max=20,
            dim_supported=[10, 20],
            f_bias=f_bias,
            shift=shift,
        )
        self.check_ndim_and_bounds(
            ndim, self.dim_max, bounds, np.array([[-100.0, 100.0] for _ in range(self.dim_default)])
        )
        self.make_support_data_path("data_2021")
        self.f_shift = self.check_shift_data(f_shift)[: self.ndim]
        self.f_matrix = self.check_matrix_data(f_matrix, needed_dim=True)
        self.f_shuffle = self.check_shuffle_data(f_shuffle, needed_dim=True)
        self.f_shuffle = (self.f_shuffle - 1).astype(int)
        self._x_global = np.asarray(self.f_shift)
        self.n_funcs = 3
        self.p = np.array([0.3, 0.3, 0.4])
        self.n1 = int(np.ceil(self.p[0] * self.ndim))
        self.n2 = int(np.ceil(self.p[1] * self.ndim)) + self.n1
        self.idx1, self.idx2, self.idx3 = (
            self.f_shuffle[: self.n1],
            self.f_shuffle[self.n1 : self.n2],
            self.f_shuffle[self.n2 : self.ndim],
        )
        self.g1 = operator.modified_schwefel_func
        self.g2 = operator.rastrigin_func
        self.g3 = operator.elliptic_func
        self._bind_kernel(
            compute,
            ["f_shift", "idx1", "idx2", "idx3", "f_matrix", "n1", "n2", "f_bias"],
            paras={
                "f_shift": self.f_shift,
                "f_bias": self.f_bias,
                "f_matrix": self.f_matrix,
                "f_shuffle": self.f_shuffle,
            },
        )


class F62021(CecBenchmark):
    """
    .. [1] Problem Definitions and Evaluation Criteria for the CEC 2021
    Special Session and Competition on Single Objective Bound Constrained Numerical Optimization
    """

    name = "F16: Hybrid Function 2 (F15 CEC-2017)"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = bias = 1600.0"

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
        f_shift: typing.Any = "shift_data_6",
        f_matrix: typing.Any = "M_6_D",
        f_shuffle: typing.Any = "shuffle_data_6_D",
        f_bias: float = 1600.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
        shift: typing.Any = None,
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
                operator.expanded_schaffer_f6_func(mz[idx1])
                + operator.hgbat_func(mz[idx2], shift=-1.0)
                + operator.rosenbrock_func(mz[idx3], shift=1.0)
                + operator.modified_schwefel_func(mz[idx4])
                + f_bias
            )

        super().__init__(
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            dim_changeable=True,
            dim_default=10,
            dim_max=20,
            dim_supported=[10, 20],
            f_bias=f_bias,
            shift=shift,
        )
        self.check_ndim_and_bounds(
            ndim, self.dim_max, bounds, np.array([[-100.0, 100.0] for _ in range(self.dim_default)])
        )
        self.make_support_data_path("data_2021")
        self.f_shift = self.check_shift_data(f_shift)[: self.ndim]
        self.f_matrix = self.check_matrix_data(f_matrix, needed_dim=True)
        self.f_shuffle = self.check_shuffle_data(f_shuffle, needed_dim=True)
        self.f_shuffle = (self.f_shuffle - 1).astype(int)
        self._x_global = np.asarray(self.f_shift)
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


class F72021(CecBenchmark):
    """
    .. [1] Problem Definitions and Evaluation Criteria for the CEC 2021
    Special Session and Competition on Single Objective Bound Constrained Numerical Optimization
    """

    name = "F7: Hybrid Function 3 (F21 CEC-2014)"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = bias = 2100.0"

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

    characteristics = ["Different properties for different variables subcomponents"]

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "shift_data_7",
        f_matrix: typing.Any = "M_7_D",
        f_shuffle: typing.Any = "shuffle_data_7_D",
        f_bias: float = 2100.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
        shift: typing.Any = None,
    ) -> None:
        def compute(
            x: np.ndarray,
            f_shift: typing.Any,
            idx1: typing.Any,
            idx2: typing.Any,
            idx3: typing.Any,
            idx4: typing.Any,
            idx5: typing.Any,
            f_matrix: typing.Any,
            n1: int,
            n2: typing.Any,
            n3: typing.Any,
            n4: typing.Any,
            f_bias: typing.Any,
            out: np.ndarray,
        ) -> None:
            z = x - f_shift
            z1 = np.concatenate((z[idx1], z[idx2], z[idx3], z[idx4], z[idx5]))
            mz = operator.dot_mv(f_matrix, z1)
            out[0] = (
                operator.expanded_schaffer_f6_func(mz[:n1])
                + operator.hgbat_func(mz[n1:n2], shift=-1.0)
                + operator.rosenbrock_func(mz[n2:n3], shift=1.0)
                + operator.modified_schwefel_func(mz[n3:n4])
                + operator.elliptic_func(mz[n4:])
                + f_bias
            )

        super().__init__(
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            dim_changeable=True,
            dim_default=10,
            dim_max=20,
            dim_supported=[10, 20],
            f_bias=f_bias,
            shift=shift,
        )
        self.check_ndim_and_bounds(
            ndim, self.dim_max, bounds, np.array([[-100.0, 100.0] for _ in range(self.dim_default)])
        )
        self.make_support_data_path("data_2021")
        self.f_shift = self.check_shift_data(f_shift)[: self.ndim]
        self.f_matrix = self.check_matrix_data(f_matrix, needed_dim=True)
        self.f_shuffle = self.check_shuffle_data(f_shuffle, needed_dim=True)
        self.f_shuffle = (self.f_shuffle - 1).astype(int)
        self._x_global = np.asarray(self.f_shift)
        self.n_funcs = 5
        self.p = np.array([0.1, 0.2, 0.2, 0.2, 0.3])
        self.n1 = int(np.ceil(self.p[0] * self.ndim))
        self.n2 = int(np.ceil(self.p[1] * self.ndim)) + self.n1
        self.n3 = int(np.ceil(self.p[2] * self.ndim)) + self.n2
        self.n4 = int(np.ceil(self.p[3] * self.ndim)) + self.n3
        self.idx1, self.idx2, self.idx3 = (
            self.f_shuffle[: self.n1],
            self.f_shuffle[self.n1 : self.n2],
            self.f_shuffle[self.n2 : self.n3],
        )
        self.idx4, self.idx5 = self.f_shuffle[self.n3 : self.n4], self.f_shuffle[self.n4 : self.ndim]
        self._bind_kernel(
            compute,
            ["f_shift", "idx1", "idx2", "idx3", "idx4", "idx5", "f_matrix", "n1", "n2", "n3", "n4", "f_bias"],
            paras={
                "f_shift": self.f_shift,
                "f_bias": self.f_bias,
                "f_matrix": self.f_matrix,
                "f_shuffle": self.f_shuffle,
            },
        )


class F82021(CecBenchmark):
    """
    .. [1] Problem Definitions and Evaluation Criteria for the CEC 2021
    Special Session and Competition on Single Objective Bound Constrained Numerical Optimization
    """

    name = "F8: Composition Function 1 (F21 CEC-2017)"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = bias = 2200.0"

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
    characteristics = ["Asymmetrical", "Different properties around different local optima"]

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "shift_data_8",
        f_matrix: typing.Any = "M_8_D",
        f_bias: float = 2200.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
        shift: typing.Any = None,
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
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            dim_changeable=True,
            dim_default=10,
            dim_max=20,
            dim_supported=[2, 10, 20],
            f_bias=f_bias,
            shift=shift,
        )
        self.check_ndim_and_bounds(
            ndim, self.dim_max, bounds, np.array([[-100.0, 100.0] for _ in range(self.dim_default)])
        )
        self.make_support_data_path("data_2021")
        self.f_shift = self.check_shift_matrix(f_shift)[:, : self.ndim]
        self.f_matrix = self.check_matrix_data(f_matrix)[:, : self.ndim]
        self._x_global = np.asarray(self.f_shift[0])
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


class F92021(CecBenchmark):
    """
    .. [1] Problem Definitions and Evaluation Criteria for the CEC 2021
    Special Session and Competition on Single Objective Bound Constrained Numerical Optimization
    """

    name = "F9: Composition Function 2 (F23 CEC-2017)"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = bias = 2400.0"

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
        f_shift: typing.Any = "shift_data_9",
        f_matrix: typing.Any = "M_9_D",
        f_bias: float = 2400.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
        shift: typing.Any = None,
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
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            dim_changeable=True,
            dim_default=10,
            dim_max=20,
            dim_supported=[2, 10, 20],
            f_bias=f_bias,
            shift=shift,
        )
        self.check_ndim_and_bounds(
            ndim, self.dim_max, bounds, np.array([[-100.0, 100.0] for _ in range(self.dim_default)])
        )
        self.make_support_data_path("data_2021")
        self.f_shift = self.check_shift_matrix(f_shift)[:, : self.ndim]
        self.f_matrix = self.check_matrix_data(f_matrix)[:, : self.ndim]
        self._x_global = np.asarray(self.f_shift[0])
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


class F102021(CecBenchmark):
    """
    .. [1] Problem Definitions and Evaluation Criteria for the CEC 2021
    Special Session and Competition on Single Objective Bound Constrained Numerical Optimization
    """

    name = "F10: Composition Function 3 (F24 CEC-2017)"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = bias = 2500.0"

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
        f_shift: typing.Any = "shift_data_10",
        f_matrix: typing.Any = "M_10_D",
        f_bias: float = 2500.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
        shift: typing.Any = None,
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
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            dim_changeable=True,
            dim_default=10,
            dim_max=20,
            dim_supported=[2, 10, 20],
            f_bias=f_bias,
            shift=shift,
        )
        self.check_ndim_and_bounds(
            ndim, self.dim_max, bounds, np.array([[-100.0, 100.0] for _ in range(self.dim_default)])
        )
        self.make_support_data_path("data_2021")
        self.f_shift = self.check_shift_matrix(f_shift)[:, : self.ndim]
        self.f_matrix = self.check_matrix_data(f_matrix)[:, : self.ndim]
        self._x_global = np.asarray(self.f_shift[0])
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
