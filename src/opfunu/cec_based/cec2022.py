#!/usr/bin/env python
# Created by "Thieu" at 09:53, 13/07/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import typing

import numpy as np

from opfunu.benchmark.cec import CecBenchmark
from opfunu.utils import operator


class F12022(CecBenchmark):
    """
    .. [1] Problem Definitions and Evaluation Criteria for the CEC 2022
    Special Session and Competition on Single Objective Bound Constrained Numerical Optimization
    """

    name = "F1: Shifted and full Rotated Zakharov Function"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = bias = 300.0"

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
    characteristics = []

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "shift_data_1",
        f_matrix: typing.Any = "M_1_D",
        f_bias: float = 300.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
    ) -> None:
        def compute(
            x: np.ndarray, f_matrix: typing.Any, f_shift: typing.Any, f_bias: typing.Any, out: np.ndarray
        ) -> None:
            z = operator.dot_mv(f_matrix, x - f_shift)
            out[0] = operator.zakharov_func(z) + f_bias

        super().__init__(
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            dim_changeable=True,
            dim_default=10,
            dim_max=20,
            dim_supported=[2, 10, 20],
            f_bias=f_bias,
        )
        self.check_ndim_and_bounds(
            ndim, self.dim_max, bounds, np.array([[-100.0, 100.0] for _ in range(self.dim_default)])
        )
        self.make_support_data_path("data_2022")
        self.f_shift = self.check_shift_data(f_shift)[: self.ndim]
        self.f_matrix = self.check_matrix_data(f_matrix, needed_dim=True)
        self._x_global = np.asarray(self.f_shift)
        self._bind_kernel(
            compute,
            ["f_matrix", "f_shift", "f_bias"],
            paras={"f_shift": self.f_shift, "f_bias": self.f_bias, "f_matrix": self.f_matrix},
        )


class F22022(F12022):
    """
    .. [1] Problem Definitions and Evaluation Criteria for the CEC 2022
    Special Session and Competition on Single Objective Bound Constrained Numerical Optimization
    """

    name = "F2: Shifted and Rotated Rosenbrock’s Function"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = bias = 400.0"

    unimodal = False
    modality = True  # Number of ambiguous peaks, unknown # peaks
    characteristics = ["Local optima’s number is huge"]

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "shift_data_2",
        f_matrix: typing.Any = "M_2_D",
        f_bias: float = 400.0,
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


class F32022(F12022):
    """
    .. [1] Problem Definitions and Evaluation Criteria for the CEC 2022
    Special Session and Competition on Single Objective Bound Constrained Numerical Optimization
    """

    name = "F3: Shifted and full Rotated Expanded Schaffer’s F7"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = bias = 600.0"

    unimodal = False
    convex = False
    modality = True  # Number of ambiguous peaks, unknown # peaks
    characteristics = ["Asymmetrical", "Local optima’s number is huge"]

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "shift_data_3",
        f_matrix: typing.Any = "M_3_D",
        f_bias: float = 600.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
    ) -> None:
        def compute(
            x: np.ndarray, f_matrix: typing.Any, f_shift: typing.Any, f_bias: typing.Any, out: np.ndarray
        ) -> None:
            z = operator.dot_mv(f_matrix, 0.5 * (x - f_shift) / 100)
            out[0] = operator.rotated_expanded_schaffer_func(z) + f_bias

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


class F42022(F12022):
    """
    .. [1] Problem Definitions and Evaluation Criteria for the CEC 2022
    Special Session and Competition on Single Objective Bound Constrained Numerical Optimization
    """

    name = "F4: Shifted and Rotated Non-Continuous Rastrigin’s Function"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = bias = 800.0"

    unimodal = False
    convex = False
    modality = True  # Number of ambiguous peaks, unknown # peaks
    characteristics = ["Asymmetrical", "Local optima’s number is huge"]

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "shift_data_4",
        f_matrix: typing.Any = "M_4_D",
        f_bias: float = 800.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
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
        )
        self._bind_kernel(compute, ["f_matrix", "f_shift", "f_bias"])


class F52022(F12022):
    """
    .. [1] Problem Definitions and Evaluation Criteria for the CEC 2022
    Special Session and Competition on Single Objective Bound Constrained Numerical Optimization
    """

    name = "F5: Shifted and Rotated Levy Function"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = bias = 900.0"

    unimodal = False
    convex = False
    modality = True  # Number of ambiguous peaks, unknown # peaks
    characteristics = ["Local optima’s number is huge"]

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "shift_data_5",
        f_matrix: typing.Any = "M_5_D",
        f_bias: float = 900.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
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
        )
        self._bind_kernel(compute, ["f_matrix", "f_shift", "f_bias"])


class F62022(CecBenchmark):
    """
    .. [1] Problem Definitions and Evaluation Criteria for the CEC 2022
    Special Session and Competition on Single Objective Bound Constrained Numerical Optimization
    """

    name = "F6: Hybrid Function 1"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = bias = 1800.0"

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
        f_shift: typing.Any = "shift_data_6",
        f_matrix: typing.Any = "M_6_D",
        f_shuffle: typing.Any = "shuffle_data_6_D",
        f_bias: float = 1800.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
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
                operator.bent_cigar_func(mz[:n1])
                + operator.hgbat_func(mz[n1:n2], shift=-1.0)
                + operator.rastrigin_func(mz[n2:])
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
        )
        self.check_ndim_and_bounds(
            ndim, self.dim_max, bounds, np.array([[-100.0, 100.0] for _ in range(self.dim_default)])
        )
        self.make_support_data_path("data_2022")
        self.f_shift = self.check_shift_data(f_shift)[: self.ndim]
        self.f_matrix = self.check_matrix_data(f_matrix, needed_dim=True)
        self.f_shuffle = self.check_shuffle_data(f_shuffle, needed_dim=True)
        self.f_shuffle = (self.f_shuffle - 1).astype(int)
        self._x_global = np.asarray(self.f_shift)
        self.n_funcs = 3
        self.p = np.array([0.4, 0.4, 0.2])
        self.n1 = int(np.ceil(self.p[0] * self.ndim))
        self.n2 = int(np.ceil(self.p[1] * self.ndim)) + self.n1
        self.idx1, self.idx2, self.idx3 = (
            self.f_shuffle[: self.n1],
            self.f_shuffle[self.n1 : self.n2],
            self.f_shuffle[self.n2 : self.ndim],
        )
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


class F72022(CecBenchmark):
    """
    .. [1] Problem Definitions and Evaluation Criteria for the CEC 2022
    Special Session and Competition on Single Objective Bound Constrained Numerical Optimization
    """

    name = "F7: Hybrid Function 2"
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
        f_bias: float = 2000.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
    ) -> None:
        def compute(
            x: np.ndarray,
            f_shift: typing.Any,
            idx1: typing.Any,
            idx2: typing.Any,
            idx3: typing.Any,
            idx4: typing.Any,
            idx5: typing.Any,
            idx6: typing.Any,
            f_matrix: typing.Any,
            n1: int,
            n2: typing.Any,
            n3: typing.Any,
            n4: typing.Any,
            n5: typing.Any,
            f_bias: typing.Any,
            out: np.ndarray,
        ) -> None:
            z = x - f_shift
            z1 = np.concatenate((z[idx1], z[idx2], z[idx3], z[idx4], z[idx5], z[idx6]))
            mz = operator.dot_mv(f_matrix, z1)
            out[0] = (
                operator.hgbat_func(mz[:n1], shift=-1.0)
                + operator.katsuura_func(mz[n1:n2])
                + operator.ackley_func(mz[n2:n3])
                + operator.rastrigin_func(mz[n3:n4])
                + operator.modified_schwefel_func(mz[n4:n5])
                + operator.schaffer_f7_func(mz[n5 : x.shape[0]])
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
        )
        self.check_ndim_and_bounds(
            ndim, self.dim_max, bounds, np.array([[-100.0, 100.0] for _ in range(self.dim_default)])
        )
        self.make_support_data_path("data_2022")
        self.f_shift = self.check_shift_data(f_shift)[: self.ndim]
        self.f_matrix = self.check_matrix_data(f_matrix, needed_dim=True)
        self.f_shuffle = self.check_shuffle_data(f_shuffle, needed_dim=True)
        self.f_shuffle = (self.f_shuffle - 1).astype(int)
        self._x_global = np.asarray(self.f_shift)
        self.n_funcs = 6
        self.p = np.array([0.1, 0.2, 0.2, 0.2, 0.1, 0.2])
        self.n1 = int(np.ceil(self.p[0] * self.ndim))
        self.n2 = int(np.ceil(self.p[1] * self.ndim)) + self.n1
        self.n3 = int(np.ceil(self.p[2] * self.ndim)) + self.n2
        self.n4 = int(np.ceil(self.p[3] * self.ndim)) + self.n3
        self.n5 = int(np.ceil(self.p[4] * self.ndim)) + self.n4
        self.idx1, self.idx2, self.idx3 = (
            self.f_shuffle[: self.n1],
            self.f_shuffle[self.n1 : self.n2],
            self.f_shuffle[self.n2 : self.n3],
        )
        self.idx4, self.idx5, self.idx6 = (
            self.f_shuffle[self.n3 : self.n4],
            self.f_shuffle[self.n4 : self.n5],
            self.f_shuffle[self.n5 : self.ndim],
        )
        self._bind_kernel(
            compute,
            [
                "f_shift",
                "idx1",
                "idx2",
                "idx3",
                "idx4",
                "idx5",
                "idx6",
                "f_matrix",
                "n1",
                "n2",
                "n3",
                "n4",
                "n5",
                "f_bias",
            ],
            paras={
                "f_shift": self.f_shift,
                "f_bias": self.f_bias,
                "f_matrix": self.f_matrix,
                "f_shuffle": self.f_shuffle,
            },
        )


class F82022(CecBenchmark):
    """
    .. [1] Problem Definitions and Evaluation Criteria for the CEC 2022
    Special Session and Competition on Single Objective Bound Constrained Numerical Optimization
    """

    name = "F8: Hybrid Function 3"
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
    # n_basins = 1
    # n_valleys = 1

    characteristics = ["Different properties for different variables subcomponents"]

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "shift_data_8",
        f_matrix: typing.Any = "M_8_D",
        f_shuffle: typing.Any = "shuffle_data_8_D",
        f_bias: float = 2200.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
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
                operator.katsuura_func(mz[:n1])
                + operator.happy_cat_func(mz[n1:n2], shift=-1.0)
                + operator.grie_rosen_cec_func(mz[n2:n3])
                + operator.modified_schwefel_func(mz[n3:n4])
                + operator.ackley_func(mz[n4 : x.shape[0]])
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
        )
        self.check_ndim_and_bounds(
            ndim, self.dim_max, bounds, np.array([[-100.0, 100.0] for _ in range(self.dim_default)])
        )
        self.make_support_data_path("data_2022")
        self.f_shift = self.check_shift_data(f_shift)[: self.ndim]
        self.f_matrix = self.check_matrix_data(f_matrix, needed_dim=True)
        self.f_shuffle = self.check_shuffle_data(f_shuffle, needed_dim=True)
        self.f_shuffle = (self.f_shuffle - 1).astype(int)
        self._x_global = np.asarray(self.f_shift)
        self.n_funcs = 6
        self.p = np.array([0.3, 0.2, 0.2, 0.1, 0.2])
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


class F92022(CecBenchmark):
    """
    .. [1] Problem Definitions and Evaluation Criteria for the CEC 2022
    Special Session and Competition on Single Objective Bound Constrained Numerical Optimization
    """

    name = "F9: Composition Function 1"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = bias = 2300.0"

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
        f_bias: float = 2300.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
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

            # 2. High Conditioned Elliptic Function f8
            z1 = operator.dot_mv(f_matrix[x.shape[0] : 2 * x.shape[0], :], x - f_shift[0])
            g1 = lamdas[1] * operator.elliptic_func(z1) + bias[1]
            w1 = operator.calculate_weight(x - f_shift[1], xichmas[1])

            # 3. Rotated Bent Cigar Function f6
            z2 = operator.dot_mv(f_matrix[2 * x.shape[0] : 3 * x.shape[0], :], x - f_shift[0])
            g2 = lamdas[2] * operator.bent_cigar_func(z2) + bias[2]
            w2 = operator.calculate_weight(x - f_shift[2], xichmas[2])

            # 4. Rotated Discus Function f14
            z3 = operator.dot_mv(f_matrix[3 * x.shape[0] : 4 * x.shape[0], :], x - f_shift[0])
            g3 = lamdas[3] * operator.discus_func(z3) + bias[3]
            w3 = operator.calculate_weight(x - f_shift[3], xichmas[3])

            # 5. High Conditioned Elliptic Function f8
            z4 = operator.dot_mv(f_matrix[4 * x.shape[0] : 5 * x.shape[0], :], x - f_shift[0])
            g4 = lamdas[4] * operator.elliptic_func(z4) + bias[4]
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
        )
        self.check_ndim_and_bounds(
            ndim, self.dim_max, bounds, np.array([[-100.0, 100.0] for _ in range(self.dim_default)])
        )
        self.make_support_data_path("data_2022")
        self.f_shift = self.check_shift_matrix(f_shift)[:, : self.ndim]
        self.f_matrix = self.check_matrix_data(f_matrix)[:, : self.ndim]
        self._x_global = np.asarray(self.f_shift[0])
        self.n_funcs = 5
        self.xichmas = [10, 20, 30, 40, 50]  # aka delta in original CEC2022 logic
        self.lamdas = [1, 1e-6, 1e-6, 1e-6, 1e-6]
        self.bias = [0, 200, 300, 100, 400]
        self.g0 = operator.rosenbrock_func
        self.g1 = operator.elliptic_func
        self.g2 = operator.bent_cigar_func
        self.g3 = operator.discus_func
        self.g4 = operator.elliptic_func
        self._bind_kernel(
            compute,
            ["f_matrix", "f_shift", "lamdas", "bias", "xichmas", "f_bias"],
            paras={"f_shift": self.f_shift, "f_bias": self.f_bias, "f_matrix": self.f_matrix},
        )


class F102022(CecBenchmark):
    """
    .. [1] Problem Definitions and Evaluation Criteria for the CEC 2022
    Special Session and Competition on Single Objective Bound Constrained Numerical Optimization
    """

    name = "F10: Composition Function 2"
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
        f_shift: typing.Any = "shift_data_10",
        f_matrix: typing.Any = "M_10_D",
        f_bias: float = 2400.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
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
            z0 = operator.dot_mv(f_matrix[: x.shape[0], :], (1000.0 / 100) * (x - f_shift[0]))
            g0 = lamdas[0] * operator.modified_schwefel_func(z0) + bias[0]
            w0 = operator.calculate_weight(x - f_shift[0], xichmas[0])

            # 2. Rotated Rastrigin’s Function f4
            z1 = operator.dot_mv(f_matrix[x.shape[0] : 2 * x.shape[0], :], (5.12 / 100) * (x - f_shift[0]))
            g1 = lamdas[1] * operator.rastrigin_func(z1) + bias[1]
            w1 = operator.calculate_weight(x - f_shift[1], xichmas[1])

            # 3. HGBat Function f7
            z2 = operator.dot_mv(f_matrix[2 * x.shape[0] : 3 * x.shape[0], :], (5 / 100) * (x - f_shift[0]))
            g2 = lamdas[2] * operator.hgbat_func(z2, shift=-1.0) + bias[2]
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
        )
        self.check_ndim_and_bounds(
            ndim, self.dim_max, bounds, np.array([[-100.0, 100.0] for _ in range(self.dim_default)])
        )
        self.make_support_data_path("data_2022")
        self.f_shift = self.check_shift_matrix(f_shift)[:, : self.ndim]
        self.f_matrix = self.check_matrix_data(f_matrix)[:, : self.ndim]
        self._x_global = np.asarray(self.f_shift[0])
        self.n_funcs = 3
        self.xichmas = [20, 10, 10]
        self.lamdas = [1, 1, 1]
        self.bias = [0, 200, 100]
        self._bind_kernel(
            compute,
            ["f_matrix", "f_shift", "lamdas", "bias", "xichmas", "f_bias"],
            paras={"f_shift": self.f_shift, "f_bias": self.f_bias, "f_matrix": self.f_matrix},
        )


class F112022(CecBenchmark):
    """
    .. [1] Problem Definitions and Evaluation Criteria for the CEC 2022
    Special Session and Competition on Single Objective Bound Constrained Numerical Optimization
    """

    name = "F11: Composition Function 3"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = bias = 2600.0"

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
        f_shift: typing.Any = "shift_data_11",
        f_matrix: typing.Any = "M_11_D",
        f_bias: float = 2600.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
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
            z0 = operator.dot_mv(f_matrix[: x.shape[0], :], 0.5 * (x - f_shift[0]) / 100)
            g0 = lamdas[0] * operator.rotated_expanded_schaffer_func(z0) + bias[0]
            w0 = operator.calculate_weight(x - f_shift[0], xichmas[0])

            # 2. Modified Schwefel's Function f12
            z1 = operator.dot_mv(f_matrix[x.shape[0] : 2 * x.shape[0], :], 1000.0 * (x - f_shift[0]) / 100)
            g1 = lamdas[1] * operator.modified_schwefel_func(z1) + bias[1]
            w1 = operator.calculate_weight(x - f_shift[1], xichmas[1])

            # 3. Griewank’s Function f15
            z2 = operator.dot_mv(f_matrix[2 * x.shape[0] : 3 * x.shape[0], :], 600.0 * (x - f_shift[0]) / 100)
            g2 = lamdas[2] * operator.griewank_func(z2) + bias[2]
            w2 = operator.calculate_weight(x - f_shift[2], xichmas[2])

            # 4. Rosenbrock’s Function f2
            z3 = operator.dot_mv(f_matrix[3 * x.shape[0] : 4 * x.shape[0], :], 2.048 * (x - f_shift[0]) / 100)
            g3 = lamdas[3] * operator.rosenbrock_func(z3) + bias[3]
            w3 = operator.calculate_weight(x - f_shift[3], xichmas[3])

            # 5. Rastrigin’s Function f4
            z4 = operator.dot_mv(f_matrix[4 * x.shape[0] : 5 * x.shape[0], :], x - f_shift[0])
            g4 = lamdas[4] * operator.rastrigin_func(z4) + bias[4]
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
        )
        self.check_ndim_and_bounds(
            ndim, self.dim_max, bounds, np.array([[-100.0, 100.0] for _ in range(self.dim_default)])
        )
        self.make_support_data_path("data_2022")
        self.f_shift = self.check_shift_matrix(f_shift)[:, : self.ndim]
        self.f_matrix = self.check_matrix_data(f_matrix)[:, : self.ndim]
        self._x_global = np.asarray(self.f_shift[0])
        self.n_funcs = 5
        self.xichmas = [20, 20, 30, 30, 20]
        self.lamdas = [1e-26, 10, 1e-6, 10, 5e-4]
        self.bias = [0, 200, 300, 400, 200]
        self.g0 = operator.rotated_expanded_schaffer_func
        self.g1 = operator.modified_schwefel_func
        self.g2 = operator.griewank_func
        self.g3 = operator.rosenbrock_func
        self.g4 = operator.rastrigin_func
        self._bind_kernel(
            compute,
            ["f_matrix", "f_shift", "lamdas", "bias", "xichmas", "f_bias"],
            paras={"f_shift": self.f_shift, "f_bias": self.f_bias, "f_matrix": self.f_matrix},
        )


class F122022(CecBenchmark):
    """
    .. [1] Problem Definitions and Evaluation Criteria for the CEC 2022
    Special Session and Competition on Single Objective Bound Constrained Numerical Optimization
    """

    name = "F12: Composition Function 4"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = bias = 2700.0"

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
        f_shift: typing.Any = "shift_data_12",
        f_matrix: typing.Any = "M_12_D",
        f_bias: float = 2700.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
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
            z0 = operator.dot_mv(f_matrix[: x.shape[0], :], 5.0 * (x - f_shift[0]) / 100)
            g0 = lamdas[0] * operator.hgbat_func(z0, shift=-1.0) + bias[0]
            w0 = operator.calculate_weight(x - f_shift[0], xichmas[0])

            # 2. Rastrigin’s Function f4
            z1 = operator.dot_mv(f_matrix[x.shape[0] : 2 * x.shape[0], :], 5.12 * (x - f_shift[0]) / 100)
            g1 = lamdas[1] * operator.rastrigin_func(z1) + bias[1]
            w1 = operator.calculate_weight(x - f_shift[1], xichmas[1])

            # 3. Modified Schwefel's Function f12
            z2 = operator.dot_mv(f_matrix[2 * x.shape[0] : 3 * x.shape[0], :], 1000.0 * (x - f_shift[0]) / 100)
            g2 = lamdas[2] * operator.modified_schwefel_func(z2) + bias[2]
            w2 = operator.calculate_weight(x - f_shift[2], xichmas[2])

            # 4. Bent Cigar Function f6
            z3 = operator.dot_mv(f_matrix[3 * x.shape[0] : 4 * x.shape[0], :], x - f_shift[0])
            g3 = lamdas[3] * operator.bent_cigar_func(z3) + bias[3]
            w3 = operator.calculate_weight(x - f_shift[3], xichmas[3])

            # 5.  High Conditioned Elliptic Function f8
            z4 = operator.dot_mv(f_matrix[4 * x.shape[0] : 5 * x.shape[0], :], x - f_shift[0])
            g4 = lamdas[4] * operator.elliptic_func(z4) + bias[4]
            w4 = operator.calculate_weight(x - f_shift[4], xichmas[4])

            # 6.  Expanded Schaffer’s F6 Function f3
            z5 = operator.dot_mv(f_matrix[5 * x.shape[0] : 6 * x.shape[0], :], x - f_shift[0])
            g5 = lamdas[5] * operator.rotated_expanded_schaffer_func(z5) + bias[5]
            w5 = operator.calculate_weight(x - f_shift[5], xichmas[5])

            ws = np.array([w0, w1, w2, w3, w4, w5])
            ws = ws / np.sum(ws)
            gs = np.array([g0, g1, g2, g3, g4, g5])
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
        )
        self.check_ndim_and_bounds(
            ndim, self.dim_max, bounds, np.array([[-100.0, 100.0] for _ in range(self.dim_default)])
        )
        self.make_support_data_path("data_2022")
        self.f_shift = self.check_shift_matrix(f_shift)[:, : self.ndim]
        self.f_matrix = self.check_matrix_data(f_matrix)[:, : self.ndim]
        self._x_global = np.asarray(self.f_shift[0])
        self.n_funcs = 6
        self.xichmas = [10, 20, 30, 40, 50, 60]
        self.lamdas = [10, 10, 2.5, 1e-26, 1e-6, 5e-4]
        self.bias = [0, 300, 500, 100, 400, 200]
        self._bind_kernel(
            compute,
            ["f_matrix", "f_shift", "lamdas", "bias", "xichmas", "f_bias"],
            paras={"f_shift": self.f_shift, "f_bias": self.f_bias, "f_matrix": self.f_matrix},
        )
