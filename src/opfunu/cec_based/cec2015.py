#!/usr/bin/env python
# Created by "Thieu" at 14:45, 07/07/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import typing

import numpy as np

from opfunu.benchmark.cec import CecBenchmark
from opfunu.utils import operator


class F12015(CecBenchmark):
    """
    .. [1] Chen, Q., Liu, B., Zhang, Q., Liang, J., Suganthan, P., & Qu, B. (2014). Problem definitions and evaluation criteria for CEC 2015
    special session on bound constrained single-objective computationally expensive numerical optimization. Technical Report,
    Computational Intelligence Laboratory, Zhengzhou University, Zhengzhou, China and Technical Report, Nanyang Technological University.
    """

    name = "F1: Rotated Bent Cigar Function"
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
        f_shift: typing.Any = "shift_data_1_D",
        f_matrix: typing.Any = "M_1_D",
        f_bias: float = 100.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
    ) -> None:
        def compute(
            x: np.ndarray, f_matrix: typing.Any, f_shift: typing.Any, f_bias: typing.Any, out: np.ndarray
        ) -> None:
            z = operator.dot_mv(f_matrix, x - f_shift)
            out[0] = operator.bent_cigar_func(z) + f_bias

        super().__init__(parallel=parallel, fastmath=fastmath, dtype=dtype)
        self.dim_changeable = True
        self.dim_default = 10
        self.dim_max = 30
        self.dim_supported = [10, 30]
        self.check_ndim_and_bounds(
            ndim, self.dim_max, bounds, np.array([[-100.0, 100.0] for _ in range(self.dim_default)])
        )
        self.make_support_data_path("data_2015")
        self.f_shift = self.check_matrix_data(f_shift, needed_dim=True).ravel()
        self.f_matrix = self.check_matrix_data(f_matrix, needed_dim=True)
        self.f_bias = f_bias
        self.f_global = f_bias
        self.x_global = self.f_shift
        self._bind_kernel(
            compute,
            ["f_matrix", "f_shift", "f_bias"],
            paras={"f_shift": self.f_shift, "f_bias": self.f_bias, "f_matrix": self.f_matrix},
        )


class F22015(F12015):
    """
    .. [1] Chen, Q., Liu, B., Zhang, Q., Liang, J., Suganthan, P., & Qu, B. (2014). Problem definitions and evaluation criteria for CEC 2015
    special session on bound constrained single-objective computationally expensive numerical optimization. Technical Report,
    Computational Intelligence Laboratory, Zhengzhou University, Zhengzhou, China and Technical Report, Nanyang Technological University.
    """

    name = "F2: Rotated Discus Function"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = bias = 200.0"

    characteristics = ["With one sensitive direction"]

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "shift_data_2_D",
        f_matrix: typing.Any = "M_2_D",
        f_bias: float = 200.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
    ) -> None:
        def compute(
            x: np.ndarray, f_matrix: typing.Any, f_shift: typing.Any, f_bias: typing.Any, out: np.ndarray
        ) -> None:
            z = operator.dot_mv(f_matrix, x - f_shift)
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


class F32015(F12015):
    """
    .. [1] Chen, Q., Liu, B., Zhang, Q., Liang, J., Suganthan, P., & Qu, B. (2014). Problem definitions and evaluation criteria for CEC 2015
    special session on bound constrained single-objective computationally expensive numerical optimization. Technical Report,
    Computational Intelligence Laboratory, Zhengzhou University, Zhengzhou, China and Technical Report, Nanyang Technological University.
    """

    name = "F3: Shifted and Rotated Weierstrass Function"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = bias = 300.0"

    convex = False
    unimodal = False

    characteristics = ["Continuous but differentiable only on a set of points"]

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "shift_data_3_D",
        f_matrix: typing.Any = "M_3_D",
        f_bias: float = 300.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
    ) -> None:
        def compute(
            x: np.ndarray, f_matrix: typing.Any, f_shift: typing.Any, f_bias: typing.Any, out: np.ndarray
        ) -> None:
            z = operator.dot_mv(f_matrix, 0.5 * (x - f_shift) / 100)
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
        )
        self._bind_kernel(compute, ["f_matrix", "f_shift", "f_bias"])


class F42015(F12015):
    """
    .. [1] Chen, Q., Liu, B., Zhang, Q., Liang, J., Suganthan, P., & Qu, B. (2014). Problem definitions and evaluation criteria for CEC 2015
    special session on bound constrained single-objective computationally expensive numerical optimization. Technical Report,
    Computational Intelligence Laboratory, Zhengzhou University, Zhengzhou, China and Technical Report, Nanyang Technological University.
    """

    name = "F4: Shifted and Rotated Schwefel’s Function"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = bias = 400.0"

    convex = False
    unimodal = False
    modality = True

    characteristics = [
        "Local optima’s number is huge",
        "The second better local optimum is far from the global optimum",
    ]

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "shift_data_4_D",
        f_matrix: typing.Any = "M_4_D",
        f_bias: float = 400.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
    ) -> None:
        def compute(
            x: np.ndarray, f_matrix: typing.Any, f_shift: typing.Any, f_bias: typing.Any, out: np.ndarray
        ) -> None:
            z = operator.dot_mv(f_matrix, 1000 * (x - f_shift) / 100)
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


class F52015(F12015):
    """
    .. [1] Chen, Q., Liu, B., Zhang, Q., Liang, J., Suganthan, P., & Qu, B. (2014). Problem definitions and evaluation criteria for CEC 2015
    special session on bound constrained single-objective computationally expensive numerical optimization. Technical Report,
    Computational Intelligence Laboratory, Zhengzhou University, Zhengzhou, China and Technical Report, Nanyang Technological University.
    """

    name = "F5: Shifted and Rotated Katsuura Function"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = bias = 500.0"

    convex = False
    unimodal = False
    differentiable = False
    modality = True

    characteristics = ["Continuous everywhere yet differentiable nowhere"]

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "shift_data_5_D",
        f_matrix: typing.Any = "M_5_D",
        f_bias: float = 500.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
    ) -> None:
        def compute(
            x: np.ndarray, f_matrix: typing.Any, f_shift: typing.Any, f_bias: typing.Any, out: np.ndarray
        ) -> None:
            z = operator.dot_mv(f_matrix, 5 * (x - f_shift) / 100)
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


class F62015(F12015):
    """
    .. [1] Chen, Q., Liu, B., Zhang, Q., Liang, J., Suganthan, P., & Qu, B. (2014). Problem definitions and evaluation criteria for CEC 2015
    special session on bound constrained single-objective computationally expensive numerical optimization. Technical Report,
    Computational Intelligence Laboratory, Zhengzhou University, Zhengzhou, China and Technical Report, Nanyang Technological University.
    """

    name = "F6: Shifted and Rotated HappyCat Function"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = bias = 600.0"

    convex = False
    unimodal = False
    separable = False
    differentiable = False

    characteristics = ["Continuous everywhere yet differentiable nowhere"]

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "shift_data_6_D",
        f_matrix: typing.Any = "M_6_D",
        f_bias: float = 600.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
    ) -> None:
        def compute(
            x: np.ndarray, f_matrix: typing.Any, f_shift: typing.Any, f_bias: typing.Any, out: np.ndarray
        ) -> None:
            z = operator.dot_mv(f_matrix, 5 * (x - f_shift) / 100)
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
        )
        self._bind_kernel(compute, ["f_matrix", "f_shift", "f_bias"])


class F72015(F12015):
    """
    .. [1] Chen, Q., Liu, B., Zhang, Q., Liang, J., Suganthan, P., & Qu, B. (2014). Problem definitions and evaluation criteria for CEC 2015
    special session on bound constrained single-objective computationally expensive numerical optimization. Technical Report,
    Computational Intelligence Laboratory, Zhengzhou University, Zhengzhou, China and Technical Report, Nanyang Technological University.
    """

    name = "F7: Shifted and Rotated HGBat Function"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = bias = 700.0"

    unimodal = False
    separable = False
    differentiable = True

    characteristics = []

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "shift_data_7_D",
        f_matrix: typing.Any = "M_7_D",
        f_bias: float = 700.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
    ) -> None:
        def compute(
            x: np.ndarray, f_matrix: typing.Any, f_shift: typing.Any, f_bias: typing.Any, out: np.ndarray
        ) -> None:
            z = operator.dot_mv(f_matrix, 5 * (x - f_shift) / 100)
            out[0] = operator.hgbat_func(z, shift=-1.0) + f_bias

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


class F82015(F12015):
    """
    .. [1] Chen, Q., Liu, B., Zhang, Q., Liang, J., Suganthan, P., & Qu, B. (2014). Problem definitions and evaluation criteria for CEC 2015
    special session on bound constrained single-objective computationally expensive numerical optimization. Technical Report,
    Computational Intelligence Laboratory, Zhengzhou University, Zhengzhou, China and Technical Report, Nanyang Technological University.
    """

    name = "F8: Shifted and Rotated Expanded Griewank’s plus Rosenbrock’s Function"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = bias = 800.0"

    unimodal = False
    separable = False
    differentiable = True

    characteristics = []

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "shift_data_8_D",
        f_matrix: typing.Any = "M_8_D",
        f_bias: float = 800.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
    ) -> None:
        def compute(
            x: np.ndarray, f_matrix: typing.Any, f_shift: typing.Any, f_bias: typing.Any, out: np.ndarray
        ) -> None:
            z = operator.dot_mv(f_matrix, 5.0 * (x - f_shift) / 100)
            out[0] = operator.expanded_griewank_rosenbrock_func(z) + f_bias

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


class F92015(F12015):
    """
    .. [1] Chen, Q., Liu, B., Zhang, Q., Liang, J., Suganthan, P., & Qu, B. (2014). Problem definitions and evaluation criteria for CEC 2015
    special session on bound constrained single-objective computationally expensive numerical optimization. Technical Report,
    Computational Intelligence Laboratory, Zhengzhou University, Zhengzhou, China and Technical Report, Nanyang Technological University.
    """

    name = "F9: Shifted and Rotated Expanded Scaffer’s F6 Function"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = bias = 900.0"

    unimodal = False
    convex = False
    separable = False
    differentiable = True
    modality = True
    characteristics = []

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "shift_data_9_D",
        f_matrix: typing.Any = "M_9_D",
        f_bias: float = 900.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
    ) -> None:
        def compute(
            x: np.ndarray, f_matrix: typing.Any, f_shift: typing.Any, f_bias: typing.Any, out: np.ndarray
        ) -> None:
            z = operator.dot_mv(f_matrix, x - f_shift)
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


class F102015(CecBenchmark):
    """
    .. [1] Liang, J. J., Qu, B. Y., & Suganthan, P. N. (2013). Problem definitions and evaluation criteria for the CEC 2014
    special session and competition on single objective real-parameter numerical optimization. Computational Intelligence Laboratory,
    Zhengzhou University, Zhengzhou China and Technical Report, Nanyang Technological University, Singapore, 635, 490.
    """

    name = "F10: Hybrid Function 1 (N=3)"
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
        f_shift: typing.Any = "shift_data_10_D",
        f_matrix: typing.Any = "M_10_D",
        f_shuffle: typing.Any = "shuffle_data_10_D",
        f_bias: float = 1000.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
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
                operator.modified_schwefel_func(mz[idx1])
                + operator.rastrigin_func(mz[idx2])
                + operator.elliptic_func(mz[idx3])
                + f_bias
            )

        super().__init__(parallel=parallel, fastmath=fastmath, dtype=dtype)
        self.dim_changeable = True
        self.dim_default = 10
        self.dim_max = 30
        self.dim_supported = [10, 30]
        self.check_ndim_and_bounds(
            ndim, self.dim_max, bounds, np.array([[-100.0, 100.0] for _ in range(self.dim_default)])
        )
        self.make_support_data_path("data_2015")
        self.f_shift = self.check_matrix_data(f_shift, needed_dim=True).ravel()
        self.f_matrix = self.check_matrix_data(f_matrix, needed_dim=True)
        self.f_shuffle = self.check_shuffle_data(f_shuffle, needed_dim=True)
        self.f_shuffle = (self.f_shuffle - 1).astype(int)
        self.f_bias = f_bias
        self.f_global = f_bias
        self.x_global = self.f_shift
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
            ["f_matrix", "f_shift", "idx1", "idx2", "idx3", "f_bias"],
            paras={
                "f_shift": self.f_shift,
                "f_bias": self.f_bias,
                "f_matrix": self.f_matrix,
                "f_shuffle": self.f_shuffle,
            },
        )


class F112015(F102015):
    """
    .. [1] Liang, J. J., Qu, B. Y., & Suganthan, P. N. (2013). Problem definitions and evaluation criteria for the CEC 2014
    special session and competition on single objective real-parameter numerical optimization. Computational Intelligence Laboratory,
    Zhengzhou University, Zhengzhou China and Technical Report, Nanyang Technological University, Singapore, 635, 490.
    """

    name = "F11: Hybrid Function 2 (N=4)"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = bias = 1100.0"

    characteristics = []

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "shift_data_11_D",
        f_matrix: typing.Any = "M_11_D",
        f_shuffle: typing.Any = "shuffle_data_11_D",
        f_bias: float = 1100.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
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
                operator.griewank_func(mz[idx1])
                + operator.weierstrass_func(mz[idx2])
                + operator.rosenbrock_func(mz[idx3], shift=1.0)
                + operator.expanded_scaffer_f6_func(mz[idx4])
                + f_bias
            )

        super().__init__(
            ndim, bounds, f_shift, f_matrix, f_shuffle, f_bias, parallel=parallel, fastmath=fastmath, dtype=dtype
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


class F122015(F102015):
    """
    .. [1] Liang, J. J., Qu, B. Y., & Suganthan, P. N. (2013). Problem definitions and evaluation criteria for the CEC 2014
    special session and competition on single objective real-parameter numerical optimization. Computational Intelligence Laboratory,
    Zhengzhou University, Zhengzhou China and Technical Report, Nanyang Technological University, Singapore, 635, 490.
    """

    name = "F12: Hybrid Function 3 (N=5)"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = bias = 1200.0"

    characteristics = []

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "shift_data_11_D",
        f_matrix: typing.Any = "M_11_D",
        f_shuffle: typing.Any = "shuffle_data_11_D",
        f_bias: float = 1200.0,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
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
                + operator.happy_cat_func(mz[idx2], shift=-1.0)
                + operator.expanded_griewank_rosenbrock_func(mz[idx3])
                + operator.modified_schwefel_func(mz[idx4])
                + operator.ackley_func(mz[idx5])
                + f_bias
            )

        super().__init__(
            ndim, bounds, f_shift, f_matrix, f_shuffle, f_bias, parallel=parallel, fastmath=fastmath, dtype=dtype
        )
        self.n_funcs = 5
        self.p = np.array([0.1, 0.2, 0.2, 0.2, 0.3])
        self.n1 = int(np.ceil(self.p[0] * self.ndim))
        self.n2 = int(np.ceil(self.p[1] * self.ndim)) + self.n1
        self.n3 = int(np.ceil(self.p[2] * self.ndim)) + self.n2
        self.n4 = int(np.ceil(self.p[3] * self.ndim)) + self.n3
        self.idx1, self.idx2 = self.f_shuffle[: self.n1], self.f_shuffle[self.n1 : self.n2]
        self.idx3, self.idx4, self.idx5 = (
            self.f_shuffle[self.n2 : self.n3],
            self.f_shuffle[self.n3 : self.n4],
            self.f_shuffle[self.n4 : self.ndim],
        )
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


class F132015(CecBenchmark):
    """
    .. [1] Liang, J. J., Qu, B. Y., & Suganthan, P. N. (2013). Problem definitions and evaluation criteria for the CEC 2014
    special session and competition on single objective real-parameter numerical optimization. Computational Intelligence Laboratory,
    Zhengzhou University, Zhengzhou China and Technical Report, Nanyang Technological University, Singapore, 635, 490.
    """

    name = "F13: Composition Function 1 (N=5)"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = bias = 1300.0"

    continuous = False
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
        f_shift: typing.Any = "shift_data_13_D",
        f_matrix: typing.Any = "M_13_D",
        f_bias: float = 1300.0,
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
            z0 = operator.dot_mv(f_matrix[: x.shape[0], :], x - f_shift[0])
            g0 = lamdas[0] * operator.rosenbrock_func(z0, shift=1.0) + bias[0]
            w0 = operator.calculate_weight(x - f_shift[0], xichmas[0])

            # 2. High Conditioned Elliptic Function f13
            g1 = lamdas[1] * operator.elliptic_func(x) + bias[1]
            w1 = operator.calculate_weight(x - f_shift[1], xichmas[1])

            # 3. Rotated Bent Cigar Function f1
            z2 = operator.dot_mv(f_matrix[2 * x.shape[0] : 3 * x.shape[0], :], x - f_shift[2])
            g2 = lamdas[2] * operator.bent_cigar_func(z2) + bias[2]
            w2 = operator.calculate_weight(x - f_shift[2], xichmas[2])

            # 4. Rotated Discus Function f2
            z3 = operator.dot_mv(f_matrix[3 * x.shape[0] : 4 * x.shape[0], :], x - f_shift[3])
            g3 = lamdas[3] * operator.discus_func(z3) + bias[3]
            w3 = operator.calculate_weight(x - f_shift[3], xichmas[3])

            # 5. High Conditioned Elliptic Function f13
            g4 = lamdas[4] * operator.elliptic_func(x) + bias[4]
            w4 = operator.calculate_weight(x - f_shift[4], xichmas[4])

            ws = np.array([w0, w1, w2, w3, w4])
            ws = ws / np.sum(ws)
            gs = np.array([g0, g1, g2, g3, g4])
            out[0] = operator.dot_vv(ws, gs) + f_bias

        super().__init__(parallel=parallel, fastmath=fastmath, dtype=dtype)
        self.dim_changeable = True
        self.dim_default = 10
        self.dim_max = 30
        self.dim_supported = [10, 30]
        self.check_ndim_and_bounds(
            ndim, self.dim_max, bounds, np.array([[-100.0, 100.0] for _ in range(self.dim_default)])
        )
        self.make_support_data_path("data_2015")
        self.f_shift = self.check_matrix_data(f_shift, needed_dim=True).ravel().reshape((5, -1))
        self.f_matrix = self.check_matrix_data(f_matrix, needed_dim=True)
        self.f_bias = f_bias
        self.f_global = f_bias
        self.x_global = self.f_shift[0]
        self.n_funcs = 5
        self.xichmas = [10, 20, 30, 40, 50]
        self.lamdas = [1.0, 1e-6, 1e-26, 1e-6, 1e-6]
        self.bias = [0, 100, 200, 300, 400]
        self._bind_kernel(
            compute,
            ["f_matrix", "f_shift", "lamdas", "bias", "xichmas", "f_bias"],
            paras={"f_shift": self.f_shift, "f_bias": self.f_bias, "f_matrix": self.f_matrix},
        )


class F142015(F132015):
    """
    .. [1] Liang, J. J., Qu, B. Y., & Suganthan, P. N. (2013). Problem definitions and evaluation criteria for the CEC 2014
    special session and competition on single objective real-parameter numerical optimization. Computational Intelligence Laboratory,
    Zhengzhou University, Zhengzhou China and Technical Report, Nanyang Technological University, Singapore, 635, 490.
    """

    name = "F14: Composition Function 2 (N=3)"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = bias = 1400.0"

    modality = False  # Number of ambiguous peaks, unknown # peaks

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "shift_data_14_D",
        f_matrix: typing.Any = "M_14_D",
        f_bias: float = 1400.0,
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
            z0 = operator.dot_mv(f_matrix[: x.shape[0], :], x - f_shift[0])
            g0 = lamdas[0] * operator.modified_schwefel_func(z0) + bias[0]
            w0 = operator.calculate_weight(x - f_shift[0], xichmas[0])

            # 2. Rotated Rastrigin’s Function f12
            z1 = operator.dot_mv(f_matrix[x.shape[0] : 2 * x.shape[0], :], x - f_shift[1])
            g1 = lamdas[1] * operator.rastrigin_func(z1) + bias[1]
            w1 = operator.calculate_weight(x - f_shift[1], xichmas[1])

            # 3. Rotated High Conditioned Elliptic Function f13
            z2 = operator.dot_mv(f_matrix[2 * x.shape[0] : 3 * x.shape[0], :], x - f_shift[2])
            g2 = lamdas[2] * operator.elliptic_func(z2) + bias[2]
            w2 = operator.calculate_weight(x - f_shift[2], xichmas[2])

            ws = np.array([w0, w1, w2])
            ws = ws / np.sum(ws)
            gs = np.array([g0, g1, g2])
            out[0] = operator.dot_vv(ws, gs) + f_bias

        super().__init__(ndim, bounds, f_shift, f_matrix, f_bias, parallel=parallel, fastmath=fastmath, dtype=dtype)
        self.f_shift = self.check_matrix_data(f_shift, needed_dim=True).ravel().reshape((3, -1))
        self.f_matrix = self.check_matrix_data(f_matrix, needed_dim=True)
        self.f_bias = f_bias
        self.f_global = f_bias
        self.x_global = self.f_shift[0]
        self.n_funcs = 3
        self.xichmas = [10, 30, 50]
        self.lamdas = [0.25, 1.0, 1e-7]
        self.bias = [0, 100, 200]
        self.g0 = operator.modified_schwefel_func
        self.g1 = operator.rastrigin_func
        self.g2 = operator.elliptic_func
        self._bind_kernel(
            compute,
            ["f_matrix", "f_shift", "lamdas", "bias", "xichmas", "f_bias"],
            paras={"f_shift": self.f_shift, "f_bias": self.f_bias, "f_matrix": self.f_matrix},
        )


class F152015(CecBenchmark):
    """
    .. [1] Liang, J. J., Qu, B. Y., & Suganthan, P. N. (2013). Problem definitions and evaluation criteria for the CEC 2014
    special session and competition on single objective real-parameter numerical optimization. Computational Intelligence Laboratory,
    Zhengzhou University, Zhengzhou China and Technical Report, Nanyang Technological University, Singapore, 635, 490.
    """

    name = "F15: Composition Function 3 (N=5)"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = bias = 1500.0"

    modality = False

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "shift_data_15_D",
        f_matrix: typing.Any = "M_15_D",
        f_bias: float = 1500.0,
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
            z0 = operator.dot_mv(f_matrix[: x.shape[0], :], x - f_shift[0])
            g0 = lamdas[0] * operator.hgbat_func(z0, shift=-1.0) + bias[0]
            w0 = operator.calculate_weight(x - f_shift[0], xichmas[0])

            # 2. Rotated Rastrigin’s Function f12
            z1 = operator.dot_mv(f_matrix[x.shape[0] : 2 * x.shape[0], :], x - f_shift[1])
            g1 = lamdas[1] * operator.rastrigin_func(z1) + bias[1]
            w1 = operator.calculate_weight(x - f_shift[1], xichmas[1])

            # 3. Rotated Schwefel's Function f4
            z2 = operator.dot_mv(f_matrix[2 * x.shape[0] : 3 * x.shape[0], :], x - f_shift[2])
            g2 = lamdas[2] * operator.modified_schwefel_func(z2) + bias[2]
            w2 = operator.calculate_weight(x - f_shift[2], xichmas[2])

            # 4. Rotated Weierstrass Function f3
            z3 = operator.dot_mv(f_matrix[3 * x.shape[0] : 4 * x.shape[0], :], x - f_shift[3])
            g3 = lamdas[3] * operator.weierstrass_func(z3) + bias[3]
            w3 = operator.calculate_weight(x - f_shift[3], xichmas[3])

            # 5. Rotated High Conditioned Elliptic Function f13
            z4 = operator.dot_mv(f_matrix[4 * x.shape[0] : 5 * x.shape[0], :], x - f_shift[4])
            g4 = lamdas[4] * operator.elliptic_func(z4) + bias[4]
            w4 = operator.calculate_weight(x - f_shift[4], xichmas[4])

            ws = np.array([w0, w1, w2, w3, w4])
            ws = ws / np.sum(ws)
            gs = np.array([g0, g1, g2, g3, g4])
            out[0] = operator.dot_vv(ws, gs) + f_bias

        super().__init__(parallel=parallel, fastmath=fastmath, dtype=dtype)
        self.dim_changeable = True
        self.dim_default = 10
        self.dim_max = 30
        self.dim_supported = [10, 30]
        self.check_ndim_and_bounds(
            ndim, self.dim_max, bounds, np.array([[-100.0, 100.0] for _ in range(self.dim_default)])
        )
        self.make_support_data_path("data_2015")
        self.f_shift = self.check_matrix_data(f_shift, needed_dim=True).ravel().reshape((5, -1))
        self.f_matrix = self.check_matrix_data(f_matrix, needed_dim=True)
        self.f_bias = f_bias
        self.f_global = f_bias
        self.x_global = self.f_shift[0]
        self.n_funcs = 5
        self.xichmas = [10, 10, 10, 20, 20]
        self.lamdas = [10, 10, 2.5, 25, 1e-6]
        self.bias = [0, 100, 200, 300, 400]
        self._bind_kernel(
            compute,
            ["f_matrix", "f_shift", "lamdas", "bias", "xichmas", "f_bias"],
            paras={"f_shift": self.f_shift, "f_bias": self.f_bias, "f_matrix": self.f_matrix},
        )
