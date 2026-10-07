#!/usr/bin/env python
# Created by "Thieu" at 09:55, 02/07/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import typing

import numpy as np

from opfunu.benchmark.cec import CecBenchmark
from opfunu.utils import operator


class F12010(CecBenchmark):
    """
    .. [1] Benchmark Functions for the CEC’2010 Special Session and Competition on Large-Scale Global Optimization
    """

    name = "F1: Shifted Elliptic Function"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = 0"

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
        f_shift: typing.Any = "f01_o",
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
        shift: typing.Any = None,
        rotate: typing.Any = None,
        rotate_bounds: bool = True,
    ) -> None:
        def compute(x: np.ndarray, f_shift: typing.Any, out: np.ndarray) -> None:
            out[0] = operator.elliptic_func(x - f_shift)

        super().__init__(
            ndim=ndim,
            bounds=bounds,
            f_shift=f_shift,
            dim_default=1000,
            dim_max=1000,
            data_name="data_2010",
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            compute=compute,
            param_names=["f_shift"],
            shift=shift,
            rotate=rotate,
            rotate_bounds=rotate_bounds,
        )


class F22010(F12010):
    """
    .. [1] Benchmark Functions for the CEC’2010 Special Session and Competition on Large-Scale Global Optimization
    """

    name = "F2: Shifted Rastrigin’s Function"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = 0"

    unimodal = False

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "f02_o",
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
        shift: typing.Any = None,
        rotate: typing.Any = None,
        rotate_bounds: bool = True,
    ) -> None:
        def compute(x: np.ndarray, f_shift: typing.Any, out: np.ndarray) -> None:
            out[0] = operator.rastrigin_func(x - f_shift)

        super().__init__(
            ndim=ndim,
            bounds=bounds,
            f_shift=f_shift,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            shift=shift,
            rotate=rotate,
            rotate_bounds=rotate_bounds,
        )
        self.check_ndim_and_bounds(ndim, self.dim_max, bounds, np.array([[-5.0, 5.0] for _ in range(self.dim_default)]))
        self._bind_kernel(compute, ["f_shift"])


class F32010(F12010):
    """
    .. [1] Benchmark Functions for the CEC’2010 Special Session and Competition on Large-Scale Global Optimization
    """

    name = "F3: Shifted Ackley’s Function"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = 0"

    unimodal = False

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "f03_o",
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
        shift: typing.Any = None,
        rotate: typing.Any = None,
        rotate_bounds: bool = True,
    ) -> None:
        def compute(x: np.ndarray, f_shift: typing.Any, out: np.ndarray) -> None:
            out[0] = operator.ackley_func(x - f_shift)

        super().__init__(
            ndim=ndim,
            bounds=bounds,
            f_shift=f_shift,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            shift=shift,
            rotate=rotate,
            rotate_bounds=rotate_bounds,
        )
        self.check_ndim_and_bounds(
            ndim, self.dim_max, bounds, np.array([[-32.0, 32.0] for _ in range(self.dim_default)])
        )
        self._bind_kernel(compute, ["f_shift"])


class F42010(CecBenchmark):
    """
    .. [1] Benchmark Functions for the CEC’2010 Special Session and Competition on Large-Scale Global Optimization
    """

    name = "F4: Single-group Shifted and m-rotated Elliptic Function"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = 0"

    continuous = True
    linear = False
    convex = False
    unimodal = True
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
        f_shift: typing.Any = "f04_op",
        f_matrix: typing.Any = "f04_m",
        m_group: int = 50,
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
            P: typing.Any,
            m_group: typing.Any,
            f_matrix: typing.Any,
            out: np.ndarray,
        ) -> None:
            z = x - f_shift
            idx1 = P[:m_group]
            idx2 = P[m_group:]
            z_rot_elliptic = operator.dot_vm(z[idx1], f_matrix[:m_group, :m_group])
            z_elliptic = z[idx2]
            out[0] = operator.elliptic_func(z_rot_elliptic) * 10**6 + operator.elliptic_func(z_elliptic)

        super().__init__(
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            dim_changeable=True,
            dim_default=1000,
            dim_max=1000,
            shift=shift,
            rotate=rotate,
            rotate_bounds=rotate_bounds,
        )
        self.check_ndim_and_bounds(
            ndim, self.dim_max, bounds, np.array([[-100.0, 100.0] for _ in range(self.dim_default)])
        )
        self.make_support_data_path("data_2010")
        f_shift = self.load_matrix_data(f_shift)
        self.f_matrix = self.check_matrix_data(f_matrix, False)
        self.f_shift = f_shift[:1, :].ravel()[: self.ndim]
        if self.ndim == 1000:
            self.P = (f_shift[1:, :].ravel() - np.ones(self.ndim)).astype(int)
        else:
            np.random.seed(0)
            self.P = np.random.permutation(self.ndim)
        self.m_group = self.check_m_group(m_group)
        self._f_global = 0.0
        self._x_global = np.asarray(self.f_shift)
        self._bind_kernel(
            compute,
            ["f_shift", "P", "m_group", "f_matrix"],
            paras={"f_shift": self.f_shift, "P": self.P, "f_matrix": self.f_matrix, "m_group": self.m_group},
        )


class F52010(F42010):
    """
    .. [1] Benchmark Functions for the CEC’2010 Special Session and Competition on Large-Scale Global Optimization
    """

    name = "F5: Single-group Shifted and m-rotated Rastrigin’s Function"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = 0"

    unimodal = False

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "f05_op",
        f_matrix: typing.Any = "f05_m",
        m_group: int = 50,
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
            P: typing.Any,
            m_group: typing.Any,
            f_matrix: typing.Any,
            out: np.ndarray,
        ) -> None:
            z = x - f_shift
            idx1 = P[:m_group]
            idx2 = P[m_group:]
            z_rot_ras = operator.dot_vm(z[idx1], f_matrix[:m_group, :m_group])
            z_ras = z[idx2]
            out[0] = operator.rastrigin_func(z_rot_ras) * 10**6 + operator.rastrigin_func(z_ras)

        super().__init__(
            ndim,
            bounds,
            f_shift,
            f_matrix,
            m_group,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            shift=shift,
            rotate=rotate,
            rotate_bounds=rotate_bounds,
        )
        self.check_ndim_and_bounds(ndim, self.dim_max, bounds, np.array([[-5.0, 5.0] for _ in range(self.dim_default)]))
        self._bind_kernel(compute, ["f_shift", "P", "m_group", "f_matrix"])


class F62010(F42010):
    """
    .. [1] Benchmark Functions for the CEC’2010 Special Session and Competition on Large-Scale Global Optimization
    """

    name = "F6: Single-group Shifted and m-rotated Ackley’s Function"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = 0"

    unimodal = False

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "f06_op",
        f_matrix: typing.Any = "f06_m",
        m_group: int = 50,
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
            P: typing.Any,
            m_group: typing.Any,
            f_matrix: typing.Any,
            out: np.ndarray,
        ) -> None:
            z = x - f_shift
            idx1 = P[:m_group]
            idx2 = P[m_group:]
            z_rot_ras = operator.dot_vm(z[idx1], f_matrix[:m_group, :m_group])
            z_ras = z[idx2]
            out[0] = operator.ackley_func(z_rot_ras) * 10**6 + operator.ackley_func(z_ras)

        super().__init__(
            ndim,
            bounds,
            f_shift,
            f_matrix,
            m_group,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            shift=shift,
            rotate=rotate,
            rotate_bounds=rotate_bounds,
        )
        self.check_ndim_and_bounds(
            ndim, self.dim_max, bounds, np.array([[-32.0, 32.0] for _ in range(self.dim_default)])
        )
        self._bind_kernel(compute, ["f_shift", "P", "m_group", "f_matrix"])


class F72010(CecBenchmark):
    """
    .. [1] Benchmark Functions for the CEC’2010 Special Session and Competition on Large-Scale Global Optimization
    """

    name = "F7: Single-group Shifted m-dimensional Schwefel’s Problem 1.2"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = 0"

    continuous = True
    linear = False
    convex = False
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
        f_shift: typing.Any = "f07_op",
        m_group: int = 50,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
        shift: typing.Any = None,
        rotate: typing.Any = None,
        rotate_bounds: bool = True,
    ) -> None:
        def compute(x: np.ndarray, f_shift: typing.Any, P: typing.Any, m_group: typing.Any, out: np.ndarray) -> None:
            z = x - f_shift
            z_schwefel = z[P[:m_group]]
            z_sphere = z[P[m_group:]]
            out[0] = operator.schwefel_12_func(z_schwefel) * 10**6 + operator.sphere_func(z_sphere)

        super().__init__(
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            dim_changeable=True,
            dim_default=1000,
            dim_max=1000,
            shift=shift,
            rotate=rotate,
            rotate_bounds=rotate_bounds,
        )
        self.check_ndim_and_bounds(
            ndim, self.dim_max, bounds, np.array([[-100.0, 100.0] for _ in range(self.dim_default)])
        )
        self.make_support_data_path("data_2010")
        f_shift = self.load_matrix_data(f_shift)
        self.f_shift = f_shift[:1, :].ravel()[: self.ndim]
        if self.ndim == 1000:
            self.P = (f_shift[1:, :].ravel() - np.ones(self.ndim)).astype(int)
        else:
            np.random.seed(0)
            self.P = np.random.permutation(self.ndim)
        self.m_group = self.check_m_group(m_group)
        self._f_global = 0.0
        self._x_global = np.asarray(self.f_shift)
        self._bind_kernel(
            compute, ["f_shift", "P", "m_group"], paras={"f_shift": self.f_shift, "P": self.P, "m_group": self.m_group}
        )


class F82010(F72010):
    """
    .. [1] Benchmark Functions for the CEC’2010 Special Session and Competition on Large-Scale Global Optimization
    """

    name = "F8: Single-group Shifted m-dimensional Rosenbrock’s Function"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = 0"

    unimodal = False

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "f08_op",
        m_group: int = 50,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
        shift: typing.Any = None,
        rotate: typing.Any = None,
        rotate_bounds: bool = True,
    ) -> None:
        def compute(x: np.ndarray, f_shift: typing.Any, P: typing.Any, m_group: typing.Any, out: np.ndarray) -> None:
            z = x - f_shift
            z_rosen = z[P[:m_group]]
            z_sphere = z[P[m_group:]]
            out[0] = operator.rosenbrock_func(z_rosen) * 10**6 + operator.sphere_func(z_sphere)

        super().__init__(
            ndim,
            bounds,
            f_shift,
            m_group,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            shift=shift,
            rotate=rotate,
            rotate_bounds=rotate_bounds,
        )
        self._x_global = np.asarray(self.f_shift.copy())
        self.x_global[self.P[: self.m_group]] = self.f_shift[self.P[: self.m_group]] + 1
        self.x_global[self.P[self.m_group :]] = self.f_shift[self.P[self.m_group :]]
        self._bind_kernel(compute, ["f_shift", "P", "m_group"])


class F92010(CecBenchmark):
    """
    .. [1] Benchmark Functions for the CEC’2010 Special Session and Competition on Large-Scale Global Optimization
    """

    name = "F9: D/2m-group Shifted and m-rotated Elliptic Function"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = 0"

    continuous = True
    linear = False
    convex = False
    unimodal = True
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
        f_shift: typing.Any = "f09_op",
        f_matrix: typing.Any = "f09_m",
        m_group: int = 50,
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
            count_up: int,
            P: typing.Any,
            m_group: typing.Any,
            f_matrix: typing.Any,
            out: np.ndarray,
        ) -> None:
            z = x - f_shift
            result = 0.0
            for k in range(0, count_up):
                idx1 = P[k * m_group : (k + 1) * m_group]
                z1 = operator.dot_vm(z[idx1], f_matrix[: len(idx1), : len(idx1)])
                result += operator.elliptic_func(z1)
            z2 = z[P[int(x.shape[0] / 2) :]]
            out[0] = result + operator.elliptic_func(z2)

        super().__init__(
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            dim_changeable=True,
            dim_default=1000,
            dim_max=1000,
            shift=shift,
            rotate=rotate,
            rotate_bounds=rotate_bounds,
        )
        self.check_ndim_and_bounds(
            ndim, self.dim_max, bounds, np.array([[-100.0, 100.0] for _ in range(self.dim_default)])
        )
        self.make_support_data_path("data_2010")
        self.f_matrix = self.check_matrix_data(f_matrix, False)
        f_shift = self.load_matrix_data(f_shift)
        self.f_shift = f_shift[:1, :].ravel()[: self.ndim]
        if self.ndim == 1000:
            self.P = (f_shift[1:, :].ravel() - np.ones(self.ndim)).astype(int)
        else:
            np.random.seed(0)
            self.P = np.random.permutation(self.ndim)
        self.m_group = self.check_m_group(m_group)
        self._f_global = 0.0
        self._x_global = np.asarray(self.f_shift)
        self.count_up = int(self.ndim / (2 * self.m_group))
        self._bind_kernel(
            compute,
            ["f_shift", "count_up", "P", "m_group", "f_matrix"],
            paras={"f_shift": self.f_shift, "P": self.P, "f_matrix": self.f_matrix, "m_group": self.m_group},
        )


class F102010(F92010):
    """
    .. [1] Benchmark Functions for the CEC’2010 Special Session and Competition on Large-Scale Global Optimization
    """

    name = "F10: D/2m-group Shifted and m-rotated Rastrigin’s Function"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = 0"

    unimodal = False

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "f10_op",
        f_matrix: typing.Any = "f10_m",
        m_group: int = 50,
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
            count_up: typing.Any,
            P: typing.Any,
            m_group: typing.Any,
            f_matrix: typing.Any,
            out: np.ndarray,
        ) -> None:
            z = x - f_shift
            result = 0.0
            for k in range(0, count_up):
                idx1 = P[k * m_group : (k + 1) * m_group]
                z1 = operator.dot_vm(z[idx1], f_matrix[: len(idx1), : len(idx1)])
                result += operator.rastrigin_func(z1)
            z2 = z[P[int(x.shape[0] / 2) :]]
            out[0] = result + operator.rastrigin_func(z2)

        super().__init__(
            ndim,
            bounds,
            f_shift,
            f_matrix,
            m_group,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            shift=shift,
            rotate=rotate,
            rotate_bounds=rotate_bounds,
        )
        self.check_ndim_and_bounds(ndim, self.dim_max, bounds, np.array([[-5.0, 5.0] for _ in range(self.dim_default)]))
        self._bind_kernel(compute, ["f_shift", "count_up", "P", "m_group", "f_matrix"])


class F112010(F92010):
    """
    .. [1] Benchmark Functions for the CEC’2010 Special Session and Competition on Large-Scale Global Optimization
    """

    name = "F11: D/2m-group Shifted and m-rotated Ackley’s Function"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = 0"

    unimodal = False

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "f11_op",
        f_matrix: typing.Any = "f11_m",
        m_group: int = 50,
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
            count_up: typing.Any,
            P: typing.Any,
            m_group: typing.Any,
            f_matrix: typing.Any,
            out: np.ndarray,
        ) -> None:
            z = x - f_shift
            result = 0.0
            for k in range(0, count_up):
                idx1 = P[k * m_group : (k + 1) * m_group]
                z1 = operator.dot_vm(z[idx1], f_matrix[: len(idx1), : len(idx1)])
                result += operator.ackley_func(z1)
            z2 = z[P[int(x.shape[0] / 2) :]]
            out[0] = result + operator.ackley_func(z2)

        super().__init__(
            ndim,
            bounds,
            f_shift,
            f_matrix,
            m_group,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            shift=shift,
            rotate=rotate,
            rotate_bounds=rotate_bounds,
        )
        self.check_ndim_and_bounds(
            ndim, self.dim_max, bounds, np.array([[-32.0, 32.0] for _ in range(self.dim_default)])
        )
        self._bind_kernel(compute, ["f_shift", "count_up", "P", "m_group", "f_matrix"])


class F122010(F72010):
    """
    .. [1] Benchmark Functions for the CEC’2010 Special Session and Competition on Large-Scale Global Optimization
    """

    name = "F12: D/2m-group Shifted m-dimensional Schwefel’s Problem 1.2"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = 0"

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "f11_op",
        m_group: int = 50,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
        shift: typing.Any = None,
        rotate: typing.Any = None,
        rotate_bounds: bool = True,
    ) -> None:
        def compute(
            x: np.ndarray, f_shift: typing.Any, count_up: int, P: typing.Any, m_group: typing.Any, out: np.ndarray
        ) -> None:
            z = x - f_shift
            result = 0.0
            for k in range(0, count_up):
                idx1 = P[k * m_group : (k + 1) * m_group]
                result += operator.schwefel_12_func(z[idx1])
            z2 = z[P[int(x.shape[0] / 2) :]]
            out[0] = result + operator.sphere_func(z2)

        super().__init__(
            ndim,
            bounds,
            f_shift,
            m_group,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            shift=shift,
            rotate=rotate,
            rotate_bounds=rotate_bounds,
        )
        self.count_up = int(self.ndim / (2 * self.m_group))
        self._bind_kernel(compute, ["f_shift", "count_up", "P", "m_group"])


class F132010(F72010):
    """
    .. [1] Benchmark Functions for the CEC’2010 Special Session and Competition on Large-Scale Global Optimization
    """

    name = "F13: D/2m-group Shifted m-dimensional Rosenbrock’s Function"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = 0"
    unimodal = False

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "f13_op",
        m_group: int = 50,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
        shift: typing.Any = None,
        rotate: typing.Any = None,
        rotate_bounds: bool = True,
    ) -> None:
        def compute(
            x: np.ndarray, f_shift: typing.Any, count_up: int, P: typing.Any, m_group: typing.Any, out: np.ndarray
        ) -> None:
            z = x - f_shift
            result = 0.0
            for k in range(0, count_up):
                idx1 = P[k * m_group : (k + 1) * m_group]
                result += operator.rosenbrock_func(z[idx1])
            z2 = z[P[int(x.shape[0] / 2) :]]
            out[0] = result + operator.sphere_func(z2)

        super().__init__(
            ndim,
            bounds,
            f_shift,
            m_group,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            shift=shift,
            rotate=rotate,
            rotate_bounds=rotate_bounds,
        )
        self.count_up = int(self.ndim / (2 * self.m_group))
        self._x_global = np.asarray(self.f_shift.copy())
        self.x_global[self.P[: int(self.ndim / 2)]] = self.f_shift[self.P[: int(self.ndim / 2)]] + 1
        self.x_global[self.P[int(self.ndim / 2) :]] = self.f_shift[self.P[int(self.ndim / 2) :]]
        self._bind_kernel(compute, ["f_shift", "count_up", "P", "m_group"])


class F142010(F92010):
    """
    .. [1] Benchmark Functions for the CEC’2010 Special Session and Competition on Large-Scale Global Optimization
    """

    name = "F14: D/m-group Shifted and m-rotated Elliptic Function"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = 0"

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "f14_op",
        f_matrix: typing.Any = "f14_m",
        m_group: int = 50,
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
            count_up: int,
            P: typing.Any,
            m_group: typing.Any,
            f_matrix: typing.Any,
            out: np.ndarray,
        ) -> None:
            z = x - f_shift
            result = 0.0
            for k in range(0, count_up):
                idx1 = P[k * m_group : (k + 1) * m_group]
                z1 = operator.dot_vm(z[idx1], f_matrix[: len(idx1), : len(idx1)])
                result += operator.elliptic_func(z1)
            out[0] = result

        super().__init__(
            ndim,
            bounds,
            f_shift,
            f_matrix,
            m_group,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            shift=shift,
            rotate=rotate,
            rotate_bounds=rotate_bounds,
        )
        self.count_up = int(self.ndim / self.m_group)
        self._bind_kernel(compute, ["f_shift", "count_up", "P", "m_group", "f_matrix"])


class F152010(F92010):
    """
    .. [1] Benchmark Functions for the CEC’2010 Special Session and Competition on Large-Scale Global Optimization
    """

    name = "F15: D/m-group Shifted and m-rotated Rastrigin’s Function"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = 0"

    unimodal = False

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "f15_op",
        f_matrix: typing.Any = "f15_m",
        m_group: int = 50,
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
            count_up: int,
            P: typing.Any,
            m_group: typing.Any,
            f_matrix: typing.Any,
            out: np.ndarray,
        ) -> None:
            z = x - f_shift
            result = 0.0
            for k in range(0, count_up):
                idx1 = P[k * m_group : (k + 1) * m_group]
                z1 = operator.dot_vm(z[idx1], f_matrix[: len(idx1), : len(idx1)])
                result += operator.rastrigin_func(z1)
            out[0] = result

        super().__init__(
            ndim,
            bounds,
            f_shift,
            f_matrix,
            m_group,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            shift=shift,
            rotate=rotate,
            rotate_bounds=rotate_bounds,
        )
        self.check_ndim_and_bounds(ndim, self.dim_max, bounds, np.array([[-5.0, 5.0] for _ in range(self.dim_default)]))
        self.count_up = int(self.ndim / self.m_group)
        self._bind_kernel(compute, ["f_shift", "count_up", "P", "m_group", "f_matrix"])


class F162010(F92010):
    """
    .. [1] Benchmark Functions for the CEC’2010 Special Session and Competition on Large-Scale Global Optimization
    """

    name = "F16: D/m-group Shifted and m-rotated Ackley’s Function"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = 0"

    unimodal = False

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "f16_op",
        f_matrix: typing.Any = "f16_m",
        m_group: int = 50,
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
            count_up: int,
            P: typing.Any,
            m_group: typing.Any,
            f_matrix: typing.Any,
            out: np.ndarray,
        ) -> None:
            z = x - f_shift
            result = 0.0
            for k in range(0, count_up):
                idx1 = P[k * m_group : (k + 1) * m_group]
                z1 = operator.dot_vm(z[idx1], f_matrix[: len(idx1), : len(idx1)])
                result += operator.ackley_func(z1)
            out[0] = result

        super().__init__(
            ndim,
            bounds,
            f_shift,
            f_matrix,
            m_group,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            shift=shift,
            rotate=rotate,
            rotate_bounds=rotate_bounds,
        )
        self.check_ndim_and_bounds(
            ndim, self.dim_max, bounds, np.array([[-32.0, 32.0] for _ in range(self.dim_default)])
        )
        self.count_up = int(self.ndim / self.m_group)
        self._bind_kernel(compute, ["f_shift", "count_up", "P", "m_group", "f_matrix"])


class F172010(F72010):
    """
    .. [1] Benchmark Functions for the CEC’2010 Special Session and Competition on Large-Scale Global Optimization
    """

    name = "F17: D/m-group Shifted m-dimensional Schwefel’s Problem 1.2"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = 0"

    unimodal = True

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "f17_op",
        m_group: int = 50,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
        shift: typing.Any = None,
        rotate: typing.Any = None,
        rotate_bounds: bool = True,
    ) -> None:
        def compute(
            x: np.ndarray, f_shift: typing.Any, count_up: int, P: typing.Any, m_group: typing.Any, out: np.ndarray
        ) -> None:
            z = x - f_shift
            result = 0.0
            for k in range(0, count_up):
                idx1 = P[k * m_group : (k + 1) * m_group]
                result += operator.ackley_func(z[idx1])
            out[0] = result

        super().__init__(
            ndim,
            bounds,
            f_shift,
            m_group,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            shift=shift,
            rotate=rotate,
            rotate_bounds=rotate_bounds,
        )
        self.count_up = int(self.ndim / self.m_group)
        self._bind_kernel(compute, ["f_shift", "count_up", "P", "m_group"])


class F182010(F72010):
    """
    .. [1] Benchmark Functions for the CEC’2010 Special Session and Competition on Large-Scale Global Optimization
    """

    name = "F18: D/m-group Shifted m-dimensional Rosenbrock’s Function"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = 0"

    unimodal = False

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "f18_op",
        m_group: int = 50,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
        shift: typing.Any = None,
        rotate: typing.Any = None,
        rotate_bounds: bool = True,
    ) -> None:
        def compute(
            x: np.ndarray, f_shift: typing.Any, count_up: int, P: typing.Any, m_group: typing.Any, out: np.ndarray
        ) -> None:
            z = x - f_shift
            result = 0.0
            for k in range(0, count_up):
                idx1 = P[k * m_group : (k + 1) * m_group]
                result += operator.rosenbrock_func(z[idx1])
            out[0] = result

        super().__init__(
            ndim,
            bounds,
            f_shift,
            m_group,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            shift=shift,
            rotate=rotate,
            rotate_bounds=rotate_bounds,
        )
        self.count_up = int(self.ndim / self.m_group)
        self._x_global = np.asarray(self.f_shift + 1)
        self._bind_kernel(compute, ["f_shift", "count_up", "P", "m_group"])


class F192010(F12010):
    """
    .. [1] Benchmark Functions for the CEC’2010 Special Session and Competition on Large-Scale Global Optimization
    """

    name = "F19: Shifted Schwefel’s Problem 1.2"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = 0"

    separable = False

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "f19_o",
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
        shift: typing.Any = None,
        rotate: typing.Any = None,
        rotate_bounds: bool = True,
    ) -> None:
        def compute(x: np.ndarray, f_shift: typing.Any, out: np.ndarray) -> None:
            out[0] = operator.schwefel_12_func(x - f_shift)

        super().__init__(
            ndim,
            bounds,
            f_shift,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            shift=shift,
            rotate=rotate,
            rotate_bounds=rotate_bounds,
        )
        self._bind_kernel(compute, ["f_shift"])


class F202010(F12010):
    """
    .. [1] Benchmark Functions for the CEC’2010 Special Session and Competition on Large-Scale Global Optimization
    """

    name = "F20: Shifted Rosenbrock’s Function"
    latex_formula = r"F_1(x) = \sum_{i=1}^D z_i^2 + bias, z=x-o,\\ x=[x_1, ..., x_D]; o=[o_1, ..., o_D]: \text{the shifted global optimum}"
    latex_formula_dimension = r"2 <= D <= 100"
    latex_formula_bounds = r"x_i \in [-100.0, 100.0], \forall i \in  [1, D]"
    latex_formula_global_optimum = r"\text{Global optimum: } x^* = o, F_1(x^*) = 0"

    separable = False
    unimodal = False

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: typing.Any = "f20_o",
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
        shift: typing.Any = None,
        rotate: typing.Any = None,
        rotate_bounds: bool = True,
    ) -> None:
        def compute(x: np.ndarray, f_shift: typing.Any, out: np.ndarray) -> None:
            out[0] = operator.rosenbrock_func(x - f_shift)

        super().__init__(
            ndim,
            bounds,
            f_shift,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            shift=shift,
            rotate=rotate,
            rotate_bounds=rotate_bounds,
        )
        self._bind_kernel(compute, ["f_shift"])
        self._x_global = np.asarray(self.f_shift + 1)
