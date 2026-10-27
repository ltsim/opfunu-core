#!/usr/bin/env python
# Created by "Thieu" at 09:55, 02/07/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import sys
import typing

if sys.version_info >= (3, 11):
    from importlib.resources.abc import Traversable as _Traversable
else:  # Python 3.10: Traversable lives in importlib.abc
    from importlib.abc import Traversable as _Traversable

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
        prepare: bool | None = None,
        *,
        default_bounds: typing.Any = None,
        f_bias: float | None = None,
        f_global: typing.Any = None,
        x_global: typing.Any = None,
    ) -> None:
        def compute(x: np.ndarray, shift: np.ndarray, f_shift: typing.Any, out: np.ndarray) -> None:
            x = x - shift
            out[0] = operator.elliptic_func(x - f_shift)

        super().__init__(
            ndim=ndim,
            bounds=bounds,
            f_shift=f_shift,
            f_bias=0.0 if f_bias is None else f_bias,
            default_bounds=default_bounds,
            dim_default=1000,
            dim_max=1000,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            compute=compute,
            param_names=["f_shift"],
            shift=shift,
            prepare=prepare,
            f_global=0.0 if f_global is None else f_global,
            x_global=x_global,
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
        prepare: bool | None = None,
    ) -> None:
        def compute(x: np.ndarray, shift: np.ndarray, f_shift: typing.Any, out: np.ndarray) -> None:
            x = x - shift
            out[0] = operator.rastrigin_func(x - f_shift)

        super().__init__(
            ndim=ndim,
            bounds=bounds,
            f_shift=f_shift,
            f_bias=0.0,
            default_bounds=np.array([-5.0, 5.0]),
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            shift=shift,
            prepare=prepare,
            f_global=0.0,
        )
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
        prepare: bool | None = None,
    ) -> None:
        def compute(x: np.ndarray, shift: np.ndarray, f_shift: typing.Any, out: np.ndarray) -> None:
            x = x - shift
            out[0] = operator.ackley_func(x - f_shift)

        super().__init__(
            ndim=ndim,
            bounds=bounds,
            f_shift=f_shift,
            f_bias=0.0,
            default_bounds=np.array([-32.0, 32.0]),
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            shift=shift,
            prepare=prepare,
            f_global=0.0,
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
        prepare: bool | None = None,
        *,
        default_bounds: typing.Any = None,
        x_global: typing.Any = None,
        f_global: typing.Any = None,
        load: typing.Callable[["CecBenchmark", int, _Traversable | None], None] | None = None,
        paras: typing.Union[
            dict[str, typing.Any], list[str], tuple[str, ...], typing.Callable[..., dict[str, typing.Any]], None
        ] = None,
    ) -> None:
        def compute(
            x: np.ndarray,
            shift: np.ndarray,
            f_shift: typing.Any,
            P: typing.Any,
            m_group: typing.Any,
            f_matrix: typing.Any,
            out: np.ndarray,
        ) -> None:
            x = x - shift
            z = x - f_shift
            idx1 = P[:m_group]
            idx2 = P[m_group:]
            z_rot_elliptic = operator.dot_vm(z[idx1], f_matrix[:m_group, :m_group])
            z_elliptic = z[idx2]
            out[0] = operator.elliptic_func(z_rot_elliptic) * 10**6 + operator.elliptic_func(z_elliptic)

        self.m_group = m_group

        def _load(problem: "CecBenchmark", resolved_ndim: int, support_path: _Traversable | None) -> None:
            f_shift_data = problem.load_matrix_data(f_shift, support_path=support_path)
            problem.f_matrix = problem.check_matrix_data(f_matrix, False, support_path=support_path)
            problem.f_shift = f_shift_data[:1, :].ravel()[:resolved_ndim]
            if resolved_ndim == 1000:
                problem.P = (f_shift_data[1:, :].ravel() - np.ones(resolved_ndim)).astype(int)
            else:
                np.random.seed(0)
                problem.P = np.random.permutation(resolved_ndim)
            problem.m_group = problem.check_m_group(problem.m_group, ndim=resolved_ndim)
            if load is not None:
                load(problem, resolved_ndim, support_path)

        super().__init__(
            ndim=ndim,
            bounds=bounds,
            f_shift=None,
            f_matrix=None,
            f_bias=0.0,
            default_bounds=np.array([-100.0, 100.0]) if default_bounds is None else default_bounds,
            dim_changeable=True,
            dim_default=1000,
            dim_max=1000,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            shift=shift,
            prepare=prepare,
            load=_load,
            f_global=0.0 if f_global is None else f_global,
            x_global=(lambda problem, dim: np.asarray(problem.f_shift)) if x_global is None else x_global,
        )
        if paras is None:
            self._bind_kernel(
                compute,
                ["f_shift", "P", "m_group", "f_matrix"],
                paras={"f_shift": self.f_shift, "P": self.P, "f_matrix": self.f_matrix, "m_group": self.m_group},
            )
        else:
            self._bind_kernel(compute, ["f_shift", "P", "m_group", "f_matrix"], paras=paras)


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
        prepare: bool | None = None,
        *,
        default_bounds: typing.Any = None,
        x_global: typing.Any = None,
        f_global: typing.Any = None,
        load: typing.Callable[["CecBenchmark", int, _Traversable | None], None] | None = None,
        paras: typing.Union[
            dict[str, typing.Any], list[str], tuple[str, ...], typing.Callable[..., dict[str, typing.Any]], None
        ] = None,
    ) -> None:
        def compute(
            x: np.ndarray,
            shift: np.ndarray,
            f_shift: typing.Any,
            P: typing.Any,
            m_group: typing.Any,
            f_matrix: typing.Any,
            out: np.ndarray,
        ) -> None:
            x = x - shift
            z = x - f_shift
            idx1 = P[:m_group]
            idx2 = P[m_group:]
            z_rot_ras = operator.dot_vm(z[idx1], f_matrix[:m_group, :m_group])
            z_ras = z[idx2]
            out[0] = operator.rastrigin_func(z_rot_ras) * 10**6 + operator.rastrigin_func(z_ras)

        super().__init__(
            ndim=ndim,
            bounds=bounds,
            f_shift=f_shift,
            f_matrix=f_matrix,
            m_group=m_group,
            default_bounds=np.array([-5.0, 5.0]) if default_bounds is None else default_bounds,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            shift=shift,
            prepare=prepare,
            x_global=x_global,
            f_global=f_global,
            load=load,
            paras=paras,
        )
        self._bind_kernel(compute, ["f_shift", "P", "m_group", "f_matrix"], paras=paras)


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
        prepare: bool | None = None,
        *,
        default_bounds: typing.Any = None,
        x_global: typing.Any = None,
        f_global: typing.Any = None,
        load: typing.Callable[["CecBenchmark", int, _Traversable | None], None] | None = None,
        paras: typing.Union[
            dict[str, typing.Any], list[str], tuple[str, ...], typing.Callable[..., dict[str, typing.Any]], None
        ] = None,
    ) -> None:
        def compute(
            x: np.ndarray,
            shift: np.ndarray,
            f_shift: typing.Any,
            P: typing.Any,
            m_group: typing.Any,
            f_matrix: typing.Any,
            out: np.ndarray,
        ) -> None:
            x = x - shift
            z = x - f_shift
            idx1 = P[:m_group]
            idx2 = P[m_group:]
            z_rot_ras = operator.dot_vm(z[idx1], f_matrix[:m_group, :m_group])
            z_ras = z[idx2]
            out[0] = operator.ackley_func(z_rot_ras) * 10**6 + operator.ackley_func(z_ras)

        super().__init__(
            ndim=ndim,
            bounds=bounds,
            f_shift=f_shift,
            f_matrix=f_matrix,
            m_group=m_group,
            default_bounds=np.array([-32.0, 32.0]) if default_bounds is None else default_bounds,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            shift=shift,
            prepare=prepare,
            x_global=x_global,
            f_global=f_global,
            load=load,
            paras=paras,
        )
        self._bind_kernel(compute, ["f_shift", "P", "m_group", "f_matrix"], paras=paras)


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
        prepare: bool | None = None,
        *,
        default_bounds: typing.Any = None,
        x_global: typing.Any = None,
        f_global: typing.Any = None,
        load: typing.Callable[["CecBenchmark", int, _Traversable | None], None] | None = None,
        paras: typing.Union[
            dict[str, typing.Any], list[str], tuple[str, ...], typing.Callable[..., dict[str, typing.Any]], None
        ] = None,
    ) -> None:
        def compute(
            x: np.ndarray, shift: np.ndarray, f_shift: typing.Any, P: typing.Any, m_group: typing.Any, out: np.ndarray
        ) -> None:
            x = x - shift
            z = x - f_shift
            z_schwefel = z[P[:m_group]]
            z_sphere = z[P[m_group:]]
            out[0] = operator.schwefel_12_func(z_schwefel) * 10**6 + operator.sphere_func(z_sphere)

        self.m_group = m_group

        def _load(problem: "CecBenchmark", resolved_ndim: int, support_path: _Traversable | None) -> None:
            f_shift_data = problem.load_matrix_data(f_shift, support_path=support_path)
            problem.f_shift = f_shift_data[:1, :].ravel()[:resolved_ndim]
            if resolved_ndim == 1000:
                problem.P = (f_shift_data[1:, :].ravel() - np.ones(resolved_ndim)).astype(int)
            else:
                np.random.seed(0)
                problem.P = np.random.permutation(resolved_ndim)
            problem.m_group = problem.check_m_group(problem.m_group, ndim=resolved_ndim)
            if load is not None:
                load(problem, resolved_ndim, support_path)

        super().__init__(
            ndim=ndim,
            bounds=bounds,
            f_shift=None,
            f_bias=0.0,
            default_bounds=np.array([-100.0, 100.0]) if default_bounds is None else default_bounds,
            dim_changeable=True,
            dim_default=1000,
            dim_max=1000,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            shift=shift,
            prepare=prepare,
            load=_load,
            f_global=0.0 if f_global is None else f_global,
            x_global=(lambda problem, dim: np.asarray(problem.f_shift)) if x_global is None else x_global,
        )
        if paras is None:
            self._bind_kernel(
                compute,
                ["f_shift", "P", "m_group"],
                paras={"f_shift": self.f_shift, "P": self.P, "m_group": self.m_group},
            )
        else:
            self._bind_kernel(compute, ["f_shift", "P", "m_group"], paras=paras)


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
        prepare: bool | None = None,
        *,
        default_bounds: typing.Any = None,
        x_global: typing.Any = None,
        f_global: typing.Any = None,
        load: typing.Callable[["CecBenchmark", int, _Traversable | None], None] | None = None,
        paras: typing.Union[
            dict[str, typing.Any], list[str], tuple[str, ...], typing.Callable[..., dict[str, typing.Any]], None
        ] = None,
    ) -> None:
        def compute(
            x: np.ndarray, shift: np.ndarray, f_shift: typing.Any, P: typing.Any, m_group: typing.Any, out: np.ndarray
        ) -> None:
            x = x - shift
            z = x - f_shift
            z_rosen = z[P[:m_group]]
            z_sphere = z[P[m_group:]]
            out[0] = operator.rosenbrock_func(z_rosen) * 10**6 + operator.sphere_func(z_sphere)

        def _x_global_opt(problem: "CecBenchmark", dim: int) -> np.ndarray:
            xg = problem.f_shift.copy()
            xg[problem.P[: problem.m_group]] = problem.f_shift[problem.P[: problem.m_group]] + 1
            xg[problem.P[problem.m_group :]] = problem.f_shift[problem.P[problem.m_group :]]
            return np.asarray(xg)

        super().__init__(
            ndim=ndim,
            bounds=bounds,
            f_shift=f_shift,
            m_group=m_group,
            default_bounds=default_bounds,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            shift=shift,
            prepare=prepare,
            x_global=_x_global_opt if x_global is None else x_global,
            f_global=f_global,
            load=load,
            paras=paras,
        )
        self._bind_kernel(compute, ["f_shift", "P", "m_group"], paras=paras)


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
        prepare: bool | None = None,
        *,
        default_bounds: typing.Any = None,
        x_global: typing.Any = None,
        f_global: typing.Any = None,
        load: typing.Callable[["CecBenchmark", int, _Traversable | None], None] | None = None,
        paras: typing.Union[
            dict[str, typing.Any], list[str], tuple[str, ...], typing.Callable[..., dict[str, typing.Any]], None
        ] = None,
    ) -> None:
        def compute(
            x: np.ndarray,
            shift: np.ndarray,
            f_shift: typing.Any,
            count_up: int,
            P: typing.Any,
            m_group: typing.Any,
            f_matrix: typing.Any,
            out: np.ndarray,
        ) -> None:
            x = x - shift
            z = x - f_shift
            result = 0.0
            for k in range(0, count_up):
                idx1 = P[k * m_group : (k + 1) * m_group]
                z1 = operator.dot_vm(z[idx1], f_matrix[: len(idx1), : len(idx1)])
                result += operator.elliptic_func(z1)
            z2 = z[P[int(x.shape[0] / 2) :]]
            out[0] = result + operator.elliptic_func(z2)

        self.m_group = m_group

        def _load(problem: "CecBenchmark", resolved_ndim: int, support_path: _Traversable | None) -> None:
            problem.f_matrix = problem.check_matrix_data(f_matrix, False, support_path=support_path)
            f_shift_data = problem.load_matrix_data(f_shift, support_path=support_path)
            problem.f_shift = f_shift_data[:1, :].ravel()[:resolved_ndim]
            if resolved_ndim == 1000:
                problem.P = (f_shift_data[1:, :].ravel() - np.ones(resolved_ndim)).astype(int)
            else:
                np.random.seed(0)
                problem.P = np.random.permutation(resolved_ndim)
            problem.m_group = problem.check_m_group(problem.m_group, ndim=resolved_ndim)
            problem.count_up = int(resolved_ndim / (2 * problem.m_group))
            if load is not None:
                load(problem, resolved_ndim, support_path)

        super().__init__(
            ndim=ndim,
            bounds=bounds,
            f_shift=None,
            f_matrix=None,
            f_bias=0.0,
            default_bounds=np.array([-100.0, 100.0]) if default_bounds is None else default_bounds,
            dim_changeable=True,
            dim_default=1000,
            dim_max=1000,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            shift=shift,
            prepare=prepare,
            load=_load,
            f_global=0.0 if f_global is None else f_global,
            x_global=(lambda problem, dim: np.asarray(problem.f_shift)) if x_global is None else x_global,
        )
        if paras is None:
            self._bind_kernel(
                compute,
                ["f_shift", "count_up", "P", "m_group", "f_matrix"],
                paras={"f_shift": self.f_shift, "P": self.P, "f_matrix": self.f_matrix, "m_group": self.m_group},
            )
        else:
            self._bind_kernel(compute, ["f_shift", "count_up", "P", "m_group", "f_matrix"], paras=paras)


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
        prepare: bool | None = None,
        *,
        default_bounds: typing.Any = None,
        x_global: typing.Any = None,
        f_global: typing.Any = None,
        load: typing.Callable[["CecBenchmark", int, _Traversable | None], None] | None = None,
        paras: typing.Union[
            dict[str, typing.Any], list[str], tuple[str, ...], typing.Callable[..., dict[str, typing.Any]], None
        ] = None,
    ) -> None:
        def compute(
            x: np.ndarray,
            shift: np.ndarray,
            f_shift: typing.Any,
            count_up: typing.Any,
            P: typing.Any,
            m_group: typing.Any,
            f_matrix: typing.Any,
            out: np.ndarray,
        ) -> None:
            x = x - shift
            z = x - f_shift
            result = 0.0
            for k in range(0, count_up):
                idx1 = P[k * m_group : (k + 1) * m_group]
                z1 = operator.dot_vm(z[idx1], f_matrix[: len(idx1), : len(idx1)])
                result += operator.rastrigin_func(z1)
            z2 = z[P[int(x.shape[0] / 2) :]]
            out[0] = result + operator.rastrigin_func(z2)

        super().__init__(
            ndim=ndim,
            bounds=bounds,
            f_shift=f_shift,
            f_matrix=f_matrix,
            m_group=m_group,
            default_bounds=np.array([-5.0, 5.0]) if default_bounds is None else default_bounds,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            shift=shift,
            prepare=prepare,
            x_global=x_global,
            f_global=f_global,
            load=load,
            paras=paras,
        )
        self._bind_kernel(compute, ["f_shift", "count_up", "P", "m_group", "f_matrix"], paras=paras)


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
        prepare: bool | None = None,
        *,
        default_bounds: typing.Any = None,
        x_global: typing.Any = None,
        f_global: typing.Any = None,
        load: typing.Callable[["CecBenchmark", int, _Traversable | None], None] | None = None,
        paras: typing.Union[
            dict[str, typing.Any], list[str], tuple[str, ...], typing.Callable[..., dict[str, typing.Any]], None
        ] = None,
    ) -> None:
        def compute(
            x: np.ndarray,
            shift: np.ndarray,
            f_shift: typing.Any,
            count_up: typing.Any,
            P: typing.Any,
            m_group: typing.Any,
            f_matrix: typing.Any,
            out: np.ndarray,
        ) -> None:
            x = x - shift
            z = x - f_shift
            result = 0.0
            for k in range(0, count_up):
                idx1 = P[k * m_group : (k + 1) * m_group]
                z1 = operator.dot_vm(z[idx1], f_matrix[: len(idx1), : len(idx1)])
                result += operator.ackley_func(z1)
            z2 = z[P[int(x.shape[0] / 2) :]]
            out[0] = result + operator.ackley_func(z2)

        super().__init__(
            ndim=ndim,
            bounds=bounds,
            f_shift=f_shift,
            f_matrix=f_matrix,
            m_group=m_group,
            default_bounds=np.array([-32.0, 32.0]) if default_bounds is None else default_bounds,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            shift=shift,
            prepare=prepare,
            x_global=x_global,
            f_global=f_global,
            load=load,
            paras=paras,
        )
        self._bind_kernel(compute, ["f_shift", "count_up", "P", "m_group", "f_matrix"], paras=paras)


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
        prepare: bool | None = None,
        *,
        default_bounds: typing.Any = None,
        x_global: typing.Any = None,
        f_global: typing.Any = None,
        load: typing.Callable[["CecBenchmark", int, _Traversable | None], None] | None = None,
        paras: typing.Union[
            dict[str, typing.Any], list[str], tuple[str, ...], typing.Callable[..., dict[str, typing.Any]], None
        ] = None,
    ) -> None:
        def compute(
            x: np.ndarray,
            shift: np.ndarray,
            f_shift: typing.Any,
            count_up: int,
            P: typing.Any,
            m_group: typing.Any,
            out: np.ndarray,
        ) -> None:
            x = x - shift
            z = x - f_shift
            result = 0.0
            for k in range(0, count_up):
                idx1 = P[k * m_group : (k + 1) * m_group]
                result += operator.schwefel_12_func(z[idx1])
            z2 = z[P[int(x.shape[0] / 2) :]]
            out[0] = result + operator.sphere_func(z2)

        def _load_extra(problem: "CecBenchmark", resolved_ndim: int, support_path: _Traversable | None) -> None:
            problem.count_up = int(resolved_ndim / (2 * problem.m_group))
            if load is not None:
                load(problem, resolved_ndim, support_path)

        super().__init__(
            ndim=ndim,
            bounds=bounds,
            f_shift=f_shift,
            m_group=m_group,
            default_bounds=default_bounds,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            shift=shift,
            prepare=prepare,
            x_global=x_global,
            f_global=f_global,
            load=_load_extra,
            paras=paras,
        )
        self._bind_kernel(compute, ["f_shift", "count_up", "P", "m_group"], paras=paras)


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
        prepare: bool | None = None,
    ) -> None:
        def compute(
            x: np.ndarray,
            shift: np.ndarray,
            f_shift: typing.Any,
            count_up: int,
            P: typing.Any,
            m_group: typing.Any,
            out: np.ndarray,
        ) -> None:
            x = x - shift
            z = x - f_shift
            result = 0.0
            for k in range(0, count_up):
                idx1 = P[k * m_group : (k + 1) * m_group]
                result += operator.rosenbrock_func(z[idx1])
            z2 = z[P[int(x.shape[0] / 2) :]]
            out[0] = result + operator.sphere_func(z2)

        def _optimum(problem: "CecBenchmark", dim: int) -> np.ndarray:
            half = int(dim / 2)
            xg = np.asarray(problem.f_shift).copy()
            xg[problem.P[:half]] = np.asarray(problem.f_shift)[problem.P[:half]] + 1
            xg[problem.P[half:]] = np.asarray(problem.f_shift)[problem.P[half:]]
            return xg

        super().__init__(
            ndim,
            bounds,
            f_shift,
            m_group,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            shift=shift,
            prepare=prepare,
            x_global=_optimum,
        )
        self.count_up = int(self.ndim / (2 * self.m_group))
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
        prepare: bool | None = None,
    ) -> None:
        def compute(
            x: np.ndarray,
            shift: np.ndarray,
            f_shift: typing.Any,
            count_up: int,
            P: typing.Any,
            m_group: typing.Any,
            f_matrix: typing.Any,
            out: np.ndarray,
        ) -> None:
            x = x - shift
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
            prepare=prepare,
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
        prepare: bool | None = None,
    ) -> None:
        def compute(
            x: np.ndarray,
            shift: np.ndarray,
            f_shift: typing.Any,
            count_up: int,
            P: typing.Any,
            m_group: typing.Any,
            f_matrix: typing.Any,
            out: np.ndarray,
        ) -> None:
            x = x - shift
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
            default_bounds=np.array([-5.0, 5.0]),
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            shift=shift,
            prepare=prepare,
        )
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
        prepare: bool | None = None,
    ) -> None:
        def compute(
            x: np.ndarray,
            shift: np.ndarray,
            f_shift: typing.Any,
            count_up: int,
            P: typing.Any,
            m_group: typing.Any,
            f_matrix: typing.Any,
            out: np.ndarray,
        ) -> None:
            x = x - shift
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
            default_bounds=np.array([-32.0, 32.0]),
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            shift=shift,
            prepare=prepare,
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
        prepare: bool | None = None,
    ) -> None:
        def compute(
            x: np.ndarray,
            shift: np.ndarray,
            f_shift: typing.Any,
            count_up: int,
            P: typing.Any,
            m_group: typing.Any,
            out: np.ndarray,
        ) -> None:
            x = x - shift
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
            prepare=prepare,
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
        prepare: bool | None = None,
    ) -> None:
        def compute(
            x: np.ndarray,
            shift: np.ndarray,
            f_shift: typing.Any,
            count_up: int,
            P: typing.Any,
            m_group: typing.Any,
            out: np.ndarray,
        ) -> None:
            x = x - shift
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
            prepare=prepare,
            x_global=lambda problem, dim: np.asarray(problem.f_shift + 1),
        )
        self.count_up = int(self.ndim / self.m_group)
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
        prepare: bool | None = None,
    ) -> None:
        def compute(x: np.ndarray, shift: np.ndarray, f_shift: typing.Any, out: np.ndarray) -> None:
            x = x - shift
            out[0] = operator.schwefel_12_func(x - f_shift)

        super().__init__(
            ndim,
            bounds,
            f_shift,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            shift=shift,
            prepare=prepare,
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
        prepare: bool | None = None,
    ) -> None:
        def compute(x: np.ndarray, shift: np.ndarray, f_shift: typing.Any, out: np.ndarray) -> None:
            x = x - shift
            out[0] = operator.rosenbrock_func(x - f_shift)

        super().__init__(
            ndim,
            bounds,
            f_shift,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            shift=shift,
            prepare=prepare,
            x_global=lambda problem, dim: np.asarray(problem.f_shift + 1),
        )
        self._bind_kernel(compute, ["f_shift"])
