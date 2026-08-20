#!/usr/bin/env python
# Created by "Thieu" at 06:43, 30/06/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import importlib.resources
import typing

import numpy as np

from opfunu.benchmark import Benchmark


class CecBenchmark(Benchmark):
    """
    Defines an abstract class for optimization benchmark problem.

    All subclasses should implement the ``evaluate`` method for a particular optimization problem.

    Attributes
    ----------
    bounds : list
        The lower/upper bounds of the problem. This a 2D-matrix of [lower, upper] array that contain the lower and
        upper bounds.
        By default, each problem has its own bounds. But user can try to put different bounds to test the problem.
    ndim : int
        The dimensionality of the problem. It is calculated from bounds
    lb : np.ndarray
        The lower bounds for the problem
    ub : np.ndarray
        The upper bounds for the problem
    f_global : float
        The global optimum of the evaluated function.
    x_global : np.ndarray
        A list of vectors that provide the locations of the global minimum.
        Note that some problems have multiple global minima, not all of which may be listed.
    n_fe : int
        The number of function evaluations that the object has been asked to calculate.
    dim_changeable : bool
        Whether we can change the benchmark function `x` variable length (i.e., the dimensionality of the problem)
    """

    name: str = "Benchmark name"
    latex_formula: str = r"f(\mathbf{x})"
    latex_formula_dimension: str = r"d \in \mathbb{N}_{+}^{*}"
    latex_formula_bounds: str = r"x_i \in [-2\pi, 2\pi], \forall i \in \llbracket 1, d\rrbracket"
    latex_formula_global_optimum: str = r"f(0, ..., 0)=-1, \text{ for}, m=5, \beta=15"

    continuous: bool = True
    linear: bool = False
    convex: bool = True
    unimodal: bool = False
    separable: bool = False

    differentiable: bool = True
    scalable: bool = True
    randomized_term: bool = False
    parametric: bool = True
    shifted: bool = True
    rotated: bool = False

    modality: bool = True  # Number of ambiguous peaks, unknown # peaks

    characteristics: typing.ClassVar[list[str]]

    # n_basins = 1
    # n_valleys = 1

    f_shift: np.ndarray
    f_shuffle: np.ndarray
    f_matrix: np.ndarray
    f_matrix_a: np.ndarray
    f_matrix_b: np.ndarray
    f_bias: float
    f_global: float
    x_global: np.ndarray
    n_fe: int

    _ndim: int
    _bounds: np.ndarray
    _dim_changeable: bool
    _dim_default: int
    _dim_max: int
    _dim_supported: list[int] | None

    g0: typing.Any
    g1: typing.Any
    g2: typing.Any
    g3: typing.Any
    g4: typing.Any

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        f_shift: str | np.ndarray | None = None,
        f_matrix: str | np.ndarray | None = None,
        f_shuffle: str | np.ndarray | None = None,
        f_bias: float | None = None,
        default_bounds: typing.Any = None,
        dim_changeable: bool = True,
        dim_default: int = 30,
        dim_max: int = 100,
        dim_supported: list[int] | None = None,
        data_name: str = "",
        load_two_matrix: bool = False,
        load_tow_matrix: bool = False,
    ) -> None:
        super().__init__()

        self._dim_changeable = dim_changeable
        self._dim_default = dim_default
        self._dim_max = dim_max
        self._dim_supported = dim_supported

        self.check_ndim_and_bounds(ndim, dim_max, bounds, default_bounds)

        if not data_name:
            data_name = getattr(self, "data_name", "")
        if not data_name:
            import re

            match = re.search(r"20\d{2}", self.__class__.__name__) or re.search(r"20\d{2}", self.__module__)
            if match:
                data_name = f"data_{match.group(0)}"

        if data_name:
            self.make_support_data_path(data_name)

        self.f_matrix = np.array([])
        self.f_shuffle = np.array([])
        self.f_shift = np.array([])
        self.f_matrix_a = np.array([])
        self.f_matrix_b = np.array([])

        if load_two_matrix or load_tow_matrix:
            if not isinstance(f_shift, str):
                raise ValueError("The shift data should be a file name when loading two matrices!")
            shift_data, a_matrix, b_matrix = self.load_two_matrix_and_shift_data(f_shift)
            self.f_shift = shift_data[: self.ndim]
            self.f_matrix_a = a_matrix[: self.ndim, : self.ndim]
            self.f_matrix_b = b_matrix[: self.ndim, : self.ndim]
        else:
            if f_shift is not None:
                self.f_shift = self.check_shift_data(f_shift)[: self.ndim]
            if f_matrix is not None:
                self.f_matrix = self.check_matrix_data(f_matrix)
            if f_shuffle is not None:
                self.f_shuffle = self.check_shuffle_data(f_shuffle)

        self.f_bias = f_bias if f_bias is not None else 0.0
        self.f_global = f_bias if f_bias is not None else 0.0
        self.x_global = self.f_shift

        self.n_fe = 0

    @property
    def dim_max(self) -> int:
        return self._dim_max

    @dim_max.setter
    def dim_max(self, value: int) -> None:
        self._dim_max = value

    @property
    def dim_supported(self) -> list[int] | None:
        return self._dim_supported

    @dim_supported.setter
    def dim_supported(self, value: list[int] | None) -> None:
        self._dim_supported = value

    @property
    def dim_default(self) -> int:
        return self._dim_default

    @dim_default.setter
    def dim_default(self, value: int) -> None:
        self._dim_default = value

    @property
    def dim_changeable(self) -> bool:
        return self._dim_changeable

    @dim_changeable.setter
    def dim_changeable(self, value: bool) -> None:
        self._dim_changeable = value

    def make_support_data_path(self, data_name: str) -> None:
        self.support_path = importlib.resources.files("opfunu").joinpath(f"cec_based/{data_name}")

    def check_shift_data(self, f_shift: str | list | tuple | np.ndarray) -> np.ndarray:
        if isinstance(f_shift, str):
            return self.load_shift_data(f_shift)
        else:
            if isinstance(f_shift, (list, tuple, np.ndarray)):
                return np.squeeze(f_shift)
            else:
                raise ValueError("The shift data should be a list/tuple or np.array!")

    def check_shift_matrix(
        self,
        f_shift: str | list | tuple | np.ndarray,
        selected_idx: int | None = None,
    ) -> np.ndarray:
        if isinstance(f_shift, str):
            if selected_idx is None:
                return self.load_matrix_data(f_shift)
            else:
                return self.load_matrix_data(f_shift)[selected_idx, : self.ndim]
        else:
            if isinstance(f_shift, (list, tuple, np.ndarray)):
                return np.squeeze(f_shift)
            else:
                raise ValueError("The shift data should be a list/tuple or np.array!")

    def check_matrix_data(self, f_matrix: str | np.ndarray, needed_dim: bool = True) -> np.ndarray:
        if isinstance(f_matrix, str):
            if needed_dim:
                return self.load_matrix_data(f"{f_matrix}{self.ndim}")
            else:
                return self.load_matrix_data(f_matrix)
        else:
            if isinstance(f_matrix, np.ndarray):
                return np.squeeze(f_matrix)
            else:
                raise ValueError("The matrix data should be an orthogonal matrix (2D np.array)!")

    def check_shuffle_data(self, f_shuffle: str | list | tuple | np.ndarray, needed_dim: bool = True) -> np.ndarray:
        if isinstance(f_shuffle, str):
            if needed_dim:
                return self.load_shift_data(f"{f_shuffle}{self.ndim}")
            else:
                return self.load_shift_data(f_shuffle)
        else:
            if isinstance(f_shuffle, (list, tuple, np.ndarray)):
                return np.squeeze(f_shuffle)
            else:
                raise ValueError("The shuffle data should be a list/tuple or np.array!")

    def check_m_group(self, m_group: int | None = None) -> int:
        if isinstance(m_group, int):
            if int(self.ndim / m_group) > 1:
                return m_group
            else:
                raise ValueError("ndim is too small or m_group is too large!")
        else:
            raise ValueError("m_group is positive integer!")

    def load_shift_data(self, filename: str) -> np.ndarray:
        assert self.support_path is not None
        fname = filename if filename.endswith(".txt") else f"{filename}.txt"
        filepath = self.support_path.joinpath(fname)
        if hasattr(filepath, "open"):
            with filepath.open("r") as f:
                data = np.genfromtxt(f, dtype=float)
        else:
            data = np.genfromtxt(f"{self.support_path}/{fname}", dtype=float)
        return data.reshape((-1))

    def load_matrix_data(self, filename: str) -> np.ndarray:
        assert self.support_path is not None
        fname = filename if filename.endswith(".txt") else f"{filename}.txt"
        filepath = self.support_path.joinpath(fname)
        try:
            if hasattr(filepath, "open"):
                with filepath.open("r") as f:
                    data = np.genfromtxt(f, dtype=float)
            else:
                data = np.genfromtxt(f"{self.support_path}/{fname}", dtype=float)
            return data
        except (FileNotFoundError, OSError):
            print(f"The file named: {fname} is not found.")
            print(f"{self.__class__.__name__} problem is only supported ndim in {self.dim_supported}!")
            exit(1)

    def load_shift_and_matrix_data(self, filename: str) -> tuple[np.ndarray, np.ndarray]:
        assert self.support_path is not None
        fname = filename if filename.endswith(".txt") else f"{filename}.txt"
        filepath = self.support_path.joinpath(fname)
        if hasattr(filepath, "open"):
            with filepath.open("r") as f:
                data = np.genfromtxt(f, dtype=float)
        else:
            data = np.genfromtxt(f"{self.support_path}/{fname}", dtype=float)
        shift_data = data[:1, :].ravel()
        matrix_data = data[1:, :]
        return shift_data, matrix_data

    def load_two_matrix_and_shift_data(self, filename: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert self.support_path is not None
        fname = filename if filename.endswith(".txt") else f"{filename}.txt"
        filepath = self.support_path.joinpath(fname)
        if hasattr(filepath, "open"):
            with filepath.open("r") as f:
                data = np.genfromtxt(f, dtype=float)
        else:
            data = np.genfromtxt(f"{self.support_path}/{fname}", dtype=float)
        a_matrix = data[:100, :]
        b_matrix = data[100:200, :]
        shift_data = data[200:, :].ravel()
        return shift_data, a_matrix, b_matrix

    def check_solution(self, x: np.ndarray, dim_max: int | None = None, dim_support: list[int] | None = None) -> None:
        """
        Raise the error if the problem size is not equal to the solution length

        Parameters
        ----------
        x : np.ndarray
            The solution
        dim_max : The maximum number of variables that the function is supported
        dim_support : List of the supported dimensions
        """

        if len(x) != self._ndim:
            raise ValueError(
                f"{self.__class__.__name__} problem, the length of solution should have {self._ndim} variables!"
            )

        if (dim_max is not None) and (len(x) > dim_max):
            raise ValueError(f"{self.__class__.__name__} problem is not supported ndim > {dim_max}!")

        if (dim_support is not None) and (len(x) not in dim_support):
            raise ValueError(f"{self.__class__.__name__} problem is only supported ndim in {dim_support}!")

    def evaluate(self, x: np.ndarray) -> float:
        raise NotImplementedError

    def is_ndim_compatible(self, ndim: int | None) -> bool:
        assert (ndim is None) or (isinstance(ndim, int) and (not ndim < 0)), (
            "The dimension ndim must be None or a positive integer"
        )
        if ndim is None:
            return True
        else:
            if self.dim_changeable:
                return ndim > 0
            else:
                return ndim == self.ndim

    def is_succeed(self, x: np.ndarray, tol: float = 1.0e-5) -> bool:
        if np.any(x > self.ub) or np.any(x < self.lb):
            return False

        val = self.evaluate(np.squeeze(x))
        if np.abs(val - self.f_global) < tol:
            return True

        # you found a lower global minimum.  This shouldn't happen.
        if val < self.f_global:
            raise ValueError("Found a lower global minimum", x, val, self.f_global)
        return False

    def check_ndim_and_bounds(
        self,
        ndim: int | None = None,
        dim_max: int | None = None,
        bounds: typing.Any = None,
        default_bounds: typing.Any = None,
    ) -> None:
        """
        Check the bounds when initializing the object.

        Parameters
        ----------
        ndim : int
            The number of dimensions (variables)
        dim_max : int
            The maximum number of dimensions (variables) that the problem is supported
        bounds : list, tuple, np.ndarray
            List of lower bound and upper bound, should use default None value
        default_bounds : np.ndarray
            List of initial lower bound and upper bound values
        """

        if default_bounds is None:
            default_bounds = np.array([[-100.0, 100.0] for _ in range(self.dim_default)])
        else:
            default_bounds = np.array(default_bounds)
            if default_bounds.ndim == 1 and len(default_bounds) == 2:
                default_bounds = np.array([default_bounds for _ in range(self.dim_default)])
            elif default_bounds.ndim == 2 and default_bounds.shape[0] == 1:
                default_bounds = np.array([default_bounds[0] for _ in range(self.dim_default)])

        if ndim is None:
            self._bounds = default_bounds if bounds is None else np.array(bounds).T
            self._ndim = self._bounds.shape[0]

            if dim_max is not None and self._ndim > dim_max:
                raise ValueError(f"{self.__class__.__name__} problem supports maximum {dim_max} variables!")
        else:
            if bounds is None:
                if self.dim_changeable:
                    if isinstance(ndim, int) and ndim > 1:
                        if dim_max is None or ndim <= dim_max:
                            self._ndim = int(ndim)
                            self._bounds = np.array([default_bounds[0] for _ in range(self._ndim)])
                        else:
                            raise ValueError(f"{self.__class__.__name__} problem supports maximum {dim_max} variables!")
                    else:
                        raise ValueError("ndim must be an integer and > 1!")
                else:
                    self._ndim = self.dim_default
                    self._bounds = default_bounds
                    if self.verbose:
                        print(f"{self.__class__.__name__} is fixed problem with {self.dim_default} variables!")
            else:
                if self.dim_changeable:
                    self._bounds = np.array(bounds).T
                    self._ndim = self._bounds.shape[0]
                    if dim_max is not None and self._ndim > dim_max:
                        raise ValueError(f"{self.__class__.__name__} problem supports maximum {dim_max} variables!")
                    else:
                        print(f"{self.__class__.__name__} problem is set with {self._ndim} variables!")
                else:
                    self._bounds = np.array(bounds).T
                    if self._bounds.shape[0] == self.dim_default:
                        self._ndim = self.dim_default
                    else:
                        raise ValueError(
                            f"{self.__class__.__name__} is fixed problem with {self._ndim} variables. "
                            "Please setup the correct bounds!"
                        )
        return

    @property
    def bounds(self) -> np.ndarray:
        """
        The lower/upper bounds to be used for optimization problem. This a 2D-matrix of [lower, upper] array that
        contain the lower and upper bounds for the problem. The problem should not be asked for evaluation outside
        these bounds. ``len(bounds) == ndim``.
        """
        return self._bounds

    @property
    def ndim(self) -> int:
        """
        The dimensionality of the problem.

        Returns
        -------
        ndim : int
            The dimensionality of the problem
        """
        return self._ndim

    @property
    def lb(self) -> np.ndarray:
        """
        The lower bounds for the problem

        Returns
        -------
        lb : 1D-vector
            The lower bounds for the problem
        """
        return np.array([x[0] for x in self.bounds])

    @property
    def ub(self) -> np.ndarray:
        """
        The upper bounds for the problem

        Returns
        -------
        ub : 1D-vector
            The upper bounds for the problem
        """
        return np.array([x[1] for x in self.bounds])
