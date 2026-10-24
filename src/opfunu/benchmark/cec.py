#!/usr/bin/env python
# Created by "Thieu" at 06:43, 30/06/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import importlib.resources
import re
import typing

import numpy as np

from opfunu.benchmark import Benchmark

#: Trailing competition year of a CEC class name (``F202017`` -> ``2017``), used
#: to auto-resolve the bundled ``data_*`` folder when ``data_name`` is not passed.
_YEAR_SUFFIX = re.compile(r"(20\d{2})$")


class CecBenchmark(Benchmark):
    """
    Defines an abstract class for optimization benchmark problem.

    All subclasses should implement the ``compute`` method for a particular problem; the shared
    ``evaluate``, shape (``bounds``/``ndim``/``lb``/``ub``) and ``is_succeed`` machinery comes from
    ``Benchmark``. CEC problems add the competition data handling: ``dim_max``/``dim_supported``
    limits and the shift/matrix/shuffle/bias terms loaded from the bundled ``data_*`` folders.
    """

    shifted: bool = True
    rotated: bool = False

    characteristics: typing.ClassVar[list[str]]

    __dim_max: int
    __dim_supported: list[int] | None
    __f_shift: np.ndarray
    __f_shuffle: np.ndarray
    __f_matrix: np.ndarray
    __f_matrix_a: np.ndarray
    __f_matrix_b: np.ndarray
    __f_bias: float

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
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
        compute: typing.Callable[..., None] | None = None,
        param_names: list[str] | None = None,
        plain: bool = False,
        paras: dict[str, typing.Any] | list[str] | tuple[str, ...] | None = None,
        verbose: bool = False,
        shift: typing.Any = None,
        prepare: bool | None = None,
    ) -> None:
        super().__init__(
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            compute=compute,
            dim_changeable=dim_changeable,
            dim_default=dim_default,
            verbose=verbose,
            shift=shift,
            prepare=prepare,
        )

        self.__dim_max: int = dim_max
        self.__dim_supported: list[int] | None = dim_supported

        self.check_ndim_and_bounds(ndim, dim_max, bounds, default_bounds)

        if not data_name:
            match = _YEAR_SUFFIX.search(self.__class__.__name__) or _YEAR_SUFFIX.search(self.__module__)
            if match:
                data_name = f"data_{match.group(1)}"

        if data_name:
            self.make_support_data_path(data_name)

        self.__f_matrix = np.array([])
        self.__f_shuffle = np.array([])
        self.__f_shift = np.array([])
        self.__f_matrix_a = np.array([])
        self.__f_matrix_b = np.array([])

        if load_two_matrix:
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
        self._f_global = float(f_bias if f_bias is not None else 0.0)
        self._x_global = self.f_shift

        if type(self) is CecBenchmark:
            raise TypeError("CecBenchmark is abstract; subclass it and provide compute()")

        if compute is not None or param_names is not None or plain or paras is not None:
            self._bind_kernel(compute, param_names or [], plain=plain, paras=paras)

    @property
    def dim_max(self) -> int:
        return self.__dim_max

    @property
    def dim_supported(self) -> list[int] | None:
        return self.__dim_supported

    @dim_supported.setter
    def dim_supported(self, value: list[int] | None) -> None:
        self.__dim_supported = value

    @property
    def f_shift(self) -> np.ndarray:
        return self.__f_shift

    @f_shift.setter
    def f_shift(self, value: typing.Any) -> None:
        self.__f_shift = np.asarray(value) if not isinstance(value, np.ndarray) else value

    @property
    def f_shuffle(self) -> np.ndarray:
        return self.__f_shuffle

    @f_shuffle.setter
    def f_shuffle(self, value: typing.Any) -> None:
        self.__f_shuffle = np.asarray(value) if not isinstance(value, np.ndarray) else value

    @property
    def f_matrix(self) -> np.ndarray:
        return self.__f_matrix

    @f_matrix.setter
    def f_matrix(self, value: typing.Any) -> None:
        self.__f_matrix = np.asarray(value) if not isinstance(value, np.ndarray) else value

    @property
    def f_matrix_a(self) -> np.ndarray:
        return self.__f_matrix_a

    @f_matrix_a.setter
    def f_matrix_a(self, value: typing.Any) -> None:
        self.__f_matrix_a = np.asarray(value) if not isinstance(value, np.ndarray) else value

    @property
    def f_matrix_b(self) -> np.ndarray:
        return self.__f_matrix_b

    @f_matrix_b.setter
    def f_matrix_b(self, value: typing.Any) -> None:
        self.__f_matrix_b = np.asarray(value) if not isinstance(value, np.ndarray) else value

    @property
    def f_bias(self) -> float:
        return self.__f_bias

    @f_bias.setter
    def f_bias(self, value: float) -> None:
        self.__f_bias = float(value)

    def make_support_data_path(self, data_name: str) -> None:
        self._support_path = importlib.resources.files("opfunu").joinpath(f"cec_based/{data_name}")

    def check_shift_data(self, f_shift: str | list[typing.Any] | tuple[typing.Any, ...] | np.ndarray) -> np.ndarray:
        if isinstance(f_shift, str):
            return self.load_shift_data(f_shift)
        else:
            if isinstance(f_shift, (list, tuple, np.ndarray)):
                return np.squeeze(f_shift)
            else:
                raise ValueError("The shift data should be a list/tuple or np.array!")

    def check_shift_matrix(
        self,
        f_shift: str | list[typing.Any] | tuple[typing.Any, ...] | np.ndarray,
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

    def check_shuffle_data(
        self, f_shuffle: str | list[typing.Any] | tuple[typing.Any, ...] | np.ndarray, needed_dim: bool = True
    ) -> np.ndarray:
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

    def _load_table(self, filename: str) -> np.ndarray:
        """Read one bundled ``data_*`` text table as a float array."""
        assert self.support_path is not None
        fname = filename if filename.endswith(".txt") else f"{filename}.txt"
        with self.support_path.joinpath(fname).open("r") as f:
            return np.genfromtxt(f, dtype=float)

    def load_shift_data(self, filename: str) -> np.ndarray:
        return self._load_table(filename).reshape((-1))

    def load_matrix_data(self, filename: str) -> np.ndarray:
        try:
            return self._load_table(filename)
        except (FileNotFoundError, OSError):
            fname = filename if filename.endswith(".txt") else f"{filename}.txt"
            print(f"The file named: {fname} is not found.")
            print(f"{self.__class__.__name__} problem is only supported ndim in {self.dim_supported}!")
            exit(1)

    def load_shift_and_matrix_data(self, filename: str) -> tuple[np.ndarray, np.ndarray]:
        data = self._load_table(filename)
        shift_data = data[:1, :].ravel()
        matrix_data = data[1:, :]
        return shift_data, matrix_data

    def load_two_matrix_and_shift_data(self, filename: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        data = self._load_table(filename)
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

        if dim_max is None:
            dim_max = self.dim_max
        if dim_support is None:
            dim_support = self.dim_supported

        if len(x) != self._ndim:
            raise ValueError(
                f"{self.__class__.__name__} problem, the length of solution should have {self._ndim} variables!"
            )

        if (dim_max is not None) and (len(x) > dim_max):
            raise ValueError(f"{self.__class__.__name__} problem is not supported ndim > {dim_max}!")

        if (dim_support is not None) and (len(x) not in dim_support):
            raise ValueError(f"{self.__class__.__name__} problem is only supported ndim in {dim_support}!")

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
