#!/usr/bin/env python
# Created by "Thieu" at 06:43, 30/06/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import importlib.resources
import re
import sys
import typing

if sys.version_info >= (3, 11):
    from importlib.resources.abc import Traversable
else:  # Python 3.10: Traversable lives in importlib.abc
    from importlib.abc import Traversable

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
    #: Construction-time context: the shape and data folder resolved by ``__init__``
    #: before ``Benchmark.__init__`` installs them, so the loaders (and the class
    #: ``load`` hooks) can be called during construction without re-passing them.
    __ctx_ndim: int | None
    __ctx_path: Traversable | None

    #: Per-problem data a class attaches to itself — either in its ``load`` hook or in
    #: its constructor — and reads back as kernel parameters through ``_bind_kernel``.
    #: These are annotations only (no values), so nothing exists until a subclass fills
    #: them in; they are declared here because the typed ``load`` hooks receive the
    #: problem as a ``CecBenchmark``. ``a``/``b``/``p``/``y`` are the coefficient blocks,
    #: ``lamdas``/``xichmas``/``bias`` the scaling and shift vectors, ``n_funcs`` the
    #: number of composed sub-functions, ``g0``-``g4`` those sub-problems, ``n1``-``n5``
    #: and ``idx1``-``idx6`` the size/index blocks of a hybrid grouping, ``count_up`` the
    #: number of groups and ``m_group`` the group size.
    M: typing.Any
    C: typing.Any
    P: typing.Any
    y: typing.Any
    a: typing.Any
    b: typing.Any
    p: typing.Any
    lamdas: typing.Any
    xichmas: typing.Any
    bias: typing.Any
    n_funcs: int
    m_group: int
    count_up: int
    k_max: int
    n1: int
    n2: int
    n3: int
    n4: int
    n5: int
    idx1: typing.Any
    idx2: typing.Any
    idx3: typing.Any
    idx4: typing.Any
    idx5: typing.Any
    idx6: typing.Any
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
        paras: typing.Union[
            dict[str, typing.Any], list[str], tuple[str, ...], typing.Callable[..., dict[str, typing.Any]], None
        ] = None,
        verbose: bool = False,
        shift: typing.Any = None,
        prepare: bool | None = None,
        *,
        x_global: typing.Any = None,
        f_global: typing.Any = None,
        load: typing.Callable[["CecBenchmark", int, Traversable | None], None] | None = None,
    ) -> None:
        # Own configuration first: the competition data is loaded below, before the
        # immutable Benchmark state (shape, optimum, support folder) is installed by
        # ``super().__init__()``. There is no post-construction state to patch.
        self.__ctx_ndim: int | None = None
        self.__ctx_path: Traversable | None = None
        self.__dim_max: int = dim_max
        self.__dim_supported: list[int] | None = dim_supported
        self.__f_matrix = np.array([])
        self.__f_shuffle = np.array([])
        self.__f_shift = np.array([])
        self.__f_matrix_a = np.array([])
        self.__f_matrix_b = np.array([])
        self.f_bias = f_bias if f_bias is not None else 0.0

        if type(self) is CecBenchmark:
            raise TypeError("CecBenchmark is abstract; subclass it and provide compute()")

        resolved_ndim, resolved_bounds = self.check_ndim_and_bounds(
            ndim,
            dim_max,
            bounds,
            default_bounds,
            dim_changeable=dim_changeable,
            dim_default=dim_default,
            verbose=verbose,
        )

        if not data_name:
            match = _YEAR_SUFFIX.search(self.__class__.__name__) or _YEAR_SUFFIX.search(self.__module__)
            if match:
                data_name = f"data_{match.group(1)}"
        support_path = self.make_support_data_path(data_name) if data_name else None
        # Loaders and class hooks run before Benchmark.__init__ installs the state, so
        # they read the shape and data folder through this construction-time context.
        self.__ctx_ndim, self.__ctx_path = resolved_ndim, support_path

        if load_two_matrix:
            if not isinstance(f_shift, str):
                raise ValueError("The shift data should be a file name when loading two matrices!")
            shift_data, a_matrix, b_matrix = self.load_two_matrix_and_shift_data(f_shift, support_path=support_path)
            self.f_shift = shift_data[:resolved_ndim]
            self.f_matrix_a = a_matrix[:resolved_ndim, :resolved_ndim]
            self.f_matrix_b = b_matrix[:resolved_ndim, :resolved_ndim]
        else:
            if f_shift is not None:
                self.f_shift = self.check_shift_data(f_shift, support_path=support_path)[:resolved_ndim]
            if f_matrix is not None:
                self.f_matrix = self.check_matrix_data(f_matrix, ndim=resolved_ndim, support_path=support_path)
            if f_shuffle is not None:
                self.f_shuffle = self.check_shuffle_data(f_shuffle, ndim=resolved_ndim, support_path=support_path)

        # Class-specific loads and adjustments (matrix row selection, reshaping,
        # permutation tweaks) run here so the optimum below sees the final data.
        if load is not None:
            load(self, resolved_ndim, support_path)

        # ``x_global``/``f_global`` are values or callables ``(problem, ndim)`` evaluated
        # on the loaded data; the CEC defaults are the official shift and the bias.
        resolved_f_global = (
            f_global(self, resolved_ndim) if callable(f_global) else (f_global if f_global is not None else self.f_bias)
        )
        resolved_x_global = (
            x_global(self, resolved_ndim)
            if callable(x_global)
            else (x_global if x_global is not None else self.f_shift)
        )

        super().__init__(
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            compute=compute,
            f_global=resolved_f_global,
            x_global=resolved_x_global,
            dim_changeable=dim_changeable,
            dim_default=dim_default,
            verbose=verbose,
            shift=shift,
            prepare=prepare,
            ndim=resolved_ndim,
            bounds=resolved_bounds,
            support_path=support_path,
        )
        self.__ctx_ndim, self.__ctx_path = None, None

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

    def make_support_data_path(self, data_name: str) -> Traversable:
        """Resolve the bundled ``data_*`` folder for this competition (pure resolver)."""
        return importlib.resources.files("opfunu").joinpath(f"cec_based/{data_name}")

    @property
    def _data_ndim(self) -> int:
        """The dimensionality to read data for: the state, or the constructor's context."""
        return self.ndim if self.__ctx_ndim is None else self.__ctx_ndim

    @property
    def _data_path(self) -> Traversable | None:
        """The support-data folder to read from: the state, or the constructor's context."""
        return self.support_path if self.__ctx_path is None else self.__ctx_path

    def check_shift_data(
        self,
        f_shift: str | list[typing.Any] | tuple[typing.Any, ...] | np.ndarray,
        *,
        support_path: Traversable | None = None,
    ) -> np.ndarray:
        if isinstance(f_shift, str):
            return self.load_shift_data(f_shift, support_path=support_path)
        else:
            if isinstance(f_shift, (list, tuple, np.ndarray)):
                return np.squeeze(f_shift)
            else:
                raise ValueError("The shift data should be a list/tuple or np.array!")

    def check_shift_matrix(
        self,
        f_shift: str | list[typing.Any] | tuple[typing.Any, ...] | np.ndarray,
        selected_idx: int | None = None,
        *,
        ndim: int | None = None,
        support_path: Traversable | None = None,
    ) -> np.ndarray:
        if isinstance(f_shift, str):
            dim = self._data_ndim if ndim is None else ndim
            if selected_idx is None:
                return self.load_matrix_data(f_shift, support_path=support_path)
            else:
                return self.load_matrix_data(f_shift, support_path=support_path)[selected_idx, :dim]
        else:
            if isinstance(f_shift, (list, tuple, np.ndarray)):
                return np.squeeze(f_shift)
            else:
                raise ValueError("The shift data should be a list/tuple or np.array!")

    def check_matrix_data(
        self,
        f_matrix: str | np.ndarray,
        needed_dim: bool = True,
        *,
        ndim: int | None = None,
        support_path: Traversable | None = None,
    ) -> np.ndarray:
        if isinstance(f_matrix, str):
            dim = self._data_ndim if ndim is None else ndim
            if needed_dim:
                return self.load_matrix_data(f"{f_matrix}{dim}", support_path=support_path)
            else:
                return self.load_matrix_data(f_matrix, support_path=support_path)
        else:
            if isinstance(f_matrix, np.ndarray):
                return np.squeeze(f_matrix)
            else:
                raise ValueError("The matrix data should be an orthogonal matrix (2D np.array)!")

    def check_shuffle_data(
        self,
        f_shuffle: str | list[typing.Any] | tuple[typing.Any, ...] | np.ndarray,
        needed_dim: bool = True,
        *,
        ndim: int | None = None,
        support_path: Traversable | None = None,
    ) -> np.ndarray:
        if isinstance(f_shuffle, str):
            dim = self._data_ndim if ndim is None else ndim
            if needed_dim:
                return self.load_shift_data(f"{f_shuffle}{dim}", support_path=support_path)
            else:
                return self.load_shift_data(f_shuffle, support_path=support_path)
        else:
            if isinstance(f_shuffle, (list, tuple, np.ndarray)):
                return np.squeeze(f_shuffle)
            else:
                raise ValueError("The shuffle data should be a list/tuple or np.array!")

    def check_m_group(self, m_group: int | None = None, *, ndim: int | None = None) -> int:
        dim = self._data_ndim if ndim is None else ndim
        if isinstance(m_group, int):
            if int(dim / m_group) > 1:
                return m_group
            else:
                raise ValueError("ndim is too small or m_group is too large!")
        else:
            raise ValueError("m_group is positive integer!")

    def _load_table(self, filename: str, *, support_path: Traversable | None = None) -> np.ndarray:
        """Read one bundled ``data_*`` text table as a float array."""
        path = self._data_path if support_path is None else support_path
        assert path is not None
        fname = filename if filename.endswith(".txt") else f"{filename}.txt"
        with path.joinpath(fname).open("r") as f:
            return np.genfromtxt(f, dtype=float)

    def load_shift_data(self, filename: str, *, support_path: Traversable | None = None) -> np.ndarray:
        return self._load_table(filename, support_path=support_path).reshape((-1))

    def load_matrix_data(self, filename: str, *, support_path: Traversable | None = None) -> np.ndarray:
        try:
            return self._load_table(filename, support_path=support_path)
        except (FileNotFoundError, OSError):
            fname = filename if filename.endswith(".txt") else f"{filename}.txt"
            print(f"The file named: {fname} is not found.")
            print(f"{self.__class__.__name__} problem is only supported ndim in {self.dim_supported}!")
            exit(1)

    def load_shift_and_matrix_data(
        self, filename: str, *, support_path: Traversable | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        data = self._load_table(filename, support_path=support_path)
        shift_data = data[:1, :].ravel()
        matrix_data = data[1:, :]
        return shift_data, matrix_data

    def load_two_matrix_and_shift_data(
        self, filename: str, *, support_path: Traversable | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        data = self._load_table(filename, support_path=support_path)
        a_matrix = data[:100, :]
        b_matrix = data[100:200, :]
        shift_data = data[200:, :].ravel()
        return shift_data, a_matrix, b_matrix

    def check_solution(self, x: np.ndarray, dim_max: int | None = None, dim_support: list[int] | None = None) -> None:
        """
        Raise the error if the problem size is not equal to the solution length.

        :param x: The solution.
        :param dim_max: The maximum number of variables that the function is supported.
        :param dim_support: List of the supported dimensions.
        """

        if dim_max is None:
            dim_max = self.dim_max
        if dim_support is None:
            dim_support = self.dim_supported

        if len(x) != self.ndim:
            raise ValueError(
                f"{self.__class__.__name__} problem, the length of solution should have {self.ndim} variables!"
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
        *,
        dim_changeable: bool = True,
        dim_default: int = 30,
        verbose: bool = False,
    ) -> tuple[int, np.ndarray]:
        """
        Resolve the dimensionality and the (ndim, 2) bounds matrix (pure resolver).

        Validates the requested shape against the competition limits and returns
        ``(ndim, bounds)`` without touching any state; ``__init__`` installs the
        result through ``super().__init__()``.

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
        dim_changeable : bool
            Whether the dimensionality can be changed at construction
        dim_default : int
            The fallback dimensionality for fixed-size problems
        verbose : bool
            Whether the fixed-dimension notice is printed
        """
        if default_bounds is None:
            resolved_defaults = np.array([[-100.0, 100.0] for _ in range(dim_default)])
        else:
            resolved_defaults = np.array(default_bounds)
            if resolved_defaults.ndim == 1 and len(resolved_defaults) == 2:
                resolved_defaults = np.array([resolved_defaults for _ in range(dim_default)])
            elif resolved_defaults.ndim == 2 and resolved_defaults.shape[0] == 1:
                resolved_defaults = np.array([resolved_defaults[0] for _ in range(dim_default)])

        if ndim is None:
            resolved_bounds = resolved_defaults if bounds is None else np.array(bounds).T
            resolved_ndim = len(resolved_bounds)
            if dim_max is not None and resolved_ndim > dim_max:
                raise ValueError(f"{self.__class__.__name__} problem supports maximum {dim_max} variables!")
            return resolved_ndim, np.asarray(resolved_bounds, dtype=float)

        if bounds is None:
            if dim_changeable:
                if isinstance(ndim, int) and ndim > 1:
                    if dim_max is None or ndim <= dim_max:
                        return ndim, np.array([resolved_defaults[0] for _ in range(ndim)])
                    raise ValueError(f"{self.__class__.__name__} problem supports maximum {dim_max} variables!")
                raise ValueError("ndim must be an integer and > 1!")
            if verbose:
                print(f"{self.__class__.__name__} is fixed problem with {dim_default} variables!")
            return dim_default, np.asarray(resolved_defaults, dtype=float)

        resolved_bounds = np.array(bounds).T
        resolved_ndim = len(resolved_bounds)
        if dim_changeable:
            if dim_max is not None and resolved_ndim > dim_max:
                raise ValueError(f"{self.__class__.__name__} problem supports maximum {dim_max} variables!")
            print(f"{self.__class__.__name__} problem is set with {resolved_ndim} variables!")
            return resolved_ndim, resolved_bounds
        if resolved_ndim == dim_default:
            return dim_default, resolved_bounds
        raise ValueError(
            f"{self.__class__.__name__} is fixed problem with {resolved_ndim} variables. "
            "Please setup the correct bounds!"
        )
