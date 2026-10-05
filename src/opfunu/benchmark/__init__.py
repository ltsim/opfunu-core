# Created by "LTSIM" at 10:47, 12/05/2026 ----------%
#       Email: tsim@cucei.udg.mx                    %
#       Github: https://github.com/ltsim            %
# --------------------------------------------------%
import abc
import dataclasses
import typing
from importlib.resources.abc import Traversable

import numpy as np

from opfunu.utils import numba_compat


@dataclasses.dataclass
class Formula:
    latex: str

    def _repr_latex_(self) -> str:
        return f"$${self.latex}$$"


class Benchmark(abc.ABC):
    latex_formula: str = r"f(\mathbf{x})"
    latex_formula_dimension: str = r"d \in \mathbb{N}_{+}^{*}"
    latex_formula_bounds: str = r"x_i \in [-2\pi, 2\pi], \forall i \in \llbracket 1, d\rrbracket"
    latex_formula_global_optimum: str = r"f(0, ..., 0)=-1, \text{ for}, m=5, \beta=15"

    epsilon: typing.Final[float] = 1e-8

    _gufunc_cache: typing.ClassVar[dict[tuple[typing.Any, ...], typing.Any]] = {}

    def __init__(
        self,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
        compute: typing.Callable[..., None] | None = None,
        f_global: float = 0.0,
        x_global: typing.Any = None,
        dim_changeable: bool = False,
        dim_default: int = 2,
    ) -> None:
        self.__parallel: bool = bool(parallel)
        self.__fastmath: bool = bool(fastmath)
        self.__dtype: np.dtype = np.dtype(dtype)
        self.__support_path: Traversable | None = None
        self.__verbose: bool = False
        self.__paras: dict[str, typing.Any] = {}
        self.__numba_compiled: bool = False
        self.__compile_error: Exception | None = None
        self.__kernel: typing.Callable[..., None] | None = compute
        self.__out: np.ndarray = np.empty(1, dtype=self.__dtype)
        self.__param_names: list[str] = []
        self.__compute: typing.Callable[..., None] | None = None
        self.__f_global: float = float(f_global)
        self.__x_global: np.ndarray = np.asarray(x_global if x_global is not None else [])
        self.__dim_changeable: bool = bool(dim_changeable)
        self.__dim_default: int = int(dim_default)

    # --- configuration exposed through properties ---------------------------
    @property
    def parallel(self) -> bool:
        """Whether the gufunc kernel is compiled with ``target="parallel"``."""
        return self.__parallel

    @parallel.setter
    def parallel(self, value: bool) -> None:
        self.__parallel = bool(value)

    @property
    def fastmath(self) -> bool:
        """Whether the gufunc kernel is compiled with ``fastmath=True``."""
        return self.__fastmath

    @fastmath.setter
    def fastmath(self, value: bool) -> None:
        self.__fastmath = bool(value)

    @property
    def dtype(self) -> np.dtype:
        """The floating scalar type used for evaluation."""
        return self.__dtype

    @dtype.setter
    def dtype(self, value: typing.Any) -> None:
        self.__dtype = np.dtype(value)

    @property
    def support_path(self) -> Traversable | None:
        """The package-data directory holding shift/rotation files (CEC only)."""
        return self.__support_path

    @support_path.setter
    def support_path(self, value: Traversable | None) -> None:
        self.__support_path = value

    @property
    def verbose(self) -> bool:
        """Whether Numba compilation fallbacks are printed."""
        return self.__verbose

    @verbose.setter
    def verbose(self, value: bool) -> None:
        self.__verbose = bool(value)

    @property
    def paras(self) -> dict[str, typing.Any]:
        """Display metadata mapping parameter names to their runtime values."""
        return self.__paras

    @paras.setter
    def paras(self, value: dict[str, typing.Any]) -> None:
        self.__paras = dict(value)

    @property
    def numba_compiled(self) -> bool:
        """Whether the bound kernel is a compiled gufunc (``False`` for plain Python)."""
        return self.__numba_compiled

    @numba_compiled.setter
    def numba_compiled(self, value: bool) -> None:
        self.__numba_compiled = bool(value)

    # --- private metadata exposed through read/write properties ---------------
    @property
    def f_global(self) -> float:
        """The known global optimum of the problem."""
        return self.__f_global

    @f_global.setter
    def f_global(self, value: float) -> None:
        self.__f_global = float(value)

    @property
    def x_global(self) -> np.ndarray:
        """A location of the global optimum (1-D array, ``len(x_global) == ndim``)."""
        return self.__x_global

    @x_global.setter
    def x_global(self, value: typing.Any) -> None:
        self.__x_global = np.asarray(value)

    @property
    def dim_changeable(self) -> bool:
        """Whether the problem dimensionality may be changed at construction."""
        return self.__dim_changeable

    @dim_changeable.setter
    def dim_changeable(self, value: bool) -> None:
        self.__dim_changeable = bool(value)

    @property
    def dim_default(self) -> int:
        """The default dimensionality used when ``ndim`` is not supplied."""
        return self.__dim_default

    @dim_default.setter
    def dim_default(self, value: int) -> None:
        self.__dim_default = int(value)

    # --- kernel state (kept names for compatibility, backed by private storage)
    @property
    def compute(self) -> typing.Callable[..., None] | None:
        """The declared kernel (read-only). Bound for execution via ``_bind_kernel``."""
        return self.__kernel

    @property
    def _kernel(self) -> typing.Callable[..., None] | None:
        """The instance kernel closure (kept name; prefer the ``compute`` property)."""
        return self.__kernel

    @_kernel.setter
    def _kernel(self, value: typing.Callable[..., None] | None) -> None:
        self.__kernel = value

    @property
    def _param_names(self) -> list[str]:
        """Names of the instance attributes the kernel takes (after ``x``, before ``out``)."""
        return self.__param_names

    @_param_names.setter
    def _param_names(self, value: list[str]) -> None:
        self.__param_names = list(value)

    @property
    def _compute(self) -> typing.Callable[..., None] | None:
        """The bound kernel (compiled gufunc or plain-Python fallback)."""
        return self.__compute

    @_compute.setter
    def _compute(self, value: typing.Callable[..., None] | None) -> None:
        self.__compute = value

    @property
    def _compile_error(self) -> Exception | None:
        """The compilation exception when falling back to plain Python (``None`` on success)."""
        return self.__compile_error

    @_compile_error.setter
    def _compile_error(self, value: Exception | None) -> None:
        self.__compile_error = value

    @property
    def _out(self) -> np.ndarray:
        """Scratch output buffer reused by single-point evaluation."""
        return self.__out

    @_out.setter
    def _out(self, value: np.ndarray) -> None:
        self.__out = value

    def _bind_kernel(
        self,
        compute: typing.Callable[..., None] | None = None,
        param_names: list[str] | None = None,
        *,
        plain: bool = False,
        params: dict[str, typing.Any] | None = None,
        paras: dict[str, typing.Any] | list[str] | tuple[str, ...] | None = None,
    ) -> None:
        """
        Install ``compute`` and bind it for execution.

        The only sanctioned caller of the private ``__build_compute``
        helper. ``params`` sets extra attributes before binding;
        ``param_names`` lists the kernel parameters in call order
        (defaults to ``params`` keys); ``paras`` is display metadata
        (a dict, or a list of attribute names resolved against ``self``;
        defaults to ``{name: getattr(self, name)}`` for ``param_names``).
        With ``plain=True`` the kernel is bound uncompiled (no Numba
        attempt); otherwise it is compiled via ``__build_compute``.
        """
        if compute is not None:
            self._kernel = compute
        if params:
            for key, val in params.items():
                setattr(self, key, val)
        names: list[str] = list(param_names) if param_names is not None else list(params.keys() if params else [])
        self._param_names = names
        if paras is None:
            if names:
                self.paras = {name: getattr(self, name) for name in names}
            else:
                self.paras = {}
        elif isinstance(paras, dict):
            self.paras = dict(paras)
        elif isinstance(paras, (list, tuple)):
            self.paras = {name: getattr(self, name) for name in paras}
        else:
            raise TypeError(f"Unsupported paras metadata type: {type(paras)}")
        if self._kernel is None:
            raise RuntimeError(f"{type(self).__name__} has no kernel; pass compute= to _bind_kernel first.")
        if plain:
            self._compute = self.__resolve_kernel()
            self.numba_compiled = False
            self._compile_error = None
        else:
            self._compute = self.__build_compute(names)

    def __resolve_kernel(self) -> typing.Callable[..., None]:
        """Return the instance-bound kernel, raising when none was installed."""
        kernel = self.__kernel
        if kernel is None:
            raise RuntimeError(f"{type(self).__name__} has no kernel; call _bind_kernel first.")
        return kernel

    def _normalize_kernel_param(self, value: typing.Any) -> tuple[typing.Any, typing.Any]:
        """
        Map a runtime attribute value to ``(call_value, numpy_dtype)``.

        Float arrays are cast to ``self.dtype``; integer arrays are cast
        to ``int64`` (index arrays must stay integer); int scalars become
        ``int64``; float scalars become ``self.dtype``; bool scalars become
        ``bool``; numeric Python lists (including nested lists / matrices)
        are converted to arrays. A zero-dimensional array unwraps to a scalar.
        Anything else raises ``TypeError``.

        The returned dtype is a plain ``numpy.dtype`` so this method works
        without Numba installed; ``__build_compute`` converts it with
        ``numba.from_dtype`` when compiling.
        """
        if isinstance(value, np.ndarray) and value.ndim == 0:
            value = value.item()
        if isinstance(value, list):
            try:
                value = np.asarray(value)
            except (TypeError, ValueError) as exc:
                raise TypeError(f"Unsupported kernel parameter: {exc}") from exc
        if isinstance(value, np.ndarray):
            kind = value.dtype.kind
            arr: np.ndarray
            npdtype: np.dtype
            if kind in ("i", "u"):
                arr = np.ascontiguousarray(value, dtype=np.int64)
                npdtype = np.dtype(np.int64)
            elif kind == "f":
                arr = np.ascontiguousarray(value, dtype=self.dtype)
                npdtype = self.dtype
            else:
                raise TypeError(f"Unsupported kernel array dtype: {value.dtype}")
            if arr.ndim not in (1, 2):
                raise TypeError(f"Unsupported kernel array with ndim={arr.ndim}")
            return arr, npdtype
        if isinstance(value, (bool, np.bool_)):
            return bool(value), np.dtype(bool)
        if isinstance(value, (int, np.integer)):
            return int(value), np.dtype(np.int64)
        if isinstance(value, (float, np.floating)):
            return self.dtype.type(value), self.dtype
        raise TypeError(f"Unsupported kernel parameter type: {type(value)}")

    def __build_compute(self, param_names: list[str]) -> typing.Callable[..., None]:
        """
        Compile the bound kernel into a ``guvectorize`` gufunc.

        ``param_names`` lists the instance attributes the kernel needs
        (in call order, after ``x`` and before ``out``). Attribute lookup
        and normalization errors propagate; only Numba compilation/warmup
        failures fall back to the plain Python kernel, keeping the same
        ``(x, *params, out)`` calling convention. When Numba is not installed
        (PyPy, minimal installs) the plain kernel is used directly without
        attempting compilation. Compilation (plus a warmup call) happens here,
        i.e. at instantiation, never on first ``evaluate``. On fallback the
        raised exception is stored in ``_compile_error`` (useful for diagnosing
        silent fallbacks; ``None`` when Numba is simply absent).
        """
        kernel = self.__resolve_kernel()
        if not numba_compat.HAS_NUMBA:
            self.numba_compiled = False
            self._compile_error = None
            # Validate eagerly so mistyped params fail fast, exactly as in the
            # compiled path (lookup/normalization errors propagate; only
            # compile/warmup failures fall back).
            for name in param_names:
                self._normalize_kernel_param(getattr(self, name))
            return kernel
        import numba as nb

        # Let AttributeError/TypeError/ValueError propagate: a mistyped name
        # or an un-normalizable value is a bug, not a compile failure.
        normed = [self._normalize_kernel_param(getattr(self, name)) for name in param_names]
        ndims = [call.ndim if isinstance(call, np.ndarray) else 0 for call, _ in normed]
        # `n` is reserved for the input vector `x` with layout `(n)`.
        alphabet = "abcdefghijklmopqrstuvwxyz"
        needed = sum(2 if d == 2 else (1 if d == 1 else 0) for d in ndims)
        if needed > len(alphabet):
            raise TypeError(f"Too many kernel dimensions for a gufunc signature: {needed} > {len(alphabet)}")
        letters = iter(alphabet)
        nb_float = nb.from_dtype(self.dtype)
        layouts = ["(n)"]
        nbtypes: list[typing.Any] = [nb_float[:]]
        for (_, npdtype), ndim in zip(normed, ndims):
            nbdtype: typing.Any = nb.from_dtype(np.dtype(npdtype))
            if ndim == 0:
                layouts.append("()")
                nbtypes.append(nbdtype)
            elif ndim == 1:
                layouts.append(f"({next(letters)})")
                nbtypes.append(nbdtype[:])
            else:
                layouts.append(f"({next(letters)},{next(letters)})")
                nbtypes.append(nbdtype[:, :])
        nbtypes.append(nb_float[:])
        sig = ",".join(layouts) + "->()"
        # Key on the kernel's code object: stable across instances of the same
        # class, yet distinct between a parent and a child that define their own
        # (otherwise identically signed) kernels. Requires kernels to be
        # self-contained, i.e. not to close over per-instance values.
        key = (getattr(kernel, "__code__", kernel), sig, str(self.dtype), self.parallel, self.fastmath)
        self._compile_error = None
        try:
            gufunc = Benchmark._gufunc_cache.get(key)
            if gufunc is None:
                target = "parallel" if self.parallel else "cpu"
                guvectorize: typing.Any = nb.guvectorize
                gufunc = guvectorize([tuple(nbtypes)], sig, nopython=True, target=target, fastmath=self.fastmath)(
                    kernel
                )
                Benchmark._gufunc_cache[key] = gufunc
            x0 = np.zeros(self.ndim, dtype=self.dtype)
            if self._out.dtype != self.dtype:
                self._out = np.empty(1, dtype=self.dtype)
            gufunc(x0, *[v for v, _ in normed], self._out)
            self.numba_compiled = True
            return typing.cast(typing.Callable[..., None], gufunc)
        except Exception as exc:
            self.numba_compiled = False
            self._compile_error = exc
            if self.verbose:
                print(f"{type(self).__name__}: Numba compilation failed, using Python kernel: {exc}")
            return kernel

    def _cast_kernel_param(self, value: typing.Any) -> typing.Any:
        """
        Cast a runtime attribute value to the exact object the kernel expects.

        Total: values that ``_normalize_kernel_param`` rejects (sub-instances,
        bound methods, ...) are passed through unchanged — they only occur for
        plain kernels bound with ``plain=True``, which handle them natively.
        (If ``__build_compute`` succeeded, every value normalizes cleanly, so
        the gufunc path always sees normalized values.)
        """
        try:
            call_value, _ = self._normalize_kernel_param(value)
            return call_value
        except (TypeError, ValueError):
            return value

    def __cast_result(self, value: typing.Any) -> np.floating | np.integer:
        """Cast the raw kernel output to the scalar type configured by ``dtype``."""
        return typing.cast("np.floating | np.integer", self.dtype.type(value))

    def __evaluate(self, x: np.ndarray) -> np.floating | np.integer:
        """Run the kernel on one candidate and return the ``dtype``-cast scalar."""
        compute = self._compute
        if compute is None:
            raise RuntimeError(f"{type(self).__name__} has no kernel; call _bind_kernel first.")
        arr = np.ascontiguousarray(x, dtype=self.dtype)
        params = [self._cast_kernel_param(getattr(self, name)) for name in self._param_names]
        compute(arr, *params, self._out)
        return self.__cast_result(self._out[0])

    def _evaluate_batch(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate a whole population in a single gufunc call.

        ``x`` may be ``(N, ndim)`` (or ``(ndim,)``, treated as ``N=1``); the
        returned ``(N,)`` array has dtype ``self.dtype``.
        """
        compute = self._compute
        if compute is None:
            raise RuntimeError(f"{type(self).__name__} has no kernel; call _bind_kernel first.")
        arr = np.ascontiguousarray(x, dtype=self.dtype)
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        if not self.numba_compiled:
            # Plain-Python kernels are scalar ``(x, *params, out)`` writers; loop.
            return np.asarray([self.__evaluate(row) for row in arr], dtype=self.dtype)
        out = np.empty(arr.shape[0], dtype=self.dtype)
        params = [self._cast_kernel_param(getattr(self, name)) for name in self._param_names]
        compute(arr, *params, out)
        return out

    def _repr_latex_(self) -> str:
        return f"$${self.latex_formula}$$"

    @property
    def formula(self) -> Formula:
        return Formula(self.latex_formula)

    @property
    def formula_dimension(self) -> Formula:
        return Formula(self.latex_formula_dimension)

    @property
    def formula_bounds(self) -> Formula:
        return Formula(self.latex_formula_bounds)

    @property
    def formula_global_optimum(self) -> Formula:
        return Formula(self.latex_formula_global_optimum)

    @abc.abstractmethod
    def check_ndim_and_bounds(self, *args: typing.Any, **kwargs: typing.Any) -> None:
        """
        Check the bounds when initializing the object.

        Parameters
        ----------
        ndim : int
            The number of dimensions (variables)
        bounds : list, tuple, np.ndarray
            List of lower bound and upper bound, should use default None value
        default_bounds : np.ndarray
            List of initial lower bound and upper bound values
        """
        ...

    @abc.abstractmethod
    def check_solution(self, x: np.ndarray) -> None:
        """
        Raise the error if the problem size is not equal to the solution length

        Parameters
        ----------
        x : np.ndarray
            The solution
        """
        ...

    def get_paras(self) -> dict[str, typing.Any]:
        """
        Return the parameters of the problem. Depended on function
        """
        default = {"bounds": self.bounds, "ndim": self.ndim}
        return {**default, **self.paras}

    def evaluate(self, x: np.ndarray, *args: typing.Any) -> np.floating | np.integer:
        """
        Evaluation of the benchmark function.

        Parameters
        ----------
        x : np.ndarray
            The candidate vector for evaluating the benchmark problem. Must have ``len(x) == self.ndim``.

        Returns
        -------
        val : np.floating | np.integer
               the evaluated benchmark function, cast to ``self.dtype``
        """
        self.check_solution(x)
        return self.__evaluate(x)

    @abc.abstractmethod
    def is_ndim_compatible(self, ndim: int | None) -> bool: ...

    @abc.abstractmethod
    def is_succeed(self, x: np.ndarray, tol: float = 1.0e-5) -> bool:
        """
        Check if a candidate solution at the global minimum.

        Parameters
        ----------
        x : np.ndarray
            The candidate vector for testing if the global minimum has been reached. Must have ``len(x) == self.ndim``
        tol : float
            The evaluated function and known global minimum must differ by less than this amount to be at a
            global minimum.

        Returns
        -------
        is_succeed : bool
            Answer the question: is the candidate vector at the global minimum?
        """
        ...

    @property
    @abc.abstractmethod
    def bounds(self) -> np.ndarray:
        """
        The lower/upper bounds to be used for optimization problem. This a 2D-matrix of [lower, upper] array that
        contain the lower and upper bounds for the problem. The problem should not be asked for evaluation outside
        these bounds. ``len(bounds) == ndim``.
        """
        ...

    @property
    @abc.abstractmethod
    def ndim(self) -> int:
        """
        The dimensionality of the problem.

        Returns
        -------
        ndim : int
            The dimensionality of the problem
        """
        ...

    @property
    @abc.abstractmethod
    def lb(self) -> np.ndarray:
        """
        The lower bounds for the problem

        Returns
        -------
        lb : 1D-vector
            The lower bounds for the problem
        """
        ...

    @property
    @abc.abstractmethod
    def ub(self) -> np.ndarray:
        """
        The upper bounds for the problem

        Returns
        -------
        ub : 1D-vector
            The upper bounds for the problem
        """
        ...

    def create_solution(self) -> np.ndarray:
        """
        Create a random solution for the current problem

        Returns
        -------
        solution: 1D-vector
            The random solution
        """
        return np.random.uniform(self.lb, self.ub)
