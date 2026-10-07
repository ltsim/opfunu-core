# Created by "LTSIM" at 10:47, 12/05/2026 ----------%
#       Email: tsim@cucei.udg.mx                    %
#       Github: https://github.com/ltsim            %
# --------------------------------------------------%
import abc
import dataclasses
import functools
import typing

try:  # Python >= 3.11
    from importlib.resources.abc import Traversable
except ImportError:  # Python 3.10: Traversable lives in importlib.abc
    from importlib.abc import Traversable

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

    __shift: np.ndarray | None
    __rotate: np.ndarray | None
    __rotate_bounds: bool
    __base_bounds: np.ndarray | None
    __base_x_global: np.ndarray | None
    __init_depth: int

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
        verbose: bool = False,
        shift: typing.Any = None,
        rotate: typing.Any = None,
        rotate_bounds: bool = True,
    ) -> None:
        self.__parallel: bool = bool(parallel)
        self.__fastmath: bool = bool(fastmath)
        self.__dtype: np.dtype = np.dtype(dtype)
        self.__verbose: bool = bool(verbose)
        self.__numba_compiled: bool = False
        self.__compile_error: Exception | None = None
        self.__kernel: typing.Callable[..., None] | None = compute
        self.__out: np.ndarray = np.empty(1, dtype=self.__dtype)
        self.__param_names: list[str] = []
        self.__compute: typing.Callable[..., None] | None = None
        self.__dim_changeable: bool = bool(dim_changeable)
        self.__dim_default: int = int(dim_default)
        # Stored raw; validation runs post-construction (see __init_subclass__),
        # once the subclass has set its bounds and global optimum.
        self.__shift: np.ndarray | None = shift
        self.__rotate: np.ndarray | None = rotate
        self.__rotate_bounds: bool = bool(rotate_bounds)
        self.__base_bounds: np.ndarray | None = None
        self.__base_x_global: np.ndarray | None = None
        # Protected storage: written by subclasses while constructing data-dependent
        # metadata (loaded shift vectors, ndim-derived optima, display parameters).
        self._support_path: Traversable | None = None
        self._paras: dict[str, typing.Any] = {}
        self._f_global: float = float(f_global)
        self._x_global: np.ndarray = np.asarray(x_global if x_global is not None else [])

    def __init_subclass__(cls, **kwargs: typing.Any) -> None:
        super().__init_subclass__(**kwargs)
        orig_init = cls.__dict__.get("__init__")
        if orig_init is not None and not getattr(orig_init, "__opfunu_shift_wrapped__", False):

            @functools.wraps(orig_init)
            def __init__(self: "Benchmark", *args: typing.Any, **kw: typing.Any) -> None:
                # Nested constructors (e.g. CEC chains calling super().__init__)
                # re-enter this wrapper; only the outermost exit applies the shift,
                # so subclass bodies mutating state after super().__init__() are seen.
                depth = getattr(self, "_Benchmark__init_depth", 0) + 1
                self.__init_depth = depth
                try:
                    orig_init(self, *args, **kw)
                finally:
                    self.__init_depth = depth - 1
                if depth == 1:
                    raw_shift = getattr(self, "_Benchmark__shift", None)
                    if raw_shift is not None:
                        self.__shift = self._normalize_shift(raw_shift)
                    raw_rotate = getattr(self, "_Benchmark__rotate", None)
                    if raw_rotate is not None:
                        self.__rotate = self._normalize_rotate(raw_rotate)
                    if (
                        getattr(self, "_Benchmark__shift", None) is not None
                        or getattr(self, "_Benchmark__rotate", None) is not None
                    ):
                        self._recompute_transform()

            __init__.__opfunu_shift_wrapped__ = True  # type: ignore[attr-defined]
            cls.__init__ = __init__  # type: ignore[method-assign]

    @property
    def shift(self) -> np.ndarray | None:
        """The coordinate shift vector, or ``None`` when the problem is unshifted."""
        return getattr(self, "_Benchmark__shift", None)

    @property
    def rotate(self) -> np.ndarray | None:
        """The coordinate rotation matrix, or ``None`` when the problem is unrotated."""
        return getattr(self, "_Benchmark__rotate", None)

    def _set_bounds(self, bounds: np.ndarray) -> None:
        """Write the (ndim, 2) bounds matrix into the subclass's private storage."""
        raise NotImplementedError

    def _normalize_shift(self, shift: typing.Any) -> np.ndarray:
        """Validate a user shift vector against the problem dimensionality."""
        ndim = int(self.ndim)
        if isinstance(shift, (int, float, np.integer, np.floating)):
            offset = np.full(ndim, float(shift))
        else:
            try:
                offset = np.asarray(shift, dtype=float).ravel()
            except (TypeError, ValueError) as exc:
                raise ValueError(f"The shift must be a scalar or a 1D array-like of length {ndim}!") from exc
            if offset.size == 1 and ndim > 1:
                offset = np.full(ndim, float(offset[0]))
        if offset.size != ndim:
            raise ValueError(f"The shift length ({offset.size}) must match ndim ({ndim})!")
        if not np.all(np.isfinite(offset)):
            raise ValueError("The shift vector must contain only finite numbers!")
        return offset

    def _normalize_rotate(self, rotate: typing.Any) -> np.ndarray:
        """Validate a user rotation matrix: shape ``(ndim, ndim)``, finite, orthogonal."""
        ndim = int(self.ndim)
        try:
            matrix = np.asarray(rotate, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"The rotation must be a ({ndim}, {ndim}) orthogonal matrix!") from exc
        if matrix.shape != (ndim, ndim):
            raise ValueError(f"The rotation shape {matrix.shape} must be ({ndim}, {ndim})!")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("The rotation matrix must contain only finite numbers!")
        if not np.allclose(matrix.T @ matrix, np.eye(ndim)):
            raise ValueError("The rotation matrix must be orthogonal (M.T @ M must be the identity)!")
        return matrix

    def _recompute_transform(self) -> None:
        """Restore the base state, then apply rotation and shift (idempotent, re-settable).

        Geometry follows ``f(x) = f_base(M @ (x - o))``: the pullback through
        ``M.T`` (rotated optimum, enclosing axis-aligned box of the rotated
        bounds) happens before the translation by ``o``.
        """
        matrix = self.__rotate
        offset = self.__shift
        if matrix is None and offset is None:
            # Reset: restore the pristine base state captured on first use.
            if self.__base_bounds is not None:
                self._set_bounds(self.__base_bounds.copy())
            if self.__base_x_global is not None and self.__base_x_global.size > 0:
                self._x_global = self.__base_x_global.copy()
            self._paras.pop("shift", None)
            self._paras.pop("rotate", None)
            return
        if self.__base_bounds is None:
            self.__base_bounds = np.array(self.bounds, dtype=float, copy=True)
        if self.__base_x_global is None:
            try:
                current = np.array(self.x_global, dtype=float, copy=True)
            except (TypeError, ValueError):
                current = np.array([])
            self.__base_x_global = current
        assert self.__base_bounds is not None and self.__base_x_global is not None
        bounds = self.__base_bounds
        x_opt = self.__base_x_global
        if matrix is not None:
            if self.__rotate_bounds:
                # Enclosing AABB of M.T @ [lb, ub]: center -> M.T @ c,
                # half-widths -> |M.T| @ h (a rotation grows the box).
                center = (bounds[:, 0] + bounds[:, 1]) / 2.0
                half = (bounds[:, 1] - bounds[:, 0]) / 2.0
                new_center = matrix.T @ center
                new_half = np.abs(matrix.T) @ half
                bounds = np.column_stack([new_center - new_half, new_center + new_half])
            # rotate_bounds=False keeps the rigid base box (CEC-style); only
            # x_global still maps through M.T either way.
            if x_opt.size > 0:
                flat = x_opt.ravel()
                if flat.size == matrix.shape[1]:
                    x_opt = matrix.T @ flat
                else:
                    raise ValueError(f"Cannot rotate x_global of length {flat.size} with a {matrix.shape} matrix!")
        if offset is not None:
            # Translate the search space and the (rotated) optimum: + o.
            bounds = bounds + offset[:, None]
            if x_opt.size > 0:
                flat = x_opt.ravel()
                if flat.size == offset.size:
                    x_opt = flat + offset
                else:
                    raise ValueError(f"Cannot shift x_global of length {flat.size} with shift of length {offset.size}!")
        self._set_bounds(bounds)
        if self.__base_x_global.size > 0:
            self._x_global = x_opt
        if offset is not None:
            self._paras["shift"] = offset.copy()
        else:
            self._paras.pop("shift", None)
        if matrix is not None:
            self._paras["rotate"] = matrix.copy()
        else:
            self._paras.pop("rotate", None)

    def set_shift(self, shift: typing.Any = None) -> None:
        """Set (or reset with ``None``) the coordinate shift after construction."""
        self.__shift = None if shift is None else self._normalize_shift(shift)
        self._recompute_transform()

    def set_rotate(self, rotate: typing.Any = None) -> None:
        """Set (or reset with ``None``) the coordinate rotation after construction."""
        self.__rotate = None if rotate is None else self._normalize_rotate(rotate)
        self._recompute_transform()

    # --- configuration exposed through read-only properties ------------------
    # Every property on this class is reader-only: values are supplied to
    # ``__init__`` (or to ``_bind_kernel``) at construction time. Subclasses
    # write the data-dependent fields through the protected ``_f_global``,
    # ``_x_global``, ``_paras`` and ``_support_path`` attributes.
    @property
    def parallel(self) -> bool:
        """Whether the gufunc kernel is compiled with ``target="parallel"`` (read-only)."""
        return self.__parallel

    @property
    def fastmath(self) -> bool:
        """Whether the gufunc kernel is compiled with ``fastmath=True`` (read-only)."""
        return self.__fastmath

    @property
    def dtype(self) -> np.dtype:
        """The floating scalar type used for evaluation (read-only)."""
        return self.__dtype

    @property
    def support_path(self) -> Traversable | None:
        """The package-data directory holding shift/rotation files (CEC only, read-only)."""
        return self._support_path

    @property
    def verbose(self) -> bool:
        """Whether Numba compilation fallbacks are printed (read-only)."""
        return self.__verbose

    @property
    def paras(self) -> dict[str, typing.Any]:
        """Display metadata mapping parameter names to their runtime values (read-only)."""
        return self._paras

    @property
    def numba_compiled(self) -> bool:
        """Whether the bound kernel is a compiled gufunc (``False`` for plain Python, read-only)."""
        return self.__numba_compiled

    @property
    def f_global(self) -> float:
        """The known global optimum of the problem (read-only)."""
        return self._f_global

    @property
    def x_global(self) -> np.ndarray:
        """A location of the global optimum (1-D array, ``len(x_global) == ndim``, read-only)."""
        return self._x_global

    @property
    def dim_changeable(self) -> bool:
        """Whether the problem dimensionality may be changed at construction (read-only)."""
        return self.__dim_changeable

    @property
    def dim_default(self) -> int:
        """The default dimensionality used when ``ndim`` is not supplied (read-only)."""
        return self.__dim_default

    # --- kernel state (kept names for compatibility, all read-only) ----------
    @property
    def compute(self) -> typing.Callable[..., None] | None:
        """The declared kernel (read-only). Bound for execution via ``_bind_kernel``."""
        return self.__kernel

    @property
    def _kernel(self) -> typing.Callable[..., None] | None:
        """The instance kernel closure (read-only; prefer the ``compute`` property)."""
        return self.__kernel

    @property
    def _param_names(self) -> list[str]:
        """Names of the instance attributes the kernel takes (after ``x``, before ``out``, read-only)."""
        return self.__param_names

    @property
    def _compute(self) -> typing.Callable[..., None] | None:
        """The bound kernel (compiled gufunc or plain-Python fallback, read-only)."""
        return self.__compute

    @property
    def _compile_error(self) -> Exception | None:
        """The compilation exception when falling back to plain Python (``None`` on success, read-only)."""
        return self.__compile_error

    @property
    def _out(self) -> np.ndarray:
        """Scratch output buffer reused by single-point evaluation (read-only)."""
        return self.__out

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
            self.__kernel = compute
        if params:
            for key, val in params.items():
                setattr(self, key, val)
        names: list[str] = list(param_names) if param_names is not None else list(params.keys() if params else [])
        self.__param_names = names
        if paras is None:
            if names:
                self._paras = {name: getattr(self, name) for name in names}
            else:
                self._paras = {}
        elif isinstance(paras, dict):
            self._paras = dict(paras)
        elif isinstance(paras, (list, tuple)):
            self._paras = {name: getattr(self, name) for name in paras}
        else:
            raise TypeError(f"Unsupported paras metadata type: {type(paras)}")
        if self.__kernel is None:
            raise RuntimeError(f"{type(self).__name__} has no kernel; pass compute= to _bind_kernel first.")
        if plain:
            self.__compute = self.__resolve_kernel()
            self.__numba_compiled = False
            self.__compile_error = None
        else:
            self.__compute = self.__build_compute(names)

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
            self.__numba_compiled = False
            self.__compile_error = None
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
        self.__compile_error = None
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
            if self.__out.dtype != self.dtype:
                self.__out = np.empty(1, dtype=self.dtype)
            gufunc(x0, *[v for v, _ in normed], self.__out)
            self.__numba_compiled = True
            return typing.cast(typing.Callable[..., None], gufunc)
        except Exception as exc:
            self.__numba_compiled = False
            self.__compile_error = exc
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
        shift = self.__shift
        if shift is not None:
            x = np.asarray(x, dtype=float) - np.asarray(shift, dtype=float)
        rotate = self.__rotate
        if rotate is not None:
            # f(x) = f_base(M @ (x - o)): translate first, then rotate.
            x = np.asarray(rotate, dtype=float) @ np.asarray(x, dtype=float)
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
        shift = self.__shift
        if shift is not None:
            arr = np.asarray(arr, dtype=float) - np.asarray(shift, dtype=float)
        rotate = self.__rotate
        if rotate is not None:
            # Row-stacked candidates: z = M @ x per row is arr @ M.T.
            arr = np.asarray(arr, dtype=float) @ np.asarray(rotate, dtype=float).T
        if shift is not None or rotate is not None:
            arr = np.ascontiguousarray(arr, dtype=self.dtype)
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
