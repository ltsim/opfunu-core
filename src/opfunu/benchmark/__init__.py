# Created by "LTSIM" at 10:47, 12/05/2026 ----------%
#       Email: tsim@cucei.udg.mx                    %
#       Github: https://github.com/ltsim            %
# --------------------------------------------------%
import abc
import dataclasses
import functools
import sys
import typing

if sys.version_info >= (3, 11):
    from importlib.resources.abc import Traversable
else:  # Python 3.10: Traversable lives in importlib.abc
    from importlib.abc import Traversable

import numpy as np

from opfunu.utils import numba_compat


@dataclasses.dataclass
class Formula:
    latex: str

    def _repr_latex_(self) -> str:
        return f"$${self.latex}$$"


class Benchmark(abc.ABC):
    """
    Abstract base class for an optimization benchmark problem.

    Subclasses supply a ``compute`` kernel and the problem-specific metadata; the
    shared evaluation machinery (``evaluate``, ``_evaluate_batch``), the shape
    storage behind ``bounds``/``ndim``/``lb``/``ub``, and the mathematical trait
    defaults below are implemented once, here.
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

    modality: bool = True  # Number of ambiguous peaks, unknown # peaks

    epsilon: typing.Final[float] = 1e-8

    #: Package-wide compilation policy for instances built without an explicit
    #: ``prepare`` argument. Set it before constructing problems to change the
    #: default for every subclass (e.g. ``Benchmark.default_prepare = False``).
    default_prepare: typing.ClassVar[bool] = True

    _gufunc_cache: typing.ClassVar[dict[tuple[typing.Any, ...], typing.Any]] = {}

    __shift: np.ndarray | None
    __base_bounds: np.ndarray | None
    __base_x_global: np.ndarray | None
    __init_depth: int

    #: Shape storage: (ndim, 2) bounds matrix and its dimensionality, written by
    #: the concrete ``check_ndim_and_bounds`` implementations and exposed through
    #: the read-only ``bounds``/``ndim``/``lb``/``ub`` properties.
    _ndim: int
    _bounds: np.ndarray

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
        prepare: bool | None = None,
    ) -> None:
        self.__parallel: bool = bool(parallel)
        self.__fastmath: bool = bool(fastmath)
        self.__dtype: np.dtype = np.dtype(dtype)
        self.__verbose: bool = bool(verbose)
        # ``None`` defers to the package-wide policy; a bool is an explicit
        # per-instance choice that always wins over the class default.
        self.__auto_prepare: bool = bool(type(self).default_prepare if prepare is None else prepare)
        self.__plain_kernel: bool = False
        self.__numba_compiled: bool = False
        self.__compile_error: Exception | None = None
        self.__kernel: typing.Callable[..., None] | None = compute
        self.__out: np.ndarray = np.empty(1, dtype=self.__dtype)
        self.__param_names: list[str] = []
        self.__compute: typing.Callable[..., None] | None = None
        self.__dim_changeable: bool = bool(dim_changeable)
        self.__dim_default: int = int(dim_default)
        # Stored raw; the translation runs post-construction (see __init_subclass__),
        # once the subclass has set its bounds and global optimum.
        self.__shift: np.ndarray | None = shift
        self.__base_bounds: np.ndarray | None = None
        self.__base_x_global: np.ndarray | None = None
        # Protected storage: written by subclasses while constructing data-dependent
        # metadata (loaded shift vectors, ndim-derived optima, display parameters).
        self._support_path: Traversable | None = None
        self._paras: dict[str, typing.Any] = {}
        self._f_global: float = float(f_global)
        self._x_global: np.ndarray = np.asarray(x_global if x_global is not None else [])
        # Shape storage; the concrete check_ndim_and_bounds of each base fills it.
        self._ndim = 0
        self._bounds = np.array([])

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
                    raw = getattr(self, "_Benchmark__shift", None)
                    if raw is not None:
                        self._apply_shift(raw)

            __init__.__opfunu_shift_wrapped__ = True  # type: ignore[attr-defined]
            cls.__init__ = __init__  # type: ignore[method-assign]

    @property
    def shift(self) -> np.ndarray | None:
        """The coordinate shift vector, or ``None`` when the problem is unshifted."""
        return getattr(self, "_Benchmark__shift", None)

    def _set_bounds(self, bounds: np.ndarray) -> None:
        """Write the (ndim, 2) bounds matrix behind the read-only ``bounds`` property."""
        self._bounds = np.asarray(bounds, dtype=float)

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

    def _apply_shift(self, shift: typing.Any) -> None:
        """Translate bounds and global optimum by ``shift`` (idempotent, re-settable)."""
        offset = self._normalize_shift(shift)
        if self.__base_bounds is None:
            self.__base_bounds = np.array(self.bounds, dtype=float, copy=True)
        if self.__base_x_global is None:
            try:
                current = np.array(self.x_global, dtype=float, copy=True)
            except (TypeError, ValueError):
                current = np.array([])
            self.__base_x_global = current
        assert self.__base_bounds is not None and self.__base_x_global is not None
        # Restore the base state first so repeated calls do not accumulate.
        self._set_bounds(self.__base_bounds.copy())
        base_x = self.__base_x_global
        if base_x.size > 0:
            self._x_global = base_x.copy()
        # Translate the search space: bounds_new = bounds_base + o.
        self._set_bounds(self.__base_bounds + offset[:, None])
        # Translate the optimum: x_global = x*_base + o.
        if base_x.size > 0:
            flat = base_x.ravel()
            if flat.size == offset.size:
                self._x_global = flat + offset
            else:
                raise ValueError(f"Cannot shift x_global of length {flat.size} with shift of length {offset.size}!")
        self.__shift = offset.copy()
        self._paras["shift"] = self.__shift.copy()

    def set_shift(self, shift: typing.Any = None) -> None:
        """Set (or reset with ``None``) the coordinate shift after construction."""
        if shift is None:
            if self.__base_bounds is not None:
                self._set_bounds(self.__base_bounds.copy())
            if self.__base_x_global is not None and self.__base_x_global.size > 0:
                self._x_global = self.__base_x_global.copy()
            self.__shift = None
            self._paras.pop("shift", None)
        else:
            self._apply_shift(shift)

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
        paras: dict[str, typing.Any] | list[str] | tuple[str, ...] | None = None,
    ) -> None:
        """
        Install ``compute`` and bind it for execution.

        The only sanctioned caller of the private ``__build_compute``
        The only sanctioned caller of the private ``__build_compute``
        helper. ``param_names`` lists the kernel parameters in call order
        (instance attributes read at bind time); ``paras`` is display
        metadata (a dict, or a list of attribute names resolved against
        ``self``; defaults to ``{name: getattr(self, name)}`` for
        ``param_names``).
        With ``plain=True`` the kernel is declared Numba-incompatible and
        bound uncompiled, whatever the ``prepare`` flag says; otherwise it is
        compiled through ``__build_compute`` when Numba is installed and the
        instance prepares eagerly (``prepare=True``, or the package-wide
        ``Benchmark.default_prepare`` policy). A deferred instance
        (``prepare=False``) binds the plain kernel here and compiles when
        ``prepare()`` is called.
        """
        if compute is not None:
            self.__kernel = compute
        names: list[str] = list(param_names) if param_names is not None else []
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
        self.__plain_kernel = bool(plain)
        if plain:
            self.__compute = self.__bind_plain(names, validate=False)
        elif numba_compat.HAS_NUMBA and self.__auto_prepare:
            self.__compute = self.__build_compute(names)
        else:
            # Numba absent, or compilation deferred (``prepare=False`` /
            # ``Benchmark.default_prepare=False``): bind the plain kernel.
            self.__compute = self.__bind_plain(names, validate=True)

    def prepare(self) -> bool:
        """
        Compile the bound kernel with Numba now (deferred compilation).

        Problems built with ``prepare=False`` (or with ``Benchmark.default_prepare``
        set to ``False``) bind a plain Python/NumPy kernel at construction and
        skip the ``guvectorize`` compilation and warm-up; this method performs
        that work on demand. It is safe to call at any time and repeatedly:
        an already-compiled kernel returns ``True`` without recompiling, and
        kernels declared Numba-incompatible (bound with ``plain=True``) are
        never compiled.

        Without Numba installed (PyPy, minimal installs) the call is a no-op
        returning ``False`` and recording no error. If compilation fails, the
        plain kernel stays bound, the exception is stored in ``_compile_error``
        (printed when ``verbose`` is enabled) and ``False`` is returned;
        evaluation is value-identical either way.

        Returns
        -------
        prepared : bool
            Whether the kernel is bound as a compiled gufunc after the call.
        """
        if self.__numba_compiled:
            return True
        if self.__plain_kernel or not numba_compat.HAS_NUMBA:
            return False
        self.__compute = self.__build_compute(self.__param_names)
        return self.__numba_compiled

    def __bind_plain(self, param_names: list[str], *, validate: bool) -> typing.Callable[..., None]:
        """
        Bind the uncompiled Python kernel and record that nothing was compiled.

        With ``validate=True`` the kernel parameters are normalized eagerly so
        mistyped names fail fast, exactly as on the compiled path. ``plain=True``
        kernels skip validation: they legitimately hold values
        ``_normalize_kernel_param`` rejects (sub-instances, bound methods).
        """
        self.__numba_compiled = False
        self.__compile_error = None
        if validate:
            for name in param_names:
                self._normalize_kernel_param(getattr(self, name))
        return self.__resolve_kernel()

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
        (in call order, after ``x`` and before ``out``). The caller decides
        whether a compilation attempt is made; this helper assumes Numba is
        installed. Attribute lookup and normalization errors propagate; only
        Numba compilation/warmup failures fall back to the plain Python
        kernel, keeping the same ``(x, *params, out)`` calling convention.
        On fallback the raised exception is stored in ``_compile_error``
        (useful for diagnosing silent fallbacks). Compilation (plus a warmup
        call) happens here — either at instantiation through ``_bind_kernel``
        or later through ``prepare()`` — never on first ``evaluate``.
        """
        kernel = self.__resolve_kernel()
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
            arr = np.ascontiguousarray(np.asarray(arr, dtype=float) - np.asarray(shift, dtype=float), dtype=self.dtype)
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

    def is_ndim_compatible(self, ndim: int | None) -> bool:
        """
        Method to support searching the functions with input ndim

        Parameters
        ----------
        ndim : int
                The number of dimensions

        Returns
        -------
        val: bool
             Always true if dim_changeable = True, Else return ndim == self.ndim
        """
        assert (ndim is None) or (isinstance(ndim, int) and (not ndim < 0)), (
            "The dimension ndim must be None or a positive integer"
        )
        if ndim is None:
            return True
        if self.dim_changeable:
            return ndim > 0
        return ndim == self.ndim

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

        # the solution should still be in bounds, otherwise immediate fail.
        if np.any(x > self.ub) or np.any(x < self.lb):
            return False

        val = float(self.evaluate(np.squeeze(x)))
        if np.abs(val - self.f_global) < tol:
            return True

        # you found a lower global minimum.  This shouldn't happen.
        if val < self.f_global:
            raise ValueError("Found a lower global minimum", x, val, self.f_global)
        return False

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

    def create_solution(self) -> np.ndarray:
        """
        Create a random solution for the current problem

        Returns
        -------
        solution: 1D-vector
            The random solution
        """
        return np.random.uniform(self.lb, self.ub)
