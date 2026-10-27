# Benchmark Base Classes

Every function in `opfunu-core` inherits from one of the base classes defined in the `opfunu.benchmark` package:

| Class            | Description                                                  | Used by                       |
|------------------|--------------------------------------------------------------|-------------------------------|
| `Benchmark`      | Abstract base class defining the common function interface   | all functions                 |
| `Formula`        | Dataclass holding a LaTeX formula (rendered by IPython)      | all functions                 |
| `FuncBenchmark`  | Base class for traditional (name-based) functions            | `opfunu.name_based.*`         |
| `CecBenchmark`   | Base class for CEC competition functions                     | `opfunu.cec_based.*`          |

## Benchmark and Formula

```{eval-rst}
.. automodule:: opfunu.benchmark
   :members: Formula, Benchmark
```

## FuncBenchmark

```{eval-rst}
.. autoclass:: opfunu.benchmark.func.FuncBenchmark
   :members:
   :undoc-members:
```

## CecBenchmark

```{eval-rst}
.. autoclass:: opfunu.benchmark.cec.CecBenchmark
   :members:
   :undoc-members:
```

## Coordinate shifts

Every benchmark function accepts an optional `shift` constructor argument (a
scalar is broadcast to all dimensions). It is an ordinary parameter of each
class's `__init__` and is forwarded through `super().__init__(...)` all the way
to `Benchmark`, which stores it privately. The shift is fixed when the problem
is constructed, passed to the kernel as its second argument, and normalized
once at construction (same scalar-broadcast / length / finite checks, still
eager). The stored bounds and optimum stay unshifted while the `bounds`/`lb`/`ub`/
`x_global` properties apply the normalized offset on read (`f_global` is
invariant, since `f(x) = f_base(x - o)`). Building a new problem is the only
way to change it.

```python
from opfunu.name_based import Ackley01

func = Ackley01(ndim=3, shift=[2.0, -1.5, 3.0])
print(func.x_global)  # base optimum + shift
print(func.bounds)  # base bounds + shift
print(func.is_succeed(func.x_global))  # True
```

`ndim`, `bounds`, `lb`, `ub` remain public read-only properties, plus a
read-only `shift` property. `get_paras()` still reports `"shift"` when the
problem is shifted.

## Implementing a problem

A name-based problem defines its kernel inline and hands the metadata to the base class:

```python
class Quadratic(FuncBenchmark):
    def __init__(self, ndim=None, bounds=None, parallel=False, fastmath=True, dtype=np.float64):
        def compute(x, shift, out):
            x = x - shift
            out[0] = (
                -3803.84
                - 138.08 * x[0]
                - 232.92 * x[1]
                + 128.08 * x[0] ** 2
                + 203.64 * x[1] ** 2
                + 182.25 * x[0] * x[1]
            )

        super().__init__(
            compute=compute,
            ndim=ndim,
            bounds=bounds,
            default_bounds=np.array([[-10.0, 10.0] for _ in range(2)]),
            f_global=-3873.72418,
            x_global=[0.19388, 0.48513],
            dim_changeable=False,
            dim_default=2,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
        )
```

`x_global` and `f_global` may also be callables of `ndim` (useful when the optimum scales with
the dimension).

Simple CEC problems declare the kernel straight in the `super().__init__` call —
the base loads the shift/rotation data, then binds:

```python
def compute(x, shift, f_shift, f_bias, out):
    x = x - shift
    out[0] = ...


super().__init__(
    ndim=ndim,
    bounds=bounds,
    f_shift=f_shift,
    f_bias=f_bias,
    compute=compute,
    param_names=["f_shift", "f_bias"],
    parallel=parallel,
    fastmath=fastmath,
    dtype=dtype,
)
```

Configuration flags (`parallel`, `fastmath`, `dtype`, `shift`, `prepare`) stay
ordinary parameters of every class's `__init__` and are forwarded to
`super().__init__(...)`; `prepare` defaults to `None` so the package-wide
`Benchmark.default_prepare` policy applies unless the caller chooses explicitly.

Problems that load or derive their own competition data (matrix row selection, reshaping,
permutation vectors, composition weights, index blocks, …) do it *before* the state is
installed, through the base's `load` hook, then bind explicitly:

```python
def load(problem, ndim, support_path):          # runs before super().__init__ installs the state
    problem.f_shift = problem.check_shift_matrix(f_shift, selected_idx=0, ndim=ndim, support_path=support_path)

super().__init__(..., load=load, x_global=lambda problem, ndim: np.asarray(problem.f_shift[0]))
self._bind_kernel(compute, ["f_shift", "P", "m_group", "f_matrix"], paras=[...])  # plain=True if Numba-incompatible
```

`x_global=` and `f_global=` accept a value or a `(problem, ndim)` callable and default to the
official shift and the bias, so most classes pass nothing at all.

A CEC class that is meant to be subclassed (the hybrid/composition families) takes the same
knobs as keyword-only arguments — `default_bounds=`, `x_global=`, `f_global=`, `load=`, and,
when it binds its own kernel, `compute=`/`param_names=`/`paras=` — and forwards them to
`super().__init__()` with `None` meaning "keep my own value". A subclass therefore re-describes
the data, the optimum or the kernel through the constructor instead of touching state:

```python
class F202010(F12010):                     # F12010 forwards default_bounds/x_global/...
    def __init__(self, ndim=None, bounds=None, f_shift="f01_o", ...):
        super().__init__(ndim, bounds, f_shift, ..., x_global=lambda problem, dim: np.asarray(problem.f_shift + 1))
        self._bind_kernel(compute, ["f_shift"])     # only when the kernel itself differs
```

A `load=` hook passed to such a class runs *after* the parent's own loading, so it can adjust
what the parent loaded (`problem.f_shift[0, 1::2] = 5`) before the optimum is derived from it.

`compute` and every other `Benchmark` property are read-only, and there are **no
protected setters**: the state is written exactly once, by `Benchmark.__init__`, from
the values the bases hand it (`ndim=`, `bounds=`, `f_global=`, `x_global=`,
`support_path=` — keyword-only). A subclass therefore resolves everything it needs
*before* calling `super().__init__(...)` instead of patching fields afterwards; `FuncBenchmark`
does that with its `default_bounds`/`f_global`/`x_global` arguments, and `CecBenchmark`
with `default_bounds`, the official `f_shift`/`f_matrix`/`f_shuffle` files, an optional
`load(problem, ndim, support_path)` hook for class-specific data (matrix row selection,
reshaping, permutation vectors, …) and optional `x_global=`/`f_global=` values or
`(problem, ndim)` callables (defaults: the official shift and the bias).
`check_ndim_and_bounds` is a **pure resolver** — it validates and returns
`(ndim, bounds)` without touching any state — and `make_support_data_path` returns the
data folder instead of storing it. The shape (`bounds`/`ndim`/`lb`/`ub`), `is_succeed`,
`is_ndim_compatible`
and `create_solution` are implemented once on `Benchmark`; `FuncBenchmark`/`CecBenchmark`
only add their own `check_ndim_and_bounds`/`check_solution` and CEC specifics. All the state —
the optimum (`__f_global`/`__x_global`), the shape (`__ndim`/`__bounds`), the display
parameters (`__paras`), the data folder (`__support_path`) and the kernel internals (the bound
`__compute`, `numba_compiled`, …) — is name-mangled on `Benchmark`, so an assignment to a
`_`-prefixed name in a subclass would only shadow the storage and leave the properties stale;
the only sanctioned writers are `Benchmark.__init__` and `Benchmark._bind_kernel`
(the latter owns `paras`, which may also be given as `paras=lambda problem: {...}`).
The shift is validated/normalized
exactly once in the constructor (after dimensionality is resolved) and passed to the kernel
as its second argument; `param_names` lists must not contain `"shift"` because the base
injects it automatically. Unshifted problems receive a vector of zeros so the compiled
gufunc signature is uniform. The stored bounds and optimum stay unshifted while
`bounds`/`lb`/`ub`/`x_global` apply the normalized offset on read (`f_global` is invariant).
The private `__build_compute` helper is only reachable through `_bind_kernel`
(`plain=True` binds the Python kernel directly, with no Numba attempt) and the
public `prepare()` method.

## Numba acceleration (optional)

Numba-based vectorization is opt-in: install with `pip install opfunu-core[numba]`
(CPython only) or `uv sync --group numba`. Without Numba every kernel binds as plain Python automatically and
the library stays fully usable on CPython and PyPy — only batch/single-point
evaluation is slower. `opfunu.HAS_NUMBA` reports whether Numba is available and
`problem.numba_compiled` reports which path a given instance bound.

### Choosing not to compile

Installing Numba does not force you to use it. Every problem constructor accepts a
`prepare` flag, so compilation stays optional even in a Numba-enabled
environment:

```python
from opfunu.name_based import Ackley01
from opfunu.benchmark import Benchmark

f1 = Ackley01(ndim=10, prepare=False)  # plain Python/NumPy kernel for now
f2 = Ackley01(ndim=10, prepare=True)  # gufunc kernel at construction (if Numba installed)

f1.prepare()  # deferred compilation: now f1 binds a compiled gufunc

Benchmark.default_prepare = False  # package-wide policy for new instances
f3 = Ackley01(ndim=10)  # follows the policy: plain kernel until prepare()
f4 = Ackley01(ndim=10, prepare=True)  # explicit choice always wins
```

`prepare` defaults to `None`, meaning "follow `Benchmark.default_prepare`"
(`True` out of the box). Set the class attribute before building problems to flip
the default for a whole session; it is read at construction, so already-built
instances keep their own setting. `problem.numba_compiled` reports what actually
bound.

`prepare=False` defers compilation, it does not forbid it: `problem.prepare()`
performs the `guvectorize` compilation and warm-up that construction skipped and
returns whether the kernel ended up compiled. The call is idempotent (an
already-compiled kernel returns `True` without recompiling) and it never touches
the instance's metadata. Kernels bound with `plain=True` (stochastic terms,
hybrid compositions holding sub-instances) stay plain Python whatever the flag
or `prepare()` says, and `prepare()` is a quiet no-op — returning `False` and
recording no error — when Numba is not installed (PyPy, minimal installs).
Declining or skipping compilation is not an error either: `_compile_error`
stays `None` and `numba_compiled` is `False`. Note that plain kernels calling
`opfunu.utils.operator` helpers still let those `njit`-decorated functions
compile on first use, so `prepare=False` keeps the *problem* kernel plain
rather than guaranteeing that no Numba code is ever generated.

`parallel=True` and `fastmath` are gufunc options and have no effect on an
instance that was built with `prepare=False` until `prepare()` compiles it.

### Evaluation contract

`evaluate` is implemented once on `Benchmark` and inherited by every problem — subclasses
only supply `compute` and must not reimplement it. It calls `self.check_solution(x)`,
runs the bound kernel through a per-instance output buffer, and returns the result cast
to `self.dtype` (`np.float64` by default; `np.float32` when constructed with
`dtype=np.float32`).

`_evaluate_batch(x)` evaluates a whole population in a single gufunc call: `(N, ndim)` in
(or `(ndim,)`, treated as `N=1`), `(N,)` out with dtype `self.dtype`. For plain-Python
fallbacks it loops the scalar kernel behind the same interface.

Compilation (plus a warmup call) happens either at instantiation, when the
instance prepares eagerly, or on demand via `prepare()` — never on first
`evaluate`; see *Choosing not to compile*.
The gufunc cache is keyed on the kernel's `__code__`, so kernels must be self-contained —
no closing over per-instance values.

### Kernel parameter handling

`_normalize_kernel_param` maps each runtime attribute to `(call_value, numpy_dtype)`
without importing Numba (`numba.from_dtype` conversion happens only inside
`__build_compute`, when `HAS_NUMBA` is true): float arrays are cast to `self.dtype`,
integer arrays to `int64` (index arrays must stay integer), int/float scalars to
`int64`/`self.dtype`, bool scalars to `bool`, and numeric Python lists (including nested
matrices) to arrays; zero-dimensional arrays unwrap to scalars. Anything else raises
`TypeError`.

Attribute-lookup and normalization errors propagate — a mistyped `param_names` entry or
an un-normalizable value is a bug, not a compile failure. Only Numba
compilation/warmup failures fall back to the plain Python kernel, keeping the same
`(x, shift, *params, out)` calling convention; the exception is stored on `_compile_error`
(`None` on success, when Numba is absent, or when compilation was declined) and
printed when `verbose=True`. The shift is *not* part of `param_names` — it is injected
automatically by the base class.
