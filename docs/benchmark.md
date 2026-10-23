# Benchmark Base Classes

Every function in `opfunu-core` inherits from one of the base classes defined in the `opfunu.benchmark` package:

| Class            | Description                                                  | Used by                       |
|------------------|--------------------------------------------------------------|-------------------------------|
| `Benchmark`      | Abstract base class defining the common function interface   | all functions                 |
| `Formula`        | Dataclass holding a LaTeX formula (rendered by IPython)      | all functions                 |
| `FuncBenchmark`  | Base class for traditional (name-based) functions            | `opfunu.name_based.*`         |
| `CecBenchmark`   | Base class for CEC competition functions                     | `opfunu.cec_based.*`          |

## Benchmark and Formula

::: opfunu.benchmark
    options:
      members: [Formula, Benchmark]

## FuncBenchmark

::: opfunu.benchmark.func.FuncBenchmark

## CecBenchmark

::: opfunu.benchmark.cec.CecBenchmark

## Coordinate shifts

Every benchmark function accepts an optional `shift` constructor argument (a
scalar is broadcast to all dimensions). It is an ordinary parameter of each
class's `__init__` and is forwarded through `super().__init__(...)` all the way
to `Benchmark`, which stores it privately. Once construction finishes, the
translation is applied automatically:

- Fitness: `f(x) = f_base(x - o)`
- Global optimum: `x_global = x*_base + o` (`f_global` is unchanged)
- Search space: `bounds_new = bounds_base + o` (per dimension)

```python
from opfunu.name_based import Ackley01

func = Ackley01(ndim=3, shift=[2.0, -1.5, 3.0])
print(func.x_global)  # base optimum + shift
print(func.bounds)  # base bounds + shift
print(func.is_succeed(func.x_global))  # True
```

`ndim`, `bounds`, `lb`, `ub` remain public read-only properties, plus a
read-only `shift` property. The shift can also be changed after construction
with `func.set_shift(o)`, and removed with `func.set_shift(None)`. Calling
`set_shift` repeatedly never accumulates: the base bounds and optimum are
snapshotted on the first shift and restored before each new translation.

## Implementing a problem

A name-based problem defines its kernel inline and hands the metadata to the base class:

```python
class Quadratic(FuncBenchmark):
    def __init__(self, ndim=None, bounds=None, parallel=False, fastmath=True, dtype=np.float64):
        def compute(x, out):
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
def compute(x, f_shift, f_bias, out):
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

Problems that compute extra kernel parameters after `super().__init__` (permutation
vectors, composition weights, index blocks, …) bind explicitly instead:

```python
self._bind_kernel(compute, ["f_shift", "P", "m_group", "f_matrix"], paras=[...])  # plain=True if Numba-incompatible
```

`compute` and every other `Benchmark` property are read-only: configuration is
supplied through `__init__` kwargs (`dtype`, `dim_changeable`, `verbose`, `prepare`, …), and
data-dependent metadata written by subclasses goes through protected fields
(`self._f_global`, `self._x_global`, `self._paras`, `self._support_path`, `self._ndim`,
`self._bounds`). The shape (`bounds`/`ndim`/`lb`/`ub`), `is_succeed`, `is_ndim_compatible`
and `create_solution` are implemented once on `Benchmark`; `FuncBenchmark`/`CecBenchmark`
only add their own `check_ndim_and_bounds`/`check_solution` and CEC specifics. All private
kernel state (the bound `__compute`, `__paras`, `numba_compiled`, …)
lives in name-mangled storage behind those properties. The private
`__build_compute` helper is only reachable through `_bind_kernel`
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
`(x, *params, out)` calling convention; the exception is stored on `_compile_error`
(`None` on success, when Numba is absent, or when compilation was declined) and
printed when `verbose=True`.
