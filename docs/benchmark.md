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
snapshotted on the first transform and restored before each recomputation.

## Coordinate rotations

Every benchmark function also accepts an optional `rotate` constructor argument
(an orthogonal `ndim x ndim` matrix — any matrix with `M.T @ M = I`, including
reflections) and a `rotate_bounds` flag (default `True`). Both are ordinary
`__init__` parameters forwarded like `shift`, and compose with it as
`f(x) = f_base(M @ (x - o))` — the shift is applied first, then the rotation:

- Fitness: `f(x) = f_base(M @ (x - o))`
- Global optimum: `x_global = o + M.T @ x*_base` (`f_global` is unchanged)
- Search space: the enclosing axis-aligned box of `M.T @ bounds_base`, then
  `+ o`. For a corner-defined box this is `center -> M.T @ c`,
  `half-widths -> |M.T| @ h`, so a rotation can only grow the box (at 45 deg
  a `[-a, a]` interval becomes `[-a*sqrt(2), a*sqrt(2)]`).

```python
import numpy as np
from opfunu.name_based import Ackley01

theta = np.pi / 4
M = np.eye(3)
M[:2, :2] = [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]

func = Ackley01(ndim=3, shift=[2.0, -1.5, 3.0], rotate=M)
print(func.x_global)  # o + M.T @ base optimum
print(func.bounds)  # enclosing AABB of the rotated box + shift
print(func.is_succeed(func.x_global))  # True
```

Validation is strict: the matrix must have shape `(ndim, ndim)`, contain only
finite numbers, and be orthogonal (`np.allclose(M.T @ M, I)`); anything else
raises `ValueError` during construction. A read-only `rotate` property exposes
the matrix, `func.get_paras()["rotate"]` carries a copy, and `set_rotate(M)` /
`set_rotate(None)` change or remove it after construction with the same
idempotency guarantee as `set_shift`.

Pass `rotate_bounds=False` to keep the base box rigid (the CEC convention,
where official rotation data only transforms the fitness landscape): bounds
stay `bounds_base + o` while `x_global` still maps through `M.T`. Caveat: the
rotated optimum may then fall outside the bounds, making
`is_succeed(x_global)` return `False` — that is expected CEC behavior.

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

Problems that compute extra kernel parameters after `super().__init__` (permutation
vectors, composition weights, index blocks, …) bind explicitly instead:

```python
self._bind_kernel(compute, ["f_shift", "P", "m_group", "f_matrix"], paras=[...])  # plain=True if Numba-incompatible
```

`compute` and every other `Benchmark` property are read-only: configuration is
supplied through `__init__` kwargs (`dtype`, `dim_changeable`, `verbose`, …), and
data-dependent metadata written by subclasses goes through protected fields
(`self._f_global`, `self._x_global`, `self._paras`, `self._support_path`).
All private kernel state (the bound `__compute`, `__paras`, `numba_compiled`, …)
lives in name-mangled storage behind those properties. The private
`__build_compute` helper is only reachable through `_bind_kernel`
(`plain=True` binds the Python kernel directly, with no Numba attempt).

## Numba acceleration (optional)

Numba-based vectorization is opt-in: install with `pip install opfunu-core[numba]`
(CPython only) or `uv sync --group numba`. Without Numba every kernel binds as plain Python automatically and
the library stays fully usable on CPython and PyPy — only batch/single-point
evaluation is slower. `opfunu.HAS_NUMBA` reports whether Numba is available and
`problem.numba_compiled` reports which path a given instance bound.

### Evaluation contract

`evaluate` is implemented once on `Benchmark` and inherited by every problem — subclasses
only supply `compute` and must not reimplement it. It calls `self.check_solution(x)`,
runs the bound kernel through a per-instance output buffer, and returns the result cast
to `self.dtype` (`np.float64` by default; `np.float32` when constructed with
`dtype=np.float32`).

`_evaluate_batch(x)` evaluates a whole population in a single gufunc call: `(N, ndim)` in
(or `(ndim,)`, treated as `N=1`), `(N,)` out with dtype `self.dtype`. For plain-Python
fallbacks it loops the scalar kernel behind the same interface.

Compilation (plus a warmup call) happens at instantiation, never on first `evaluate`.
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
(`None` on success, or when Numba is simply absent) and printed when `verbose=True`.
