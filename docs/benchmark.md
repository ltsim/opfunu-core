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

`compute` is a read-only property on `Benchmark`; all kernel state (`paras`,
`numba_compiled`, …) lives in private storage behind properties. The private
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
