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
`__build_compute` / `__use_plain_compute` helpers are only reachable through
`_bind_kernel`.
