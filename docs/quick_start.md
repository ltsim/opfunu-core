# Quick Start

## Installation

Install the current PyPI release with pip:

```sh
$ pip install opfunu-core
```

For Numba-based vectorization (CPython only):

```sh
$ pip install opfunu-core[numba]
```

Without the `[numba]` extra the library runs in pure-Python/NumPy mode, which also
works on PyPy (`opfunu.HAS_NUMBA` reports whether Numba is available).

Or, if you manage the project with [uv](https://github.com/astral-sh/uv):

```sh
$ uv add opfunu-core
```

With `uv`, the Numba extra lives in the `numba` dependency group:

```sh
$ uv sync --group numba   # opt-in guvectorize vectorization (CPython only)
```

Install directly from GitHub:

```sh
$ pip install git+https://github.com/ltsim/opfunu-core
```

## Import and check the version

```python
import opfunu

print(opfunu.__version__)

dir(opfunu)  # List all available objects
help(opfunu)  # Read the package documentation

opfunu.FUNC_DATABASE  # List all name_based functions
opfunu.CEC_DATABASE  # List all cec_based functions
opfunu.ALL_DATABASE  # List all functions in this library
```

## Query the databases

```python
# Exact class name (case-insensitive)
opfunu.get_functions_by_classname("MiShra04")

# Substring of the class name, e.g. every CEC-2015 function
opfunu.get_functions_based_classname("2015")

# Functions with an exact (fixed) number of dimensions
opfunu.get_functions_by_ndim(2)

# Functions that support a given number of dimensions
opfunu.get_functions_based_ndim(50)

# Filter by mathematical properties
opfunu.get_name_based_functions(ndim=10, continuous=True)
opfunu.get_cec_based_functions(ndim=2)
```

## Evaluate a function

1st way — import the class directly:

```python
from opfunu.cec_based.cec2014 import F12014

func = F12014(ndim=30)
func.evaluate(func.create_solution())

# or through the package namespace
from opfunu.cec_based import F102014

func = F102014(ndim=50)
func.evaluate(func.create_solution())
```

2nd way — look the function up from the database:

```python
import opfunu

funcs = opfunu.get_functions_by_classname("F12014")
func = funcs[0](ndim=10)
func.evaluate(func.create_solution())

all_2014 = opfunu.get_functions_based_classname("2014")
print(all_2014)
```

## Working with a function object

```python
import numpy as np

f12005 = opfunu.cec_based.F12005(ndim=2)
print(f12005.evaluate(np.array([5, 4])))  # evaluate a solution
print(f12005.bounds)  # [lower, upper] bounds matrix
print(f12005.lb)  # lower bounds
print(f12005.ub)  # upper bounds

print(f12005.f_global)  # global minimum value
print(f12005.x_global)  # location of the global minimum
print(f12005.is_succeed(f12005.x_global))  # did we reach the global minimum?
```

## Construction flags

Every problem accepts `parallel`, `fastmath` and `dtype` flags:

```python
func = opfunu.cec_based.F12014(ndim=30, parallel=True, fastmath=True, dtype=np.float64)
```

* `dtype` sets the floating scalar type used for evaluation; `evaluate` returns
  `self.dtype.type(...)` (`np.float64` by default, `np.float32` when requested).
* `fastmath=True` (default) compiles the kernel with Numba `fastmath`; pass
  `fastmath=False` for bit-exact agreement with the pre-migration NumPy code.
* `parallel=True` compiles with `target="parallel"` for multi-threaded batch runs.

## Batch evaluation

`_evaluate_batch` evaluates a whole population in one gufunc call — `(N, ndim)` in,
`(N,)` out with dtype `self.dtype` (a 1-D `(ndim,)` input is treated as `N=1`):

```python
pop = np.random.uniform(func.lb, func.ub, size=(256, func.ndim))
values = func._evaluate_batch(pop)
```

Without Numba the same call loops the scalar kernel, so the interface is identical
on every interpreter — only slower.

## Diagnostics

```python
print(opfunu.HAS_NUMBA)  # is Numba available in this environment?
print(func.numba_compiled)  # did this instance bind a compiled gufunc?
print(func._compile_error)  # compilation failure, or None (also None when Numba is absent)
print(func.get_paras())  # kernel parameters bound for this instance
```

`verbose` is a read-only property configured at construction time (pass
`verbose=True` to `Benchmark`/`FuncBenchmark`/`CecBenchmark`, or set a class
attribute such as `F12021.verbose = True`) to print compilation
fallbacks. Functions whose bodies cannot compile in nopython mode (stochastic terms,
hybrid compositions holding sub-instances) bind with `plain=True` automatically and
report `numba_compiled=False` while returning value-identical results.

For more usage examples, see the [Examples](examples.md) page and the
[`examples/`](https://github.com/ltsim/opfunu-core/tree/master/examples) folder.
