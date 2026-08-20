# Quick Start

## Installation

Install the current PyPI release with pip:

```sh
$ pip install opfunu-core
```

Or, if you manage the project with [uv](https://github.com/astral-sh/uv):

```sh
$ uv add opfunu-core
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

For more usage examples, see the [Examples](examples.md) page and the
[`examples/`](https://github.com/ltsim/opfunu-core/tree/master/examples) folder.
