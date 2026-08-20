# Examples

This page shows how to use **opfunu-core** (version `2026a`). More runnable scripts live in the
[examples](https://github.com/ltsim/opfunu-core/tree/master/examples) folder.

## Import and version

```python
import opfunu
import numpy as np

print(opfunu.__version__)

# List all functions grouped by database
opfunu.FUNC_DATABASE  # all name_based functions
opfunu.CEC_DATABASE  # all cec_based functions
opfunu.ALL_DATABASE  # all functions in this library
```

## Querying the databases

```python
# Exact class name (case-insensitive)
funcs = opfunu.get_functions_by_classname("MiShra04")

# Substring of the class name, e.g. every CEC-2014 function
all_2014 = opfunu.get_functions_based_classname("2014")

# Exact / supported number of dimensions
opfunu.get_functions_by_ndim(2)
opfunu.get_functions_based_ndim(50)

# Filter by mathematical properties
opfunu.get_name_based_functions(ndim=10, continuous=True)
opfunu.get_cec_based_functions(ndim=2, rotated=True)
opfunu.get_cec_based_functions(randomized_term=True)
```

## Evaluating a function

```python
# 1st way: import the class directly
from opfunu.cec_based.cec2014 import F12014

func = F12014(ndim=30)
func.evaluate(func.create_solution())

# or through the package namespace
from opfunu.cec_based import F102014

func = F102014(ndim=50)
func.evaluate(func.create_solution())

# 2nd way: look the function up from the database
funcs = opfunu.get_functions_by_classname("F12014")
func = funcs[0](ndim=10)
func.evaluate(func.create_solution())
```

## Function object API

```python
f12005 = opfunu.cec_based.F12005()
print(f12005.evaluate(np.array([5, 4, 5])))  # evaluate a solution
print(f12005.bounds)  # [lower, upper] bounds matrix
print(f12005.lb, f12005.ub)  # lower and upper bound arrays

# Custom bounds
bounds = [[-10.0] * 15, [10.0] * 15]
f12005 = opfunu.cec_based.F12005(bounds=bounds)
print(f12005.lb)

# Access/change the parameters of parametric functions
f22005 = opfunu.cec_based.F22005(ndim=2)
print(f22005.get_paras())

# Global minimum value and location
print(f22005.f_global)
print(f22005.x_global)

# LaTeX formulas of the function
print(f22005.latex_formula)
print(f22005.latex_formula_dimension)
print(f22005.latex_formula_bounds)
print(f22005.latex_formula_global_optimum)
```

## CEC data file usage

Some CEC functions read shift/rotation matrices from bundled data files.

```python
f32005 = opfunu.cec_based.F32005(ndim=10)
x = np.ones(10)
print(f32005.evaluate(x))
print(f32005.f_matrix)  # the loaded matrix
print(f32005.x_global)

problem = opfunu.cec_based.F212005(ndim=10)
print(problem.evaluate(x))
print(problem.x_global)
print(problem.is_succeed(problem.x_global))
```

## Name-based functions

```python
problem = opfunu.name_based.Ackley01(ndim=25)
x = np.ones(25)
print(problem.evaluate(x))
print(problem.evaluate(problem.x_global))
print(problem.is_succeed(problem.x_global))
```

## Optimizing with another library (mealpy)

```python
from mealpy import GA, FloatVar

from opfunu.cec_based import cec2017

f3 = cec2017.F32017(ndim=30)

problem = {
    "obj_func": f3.evaluate,
    "bounds": FloatVar(lb=f3.lb, ub=f3.ub),
    "minmax": "min",
}
model = GA.BaseGA(epoch=100, pop_size=50)
gbest = model.solve(problem)
print(f"Solution: {gbest.solution}, Fit: {gbest.target.fitness}")
```
