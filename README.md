# opfunu-core

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg?style=flat-square)](https://www.gnu.org/licenses/gpl-3.0)
![PyPI - Version](https://img.shields.io/pypi/v/opfunu-core?style=flat-square)
![PyPI - Implementation](https://img.shields.io/pypi/implementation/opfunu-core?style=flat-square)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/opfunu-core?style=flat-square)
![PyPI - Wheel](https://img.shields.io/pypi/wheel/opfunu-core?style=flat-square)
![GitHub Release Date](https://img.shields.io/github/release-date/ltsim/opfunu-core.svg?style=flat-square)
![PyPI - Downloads](https://img.shields.io/pypi/dm/opfunu-core?style=flat-square)

This library is a maintenance version, a fork of [OPFUNU (Optimization Reference Functions in NUMPy)](https://github.com/thieu1995/opfunu). Is one of the most comprehensive Python libraries of numerical optimization reference functions. It contains all the functions from the CEC competitions of 2005, 2008, 2010, 2013, 2014, 2015, 2017, 2019, 2020, 2021, and 2022. In addition, it implements over 300 traditional functions with varying dimensions.

* **Free software:** GNU General Public License (GPL) V3 license
* **Total problems**: > 500 problems
* **Documentation:** https://opfunu.readthedocs.io
* **Python versions:** >= 3.10.x

# Citation Request 

Please include these citations if you plan to use this library:

**LaTeX Style**

```bibtex
  @article{Van_Thieu_2024_Opfunu,
      author = {Van Thieu, Nguyen},
      title = {Opfunu: An Open-source Python Library for Optimization Benchmark Functions},
      doi = {10.5334/jors.508},
      journal = {Journal of Open Research Software},
      month = {May},
      year = {2024}
  }
```

**APA Style**

Van Thieu, N. (2024). **Opfunu: An Open-source Python Library for Optimization Benchmark Functions.** _Journal of Open Research Software_, _12_(1), 8. https://doi.org/10.5334/jors.508

# Installation and Usage

### Install with pip

Install the [current PyPI release](https://pypi.python.org/pypi/opfunu):
```sh
$ pip install opfunu-core
```

Install from Github:
```sh
$ pip install git+https://github.com/ltsim/opfunu-core
```

After installation, you can import and check version of Opfunu:

```sh
$ python
>>> import opfunu
>>> opfunu.__version__

>>> dir(opfunu)
>>> help(opfunu)

>>> opfunu.FUNC_DATABASE      # List all name_based functions
>>> opfunu.CEC_DATABASE       # List all cec_based functions
>>> opfunu.ALL_DATABASE       # List all functions in this library

>>> opfunu.get_functions_by_classname("MiShra04")
>>> opfunu.get_functions_based_classname("2015")
>>> opfunu.get_functions_by_ndim(2)
>>> opfunu.get_functions_based_ndim(50)

>>> opfunu.get_name_based_functions(ndim=10, continuous=True)
>>> opfunu.get_cec_based_functions(ndim=2)
```

Let's go through some examples.

### Examples

How to get the function and use it

#### 1st way

```python
from opfunu.cec_based.cec2014 import F12014

func = F12014(ndim=30)
func.evaluate(func.create_solution())

## or

from opfunu.cec_based import F102014

func = F102014(ndim=50)
func.evaluate(func.create_solution())
```

#### 2nd way

```python
import opfunu

funcs = opfunu.get_functions_by_classname("F12014")
func = funcs[0](ndim=10)
func.evaluate(func.create_solution())

## or

all_funcs_2014 = opfunu.get_functions_based_classname("2014")
print(all_funcs_2014)
```

### How to draw 2D, 3D 

Two ways if you want to draw functions that available in Opfunu.

```python
from opfunu.cec_based import F12010
f0 = F12010()

# Visualize opfunu function using method in object
f0.plot_2d(selected_dims=(2, 3), n_points=300, ct_cmap="viridis", ct_levels=30, ct_alpha=0.7,
           fixed_strategy="mean", fixed_values=None, title="Contour map of the F1 CEC 2010 function",
           x_label=None, y_label=None, figsize=(10, 8), filename="2d-f12010", exts=(".png", ".pdf"), verbose=True)

f0.plot_3d(selected_dims=(1, 6), n_points=500, ct_cmap="viridis", ct_levels=30, ct_alpha=0.7,
           fixed_strategy="mean", fixed_values=None, title="3D visualization of the F1 CEC 2010 function",
           x_label=None, y_label=None, figsize=(10, 8), filename="3d-f12010", exts=(".png", ".pdf"), verbose=True)

## Visualize opfunu function using utility function
from opfunu import draw_2d, draw_3d

draw_2d(f0.evaluate, f0.lb, f0.ub, selected_dims=(2, 3), n_points=300)
draw_3d(f0.evaluate, f0.lb, f0.ub, selected_dims=(2, 3), n_points=300)
```

### How to draw Latex

Two ways if you want to draw latex equation. 

```python
from opfunu.cec_based import F12010
from opfunu.name_based import Ackley02
from opfunu.utils.visualize import draw_latex

f0 = F12010()
f1 = Ackley02()

## Plot using function inside the object
f0.plot_latex(f0.latex_formula, figsize=(8, 3), dpi=500, title="Latex equation", exts=(".png", ".pdf"), verbose=True)
f1.plot_latex(f1.latex_formula_global_optimum, figsize=(8, 3), dpi=500, title="Global optimum", verbose=True)

## Plot using module
draw_latex(f0.latex_formula_bounds, title="Boundary for Function")
draw_latex(f1.latex_formula_dimension, title=None)
```

For more usage examples please look at [examples](/examples) folder.

# Contributing

There are lots of ways how you can contribute to Permetrics's development, and you are welcome to join in! For example, 
you can report problems or make feature requests on the [issues](/issues) pages. To facilitate contributions, 
please check for the guidelines in the [CONTRIBUTING.md](/CONTRIBUTING.md) file.

# Official channels 

* [Official source code repository](https://github.com/ltsim/opfunu-core)
* [Official document](https://opfunu.readthedocs.io/)
* [Download releases](https://pypi.org/project/opfunu-core/) 
* [Issue tracker](https://github.com/ltsim/opfunu-core/issues) 
* [Notable changes log](/CHANGELOG.md)

---

* Maintained by: [LTSIM](mailto:tsim@cucei.udg.mx) @ 2026
* Developed by: [Thieu](mailto:nguyenthieu2102@gmail.com?Subject=Opfunu_QUESTIONS) @ 2023
