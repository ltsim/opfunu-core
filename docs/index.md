# opfunu-core

**An open-source Python library of optimization benchmark functions.**

opfunu-core is a maintenance version, a fork of
[OPFUNU (Optimization Reference Functions in NumPy)](https://github.com/thieu1995/opfunu). It bundles every function
from the CEC competitions of 2005–2022 (around 190 problems) together with 125 traditional functions — all behind a
single API for benchmarking your optimization algorithms. The base dependency is NumPy only; opt-in
Numba vectorization (`pip install opfunu-core[numba]`, CPython) compiles every function for batch evaluation.

```{toctree}
:hidden:
:caption: Getting Started
:maxdepth: 1

quick_start
categories
examples
```

```{toctree}
:hidden:
:caption: API Reference
:maxdepth: 2

benchmark
name_based/index
cec_based/index
integrated
utils
plot
```

```{toctree}
:hidden:
:caption: Project
:maxdepth: 1

support
license
changelog
```

## Highlights

- **300+ problems** — every CEC competition function (2005, 2008, 2010, 2013, 2014, 2015, 2017, 2019, 2020, 2021,
  2022) plus 125 traditional test functions with varying dimensions.
- **NumPy-first** — every function exposes `lb`, `ub`, `bounds` and a single `evaluate(x)` method, with no
  mandatory dependency beyond NumPy.
- **Queryable databases** — filter functions by dimension, modality, separability, shift, rotation and more
  through the package-level query helpers.
- **Free software** — distributed under the GNU General Public License v3 (GPL-3.0).

## Quick start

Install from PyPI:

```bash
pip install opfunu-core
```

For Numba-based vectorization (CPython only):

```bash
pip install opfunu-core[numba]
```

Without the `[numba]` extra the library runs in pure-Python/NumPy mode, which also
works on PyPy.

Evaluate your first function:

```python
import numpy as np

import opfunu

func = opfunu.cec_based.F12014(ndim=30)
solution = np.random.uniform(func.lb, func.ub)
print(func.evaluate(solution))
```

## Where to go next

- [Quick Start](quick_start.md) — install, import, and evaluate your first function in minutes.
- [Examples](examples.md) — real usage patterns, from database queries to optimization with other libraries.
- [API Reference](benchmark.md) — benchmark base classes, name-based and CEC-based functions, package helpers,
  and utilities.
- [Function Categories](categories.md) — understand modality, basins, valleys, separability and dimensionality.

## Citation

Please cite this library as {cite}`Van_Thieu_2024_Opfunu` if you use it in your research.

```{bibliography}
```

## Official channels

- [Official source code repository](https://github.com/ltsim/opfunu-core)
- [Official documentation](https://ltsim.github.io/opfunu-core/)
- [Download releases](https://pypi.org/project/opfunu-core/)
- [Issue tracker](https://github.com/ltsim/opfunu-core/issues)
- [Notable changes log](https://github.com/ltsim/opfunu-core/blob/master/CHANGELOG.md)

***

- Maintained by: [LTSIM](mailto:tsim@cucei.udg.mx) @ 2026
- Developed by: [Thieu](mailto:nguyenthieu2102@gmail.com?Subject=Opfunu_QUESTIONS) @ 2023
