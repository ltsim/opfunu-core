---
hide:
  - toc
---

# opfunu-core

**An open-source Python library of optimization benchmark functions.**

opfunu-core is a maintenance version, a fork of
[OPFUNU (Optimization Reference Functions in NumPy)](https://github.com/thieu1995/opfunu). It bundles every function
from the CEC competitions of 2005–2022 together with over 300 traditional functions — all behind a single, NumPy-based
API for benchmarking your optimization algorithms.

<div class="grid cards" markdown>

-   :material-function-variant: **500+ problems**

    All CEC competition functions (2005, 2008, 2010, 2013, 2014, 2015, 2017, 2019, 2020, 2021, 2022) and over
    300 traditional test functions with varying dimensions.

-   :material-cube-outline: **NumPy-first**

    Every function exposes `lb`, `ub`, `bounds` and a single `evaluate(x)` method — no extra dependencies.

-   :material-database-search-outline: **Queryable databases**

    Filter functions by dimension, modality, separability, shift, rotation, and more through the package-level
    query helpers.

-   :material-license: **Free software**

    Distributed under the GNU General Public License v3 (GPL-3.0).

</div>

## Quick start

Install from PyPI:

```bash
pip install opfunu-core
```

Evaluate your first function:

```python
import numpy as np

import opfunu

func = opfunu.cec_based.F12014(ndim=30)
solution = np.random.uniform(func.lb, func.ub)
print(func.evaluate(solution))
```

<div class="grid cards" markdown>

-   :material-rocket-launch-outline: [**Quick Start**](quick_start.md)

    Install, import, and evaluate your first function in minutes.

-   :material-book-open-page-variant-outline: [**Examples**](examples.md)

    Real usage patterns — from database queries to optimization with other libraries.

-   :material-api: [**API Reference**](benchmark.md)

    Benchmark base classes, name-based and CEC-based functions, package helpers, and utilities.

-   :material-shape-outline: [**Function Categories**](categories.md)

    Understand modality, basins, valleys, separability, and dimensionality.

</div>

## Citation

Please cite this library as [@Van_Thieu_2024_Opfunu] if you use it in your research.

\bibliography

## Official channels

* [Official source code repository](https://github.com/ltsim/opfunu-core)
* [Official documentation](https://ltsim.github.io/opfunu-core/)
* [Download releases](https://pypi.org/project/opfunu-core/)
* [Issue tracker](https://github.com/ltsim/opfunu-core/issues)
* [Notable changes log](https://github.com/ltsim/opfunu-core/blob/master/CHANGELOG.md)

---

* Maintained by: [LTSIM](mailto:tsim@cucei.udg.mx) @ 2026
* Developed by: [Thieu](mailto:nguyenthieu2102@gmail.com?Subject=Opfunu_QUESTIONS) @ 2023
