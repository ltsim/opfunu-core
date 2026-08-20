# opfunu-core

**opfunu-core** is a maintenance version, a fork of
[OPFUNU (Optimization Reference Functions in NumPy)](https://github.com/thieu1995/opfunu).
It is one of the most comprehensive Python libraries of numerical optimization reference functions.
It contains all the functions from the CEC competitions of 2005, 2008, 2010, 2013, 2014, 2015, 2017, 2019, 2020, 2021,
and 2022. In addition, it implements over 300 traditional functions with varying dimensions.

* **Free software:** GNU General Public License (GPL) V3 license
* **Total problems:** > 500 problems
* **Dependencies:** numpy

## Citation

Please cite this library as [@Van_Thieu_2024_Opfunu] if you use it in your research.

APA style: Van Thieu, N. (2024). **Opfunu: An Open-source Python Library for Optimization Benchmark Functions.**
_Journal of Open Research Software_, _12_(1), 8. https://doi.org/10.5334/jors.508

\bibliography

## Modules

| Module                  | Description                                                       |
|-------------------------|-------------------------------------------------------------------|
| `opfunu`               | Package-level API: function databases and query helpers           |
| `opfunu.benchmark`     | Base classes `Benchmark`, `FuncBenchmark`, `CecBenchmark`         |
| `opfunu.name_based`    | More than 300 traditional functions grouped alphabetically        |
| `opfunu.cec_based`     | All CEC competition functions from 2005 to 2022                   |
| `opfunu.utils`         | Low-level mathematical operators used by the functions            |

## Official channels

* [Official source code repository](https://github.com/ltsim/opfunu-core)
* [Official documentation](https://ltsim.github.io/opfunu-core/)
* [Download releases](https://pypi.org/project/opfunu-core/)
* [Issue tracker](https://github.com/ltsim/opfunu-core/issues)
* [Notable changes log](https://github.com/ltsim/opfunu-core/blob/master/CHANGELOG.md)

---

* Maintained by: [LTSIM](mailto:tsim@cucei.udg.mx) @ 2026
* Developed by: [Thieu](mailto:nguyenthieu2102@gmail.com?Subject=Opfunu_QUESTIONS) @ 2023
