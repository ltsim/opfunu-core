# Version 2026a

+ Added automatic coordinate-shift propagation (issue #19): every benchmark function accepts a `shift`
  constructor argument, forwarded via `super().__init__(...)` to the parent class, evaluating
  `f(x) = f_base(x - o)` while `x_global` and `bounds` translate by `o`
  (`set_shift()` / `set_shift(None)` for post-construction use, `shift` exposed via `get_paras()`)
+ Migrated all benchmark functions to Numba: every problem class now implements its math in a
  mandatory `compute(x, *params, out)` kernel (a nested closure or a `@staticmethod`), compiled
  at instantiation with `numba.guvectorize` (`nopython=True`, `fastmath=True` by default,
  `parallel` opt-in)
+ `evaluate` is now implemented once on the `Benchmark` base class and inherited by every problem:
  subclasses only supply `compute`, and no longer build an output vector. The shared implementation
  reuses a per-instance output buffer and returns the result cast to the problem's `dtype`
  (`np.float32` with `dtype=np.float32`, `np.float64` by default)
+ Added `_evaluate_batch(x)` on the base to evaluate a whole `(N, ndim)` population in one gufunc call
+ Added `parallel`, `fastmath` and `dtype` constructor flags to all problems
  (e.g. `F12014(ndim=30, parallel=True, dtype=np.float32)`); guvectorized kernels also evaluate
  whole populations in one call (up to ~55x on 256-vector batches; ~2-8x on single evaluations)
+ Functions whose bodies cannot compile in nopython mode (stochastic terms, hybrid compositions
  holding sub-instances, `Cola`, 2005 `F232005`, 2008 `F72008`) bind with `plain=True` and keep
  their exact NumPy bodies behind the same interface (`numba_compiled=False`); verified
  value-identical against the pre-migration code
+ Removed the `n_fe` evaluation counter from all classes
+ Made Numba an opt-in extra: the base dependency is now NumPy only, `pip install
+   opfunu-core[numba]` (CPython) enables `guvectorize` vectorization, and without it
+   every kernel binds as plain Python automatically so the library stays fully usable
+   on CPython and PyPy (`opfunu.HAS_NUMBA`, per-instance `numba_compiled`); shared
+   operators and CEC2005 helpers use a `numba_compat.njit` pass-through shim
+ Added `numba>=0.68.0` as an optional dependency (`[numba]` extra, CPython-only marker);
+   `FuncBenchmark` joined `EXCLUDES` in the function registry (its `compute` is now abstract)
+ Migrated the project to a `src/` layout and to the `uv` package manager
+ Added `ruff` (lint + format) and `mypy` (type-check) as the default tooling; `mypy` now runs in
  strict mode across `src/`
+ `dim_changeable`, `dim_default`, `f_global` and `x_global` are now private attributes exposed
  through read/write properties on `Benchmark`
+ `Benchmark` accepts an instance-level `compute` callable; `FuncBenchmark` gained a compact
  declarative constructor and the 125 name-based problems now define their kernel inline in
  `__init__` and pass the metadata straight to `super().__init__(...)`
+ All 191 CEC problem kernels are likewise defined inline in `__init__`: simple classes
  pass `compute=...`/`param_names=[...]` straight to `super().__init__(...)` (which binds
  after loading the shift/rotation data), while classes with extra computed parameters bind
  explicitly with `self._bind_kernel(compute, [...], paras=..., plain=...)`; the gufunc cache
  is keyed on the kernel's `__code__` so a parent build cannot shadow a child's
  identically-signed kernel
+ Encapsulated all kernel/config state behind properties: `compute` is now a read-only
  property (the class-level `compute` fallback is gone) and `paras`, `numba_compiled`,
  `f_bias`, `f_shift`, `f_matrix`, `dim_max`, … live in private storage; compilation
  goes through the private `__build_compute` helper, reachable only through
  the protected `_bind_kernel` bridge (`plain=True` binds the Python kernel directly)
+ Hardened the Numba kernel layer: `_evaluate_batch` loops the scalar kernel for plain-Python
  fallbacks, `_normalize_kernel_param` returns `(call_value, numpy_dtype)` (Numba-free;
  bool scalars map to `bool`; anything else raises `TypeError`), the gufunc cache is keyed on the kernel's
  `__code__`, attribute/normalization errors propagate while only compilation/warmup
  failures fall back and are recorded on `Benchmark._compile_error`
+ Composite CEC functions that hold sub-instances (2013 F21-F24, 2014 F26-F30, 2017 F27-F28,
  plus 2005 `F232005`) now bind with `plain=True`, avoiding a guaranteed failed compilation attempt
+ Refactored the pytest test suite around shared fixtures in `tests/conftest.py`
+ Refactored the examples and the `EXAMPLES.md` guide for the current API
+ Added MkDocs documentation with the Material theme, published on GitHub Pages
+ Dropped the legacy ReadTheDocs/Sphinx configuration

---------------------------------------------------------------------

# Version 2.0.0

+ ...

---------------------------------------------------------------------

# Version 1.0.6

+ Remove visualize (Matplotlib and Pillow)

---------------------------------------------------------------------

# Version 1.0.5.1.1

+ Library dependency update

---------------------------------------------------------------------

# Version 1.0.5.1

+ Compatibility with Pypy 3.11

---------------------------------------------------------------------

# Version 1.0.5

+ Maintenance for later versions of Python 3.10
+ Updating dependencies like Numpy

---------------------------------------------------------------------

# Version 1.0.4

+ Fix p value in F10 and F17 of CEC-2017
+ Add plot_latex to Benchmark class.
+ User can use draw_latex from opfunu to draw their latex equation.
+ Update examples for draw latex function

---------------------------------------------------------------------

# Version 1.0.3

+ Optimized katsuura_func performance, at 1M ndim > 80x speedup
+ Add plot_2d, plot_3d to Benchmark class.
+ User can use draw_2d, draw_3d from opfunu to draw their function.
+ Add tutorial on how to integrate with other optimization frameworks like Mealpy, Opytimizer, Niapy
+ Update examples and update documentations
+ Update citation and paper (Got published at Journal of Open Research Software)

---------------------------------------------------------------------

# Version 1.0.2

+ Fix modified_schwefel_func() in operator.py
+ Fix Mishra07 function that has no method factorial
+ Fix Dolan function has wrong default dim
+ Fix ndim property in Benchmark class
+ Fix typo pi function in OddSquare class
+ Fix LennardJones class with specified ndim
+ Fix CEC2021 F2 has abnormal optimal value
+ Fix bug in F8 function CEC2020
+ Update docs
+ Update README and workflows.

---------------------------------------------------------------------

# Version 1.0.1

+ Add all normal functions in name_based module (From A to Z)
+ Fix bug exit() program
+ Delete all old CEC module
+ Delete type_based module
+ Delete dimension_based module
+ Delete mealpy dependency
+ Update docs, examples and tests

---------------------------------------------------------------------

# Version 1.0.0

+ Refactoring project with 1 abstract class: Benchmark
+ Two sub-packages: name_based and cec_based
+ In name_based package contains all letter modules in order of the alphabet
+ In cec_based package contains all CEC competition modules from years: 2005, 2008, 2010, 2013, 2014, 2015, 2017, 
  2019, 2020, 2021, 2022


---------------------------------------------------------------------

# Version 0.8.0

+ Update author's repo
+ Add constrained CEC-2020
+ Update test cases for CEC-2020


---------------------------------------------------------------------

# Version 0.7.1

+ Update license and requirements environemnt
+ Fix template document


---------------------------------------------------------------------


# Version 0.7.0

+ Update documents
+ Fix some bugs in CEC-2010, 2013, 2015
+ Fix some bugs in CEC basic version
+ Update examples


---------------------------------------------------------------------


# Version 0.6.7

+ Fix some bugs in CEC-2014 no-bias


---------------------------------------------------------------------

# Version 0.6.6

+ Add CEC-2014 without support data


---------------------------------------------------------------------

# Version 0.6.5

+ Fix some bugs in CEC-2013
+ Add CEC-2013 without support data


---------------------------------------------------------------------

# Version 0.6.3

+ Fix some bugs in library introduction
        
---------------------------------------------------------------------

# Version 0.6.2

+ Add CEC-2013 with support data
  
        
---------------------------------------------------------------------

# Version 0.6.1

+ Fix some bugs in CEC-2014
+ Add CEC-2015
+ Update references


        
---------------------------------------------------------------------


# Version 0.6.0

+ Add CEC-2005, CEC-2008, CEC-2010, CEC-2014 with support data
+ Update more documents (references paper)
+ Add examples with test functions
   
        
---------------------------------------------------------------------


# Version 0.5.1

+ Add CEC-2005 with support data
+ Add documents
 
        
---------------------------------------------------------------------

# Version 0.4.3

+ Fix minor error in some functions

    
---------------------------------------------------------------------

# Version 0.4.2

+ Update information and library



---------------------------------------------------------------------


# Version 0.4.1

+ Add cec2014 without support data

    
---------------------------------------------------------------------


# Version 0.3.1

+ Split project into 2 sub-packages: dimension_based and type_based
+ dimension_based has 4 modules: benchmark1d, benchmark2d, benchmark3d, and benchmarknd
+ type_based has 2 modules: uni_modal and multi_modal
    
---------------------------------------------------------------------

# Version 0.2.1

+ Split project into 4 modules: benchmark1d, benchmark2d, benchmark3d, and benchmarknd 

---------------------------------------------------------------------

# Version 0.2.0 

+ Split project into 2 modules: benchmark2d and benchmarknd 

---------------------------------------------------------------------

# Version 0.1.2

+ Add more functions to class Functions


---------------------------------------------------------------------

# Version 0.1.1 (First version)

+ Add project, some functions


