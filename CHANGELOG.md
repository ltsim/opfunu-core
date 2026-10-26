# Version v2026b1

+ **Breaking (plot):** replaced the function entry points `draw`/`interactive` released with
  `v2026b0` by the class-based `opfunu.plot.Plotting` visualization suite: one object per
  landscape — `Plotting(problem.evaluate)` infers the bounds from the problem, samples lazily
  through the `_evaluate_batch` kernel and renders per `backend`/`mode` (`"matplotlib"` static
  2D/3D by
  default, `"plotly"` interactive 2D/3D, `"pyvista"` interactive 3D). `show()` displays in the
  terminal or inline in Jupyter (HTML repr for plotly, PNG repr for matplotlib, pyvista uses
  the static notebook backend by default), `save()` picks the format from the file suffix —
  matplotlib raster plus PDF/SVG/EPS/PS and LaTeX-native PGF, plotly HTML plus images via
  optional `kaleido`, pyvista screenshots, `save_graphic` vectors and HTML — and themes come
  through matplotlib `style` (names, lists, `.mplstyle` paths or rc dicts, applied per plot
  without leaking global rcParams; third-party style packages work once imported) plus `rc`,
  plotly `template` and pyvista `theme`; `HAS_PLOTLY` reports availability and the new
  `[plotly]`/`[pyvista]` extras join `[plot]` (`[interactive]` stays the pyvista alias) —
  with `examples/plotting.py`, the notebooks `examples/plotting_matplotlib.ipynb` and
  `examples/plotting_plotly.ipynb`, `examples/styles/paper.mplstyle` and `docs/plot.md`
+ Kept the core dependencies NumPy-only: the Jupyter kernel (`ipykernel`) that had crept into
  `dependencies` while running the example notebooks moved to a dev-only `notebook` dependency
  group (`uv sync --group notebook`), so published wheel metadata never requires it
+ Docs: migrated the documentation stack from MkDocs + *Material* to **Sphinx** with the built-in
  **Nature** theme — Markdown sources parsed by MyST (`myst-parser`), the API reference rebuilt on
  `sphinx.ext.autodoc` inside `{eval-rst}` blocks (replacing the mkdocstrings `:::` directives), citations
  via `sphinxcontrib-bibtex`, and a strict `sphinx-build -W --keep-going` gate. Navigation is now
  captioned toctrees + `globaltoc` sidebar instead of the Material top tab bar. Added `sphinx-autobuild`
  local live preview (`uv run sphinx-autobuild docs docs/_build/html --watch src`, `make -C docs livehtml`)
  and a `docs/Makefile`. Benchmark docstrings were converted to valid reST (raw strings for LaTeX so
  `\text`/`\frac` no longer collapse to control characters, correctly indented `.. math::` blocks, properly
  continued `.. [1]` citations, and fixed inline `:math:` roles); a handful of obviously copy-pasted /
  truncated `latex_formula*` attributes were corrected to match their own docstrings. The GitHub Pages
  workflow (`.github/workflows/docs.yml`) now builds with Sphinx and deploys through the Pages artifact
  actions (`configure-pages` / `upload-pages-artifact` / `deploy-pages`) instead of `mkdocs gh-deploy`; the
  Pages *Source* must be set to *GitHub Actions*. The docs dependency group now pulls `sphinx`,
  `myst-parser`, `sphinxcontrib-bibtex`, `sphinx-autobuild` (drops `mkdocs`, `mkdocs-material`,
  `mkdocstrings`, `mkdocs-bibtex`, `pypandoc-binary`)
+ Docs: `docs/img/favicon.svg` — the nested contour bands descending into the global minimum —
  is now the default for both the site logo and the favicon

+ Opened a new cycle of the release model: this version lives on `release/v2026b1` and
+   publishes to PyPI under the same name, `v2026b1`, when the cycle merges into `master`

---------------------------------------------------------------------

# Version v2026b0

+ Added the class-based `opfunu.plot.Plotting` visualization suite (a decoupled, lazy-optional
  rework of this cycle's unreleased `draw`/`interactive` prototypes): one object per landscape —
  `Plotting(problem.evaluate)` infers the bounds from the problem, samples lazily through the
  `_evaluate_batch` kernel and renders per `backend`/`mode` (`"matplotlib"` static 2D/3D by
  default, `"plotly"` interactive 2D/3D, `"pyvista"` interactive 3D). `show()` displays in the
  terminal or inline in Jupyter (HTML repr for plotly, PNG repr for matplotlib, pyvista uses
  the static notebook backend by default), `save()` picks the format from the file suffix —
  matplotlib raster plus PDF/SVG/EPS/PS and LaTeX-native PGF, plotly HTML plus images via
  optional `kaleido`, pyvista screenshots, `save_graphic` vectors and HTML — and themes come
  through matplotlib `style` (names, lists, `.mplstyle` paths or rc dicts, applied per plot
  without leaking global rcParams; third-party style packages work once imported) plus `rc`,
  plotly `template` and pyvista `theme`; `HAS_PLOTLY` reports availability and the new
  `[plotly]`/`[pyvista]` extras join `[plot]` (`[interactive]` stays the pyvista alias) —
  with `examples/plotting.py`, the notebooks `examples/plotting_matplotlib.ipynb` and
  `examples/plotting_plotly.ipynb`, `examples/styles/paper.mplstyle` and `docs/plot.md`

+ Standardized the base classes (whole-repo over-engineering audit): the shape storage
+   (`_ndim`/`_bounds`) behind `bounds`/`ndim`/`lb`/`ub`, the `is_succeed`/`is_ndim_compatible`/
+   `create_solution` implementations and the shared trait defaults (`name`, `continuous`, …,
+   `modality`) now live once on `Benchmark` instead of being duplicated across `FuncBenchmark`
+   and `CecBenchmark`; the four CEC data-file loaders share one `_load_table` helper and the
+   `data_*` folder is auto-detected from the trailing competition year of the class name
+   (`F202017` was previously only correct because of its explicit declaration), so the 71
+   redundant per-class data-path declarations were removed
+ Removed dead configuration and aliases: the `_bind_kernel(params=...)` attribute-setting
+   branch (no callers), the `_kernel` property (alias of `compute`), the unused `dim_max`
+   setter, the `load_tow_matrix` typo alias, the never-matched `EXCLUDES` entries, and the
+   unused `f8f2_func`/`gz_func`/`sphere_noise_func` operators; `opfunu.plot` now re-exports
+   the lazy engine entry points directly, and the function-registry helpers share one filter
+   core (`opfunu/__init__.py`)
+ Guarded the `Traversable` import with `sys.version_info` instead of a `try`/`except ImportError`,
+   so `mypy` type-checks cleanly on Python 3.12–3.14 (`importlib.abc.Traversable` was removed in
+   3.14) while the Python 3.10 fallback keeps working; runtime behaviour is unchanged
+ Added Husky git hooks: `pre-commit` and `pre-push` run the same trio as CI — `ruff check --fix`
+   then `ruff check`, `mypy`, and the full `pytest` suite. Ruff's auto-fixes are staged only for
+   files it touched, so unrelated unstaged work is never swept into a commit. Requires a one-time
+   `npm install` (Node >= 18); `HUSKY=0` skips a run
+ Renamed the `precompile` flag to `prepare` and made compilation deferrable: `prepare=True`
+   compiles the kernel with Numba at construction, `prepare=False` binds the plain-Python kernel,
+   and `problem.prepare()` compiles such an instance on demand (idempotent; returns whether the
+   kernel ended up compiled); `prepare=None` (the default) follows the package-wide
+   `Benchmark.default_prepare` policy. The `precompile` property is dropped (`numba_compiled`
+   remains), `plain=True` kernels are never compiled, and `prepare()` is a quiet no-op without
+   Numba — keeping the PyPy path unchanged
+ Opened the second cycle of the release model: this version lives on `release/v2026b0` and
+   publishes to PyPI under the same name, `v2026b0`, when the cycle merges into `master`

---------------------------------------------------------------------

# Version v2026a0

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
+ Added `precompile` to every benchmark constructor: with Numba installed, `precompile=False`
+   keeps the plain-Python kernel for that problem and `precompile=True` requests compilation;
+   `precompile=None` (the default) follows the package-wide `Benchmark.default_precompile`
+   policy (`True` out of the box, read at construction). `problem.precompile` exposes the
+   resolved choice, and kernels bound `plain=True` stay uncompiled whatever the flag says
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


