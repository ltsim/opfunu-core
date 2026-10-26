# Changelog

The full changelog is maintained in the repository:

[CHANGELOG.md](https://github.com/ltsim/opfunu-core/blob/master/CHANGELOG.md)

## Version v2026b1

* **Breaking (plot):** the `draw`/`interactive` entry points released with `v2026b0` are replaced
  by the class-based `opfunu.plot.Plotting` suite: `matplotlib` (default, 2D/3D), `plotly`
  (interactive 2D/3D) and `pyvista` (interactive 3D mesh), each behind its own lazy extra
  (`[plot]`, `[plotly]`, `[pyvista]`); bounds are inferred from a bound `problem.evaluate`,
  sampling is cached, and figures display inline in Jupyter. `save()` picks the format from the
  file suffix (matplotlib PDF/SVG/EPS/PS/PGF, plotly HTML + optional kaleido images, pyvista
  screenshots and vector graphics); themes via `style`/`rc` (matplotlib), `template` (plotly),
  `theme` (pyvista). Ships `examples/plotting.py`, the two example notebooks,
  `examples/styles/paper.mplstyle` and `docs/plot.md`.
* Core dependencies stay NumPy-only: the Jupyter kernel (`ipykernel`) lives in a dev-only
  `notebook` dependency group instead of `dependencies`.
* Docs: moved from MkDocs/Material to Sphinx with the built-in *Nature* theme — MyST Markdown,
  autodoc API reference inside `{eval-rst}` blocks, `sphinxcontrib-bibtex` citations, and a strict
  `sphinx-build -W` build. Navigation is captioned toctrees + a `globaltoc` sidebar (no more Material
  top tabs). Added `sphinx-autobuild` live preview + a `docs/Makefile`; benchmark docstrings were
  repaired for reST (raw LaTeX, valid math/citation blocks); GitHub Pages now deploys a Sphinx artifact
  from `docs.yml` (set Pages Source to *GitHub Actions*).
* Docs: `docs/img/favicon.svg` (contour bands + minimum dot) is the default logo and favicon of
  the built site.
* Released on the `release/v2026b1` branch; published to PyPI under the same name, `v2026b1`,
  when the cycle merges into `master`.

## Version v2026b0

* Released on the `release/v2026b0` branch; published to PyPI under the same name, `v2026b0`,
  when the cycle merges into `master`.
* Base classes standardized after a whole-`src` over-engineering audit: shape storage, `is_succeed`,
  `is_ndim_compatible` and the trait defaults live once on `Benchmark`; CEC data folders are
  auto-detected from the class-name year; dead config (`_bind_kernel(params=…)`, `_kernel`,
  the `dim_max` setter, the `load_tow_matrix` typo, unused operators) removed.
* `Traversable` import guarded by `sys.version_info`, so `mypy` stays clean on Python 3.12–3.14
  (`importlib.abc.Traversable` was removed in 3.14) while Python 3.10 keeps working.
* Husky git hooks: `pre-commit` and `pre-push` run `ruff`, `mypy` and the full `pytest` suite —
  the same checks as CI — before anything is committed or pushed.
* The `precompile` flag is renamed to `prepare` and compilation becomes deferrable: `True` compiles
  the kernel at construction, `False` binds the plain-Python kernel, and `problem.prepare()`
  compiles a deferred instance on demand (idempotent; returns whether the kernel ended up compiled);
  `None` (the default) follows the package-wide `Benchmark.default_prepare` policy. `prepare()` is a
  quiet no-op without Numba, so the PyPy path is unchanged.

## Version v2026a0

* Maintenance release based on the original `opfunu` library.
* Project migrated to a `src/` layout and managed with [uv](https://github.com/astral-sh/uv).
* Tooling added: `ruff` (lint + format) and `mypy` (type-check, strict mode).
* Test suite refactored around shared fixtures in `tests/conftest.py`.
* Documentation rewritten with [MkDocs](https://www.mkdocs.org/) and the *Material* theme,
  published on GitHub Pages.
* All benchmark functions migrated to `compute(x, *params, out)` kernels compiled at instantiation
  with `numba.guvectorize`; `evaluate` is inherited from the `Benchmark` base class and
  `_evaluate_batch` evaluates whole `(N, ndim)` populations in one call. Added `parallel`,
  `fastmath` and `dtype` constructor flags.
* Numba is an opt-in extra (`pip install opfunu-core[numba]`, CPython only); without it every kernel
  binds as plain Python so the library stays fully usable on CPython and PyPy
  (`opfunu.HAS_NUMBA`, per-instance `numba_compiled`).
* Compilation stays optional even with Numba installed: every problem accepts `precompile`
  (`False` keeps the plain kernel, `True` requests compilation, `None` follows the
  package-wide `Benchmark.default_precompile` policy).
