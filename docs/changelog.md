# Changelog

The full changelog is maintained in the repository:

[CHANGELOG.md](https://github.com/ltsim/opfunu-core/blob/master/CHANGELOG.md)

## Version 2026b0

* Released on the `release/v2026b0` branch; published as PEP 440 version `2026.10` when the
  cycle merges into `master`.
* Base classes standardized after a whole-`src` over-engineering audit: shape storage, `is_succeed`,
  `is_ndim_compatible` and the trait defaults live once on `Benchmark`; CEC data folders are
  auto-detected from the class-name year; dead config (`_bind_kernel(params=…)`, `_kernel`,
  the `dim_max` setter, the `load_tow_matrix` typo, unused operators) removed.
* `Traversable` import guarded by `sys.version_info`, so `mypy` stays clean on Python 3.12–3.14
  (`importlib.abc.Traversable` was removed in 3.14) while Python 3.10 keeps working.
* Husky git hooks: `pre-commit` and `pre-push` run `ruff`, `mypy` and the full `pytest` suite —
  the same checks as CI — before anything is committed or pushed.

## Version 2026a0

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
