# Changelog

The full changelog is maintained in the repository:

[CHANGELOG.md](https://github.com/ltsim/opfunu-core/blob/master/CHANGELOG.md)

## Version 2026a

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
