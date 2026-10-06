# Utilities

The `opfunu.utils.operator` module provides the low-level mathematical operators (shift, rotation, noise, composition,
... ) used internally by the CEC competition functions and by several traditional functions. Most operators are
decorated with `njit(fastmath=True)` — imported from the `opfunu.utils.numba_compat` shim, so they compile when the
`[numba]` extra is installed and run as plain functions otherwise. Keep them nopython-compatible (`np.asarray`, no
`np.array(list)`, no `np.dot`/`np.matmul` — use the `dot_vv`/`dot_mv`/`dot_vm`/`dot_mm` loop helpers, SciPy-free by
design). `sphere_noise_func`/`fractal_1d_func` stay plain (RNG streams). Reuse these helpers — don't duplicate them.

::: opfunu.utils.operator
    options:
      members_order: alphabetical
      show_if_no_docstring: true
      filters: ["!^_", "!^np$"]

## Numba compatibility shim

`opfunu.utils.numba_compat` exposes `HAS_NUMBA` and `njit`, a drop-in replacement for `numba.njit` supporting both
`njit` and `njit(...)` call forms. When Numba is installed the call is delegated to `numba.njit` unchanged; otherwise the
decorated function is returned as-is. `benchmark/__init__.py` imports Numba lazily inside `__build_compute`, never at
module top level, so the package stays import-safe on PyPy and minimal installs.

::: opfunu.utils.numba_compat
    options:
      members_order: alphabetical
      show_if_no_docstring: true
      filters: ["!^_"]
