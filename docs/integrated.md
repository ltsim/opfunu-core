# Package-level API

The `opfunu` package exposes two module-level databases — `FUNC_DATABASE` (name-based functions) and
`CEC_DATABASE` (CEC competition functions) — together with a set of helpers to query them.
`HAS_NUMBA` reports whether the opt-in Numba vectorization is available, and `EXCLUDES` lists the
abstract base classes skipped by the query helpers.

::: opfunu
    options:
      members:
        - FUNC_DATABASE
        - CEC_DATABASE
        - ALL_DATABASE
        - EXCLUDES
        - HAS_NUMBA
        - get_functions_by_classname
        - get_functions_based_classname
        - get_functions_by_ndim
        - get_functions_based_ndim
        - get_all_name_based_functions
        - get_all_cec_based_functions
        - get_name_based_functions
        - get_cec_based_functions
