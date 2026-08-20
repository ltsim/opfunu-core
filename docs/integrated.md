# Package-level API

The `opfunu` package exposes two module-level databases — `FUNC_DATABASE` (name-based functions) and
`CEC_DATABASE` (CEC competition functions) — together with a set of helpers to query them.

::: opfunu
    options:
      members:
        - FUNC_DATABASE
        - CEC_DATABASE
        - ALL_DATABASE
        - get_functions_by_classname
        - get_functions_based_classname
        - get_functions_by_ndim
        - get_functions_based_ndim
        - get_all_name_based_functions
        - get_all_cec_based_functions
        - get_name_based_functions
        - get_cec_based_functions
