#!/usr/bin/env python
# Created by "Thieu" at 11:23, 16/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
#
# Examples:
# >>> from opfunu.cec_based.cec2014 import F12014
# >>>
# >>> f1 = F12014(ndim=30, f_bias=100)
# >>>
# >>> lower_bound = f1.lb                       # Numpy array
# >>> lower_bound_as_list = f1.lb.to_list()     # Python list
# >>> upper_bound = f1.ub
# >>>
# >>> solution = np.random.uniform(0, 1, 30)
# >>> print(f1.evaluate(solution))
# >>>
# >>> print(f1.get_paras())         # Print the parameters of function if has

__version__ = "2026a0"

import inspect
import re
from typing import Any

from . import cec_based, name_based
from .utils.numba_compat import HAS_NUMBA

__all__ = [
    "FUNC_DATABASE",
    "CEC_DATABASE",
    "ALL_DATABASE",
    "EXCLUDES",
    "HAS_NUMBA",
    "get_functions_by_classname",
    "get_functions_based_classname",
    "get_functions_by_ndim",
    "get_functions_based_ndim",
    "get_all_name_based_functions",
    "get_all_cec_based_functions",
    "get_name_based_functions",
    "get_cec_based_functions",
]

FUNC_DATABASE: list[tuple[str, Any]] = inspect.getmembers(name_based, inspect.isclass)
CEC_DATABASE: list[tuple[str, Any]] = inspect.getmembers(cec_based, inspect.isclass)
ALL_DATABASE: list[tuple[str, Any]] = FUNC_DATABASE + CEC_DATABASE
EXCLUDES: list[str] = ["FuncBenchmark", "CecBenchmark"]


def _classes(database: list[tuple[str, Any]]) -> list[Any]:
    """Concrete problem classes of a database, abstract bases filtered out."""
    return [cls for classname, cls in database if classname not in EXCLUDES]


def _match_traits(classes: list[Any], ndim: int | None, **traits: bool | None) -> list[Any]:
    """Filter classes by compatibility with ``ndim`` and the given boolean traits."""
    classes = [f for f in classes if f().is_ndim_compatible(ndim)]
    for name, expected in traits.items():
        if expected is not None:
            classes = [f for f in classes if getattr(f, name) == expected]
    return classes


def get_functions_by_classname(name: Any = None) -> list[Any]:
    """
    Parameters
    ----------
    name : The exact classname of the function (no difference among lowercase, uppercase, mix)

    Returns
    -------
        List of the functions, but all the classname are different, so the result is list of 1 function or list of empty
    """
    return [cls for classname, cls in ALL_DATABASE if classname not in EXCLUDES and classname.lower() == name.lower()]


def get_functions_based_classname(name: Any = None) -> list[Any]:
    """
    Parameters
    ----------
    name : The substring of classname of the function

    Returns
    -------
        List of the functions
    """
    return [
        cls
        for classname, cls in ALL_DATABASE
        if classname not in EXCLUDES and re.search(name.lower(), classname.lower())
    ]


def get_functions_by_ndim(ndim: int | None = None) -> list[Any]:
    """
    Parameters
    ----------
    ndim : The exact number of dimensions that function has and not able to change

    Returns
    -------
        List of the functions
    """
    functions = _classes(ALL_DATABASE)
    if isinstance(ndim, int) and ndim > 1:
        return [f for f in functions if f().dim_default == ndim and f().dim_changeable is False]
    return functions


def get_functions_based_ndim(ndim: int | None = None) -> list[Any]:
    """
    Parameters
    ----------
    ndim : Number of dimensions that function supported

    Returns
    -------
        List of the functions
    """
    functions = _classes(ALL_DATABASE)
    if isinstance(ndim, int) and ndim > 1:
        return [f for f in functions if f().dim_default == ndim or f().dim_changeable is True]
    return functions


def get_all_name_based_functions() -> list[Any]:
    return _classes(FUNC_DATABASE)


def get_all_cec_based_functions() -> list[Any]:
    return _classes(CEC_DATABASE)


def get_name_based_functions(
    ndim: int | None,
    continuous: bool | None = None,
    linear: bool | None = None,
    convex: bool | None = None,
    unimodal: bool | None = None,
    separable: bool | None = None,
    differentiable: bool | None = None,
    scalable: bool | None = None,
    randomized_term: bool | None = None,
    parametric: bool | None = None,
    modality: bool | None = None,
) -> list[Any]:
    return _match_traits(
        _classes(FUNC_DATABASE),
        ndim,
        continuous=continuous,
        linear=linear,
        convex=convex,
        unimodal=unimodal,
        separable=separable,
        differentiable=differentiable,
        scalable=scalable,
        randomized_term=randomized_term,
        parametric=parametric,
        modality=modality,
    )


def get_cec_based_functions(
    ndim: int | None = None,
    continuous: bool | None = None,
    linear: bool | None = None,
    convex: bool | None = None,
    unimodal: bool | None = None,
    separable: bool | None = None,
    differentiable: bool | None = None,
    scalable: bool | None = None,
    randomized_term: bool | None = None,
    parametric: bool | None = True,
    shifted: bool | None = True,
    rotated: bool | None = None,
    modality: bool | None = None,
) -> list[Any]:
    return _match_traits(
        _classes(CEC_DATABASE),
        ndim,
        continuous=continuous,
        linear=linear,
        convex=convex,
        unimodal=unimodal,
        separable=separable,
        differentiable=differentiable,
        scalable=scalable,
        randomized_term=randomized_term,
        parametric=parametric,
        shifted=shifted,
        rotated=rotated,
        modality=modality,
    )
