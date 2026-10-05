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

__version__ = "2026a"

import inspect
import re
from typing import Any

from . import cec_based, name_based

FUNC_DATABASE: list[tuple[str, Any]] = inspect.getmembers(name_based, inspect.isclass)
CEC_DATABASE: list[tuple[str, Any]] = inspect.getmembers(cec_based, inspect.isclass)
ALL_DATABASE: list[tuple[str, Any]] = FUNC_DATABASE + CEC_DATABASE
EXCLUDES: list[str] = ["Benchmark", "FuncBenchmark", "CecBenchmark", "ABC"]


def get_functions_by_classname(name: Any = None) -> list[Any]:
    """
    Parameters
    ----------
    name : The exact classname of the function (no difference among lowercase, uppercase, mix)

    Returns
    -------
        List of the functions, but all the classname are different, so the result is list of 1 function or list of empty
    """
    return [
        cls for classname, cls in ALL_DATABASE if (classname not in EXCLUDES and (classname.lower() == name.lower()))
    ]


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
        if (classname not in EXCLUDES and re.search(name.lower(), classname.lower()))
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
    functions = [cls for classname, cls in ALL_DATABASE if classname not in EXCLUDES]
    if isinstance(ndim, int) and ndim > 1:
        return [f for f in functions if (f().dim_default == ndim and f().dim_changeable is False)]
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
    functions = [cls for classname, cls in ALL_DATABASE if classname not in EXCLUDES]
    if isinstance(ndim, int) and ndim > 1:
        return [f for f in functions if (f().dim_default == ndim or f().dim_changeable is True)]
    return functions


def get_all_name_based_functions() -> list[Any]:
    return [cls for classname, cls in FUNC_DATABASE if classname not in EXCLUDES]


def get_all_cec_based_functions() -> list[Any]:
    return [cls for classname, cls in CEC_DATABASE if classname not in EXCLUDES]


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
    functions = [cls for classname, cls in FUNC_DATABASE if classname not in EXCLUDES]
    functions = [f for f in functions if f().is_ndim_compatible(ndim)]

    functions = [f for f in functions if (continuous is None) or (f.continuous == continuous)]
    functions = [f for f in functions if (linear is None) or (f.linear == linear)]
    functions = [f for f in functions if (convex is None) or (f.convex == convex)]
    functions = [f for f in functions if (unimodal is None) or (f.unimodal == unimodal)]
    functions = [f for f in functions if (separable is None) or (f.separable == separable)]
    functions = [f for f in functions if (differentiable is None) or (f.differentiable == differentiable)]
    functions = [f for f in functions if (scalable is None) or (f.scalable == scalable)]
    functions = [f for f in functions if (randomized_term is None) or (f.randomized_term == randomized_term)]
    functions = [f for f in functions if (parametric is None) or (f.parametric == parametric)]
    functions = [f for f in functions if (modality is None) or (f.modality == modality)]
    return functions


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
    functions = [cls for classname, cls in CEC_DATABASE if classname not in EXCLUDES]
    functions = [f for f in functions if f().is_ndim_compatible(ndim)]

    functions = [f for f in functions if (continuous is None) or (f.continuous == continuous)]
    functions = [f for f in functions if (linear is None) or (f.linear == linear)]
    functions = [f for f in functions if (convex is None) or (f.convex == convex)]
    functions = [f for f in functions if (unimodal is None) or (f.unimodal == unimodal)]
    functions = [f for f in functions if (separable is None) or (f.separable == separable)]
    functions = [f for f in functions if (differentiable is None) or (f.differentiable == differentiable)]
    functions = [f for f in functions if (scalable is None) or (f.scalable == scalable)]
    functions = [f for f in functions if (randomized_term is None) or (f.randomized_term == randomized_term)]
    functions = [f for f in functions if (parametric is None) or (f.parametric == parametric)]
    functions = [f for f in functions if (shifted is None) or (f.shifted == shifted)]
    functions = [f for f in functions if (rotated is None) or (f.rotated == rotated)]
    functions = [f for f in functions if (modality is None) or (f.modality == modality)]
    return functions
