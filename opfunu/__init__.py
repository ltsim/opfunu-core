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
# >>> fitness = f1.evaluate
# >>>
# >>> solution = np.random.uniform(0, 1, 30)
# >>> print(f1.evaluate(solution))
# >>> print(fitness.evaluate(solution))
# >>>
# >>> print(f1.get_paras())         # Print the parameters of function if has
# >>>
# >>> Plot 2d or plot 3d contours
# >>> Warning !! change n_points to reduce the computing time
# >>>
# >>> import opfunu
# >>> f2 = opfunu.cec_based.F22005(ndim=2)
# >>> f2.plot_2d(selected_dims=(2, 3), n_points=300)
# >>> f2.plot_3d(selected_dims=(1, 4), n_points=300)

__version__ = "2026"

import inspect
import re
from typing import Any

from . import cec_based, name_based

FUNC_DATABASE: list[tuple[str, Any]] = inspect.getmembers(name_based, inspect.isclass)
CEC_DATABASE: list[tuple[str, Any]] = inspect.getmembers(cec_based, inspect.isclass)
ALL_DATABASE: list[tuple[str, Any]] = FUNC_DATABASE + CEC_DATABASE
EXCLUDES: list[str] = ["Benchmark", "CecBenchmark", "ABC"]


def get_functions_by_classname(name=None):
    """
    Parameters
    ----------
    name : The exact classname of the function (no difference among lowercase, uppercase, mix)

    Returns
    -------
        List of the functions, but all the classname are different, so the result is list of 1 function or list of empty
    """
    return [cls for classname, cls in ALL_DATABASE if
            (classname not in EXCLUDES and (classname.lower() == name.lower()))]


def get_functions_based_classname(name=None):
    """
    Parameters
    ----------
    name : The substring of classname of the function

    Returns
    -------
        List of the functions
    """
    return [cls for classname, cls in ALL_DATABASE if
            (classname not in EXCLUDES and re.search(name.lower(), classname.lower()))]


def get_functions_by_ndim(ndim=None):
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


def get_functions_based_ndim(ndim=None):
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


def get_all_name_based_functions():
    return [cls for classname, cls in FUNC_DATABASE if classname not in EXCLUDES]


def get_all_cec_based_functions():
    return [cls for classname, cls in CEC_DATABASE if classname not in EXCLUDES]


def get_name_based_functions(ndim, continuous=None, linear=None, convex=None, unimodal=None, separable=None,
                             differentiable=None, scalable=None, randomized_term=None, parametric=None, modality=None):
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


def get_cec_based_functions(ndim=None, continuous=None, linear=None, convex=None, unimodal=None, separable=None,
                            differentiable=None,
                            scalable=None, randomized_term=None, parametric=True, shifted=True, rotated=None,
                            modality=None):
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
