#!/usr/bin/env python
# Created by "Travis" at 10:00, 13/09/2023 ---------%
#       Github: https://github.com/firestrand       %
# --------------------------------------------------%

import numpy as np

import opfunu
from opfunu import get_all_cec_based_functions


def test_cec_kernels_are_inline_not_class_level():
    """Every CEC problem carries its kernel on the instance, not as a class method."""
    for cls in (opfunu.cec_based.F12005, opfunu.cec_based.F12014, opfunu.cec_based.F12017, opfunu.cec_based.F12022):
        assert "compute" not in vars(cls), f"{cls.__name__} still defines a class-level compute"
        assert callable(cls(ndim=10)._kernel)


def test_cec_inherited_kernel_is_reused():
    """A subclass that does not define its own kernel reuses the parent's closure."""
    parent = opfunu.cec_based.F182005(ndim=10)
    child = opfunu.cec_based.F192005(ndim=10)
    assert child._kernel.__code__ is parent._kernel.__code__


def test_cec_subclass_kernel_is_not_poisoned_by_parent_build():
    """F82010(F72010) shares a signature; the parent build must not shadow the child kernel."""
    parent = opfunu.cec_based.F72010(ndim=1000)
    child = opfunu.cec_based.F82010(ndim=1000)
    assert parent._kernel.__code__ is not child._kernel.__code__
    assert abs(child.evaluate(child.x_global) - child.f_global) <= child.epsilon


def test_whenNdimNone_thenDefaultNdimUsed():
    allFunctions = get_all_cec_based_functions()
    for f in allFunctions:
        f_default = f()
        assert f_default.ndim == f_default.dim_default, f"{f.__name__} failed to have ndim == dim_default"


def test_whenEvaulateDefaultNdim_thenHasResult():
    failing = []
    for f in get_all_cec_based_functions():
        f_default = f()
        x = np.random.rand(f_default.dim_default)
        try:
            f_default.evaluate(x)
        except Exception as ex:
            print(f"{f_default.__name__}:{f_default.dim_default}:{ex}")
            failing.append(f.__name__)

    assert len(failing) == 0, f"{failing} failed to evaluate"


def test_whenEvaulateWith_x_global_then_f_global():
    # The following are broken or have incorrect or unknown correct , values.
    known_failing = ["F72008"]
    all_functions = [x for x in get_all_cec_based_functions() if x.__name__ not in known_failing]
    failing = []
    for f in all_functions:
        f_default = f()
        x_global = f_default.x_global
        if abs(f_default.evaluate(x_global) - f_default.f_global) >= f_default.epsilon:
            failing.append(f.__name__)
    assert len(failing) == 0, f"{failing} failed to have x_global result in f_global"
