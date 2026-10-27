#!/usr/bin/env python
# Tests for the coordinate shift (issue #19), now a constructor-only kernel argument:
# f(x) = f_base(x - o) with the kernel receiving (x, shift, *params, out),
# x_global = x*_base + o, bounds_new = bounds_base + o.

import inspect

import numpy as np
import pytest

import opfunu
from opfunu.benchmark import Benchmark


def _concrete_benchmark_classes():
    skipped = {"Benchmark", "FuncBenchmark", "CecBenchmark", "Formula", "ABC"}
    return [cls for _, cls in opfunu.ALL_DATABASE if cls.__name__ not in skipped and not inspect.isabstract(cls)]


def test_shift_updates_x_global_and_bounds():
    base = opfunu.name_based.Ackley01(ndim=3)
    shift = np.array([2.0, -1.5, 3.0])
    func = opfunu.name_based.Ackley01(ndim=3, shift=shift)
    assert np.allclose(func.x_global, np.asarray(base.x_global) + shift)
    assert np.allclose(func.lb, np.asarray(base.lb) + shift)
    assert np.allclose(func.ub, np.asarray(base.ub) + shift)
    assert func.bounds.shape == (3, 2)


def test_shifted_evaluation_matches_base():
    base = opfunu.name_based.Ackley01(ndim=3)
    shift = np.array([2.0, -1.5, 3.0])
    func = opfunu.name_based.Ackley01(ndim=3, shift=shift)
    rng = np.random.default_rng(42)
    for _ in range(5):
        x = rng.uniform(func.lb, func.ub)
        assert func.evaluate(x) == pytest.approx(base.evaluate(np.asarray(x) - shift))
    # The shifted optimum evaluates to the base global value.
    assert func.evaluate(func.x_global) == pytest.approx(base.f_global)
    assert func.is_succeed(func.x_global) is True


def test_shift_is_a_kernel_argument():
    """The shift travels to the kernel as its second argument; the storage stays unshifted."""
    base = opfunu.name_based.Ackley01(ndim=3)
    offset = np.array([2.0, -1.5, 3.0])
    func = opfunu.name_based.Ackley01(ndim=3, shift=offset)
    # Kernel contract: (x, shift, ..., out)
    params = list(inspect.signature(func.compute).parameters)
    assert params[0] == "x" and params[1] == "shift" and params[-1] == "out"
    state = vars(func)
    assert "_Benchmark__f_global" in state and "_Benchmark__x_global" in state
    assert "_f_global" not in state and "_x_global" not in state
    # The kernel vector is the normalized offset, in the problem dtype.
    assert np.allclose(state["_Benchmark__kernel_shift"], offset)
    assert state["_Benchmark__kernel_shift"].dtype == func.dtype
    assert np.allclose(state["_Benchmark__shift"], offset)
    # Stored shape/optimum remain the unshifted definition; the properties translate.
    assert np.allclose(state["_Benchmark__x_global"], np.asarray(base.x_global))
    assert np.allclose(state["_Benchmark__bounds"], base.bounds)
    assert np.allclose(func.x_global, np.asarray(base.x_global) + offset)
    # A coordinate shift translates the optimum location, never its value.
    assert func.f_global == base.f_global
    assert state["_Benchmark__f_global"] == base.f_global


def test_unshifted_kernel_receives_zeros():
    func = opfunu.name_based.Ackley01(ndim=4)
    state = vars(func)
    assert func.shift is None and state["_Benchmark__shift"] is None
    assert state["_Benchmark__kernel_shift"].shape == (4,)
    assert np.allclose(state["_Benchmark__kernel_shift"], np.zeros(4))
    # One uniform compiled signature: the shift argument is always there.
    params = list(inspect.signature(func.compute).parameters)
    assert params[1] == "shift"


def test_shift_none_keeps_base_behavior():
    base = opfunu.name_based.Ackley01(ndim=3)
    func = opfunu.name_based.Ackley01(ndim=3, shift=None)
    assert func.shift is None
    assert np.allclose(func.x_global, base.x_global)
    assert np.allclose(func.bounds, base.bounds)


def test_shift_is_immutable():
    """The shift is fixed at construction: the kernel is bound with it, so no setter exists."""
    func = opfunu.name_based.Ackley01(ndim=3, shift=[1.0, 2.0, 3.0])
    assert not hasattr(Benchmark, "set_shift")
    assert not hasattr(Benchmark, "_apply_shift")
    assert not hasattr(Benchmark, "_normalize_shift")
    # ...and neither the shift nor any other state can be patched after construction.
    for gone in ("_set_ndim", "_set_bounds", "_set_f_global", "_set_x_global", "_set_paras", "_set_support_path"):
        assert not hasattr(Benchmark, gone), gone
    assert not hasattr(func, "set_shift")
    assert "set_shift" not in dir(func)
    with pytest.raises(AttributeError):
        func.shift = np.zeros(3)
    # The post-construction wrapper and the base-state snapshots are gone.
    state = vars(func)
    for gone in ("_Benchmark__base_bounds", "_Benchmark__base_f_global", "_Benchmark__base_x_global"):
        assert gone not in state, gone
    assert "_Benchmark__init_depth" not in state
    # Re-constructing is the only way to change it.
    other = opfunu.name_based.Ackley01(ndim=3, shift=[9.0, 9.0, 9.0])
    assert not np.allclose(func.x_global, other.x_global)
    assert func.f_global == other.f_global


def test_shift_vector_is_copied_not_aliased():
    """Mutating the caller's vector afterwards must not move a bound problem."""
    offset = np.array([1.0, 1.0, 1.0])
    func = opfunu.name_based.Ackley01(ndim=3, shift=offset)
    base = opfunu.name_based.Ackley01(ndim=3)
    x_before = np.asarray(func.x_global).copy()
    bounds_before = np.asarray(func.bounds).copy()
    offset[:] = 50.0
    assert np.allclose(np.asarray(func.shift), [1.0, 1.0, 1.0])
    assert np.allclose(np.asarray(func.x_global), x_before)
    assert np.allclose(np.asarray(func.x_global), np.asarray(base.x_global) + [1.0, 1.0, 1.0])
    assert np.allclose(np.asarray(func.bounds), bounds_before)
    assert func.evaluate(np.asarray(func.x_global)) == pytest.approx(base.f_global)


def test_shift_scalar_is_broadcast():
    func = opfunu.name_based.Ackley01(ndim=3, shift=2.0)
    base = opfunu.name_based.Ackley01(ndim=3)
    assert np.allclose(func.x_global, np.asarray(base.x_global) + 2.0)
    assert np.allclose(func.lb, np.asarray(base.lb) + 2.0)
    assert np.allclose(vars(func)["_Benchmark__kernel_shift"], np.full(3, 2.0))


def test_shift_invalid_raises():
    with pytest.raises(ValueError):
        opfunu.name_based.Ackley01(ndim=3, shift=[1.0, 2.0])
    with pytest.raises(ValueError):
        opfunu.name_based.Ackley01(ndim=3, shift=[np.nan, 0.0, 0.0])
    with pytest.raises(ValueError):
        opfunu.name_based.Ackley01(ndim=3, shift="not-a-vector")


def test_shift_fixed_dimension_function():
    func = opfunu.name_based.Ackley02(shift=[1.0, -2.0])
    assert np.allclose(func.x_global, np.array([1.0, -2.0]))
    assert func.is_succeed(func.x_global) is True


def test_shift_exposed_in_paras_and_solution_in_bounds():
    func = opfunu.name_based.Ackley01(ndim=3, shift=[2.0, -1.5, 3.0])
    assert np.allclose(func.get_paras()["shift"], [2.0, -1.5, 3.0])
    sol = func.create_solution()
    assert np.all(sol >= func.lb) and np.all(sol <= func.ub)
    # The injected shift is not a user-declared kernel parameter.
    assert "shift" not in func._param_names


def test_unshifted_paras_have_no_shift_key():
    assert "shift" not in opfunu.name_based.Ackley01(ndim=3).get_paras()


def test_cec_shift_composes_with_official_shift():
    base = opfunu.cec_based.F12005(ndim=5)
    offset = np.ones(5)
    func = opfunu.cec_based.F12005(ndim=5, shift=offset)
    assert np.allclose(func.x_global, np.asarray(base.x_global) + offset)
    assert func.evaluate(func.x_global) == pytest.approx(func.f_global)
    assert func.is_succeed(func.x_global) is True
    # The loaded official shift stays the stored optimum; the offset is applied on read.
    assert np.allclose(vars(func)["_Benchmark__x_global"], np.asarray(base.x_global))


def test_cec_shift_restores_nothing_to_restore():
    """A CEC problem built with a shift reports the composed optimum and value."""
    base = opfunu.cec_based.F12013(ndim=10)
    offset = np.full(10, 0.5)
    func = opfunu.cec_based.F12013(ndim=10, shift=offset)
    assert func.f_global == base.f_global
    assert np.allclose(func.x_global, np.asarray(base.x_global) + offset)
    assert np.allclose(func.bounds, base.bounds + offset[:, None])


def test_all_constructors_accept_shift():
    """Every benchmark class must expose `shift` and forward it to the parent."""
    classes = _concrete_benchmark_classes()
    assert len(classes) > 300
    missing = [c.__name__ for c in classes if "shift" not in inspect.signature(c.__init__).parameters]
    assert missing == []


def test_shift_propagates_across_all_functions():
    """Behavior sweep: every kernel takes the shift and x_global/lb translate by o."""
    legacy = ("_f_global", "_x_global", "_paras", "_support_path", "_ndim", "_bounds")
    for cls in _concrete_benchmark_classes():
        np.random.seed(123)
        base = cls()
        offset = np.ones(base.ndim)
        np.random.seed(123)
        func = cls(shift=offset)
        assert func.shift is not None, cls.__name__
        assert np.allclose(np.asarray(func.shift, dtype=float), offset), cls.__name__
        assert np.allclose(np.asarray(func.x_global, dtype=float), np.asarray(base.x_global) + offset), cls.__name__
        assert np.allclose(np.asarray(func.lb, dtype=float), np.asarray(base.lb) + offset), cls.__name__
        # Uniform kernel contract: the shift is argument #2 of every kernel.
        params = list(inspect.signature(base.compute).parameters)
        assert params[:2] == ["x", "shift"], cls.__name__
        assert params[-1] == "out", cls.__name__
        assert "shift" not in base._param_names, cls.__name__
        # No class may leave a shadowing protected attribute behind: every value
        # has to reach the mangled Benchmark storage, or the properties go stale.
        for key in legacy:
            assert key not in vars(base), f"{cls.__name__} still writes {key}"
            assert key not in vars(func), f"{cls.__name__} still writes {key}"
        if not cls.randomized_term:
            assert func.evaluate(np.asarray(func.x_global)) == pytest.approx(
                base.evaluate(np.asarray(func.x_global) - offset)
            ), cls.__name__
