#!/usr/bin/env python
# Tests for the automatic coordinate-shift propagation (issue #19):
# f(x) = f_base(x - o), x_global = x*_base + o, bounds_new = bounds_base + o.

import inspect

import numpy as np
import pytest

import opfunu


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


def test_shift_none_keeps_base_behavior():
    base = opfunu.name_based.Ackley01(ndim=3)
    func = opfunu.name_based.Ackley01(ndim=3, shift=None)
    assert func.shift is None
    assert np.allclose(func.x_global, base.x_global)
    assert np.allclose(func.bounds, base.bounds)


def test_set_shift_is_idempotent_and_resettable():
    func = opfunu.name_based.Ackley01(ndim=3, shift=[1.0, 1.0, 1.0])
    first_bounds = func.bounds.copy()
    first_x_global = func.x_global.copy()
    func.set_shift([1.0, 1.0, 1.0])
    assert np.allclose(func.bounds, first_bounds)
    assert np.allclose(func.x_global, first_x_global)
    func.set_shift(None)
    base = opfunu.name_based.Ackley01(ndim=3)
    assert np.allclose(func.bounds, base.bounds)
    assert np.allclose(func.x_global, base.x_global)
    assert func.shift is None
    assert "shift" not in func.get_paras()


def test_shift_scalar_is_broadcast():
    func = opfunu.name_based.Ackley01(ndim=3, shift=2.0)
    base = opfunu.name_based.Ackley01(ndim=3)
    assert np.allclose(func.x_global, np.asarray(base.x_global) + 2.0)
    assert np.allclose(func.lb, np.asarray(base.lb) + 2.0)


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


def test_cec_shift_composes_with_official_shift():
    base = opfunu.cec_based.F12005(ndim=5)
    offset = np.ones(5)
    func = opfunu.cec_based.F12005(ndim=5, shift=offset)
    assert np.allclose(func.x_global, np.asarray(base.x_global) + offset)
    assert func.evaluate(func.x_global) == pytest.approx(func.f_global)
    assert func.is_succeed(func.x_global) is True


def test_all_constructors_accept_shift():
    """Every benchmark class must expose `shift` and forward it to the parent."""
    classes = _concrete_benchmark_classes()
    assert len(classes) > 300
    missing = [c.__name__ for c in classes if "shift" not in inspect.signature(c.__init__).parameters]
    assert missing == []


def test_shift_propagates_across_all_functions():
    """Behavior sweep: x_global/lb translate by o and the parent received the vector."""
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
        if not cls.randomized_term:
            assert func.evaluate(np.asarray(func.x_global)) == pytest.approx(
                base.evaluate(np.asarray(func.x_global) - offset)
            ), cls.__name__
