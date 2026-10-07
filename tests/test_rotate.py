#!/usr/bin/env python
# Tests for the automatic coordinate-rotation propagation (issue #22):
# f(x) = f_base(M @ (x - o)), x_global = o + M.T @ x*_base,
# bounds_new = AABB(M.T @ bounds_base) + o (enclosing axis-aligned box).

import inspect

import numpy as np
import pytest

import opfunu


def _concrete_benchmark_classes():
    skipped = {"Benchmark", "FuncBenchmark", "CecBenchmark", "Formula", "ABC"}
    return [cls for _, cls in opfunu.ALL_DATABASE if cls.__name__ not in skipped and not inspect.isabstract(cls)]


def _rotation_3d(theta: float) -> np.ndarray:
    """Rotation by ``theta`` in the first two coordinates, identity elsewhere."""
    c, s = np.cos(theta), np.sin(theta)
    matrix = np.eye(3)
    matrix[:2, :2] = [[c, -s], [s, c]]
    return matrix


def _aabb(matrix: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Enclosing axis-aligned box of ``matrix @ [lo, hi]``: c -> M.T @ c, h -> |M.T| @ h."""
    center, half = (lo + hi) / 2.0, (hi - lo) / 2.0
    new_center = matrix.T @ center
    new_half = np.abs(matrix.T) @ half
    return new_center - new_half, new_center + new_half


def test_rotate_updates_x_global_and_bounds():
    base = opfunu.name_based.Ackley01(ndim=3)
    matrix = _rotation_3d(np.pi / 4)
    func = opfunu.name_based.Ackley01(ndim=3, rotate=matrix)
    expected_xg = matrix.T @ np.asarray(base.x_global)
    exp_lo, exp_hi = _aabb(matrix, np.asarray(base.lb), np.asarray(base.ub))
    assert np.allclose(func.x_global, expected_xg)
    assert np.allclose(func.lb, exp_lo)
    assert np.allclose(func.ub, exp_hi)
    assert func.bounds.shape == (3, 2)


def test_rotated_evaluation_matches_base():
    base = opfunu.name_based.Ackley01(ndim=3)
    matrix = _rotation_3d(np.pi / 4)
    func = opfunu.name_based.Ackley01(ndim=3, rotate=matrix)
    rng = np.random.default_rng(42)
    for _ in range(5):
        x = rng.uniform(func.lb, func.ub)
        # f(x) = f_base(M @ (x - o)); here o is None so just M @ x.
        assert func.evaluate(x) == pytest.approx(base.evaluate(np.asarray(matrix) @ np.asarray(x)))
    # The rotated optimum evaluates to the base global value.
    assert func.evaluate(func.x_global) == pytest.approx(base.f_global)
    assert func.is_succeed(func.x_global) is True


def test_rotation_45deg_grows_bounds_to_inscribed_aabb():
    """A 45 deg rotation of the [-35, 35]^3 box has half-width 35*sqrt(2) in the rotated dims."""
    base = opfunu.name_based.Ackley01(ndim=3)
    matrix = _rotation_3d(np.pi / 4)
    func = opfunu.name_based.Ackley01(ndim=3, rotate=matrix)
    diag = 35.0 * np.sqrt(2.0)
    assert np.allclose(np.asarray(func.lb), [-diag, -diag, -35.0])
    assert np.allclose(np.asarray(func.ub), [diag, diag, 35.0])
    # Every corner of the base box maps inside the new bounds (enclosing property).
    lo, hi = np.asarray(base.lb), np.asarray(base.ub)
    for corner in np.array(np.meshgrid(*zip(lo, hi))).T.reshape(-1, 3):
        mapped = matrix.T @ corner
        assert np.all(mapped >= np.asarray(func.lb) - 1e-9)
        assert np.all(mapped <= np.asarray(func.ub) + 1e-9)


def test_rotate_none_keeps_base_behavior():
    base = opfunu.name_based.Ackley01(ndim=3)
    func = opfunu.name_based.Ackley01(ndim=3, rotate=None)
    assert func.rotate is None
    assert np.allclose(func.x_global, base.x_global)
    assert np.allclose(func.bounds, base.bounds)


def test_set_rotate_is_idempotent_and_resettable():
    matrix = _rotation_3d(np.pi / 4)
    func = opfunu.name_based.Ackley01(ndim=3, rotate=matrix)
    first_bounds = func.bounds.copy()
    first_x_global = func.x_global.copy()
    func.set_rotate(matrix)
    assert np.allclose(func.bounds, first_bounds)
    assert np.allclose(func.x_global, first_x_global)
    func.set_rotate(None)
    base = opfunu.name_based.Ackley01(ndim=3)
    assert np.allclose(func.bounds, base.bounds)
    assert np.allclose(func.x_global, base.x_global)
    assert func.rotate is None
    assert "rotate" not in func.get_paras()


def test_rotation_invalid_raises():
    with pytest.raises(ValueError):  # wrong shape
        opfunu.name_based.Ackley01(ndim=3, rotate=np.eye(2))
    with pytest.raises(ValueError):  # not orthogonal (shear)
        opfunu.name_based.Ackley01(ndim=3, rotate=np.ones((3, 3)))
    with pytest.raises(ValueError):  # non-finite entries
        opfunu.name_based.Ackley01(ndim=3, rotate=[[1.0, 0.0, 0.0], [0.0, np.nan, 0.0], [0.0, 0.0, 1.0]])
    with pytest.raises(ValueError):  # not array-like
        opfunu.name_based.Ackley01(ndim=3, rotate="not-a-matrix")
    with pytest.raises(ValueError):  # flat vector, not a matrix
        opfunu.name_based.Ackley01(ndim=3, rotate=[1.0, 2.0])


def test_rotation_reflection_is_allowed():
    """Any orthogonal matrix counts (M.T @ M = I), including det = -1 reflections."""
    reflection = np.diag([1.0, 1.0, -1.0])
    base = opfunu.name_based.Ackley01(ndim=3)
    func = opfunu.name_based.Ackley01(ndim=3, rotate=reflection)
    assert func.evaluate(func.x_global) == pytest.approx(base.f_global)
    assert np.allclose(func.x_global, reflection.T @ np.asarray(base.x_global))


def test_rotate_rigid_bounds_keep_base_box():
    """rotate_bounds=False keeps B_base rigid (CEC style) while x_global still rotates."""
    base = opfunu.name_based.Bukin04()  # optimum [-10, 0] in bounds x in [-15,-5], y in [-3,3]
    matrix = _rotation_3d(np.pi / 4)[:2, :2]
    func = opfunu.name_based.Bukin04(rotate=matrix, rotate_bounds=False)
    assert np.allclose(func.bounds, base.bounds)
    assert np.allclose(func.x_global, matrix.T @ np.asarray(base.x_global))
    # The rotated optimum [-7.07, 7.07] leaves the rigid y in [-3, 3] (documented caveat).
    assert np.all(np.asarray(func.x_global) >= func.lb)
    assert not np.all(np.asarray(func.x_global) <= func.ub)
    assert func.is_succeed(func.x_global) is False
    # With the default enclosing AABB the same optimum stays feasible.
    feasible = opfunu.name_based.Bukin04(rotate=matrix)
    assert np.all(np.asarray(feasible.x_global) >= feasible.lb)
    assert np.all(np.asarray(feasible.x_global) <= feasible.ub)
    assert feasible.is_succeed(feasible.x_global) is True


def test_shift_and_rotate_compose():
    """f(x) = f_base(M @ (x - o)): shift first, then rotation."""
    base = opfunu.name_based.Ackley01(ndim=3)
    matrix = _rotation_3d(np.pi / 4)
    offset = np.array([2.0, -1.5, 3.0])
    func = opfunu.name_based.Ackley01(ndim=3, shift=offset, rotate=matrix)
    assert np.allclose(func.x_global, offset + matrix.T @ np.asarray(base.x_global))
    rng = np.random.default_rng(7)
    for _ in range(5):
        x = rng.uniform(func.lb, func.ub)
        assert func.evaluate(x) == pytest.approx(base.evaluate(np.asarray(matrix) @ (np.asarray(x) - offset)))
    assert func.evaluate(func.x_global) == pytest.approx(base.f_global)
    # Dropping the shift keeps only the rotation.
    func.set_shift(None)
    assert np.allclose(func.x_global, matrix.T @ np.asarray(base.x_global))
    assert "shift" not in func.get_paras()
    assert np.allclose(func.get_paras()["rotate"], matrix)


def test_rotate_exposed_in_paras_and_solution_in_bounds():
    matrix = _rotation_3d(np.pi / 4)
    func = opfunu.name_based.Ackley01(ndim=3, rotate=matrix)
    assert np.allclose(func.get_paras()["rotate"], matrix)
    sol = func.create_solution()
    assert np.all(sol >= func.lb) and np.all(sol <= func.ub)


def test_batch_evaluation_matches_single_with_rotation():
    matrix = _rotation_3d(np.pi / 4)
    offset = np.array([2.0, -1.5, 3.0])
    func = opfunu.name_based.Ackley01(ndim=3, shift=offset, rotate=matrix)
    rng = np.random.default_rng(11)
    X = rng.uniform(func.lb, func.ub, size=(7, 3))
    batch = func._evaluate_batch(X)
    assert batch.shape == (7,)
    for i, row in enumerate(X):
        assert batch[i] == pytest.approx(float(func.evaluate(row)))


def test_cec_rotate_composes_with_official_shift():
    base = opfunu.cec_based.F12005(ndim=5)
    offset = np.ones(5)
    matrix = np.eye(5)[::-1]  # antidiagonal permutation: orthogonal with exact entries
    func = opfunu.cec_based.F12005(ndim=5, shift=offset, rotate=matrix)
    assert np.allclose(func.x_global, offset + matrix.T @ np.asarray(base.x_global))
    assert func.evaluate(func.x_global) == pytest.approx(func.f_global)
    assert func.is_succeed(func.x_global) is True


def test_all_constructors_accept_rotate():
    """Every benchmark class must expose `rotate`/`rotate_bounds` and forward them."""
    classes = _concrete_benchmark_classes()
    assert len(classes) > 300
    missing = [
        c.__name__ for c in classes if not {"rotate", "rotate_bounds"} <= set(inspect.signature(c.__init__).parameters)
    ]
    assert missing == []


def test_rotate_propagates_across_all_functions():
    """Behavior sweep with an exact permutation matrix: x_global/lb/ub map through M.T."""
    for cls in _concrete_benchmark_classes():
        np.random.seed(123)
        base = cls()
        ndim = base.ndim
        matrix = np.eye(ndim)[::-1]  # antidiagonal permutation, orthogonal, exact
        np.random.seed(123)
        func = cls(rotate=matrix)
        assert func.rotate is not None, cls.__name__
        assert np.allclose(np.asarray(func.rotate, dtype=float), matrix), cls.__name__
        assert np.allclose(np.asarray(func.x_global, dtype=float), matrix.T @ np.asarray(base.x_global)), cls.__name__
        assert np.allclose(np.asarray(func.lb, dtype=float), matrix.T @ np.asarray(base.lb)), cls.__name__
        assert np.allclose(np.asarray(func.ub, dtype=float), matrix.T @ np.asarray(base.ub)), cls.__name__
        if not cls.randomized_term:
            assert func.evaluate(np.asarray(func.x_global)) == pytest.approx(
                base.evaluate(matrix @ np.asarray(func.x_global))
            ), cls.__name__
