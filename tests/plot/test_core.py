import numpy as np
import pytest

from opfunu.plot import HAS_MATPLOTLIB, HAS_PYVISTA, draw, interactive
from opfunu.plot._core import sample_surface, validate_inputs


def _sphere(x):
    return float(np.sum(np.asarray(x, dtype=float) ** 2))


def test_validate_inputs_ok():
    lb, ub, d0, d1 = validate_inputs(_sphere, [-5.0, -5.0, -5.0], [5.0, 5.0, 5.0], (2, 3), 10)
    assert (d0, d1) == (1, 2)
    assert lb.shape == ub.shape == (3,)


def test_validate_inputs_rejects_bad_target():
    with pytest.raises(TypeError, match="callable"):
        validate_inputs(42.0, [-1.0], [1.0], (1, 1), 10)


def test_validate_inputs_rejects_bad_dims():
    with pytest.raises(ValueError, match="exactly two"):
        validate_inputs(_sphere, [-1.0, -1.0], [1.0, 1.0], (1,), 10)
    with pytest.raises(ValueError, match="out of range"):
        validate_inputs(_sphere, [-1.0, -1.0], [1.0, 1.0], (1, 3), 10)
    with pytest.raises(ValueError, match="must differ"):
        validate_inputs(_sphere, [-1.0, -1.0], [1.0, 1.0], (2, 2), 10)


def test_validate_inputs_rejects_bad_bounds_and_resolution():
    with pytest.raises(ValueError, match="strictly greater"):
        validate_inputs(_sphere, [-1.0, 5.0], [1.0, 4.0], (1, 2), 10)
    with pytest.raises(ValueError, match="n_points"):
        validate_inputs(_sphere, [-1.0, -1.0], [1.0, 1.0], (1, 2), 1)


def test_sample_surface_shape_and_values():
    lb = np.array([-2.0, -2.0])
    ub = np.array([2.0, 2.0])
    xg, yg, zg = sample_surface(_sphere, lb, ub, 0, 1, 7)
    assert xg.shape == yg.shape == zg.shape == (7, 7)
    assert zg[3, 3] == pytest.approx(0.0)  # origin is on the grid
    assert zg.min() == pytest.approx(0.0)


def test_sample_surface_uses_batch_kernel_when_available():
    from opfunu.name_based import Ackley01

    f = Ackley01(ndim=3)
    xg, yg, zg = sample_surface(f.evaluate, np.asarray(f.lb), np.asarray(f.ub), 0, 2, 5)
    assert zg.shape == (5, 5)
    assert np.all(np.isfinite(zg))


def test_draw_invalid_mode_raises_without_backend():
    with pytest.raises(ValueError, match='mode must be "2d" or "3d"'):
        draw(mode="4d", target=_sphere, lb=[-1.0], ub=[1.0])


def test_draw_requires_target_lb_ub():
    with pytest.raises(TypeError, match="requires target, lb and ub"):
        draw(mode="2d")


@pytest.mark.skipif(HAS_PYVISTA, reason="checks the missing-extra guidance")
def test_interactive_missing_extra_message():
    with pytest.raises(ImportError, match=r"pip install opfunu-core\[interactive\]"):
        interactive(target=_sphere, lb=[-1.0, -1.0], ub=[1.0, 1.0], n_points=3)


@pytest.mark.skipif(HAS_MATPLOTLIB, reason="checks the missing-extra guidance")
def test_draw_missing_extra_message():
    with pytest.raises(ImportError, match=r"pip install opfunu-core\[plot\]"):
        draw(mode="2d", target=_sphere, lb=[-1.0, -1.0], ub=[1.0, 1.0], n_points=3)
