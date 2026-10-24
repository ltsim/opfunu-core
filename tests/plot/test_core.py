import numpy as np
import pytest

from opfunu.plot import HAS_MATPLOTLIB, HAS_PLOTLY, HAS_PYVISTA, Plotting
from opfunu.plot._core import (
    owner_name,
    plotly_colorscale,
    resolve_bounds,
    sample_surface,
    validate_inputs,
)


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


def test_resolve_bounds_prefers_explicit_arguments():
    lb, ub = resolve_bounds(_sphere, [-1.0, -1.0], [1.0, 1.0])
    assert list(lb) == [-1.0, -1.0] and list(ub) == [1.0, 1.0]


def test_resolve_bounds_infers_from_bound_method_owner():
    from opfunu.name_based import Ackley01

    f = Ackley01(ndim=4)
    lb, ub = resolve_bounds(f.evaluate, None, None)
    assert np.asarray(lb).size == 4 and np.asarray(ub).size == 4
    assert owner_name(f.evaluate) == "Ackley01"
    assert owner_name(_sphere) is None


def test_resolve_bounds_rejects_plain_callable_without_bounds():
    with pytest.raises(TypeError, match="needs lb and ub"):
        resolve_bounds(_sphere, None, None)


def test_resolve_bounds_rejects_non_callable():
    with pytest.raises(TypeError, match="callable"):
        resolve_bounds(42.0, None, None)


def test_plotly_colorscale_maps_common_matplotlib_names():
    assert plotly_colorscale("viridis") == "Viridis"
    assert plotly_colorscale("Cividis") == "Cividis"
    assert plotly_colorscale("RdBu") == "RdBu"
    assert plotly_colorscale("inferno") == "inferno"  # unknown names pass through


@pytest.mark.skipif(HAS_MATPLOTLIB, reason="checks the missing-extra guidance")
def test_matplotlib_missing_extra_message():
    with pytest.raises(ImportError, match=r"pip install opfunu-core\[plot\]"):
        Plotting(_sphere, [-1.0, -1.0], [1.0, 1.0], mode="2d", n_points=3).render()


@pytest.mark.skipif(HAS_PLOTLY, reason="checks the missing-extra guidance")
def test_plotly_missing_extra_message():
    with pytest.raises(ImportError, match=r"pip install opfunu-core\[plotly\]"):
        Plotting(_sphere, [-1.0, -1.0], [1.0, 1.0], backend="plotly", n_points=3).render()


@pytest.mark.skipif(HAS_PYVISTA, reason="checks the missing-extra guidance")
def test_pyvista_missing_extra_message():
    with pytest.raises(ImportError, match=r"pip install opfunu-core\[pyvista\]"):
        Plotting(_sphere, [-1.0, -1.0], [1.0, 1.0], backend="pyvista", n_points=3).render()
