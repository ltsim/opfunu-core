import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib", reason="requires the [plot] extra")
matplotlib.use("Agg")

from opfunu.plot import HAS_MATPLOTLIB, draw  # noqa: E402


def _sphere(x):
    return float(np.sum(np.asarray(x, dtype=float) ** 2))


def test_plot_extra_flag_matches_import():
    assert HAS_MATPLOTLIB is True


def test_draw_2d_returns_figure():
    fig = draw(mode="2d", target=_sphere, lb=[-5.0, -5.0], ub=[5.0, 5.0], n_points=10, show=False)
    assert type(fig).__name__ == "Figure"
    assert len(fig.axes) == 2  # contour axes + colorbar axes


def test_draw_3d_returns_figure():
    fig = draw(mode="3d", target=_sphere, lb=[-5.0, -5.0], ub=[5.0, 5.0], n_points=10, show=False)
    assert type(fig).__name__ == "Figure"
    assert len(fig.axes) == 1


def test_draw_uses_problem_evaluate_and_selected_dims():
    from opfunu.name_based import Ackley01

    f = Ackley01(ndim=5)
    fig = draw(mode="2d", target=f.evaluate, lb=f.lb, ub=f.ub, selected_dims=(2, 4), n_points=8, show=False)
    assert type(fig).__name__ == "Figure"
