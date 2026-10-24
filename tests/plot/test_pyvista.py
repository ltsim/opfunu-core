import numpy as np
import pytest

pv = pytest.importorskip("pyvista", reason="requires the [pyvista] extra")

from opfunu.plot import HAS_PYVISTA, Plotting  # noqa: E402


def _sphere(x):
    return float(np.sum(np.asarray(x, dtype=float) ** 2))


def _plot():
    return Plotting(_sphere, [-2.0, -2.0], [2.0, 2.0], backend="pyvista", n_points=5, off_screen=True)


def _render_or_skip():
    plot = _plot()
    try:
        return plot, plot.render()
    except Exception as exc:  # noqa: BLE001 - VTK may lack a GL stack on this machine
        pytest.skip(f"pyvista cannot render off-screen here: {exc}")


def test_pyvista_extra_flag_matches_import():
    assert HAS_PYVISTA is True


def test_render_returns_plotter():
    _, plotter = _render_or_skip()
    assert isinstance(plotter, pv.Plotter)


def test_plotter_property_and_figure_guard():
    plot, plotter = _render_or_skip()
    assert plot.plotter is plotter is plot.native
    with pytest.raises(AttributeError, match="plotter"):
        plot.figure


def test_save_screenshot(tmp_path):
    plot, _ = _render_or_skip()
    try:
        path = plot.save(tmp_path / "fig.png")
    except Exception as exc:  # noqa: BLE001 - rendering backend unavailable
        pytest.skip(f"pyvista screenshot needs a rendering setup: {exc}")
    assert path.exists() and path.stat().st_size > 0


def test_save_graphic(tmp_path):
    plot, _ = _render_or_skip()
    try:
        path = plot.save(tmp_path / "fig.svg")
    except Exception as exc:  # noqa: BLE001 - rendering backend unavailable
        pytest.skip(f"pyvista save_graphic needs a rendering setup: {exc}")
    assert path.exists()


def test_save_rejects_unsupported_suffix(tmp_path):
    plot, _ = _render_or_skip()
    with pytest.raises(ValueError, match="pyvista cannot write"):
        plot.save(tmp_path / "fig.xyz")


def test_close_rebuilds_plotter():
    plot, first = _render_or_skip()
    try:
        plot.close()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"pyvista close needs a rendering setup: {exc}")
    assert plot.native is not first
