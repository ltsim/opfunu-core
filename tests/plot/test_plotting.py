import subprocess
import sys

import numpy as np
import pytest

from opfunu.plot import Plotting


def _sphere(x):
    return float(np.sum(np.asarray(x, dtype=float) ** 2))


def _plot(**kwargs):
    config = {"n_points": 4}
    config.update(kwargs)
    return Plotting(_sphere, [-2.0, -2.0], [2.0, 2.0], **config)


def test_defaults_construct_without_any_backend():
    plot = Plotting(_sphere, [-1.0, -1.0], [1.0, 1.0])
    assert plot.backend == "matplotlib"
    assert plot.mode == "3d"
    assert plot.selected_dims == (1, 2)
    assert plot.n_points == 100
    assert plot.title == "opfunu landscape (dims 1, 2) [3d]"
    assert "matplotlib" in repr(plot)


def test_bounds_and_title_inferred_from_bound_method():
    from opfunu.name_based import Ackley01

    f = Ackley01(ndim=4)
    plot = Plotting(f.evaluate, mode="2d")
    assert plot.title.startswith("Ackley01 landscape")
    assert plot.selected_dims == (1, 2)


def test_explicit_title_and_selected_dims_are_exposed():
    plot = _plot(mode="2d", title="mine", selected_dims=(2, 1))
    assert plot.title == "mine"
    assert plot.selected_dims == (2, 1)


def test_rejects_unknown_backend():
    with pytest.raises(ValueError, match="backend must be one of"):
        _plot(backend="bokeh")


def test_rejects_unknown_mode():
    with pytest.raises(ValueError, match='mode must be "2d" or "3d"'):
        _plot(mode="4d")


def test_rejects_pyvista_two_dimensional_mode():
    with pytest.raises(ValueError, match="3D meshes only"):
        _plot(backend="pyvista", mode="2d")


def test_requires_bounds_for_plain_callable():
    with pytest.raises(TypeError, match="needs lb and ub"):
        Plotting(_sphere)


def test_rejects_backend_argument_for_wrong_backend():
    with pytest.raises(ValueError, match="template"):
        _plot(template="plotly_white")
    with pytest.raises(ValueError, match="style"):
        _plot(backend="plotly", style="ggplot")
    with pytest.raises(ValueError, match="off_screen"):
        _plot(backend="plotly", off_screen=True)
    with pytest.raises(ValueError, match="figsize"):
        _plot(backend="pyvista", figsize=(8, 6))


def test_rejects_unknown_backend_option_at_construction():
    with pytest.raises(ValueError, match="unsupported matplotlib option"):
        _plot(bogus_option=True)
    with pytest.raises(ValueError, match="unsupported plotly option"):
        _plot(backend="plotly", bogus_option=True)


def test_figure_guard_for_pyvista():
    plot = _plot(backend="pyvista")
    with pytest.raises(AttributeError, match="plotter"):
        plot.figure


def test_plotter_guard_for_matplotlib():
    plot = _plot()
    with pytest.raises(AttributeError, match="pyvista"):
        plot.plotter


def test_save_requires_file_extension():
    with pytest.raises(ValueError, match="file extension"):
        _plot(mode="2d").save("noextension")


def test_importing_opfunu_plot_stays_lazy():
    code = (
        "import sys, opfunu.plot; "
        "loaded = [m for m in ('matplotlib', 'plotly', 'pyvista') if m in sys.modules]; "
        "sys.exit(f'imported backends: {loaded}' if loaded else 0)"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
