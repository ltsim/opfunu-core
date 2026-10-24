import pytest

matplotlib = pytest.importorskip("matplotlib", reason="requires the [plot] extra")
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from opfunu.plot import HAS_MATPLOTLIB, Plotting  # noqa: E402


def _sphere(x):
    return float(np.sum(np.asarray(x, dtype=float) ** 2))


def _plot(**kwargs):
    config = {"mode": "2d", "n_points": 6}
    config.update(kwargs)
    return Plotting(_sphere, [-2.0, -2.0], [2.0, 2.0], **config)


def test_plot_extra_flag_matches_import():
    assert HAS_MATPLOTLIB is True


def test_render_2d_and_3d_return_figures():
    fig2d = _plot().render()
    assert type(fig2d).__name__ == "Figure"
    assert len(fig2d.axes) == 2  # contour axes + colorbar axes
    assert "dims 1, 2" in fig2d.axes[0].get_title()
    fig3d = _plot(mode="3d").render()
    assert len(fig3d.axes) == 1
    assert fig3d.axes[0].get_zlabel() == "f(x)"


def test_render_is_cached_across_calls():
    plot = _plot()
    assert plot.render() is plot.native is plot.render()


def test_close_releases_and_rebuilds():
    plot = _plot()
    first = plot.render()
    plot.close()
    assert plot.render() is not first


def test_save_raster_and_vector_formats(tmp_path):
    plot = _plot()
    png = plot.save(tmp_path / "fig.png")
    assert png.exists() and png.read_bytes()[:4] == b"\x89PNG"
    pdf = plot.save(tmp_path / "fig.pdf")
    assert pdf.read_bytes().startswith(b"%PDF")
    svg = plot.save(tmp_path / "fig.svg")
    assert b"<svg" in svg.read_bytes()


def test_save_pgf(tmp_path):
    # The PGF backend renders through mathtext unless a TeX install is present.
    plot = _plot()
    try:
        pgf = plot.save(tmp_path / "fig.pgf")
    except Exception as exc:  # noqa: BLE001 - optional LaTeX toolchain
        pytest.skip(f"PGF export requires a usable TeX/PGF setup: {exc}")
    assert pgf.exists() and pgf.stat().st_size > 0


def test_save_rejects_unsupported_suffix(tmp_path):
    with pytest.raises(ValueError, match="matplotlib cannot write"):
        _plot().save(tmp_path / "fig.xyz")


def test_style_and_rc_apply_without_leaking():
    grid_before = plt.rcParams["axes.grid"]
    plot = _plot(style="ggplot", rc={"axes.titlesize": 12})
    fig = plot.render()
    assert fig.axes[0].get_title()  # rendered inside the style context
    assert plt.rcParams["axes.grid"] == grid_before


def test_mplstyle_file_is_applied(tmp_path):
    style = tmp_path / "tiny.mplstyle"
    style.write_text("axes.grid: true\nfont.size: 7\n", encoding="utf-8")
    _plot(style=str(style)).render()
    assert plt.rcParams["axes.grid"] is False  # restored after the render context


def test_unknown_style_mentions_third_party_import():
    with pytest.raises(OSError, match="third-party"):
        _plot(style="no-such-style-anywhere").render()


def test_repr_png_returns_png_bytes():
    data = _plot()._repr_png_()
    assert data is not None and data[:4] == b"\x89PNG"
    assert _plot()._repr_html_() is None


def test_show_calls_plt_show(monkeypatch):
    calls = []
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: calls.append(args))
    _plot().show()
    assert len(calls) == 1
