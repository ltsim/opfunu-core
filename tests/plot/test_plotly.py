import importlib.util

import numpy as np
import pytest

pytest.importorskip("plotly", reason="requires the [plotly] extra")

from opfunu.plot import HAS_PLOTLY, Plotting  # noqa: E402


def _sphere(x):
    return float(np.sum(np.asarray(x, dtype=float) ** 2))


def _plot(**kwargs):
    config = {"backend": "plotly", "n_points": 6}
    config.update(kwargs)
    return Plotting(_sphere, [-2.0, -2.0], [2.0, 2.0], **config)


def test_plotly_extra_flag_matches_import():
    assert HAS_PLOTLY is True


def test_render_3d_builds_surface_trace():
    fig = _plot().render()
    assert type(fig).__name__ == "Figure"
    assert fig.data[0].type == "surface"
    assert len(fig.data[0].z) == 6
    assert fig.layout.scene.zaxis.title.text == "f(x)"


def test_render_2d_builds_contour_trace():
    fig = _plot(mode="2d").render()
    assert fig.data[0].type == "contour"
    assert fig.layout.xaxis.title.text == "x1"
    assert fig.layout.yaxis.title.text == "x2"
    assert "[2d]" in fig.layout.title.text


def test_cmap_is_mapped_to_plotly_colorscale():
    # plotly resolves recognized names into color-stop sequences
    viridis = _plot().render().data[0].colorscale
    cividis = _plot(cmap="cividis").render().data[0].colorscale
    assert len(viridis) > 1 and len(cividis) > 1
    assert list(viridis) != list(cividis)


def test_custom_colorscale_option_wins():
    assert len(_plot(colorscale="Portland").render().data[0].colorscale) > 1


def test_template_is_applied():
    fig = _plot(template={"layout": {"font": {"family": "Times New Roman"}}}).render()
    assert fig.layout.template.layout.font.family == "Times New Roman"


def test_width_and_height_options():
    fig = _plot(width=640, height=480).render()
    assert (fig.layout.width, fig.layout.height) == (640, 480)


def test_save_html(tmp_path):
    text = _plot().save(tmp_path / "fig.html").read_text(encoding="utf-8")
    assert "plotly" in text.lower()
    standalone = _plot().save(tmp_path / "offline.html", include_plotlyjs=True).read_text(encoding="utf-8")
    assert len(standalone) > len(text)  # the plotly.js bundle is embedded


def test_save_rejects_unsupported_suffix(tmp_path):
    with pytest.raises(ValueError, match="plotly cannot write"):
        _plot().save(tmp_path / "fig.xyz")


def test_save_image_without_kaleido_guides_install(tmp_path):
    if importlib.util.find_spec("kaleido") is not None:
        pytest.skip("kaleido is installed; the image export cannot test the guidance message")
    with pytest.raises(ImportError, match="kaleido"):
        _plot().save(tmp_path / "fig.png")


def test_repr_html_contains_plotly_markup():
    html = _plot()._repr_html_()
    assert html is not None and "plotly" in html.lower()
    assert _plot()._repr_png_() is None


def test_show_uses_figure_show(monkeypatch):
    import plotly.graph_objects as go

    calls = []
    monkeypatch.setattr(go.Figure, "show", lambda self, *args, **kwargs: calls.append(kwargs))
    _plot().show()
    assert len(calls) == 1
    _plot(renderer="browser").show()
    assert calls[1]["renderer"] == "browser"
