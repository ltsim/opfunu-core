"""One view class per rendering backend, used by :class:`opfunu.plot.Plotting`.

Every view is constructed without touching its heavy dependency: ``matplotlib``,
``plotly`` and ``pyvista`` are imported only when the view renders, displays or
exports, so a missing extra surfaces as a guiding ``ImportError`` at the call.
"""

import contextlib
import io
import pathlib
import typing

import numpy as np

from opfunu.plot import _core

PLOT_EXTRA_MSG = (
    "matplotlib is required for Plotting(backend='matplotlib'). Install it with: pip install opfunu-core[plot]"
)
PLOTLY_EXTRA_MSG = (
    "plotly is required for Plotting(backend='plotly'). Install it with: pip install opfunu-core[plotly]"
)
PYVISTA_EXTRA_MSG = (
    "pyvista is required for Plotting(backend='pyvista'). Install it with: pip install opfunu-core[pyvista]"
)
KALEIDO_EXTRA_MSG = (
    "kaleido is required to export Plotly figures to image formats. Install it with: pip install kaleido"
)

SampleFn = typing.Callable[[], tuple[np.ndarray, np.ndarray, np.ndarray]]
StyleLike = typing.Union[str, typing.Sequence[str], typing.Mapping[str, typing.Any]]


class BaseView:
    """Shared lifecycle for a single backend: render once, show, save, close.

    Parameters
    ----------
    sample : callable
        Zero-argument callable returning the cached ``(X, Y, Z)`` surface grid.
    mode : str
        ``"2d"`` or ``"3d"`` view to render.
    title, xlabel, ylabel, cmap : str
        Labels and colormap name shared by every backend.
    options : dict
        Backend-specific options, validated against ``_OPTION_NAMES``.
    style, rc
        Matplotlib-only theme arguments; other views ignore them.
    """

    backend: typing.ClassVar[str] = "base"
    _OPTION_NAMES: typing.ClassVar[frozenset[str]] = frozenset()

    def __init__(
        self,
        *,
        sample: SampleFn,
        mode: str,
        title: str,
        xlabel: str,
        ylabel: str,
        cmap: str,
        options: dict[str, typing.Any],
        style: StyleLike | None = None,
        rc: typing.Mapping[str, typing.Any] | None = None,
    ) -> None:
        unknown = sorted(set(options) - set(self._OPTION_NAMES))
        if unknown:
            raise ValueError(f"unsupported {self.backend} option(s): {unknown}.")
        self._sample = sample
        self.mode = mode
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.cmap = cmap
        self._options = options
        self._style = style
        self._rc = rc
        self._native: typing.Any = None

    def _option(self, name: str, default: typing.Any = None) -> typing.Any:
        """Read a validated backend option with a fallback default."""
        return self._options.get(name, default)

    def render(self) -> typing.Any:
        """Build the native figure once and return it (cached across calls)."""
        if self._native is None:
            self._native = self._create()
        return self._native

    def show(self) -> None:
        """Display the landscape through the backend's native display path."""
        self._display(self.render())

    def save(self, path: str | pathlib.Path, **kwargs: typing.Any) -> pathlib.Path:
        """Write the landscape to ``path``; the file suffix selects the format."""
        target = pathlib.Path(path)
        suffix = target.suffix.lower()
        if not suffix:
            raise ValueError(f"save() needs a file extension, got {target.name!r} (e.g. 'landscape.pdf').")
        self._write(self.render(), target, suffix, **kwargs)
        return target

    def close(self) -> None:
        """Release the native object; a later ``render()`` rebuilds it (the grid stays cached)."""
        if self._native is not None:
            self._dispose(self._native)
            self._native = None

    def repr_html(self) -> str | None:
        """Inline notebook representation as HTML, when the backend has one."""
        return None

    def repr_png(self) -> bytes | None:
        """Inline notebook representation as PNG bytes, when the backend has one."""
        return None

    def _create(self) -> typing.Any:
        raise NotImplementedError

    def _display(self, native: typing.Any) -> None:
        raise NotImplementedError

    def _write(self, native: typing.Any, path: pathlib.Path, suffix: str, **kwargs: typing.Any) -> None:
        raise NotImplementedError

    def _dispose(self, native: typing.Any) -> None:
        raise NotImplementedError


class MatplotlibView(BaseView):
    """Static 2D filled contour / 3D surface views rendered through matplotlib."""

    backend: typing.ClassVar[str] = "matplotlib"
    _OPTION_NAMES: typing.ClassVar[frozenset[str]] = frozenset({"figsize", "dpi", "levels"})
    _FORMATS: typing.ClassVar[frozenset[str]] = frozenset(
        {
            ".png",
            ".jpg",
            ".jpeg",
            ".tif",
            ".tiff",
            ".webp",
            ".raw",
            ".rgba",
            ".pdf",
            ".svg",
            ".svgz",
            ".eps",
            ".ps",
            ".pgf",
        }
    )

    @staticmethod
    def _require() -> typing.Any:
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError(PLOT_EXTRA_MSG) from exc
        return plt

    def _style_context(self, plt: typing.Any) -> contextlib.ExitStack:
        """Enter ``style`` and ``rc`` contexts; caller must close the returned stack."""
        stack = contextlib.ExitStack()
        try:
            if self._style is not None:
                try:
                    stack.enter_context(plt.style.context(self._style))
                except OSError as exc:
                    raise OSError(
                        f"{exc} Styles registered by third-party packages must be imported "
                        "before plotting (their names are not part of matplotlib itself)."
                    ) from exc
            if self._rc is not None:
                stack.enter_context(plt.rc_context(dict(self._rc)))
        except BaseException:
            stack.close()
            raise
        return stack

    def _create(self) -> typing.Any:
        plt = self._require()
        x_grid, y_grid, z_grid = self._sample()
        with self._style_context(plt):
            fig = plt.figure(figsize=self._option("figsize"))
            if self.mode == "2d":
                ax = fig.add_subplot(111)
                contour = ax.contourf(x_grid, y_grid, z_grid, levels=self._option("levels", 50), cmap=self.cmap)
                fig.colorbar(contour, ax=ax)
            else:
                ax = fig.add_subplot(111, projection="3d")
                ax.plot_surface(x_grid, y_grid, z_grid, cmap=self.cmap, edgecolor="none", antialiased=True)
                ax.set_zlabel("f(x)")
            ax.set_xlabel(self.xlabel)
            ax.set_ylabel(self.ylabel)
            ax.set_title(self.title)
        return fig

    def _display(self, native: typing.Any) -> None:
        self._require().show()

    def _write(self, native: typing.Any, path: pathlib.Path, suffix: str, **kwargs: typing.Any) -> None:
        if suffix not in self._FORMATS:
            raise ValueError(
                f"matplotlib cannot write '{suffix}' files; supported suffixes: "
                f"{', '.join(sorted(self._FORMATS))}."
            )
        plt = self._require()
        kwargs.setdefault("dpi", self._option("dpi", 300))
        kwargs.setdefault("bbox_inches", "tight")
        with self._style_context(plt):
            native.savefig(path, format=suffix.lstrip("."), **kwargs)

    def _dispose(self, native: typing.Any) -> None:
        self._require().close(native)

    def repr_png(self) -> bytes | None:
        """Render the figure into in-memory PNG bytes for notebook display."""
        plt = self._require()
        buffer = io.BytesIO()
        with self._style_context(plt):
            self.render().savefig(buffer, format="png", dpi=self._option("dpi", 300))
        return buffer.getvalue()


class PlotlyView(BaseView):
    """Interactive 3D surface / 2D contour views rendered through plotly."""

    backend: typing.ClassVar[str] = "plotly"
    _OPTION_NAMES: typing.ClassVar[frozenset[str]] = frozenset(
        {"renderer", "template", "width", "height", "colorscale", "hovertemplate"}
    )
    _HTML_FORMATS: typing.ClassVar[frozenset[str]] = frozenset({".html", ".htm"})
    _IMAGE_FORMATS: typing.ClassVar[frozenset[str]] = frozenset(
        {".png", ".jpg", ".jpeg", ".webp", ".svg", ".pdf", ".eps"}
    )

    @staticmethod
    def _require() -> typing.Any:
        try:
            import plotly.graph_objects as go
        except ImportError as exc:
            raise ImportError(PLOTLY_EXTRA_MSG) from exc
        return go

    def _create(self) -> typing.Any:
        go = self._require()
        x_grid, y_grid, z_grid = self._sample()
        colorscale = self._option("colorscale")
        if colorscale is None:
            colorscale = _core.plotly_colorscale(self.cmap)
        trace_kwargs: dict[str, typing.Any] = {
            "x": x_grid[0],
            "y": y_grid[:, 0],
            "z": z_grid,
            "colorscale": colorscale,
        }
        hovertemplate = self._option("hovertemplate")
        if hovertemplate is not None:
            trace_kwargs["hovertemplate"] = hovertemplate
        if self.mode == "3d":
            trace_kwargs["colorbar"] = {"title": "f(x)"}
            trace = go.Surface(**trace_kwargs)
        else:
            trace_kwargs["contours"] = {"coloring": "heatmap"}
            trace = go.Contour(**trace_kwargs)
        fig = go.Figure(trace)
        template = self._option("template")
        if template is not None:
            fig.update_layout(template=template)
        if self.mode == "3d":
            fig.update_layout(
                scene={"xaxis_title": self.xlabel, "yaxis_title": self.ylabel, "zaxis_title": "f(x)"}
            )
        else:
            fig.update_layout(xaxis_title=self.xlabel, yaxis_title=self.ylabel)
        layout_kwargs: dict[str, typing.Any] = {"title": self.title}
        for name in ("width", "height"):
            value = self._option(name)
            if value is not None:
                layout_kwargs[name] = value
        fig.update_layout(**layout_kwargs)
        return fig

    def _display(self, native: typing.Any) -> None:
        native.show(renderer=self._option("renderer"))

    def _write(self, native: typing.Any, path: pathlib.Path, suffix: str, **kwargs: typing.Any) -> None:
        if suffix in self._HTML_FORMATS:
            kwargs.setdefault("include_plotlyjs", "cdn")
            kwargs.setdefault("full_html", True)
            native.write_html(str(path), **kwargs)
        elif suffix in self._IMAGE_FORMATS:
            try:
                native.write_image(str(path), **kwargs)
            except ImportError as exc:
                raise ImportError(KALEIDO_EXTRA_MSG) from exc
            except RuntimeError as exc:
                # plotly >= 6 raises RuntimeError when kaleido is not installed
                if "kaleido" in str(exc).lower():
                    raise ImportError(KALEIDO_EXTRA_MSG) from exc
                raise
        else:
            raise ValueError(
                f"plotly cannot write '{suffix}' files; supported suffixes: "
                f"{', '.join(sorted(self._HTML_FORMATS | self._IMAGE_FORMATS))} "
                "(image formats additionally require kaleido)."
            )

    def _dispose(self, native: typing.Any) -> None:
        return None

    def repr_html(self) -> str | None:
        """Embeddable plotly.js HTML fragment for notebook display."""
        self.render()
        try:
            import plotly.io as pio
        except ImportError as exc:
            raise ImportError(PLOTLY_EXTRA_MSG) from exc
        html: str = pio.to_html(self._native, include_plotlyjs="cdn", full_html=False)
        return html


class PyVistaView(BaseView):
    """Interactive 3D structured-grid mesh rendered through pyvista."""

    backend: typing.ClassVar[str] = "pyvista"
    _OPTION_NAMES: typing.ClassVar[frozenset[str]] = frozenset(
        {
            "off_screen",
            "jupyter_backend",
            "window_size",
            "theme",
            "show_edges",
            "show_scalar_bar",
            "show_axes",
        }
    )
    _HTML_FORMATS: typing.ClassVar[frozenset[str]] = frozenset({".html", ".htm"})
    _RASTER_FORMATS: typing.ClassVar[frozenset[str]] = frozenset({".png", ".jpg", ".jpeg", ".tif", ".tiff"})
    _VECTOR_FORMATS: typing.ClassVar[frozenset[str]] = frozenset({".svg", ".pdf", ".eps", ".ps"})

    @staticmethod
    def _require() -> typing.Any:
        try:
            import pyvista as pv
        except ImportError as exc:
            raise ImportError(PYVISTA_EXTRA_MSG) from exc
        return pv

    def _create(self) -> typing.Any:
        pv = self._require()
        x_grid, y_grid, z_grid = self._sample()
        grid = pv.StructuredGrid(x_grid, y_grid, z_grid)
        off_screen = self._option("off_screen")
        if off_screen is None:
            off_screen = _core.headless()
        plotter = pv.Plotter(
            title=self.title,
            off_screen=bool(off_screen),
            window_size=self._option("window_size"),
        )
        theme = self._option("theme")
        if theme is not None:
            plotter.set_theme(theme)
        plotter.add_mesh(
            grid,
            scalars=z_grid.ravel(),
            cmap=self.cmap,
            show_edges=bool(self._option("show_edges", False)),
            show_scalar_bar=bool(self._option("show_scalar_bar", True)),
        )
        if not bool(self._option("show_axes", True)):
            plotter.show_axes = False
        return plotter

    def _display(self, native: typing.Any) -> None:
        if _core.in_notebook():
            native.show(jupyter_backend=self._option("jupyter_backend", "static"))
        else:
            native.show()

    def _write(self, native: typing.Any, path: pathlib.Path, suffix: str, **kwargs: typing.Any) -> None:
        if suffix in self._HTML_FORMATS:
            native.export_html(str(path), **kwargs)
        elif suffix in self._RASTER_FORMATS:
            kwargs.setdefault("return_img", False)
            native.screenshot(str(path), **kwargs)
        elif suffix in self._VECTOR_FORMATS:
            native.save_graphic(str(path))
        else:
            raise ValueError(
                f"pyvista cannot write '{suffix}' files; supported suffixes: "
                f"{', '.join(sorted(self._HTML_FORMATS | self._RASTER_FORMATS | self._VECTOR_FORMATS))} "
                "(HTML export additionally requires trame)."
            )

    def _dispose(self, native: typing.Any) -> None:
        native.close()


VIEWS: dict[str, type[BaseView]] = {
    "matplotlib": MatplotlibView,
    "plotly": PlotlyView,
    "pyvista": PyVistaView,
}
