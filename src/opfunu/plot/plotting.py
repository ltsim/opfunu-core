"""Class-based entry point for the ``opfunu.plot`` visualization suite."""

import pathlib
import typing

import numpy as np

from opfunu.plot import _core
from opfunu.plot._backends import VIEWS, BaseView, StyleLike

_ARGUMENT_BACKENDS: dict[str, str] = {
    "figsize": "matplotlib",
    "dpi": "matplotlib",
    "style": "matplotlib",
    "rc": "matplotlib",
    "renderer": "plotly",
    "template": "plotly",
    "off_screen": "pyvista",
    "jupyter_backend": "pyvista",
    "theme": "pyvista",
}


class Plotting:
    """Render a benchmark landscape for one objective function.

    Construction validates the arguments but neither samples nor imports a
    backend: the grid is computed lazily on the first ``render()`` / ``show()``
    / ``save()`` call and cached for all later ones. Pass ``problem.evaluate``
    and the search bounds are inferred from the problem instance; any
    ``f(x) -> float`` callable works when ``lb``/``ub`` are given explicitly.

    Parameters
    ----------
    target : callable
        Objective function ``f(x) -> float``, typically ``problem.evaluate``.
    lb, ub : array-like, optional
        Bounds, one entry per dimension. Inferred from the owner of a bound
        ``target`` method when omitted.
    backend : str
        ``"matplotlib"`` (default, static 2D/3D), ``"plotly"`` (interactive
        2D/3D) or ``"pyvista"`` (interactive 3D mesh). Each requires its extra.
    mode : str
        ``"2d"`` or ``"3d"``; defaults to ``"3d"`` so one call works for every
        backend (pyvista renders 3D meshes only).
    selected_dims : tuple of two ints
        1-based indices of the rendered dimensions; every other dimension is
        fixed at its bound midpoint. Defaults to ``(1, 2)``.
    n_points : int
        Grid resolution per rendered dimension. Defaults to 100.
    title : str or None
        Figure title; defaults to ``"<Problem> landscape (dims i, j) [mode]"``.
    cmap : str
        Colormap name shared by the backends (matplotlib/pyvista colormaps,
        mapped onto Plotly colorscales). Defaults to ``"viridis"``.
    figsize, dpi : matplotlib only
        Figure size in inches and raster/export resolution; ``dpi`` defaults
        to 300 for saving and inline PNG rendering.
    style, rc : matplotlib only
        ``style`` accepts a registered style name, a list of names, a path to
        a ``.mplstyle`` file or a dict of rc parameters; ``rc`` layers further
        overrides on top. Both are applied with context managers around
        rendering and saving, so global ``rcParams`` stay untouched. Styles
        registered by third-party packages work once that package has been
        imported by the caller.
    renderer : plotly only
        Plotly renderer name for ``fig.show()`` (defaults to auto-detection).
    template : plotly only
        Registered template name, template dict or ``go.layout.Template``.
    off_screen : pyvista only
        Force off-screen rendering; defaults to auto-detection of headless
        Linux sessions so ``save()`` works without a display.
    jupyter_backend : pyvista only
        Passed to ``plotter.show()`` inside notebooks; defaults to ``"static"``.
    theme : pyvista only
        ``pyvista.Theme`` object applied via ``plotter.set_theme``.
    **backend_options
        Remaining backend-specific options (validated per view): matplotlib
        ``levels``; plotly ``width``, ``height``, ``colorscale``,
        ``hovertemplate``; pyvista ``window_size``, ``show_edges``,
        ``show_scalar_bar``, ``show_axes``.

    Examples
    --------
    >>> from opfunu.name_based import Ackley01
    >>> from opfunu.plot import Plotting
    >>> f = Ackley01(ndim=5)
    >>> plot = Plotting(f.evaluate, mode="2d", n_points=3)  # bounds inferred, nothing rendered yet
    >>> plot
    <Plotting backend='matplotlib' mode='2d' dims=(1, 2) n_points=3>
    >>> plot.save("landscape.pdf")  # doctest: +SKIP
    PosixPath('landscape.pdf')
    """

    def __init__(
        self,
        target: typing.Callable[..., typing.Any],
        lb: typing.Any = None,
        ub: typing.Any = None,
        *,
        backend: str = "matplotlib",
        mode: str = "3d",
        selected_dims: tuple[int, int] | list[int] = (1, 2),
        n_points: int = 100,
        title: str | None = None,
        cmap: str = "viridis",
        figsize: tuple[float, float] | None = None,
        dpi: int | None = None,
        style: StyleLike | None = None,
        rc: typing.Mapping[str, typing.Any] | None = None,
        renderer: str | None = None,
        template: typing.Any = None,
        off_screen: bool | None = None,
        jupyter_backend: str | None = None,
        theme: typing.Any = None,
        **backend_options: typing.Any,
    ) -> None:
        if backend not in VIEWS:
            raise ValueError(f"backend must be one of {sorted(VIEWS)}, got {backend!r}.")
        if mode not in ("2d", "3d"):
            raise ValueError(f'mode must be "2d" or "3d", got {mode!r}.')
        if backend == "pyvista" and mode == "2d":
            raise ValueError("pyvista renders 3D meshes only; use backend='plotly' or 'matplotlib' for mode='2d'.")
        provided: dict[str, typing.Any] = {
            "figsize": figsize,
            "dpi": dpi,
            "style": style,
            "rc": rc,
            "renderer": renderer,
            "template": template,
            "off_screen": off_screen,
            "jupyter_backend": jupyter_backend,
            "theme": theme,
        }
        for name, value in provided.items():
            if value is not None and _ARGUMENT_BACKENDS[name] != backend:
                raise ValueError(
                    f"{name!r} is a {_ARGUMENT_BACKENDS[name]} argument and cannot be used "
                    f"with backend={backend!r}."
                )
        options: dict[str, typing.Any] = dict(backend_options)
        for name, value in provided.items():
            if value is not None and name not in ("style", "rc"):
                options[name] = value

        lb, ub = _core.resolve_bounds(target, lb, ub)
        lb_arr, ub_arr, dim0, dim1 = _core.validate_inputs(target, lb, ub, selected_dims, n_points)
        if title is None:
            label = _core.owner_name(target) or "opfunu"
            title = f"{label} landscape (dims {dim0 + 1}, {dim1 + 1}) [{mode}]"

        self._target = target
        self._lb, self._ub = lb_arr, ub_arr
        self._dim0, self._dim1 = dim0, dim1
        self._n_points = int(n_points)
        self._grid: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self.backend = backend
        self.mode = mode
        self.title = title
        self.cmap = cmap
        self._view: BaseView = VIEWS[backend](
            sample=self._sample_grid,
            mode=mode,
            title=title,
            xlabel=f"x{dim0 + 1}",
            ylabel=f"x{dim1 + 1}",
            cmap=cmap,
            options=options,
            style=style,
            rc=rc,
        )

    def _sample_grid(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample (once, cached) the 2-D slice handed to the active view."""
        if self._grid is None:
            self._grid = _core.sample_surface(
                self._target, self._lb, self._ub, self._dim0, self._dim1, self._n_points
            )
        return self._grid

    @property
    def selected_dims(self) -> tuple[int, int]:
        """1-based indices of the two rendered dimensions."""
        return self._dim0 + 1, self._dim1 + 1

    @property
    def n_points(self) -> int:
        """Grid resolution per rendered dimension."""
        return self._n_points

    @property
    def native(self) -> typing.Any:
        """The backend's native object (matplotlib/plotly Figure, pyvista Plotter)."""
        return self.render()

    @property
    def figure(self) -> typing.Any:
        """Native figure of the matplotlib or plotly backend."""
        if self.backend == "pyvista":
            raise AttributeError("the pyvista backend exposes a Plotter; use the .plotter property.")
        return self.render()

    @property
    def plotter(self) -> typing.Any:
        """Native ``pyvista.Plotter`` of the pyvista backend."""
        if self.backend != "pyvista":
            raise AttributeError(".plotter is only available for backend='pyvista'.")
        return self.render()

    def render(self) -> typing.Any:
        """Build and return the native figure; later calls return the cached object."""
        return self._view.render()

    def show(self) -> None:
        """Display the landscape: plt.show(), fig.show() or plotter.show().

        Inside Jupyter notebooks the pyvista view shows a static render by
        default (``jupyter_backend="static"``) unless another backend is set.
        """
        self._view.show()

    def save(self, path: str | pathlib.Path, **kwargs: typing.Any) -> pathlib.Path:
        """Write the landscape to ``path``; the file suffix selects the format.

        Matplotlib covers png/jpg/tif/webp plus the vector formats pdf, svg,
        svgz, eps, ps and pgf (TeX installation required); Plotly writes
        standalone HTML and image formats once ``kaleido`` is installed;
        pyvista saves screenshots, vector graphics (save_graphic) and HTML
        (trame). Publication defaults: ``dpi`` and ``bbox_inches="tight"`` for
        matplotlib, ``include_plotlyjs="cdn"`` for Plotly HTML.
        """
        return self._view.save(path, **kwargs)

    def close(self) -> None:
        """Release the native object; a later render rebuilds it (the grid stays cached)."""
        self._view.close()

    def _repr_html_(self) -> str | None:
        """Notebook auto-display: interactive HTML for the plotly backend."""
        return self._view.repr_html()

    def _repr_png_(self) -> bytes | None:
        """Notebook auto-display: inline PNG for the matplotlib backend."""
        return self._view.repr_png()

    def __repr__(self) -> str:
        dims = ", ".join(str(d) for d in self.selected_dims)
        return f"<Plotting backend={self.backend!r} mode={self.mode!r} dims=({dims}) n_points={self._n_points}>"
