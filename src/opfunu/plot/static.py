"""Static 2D/3D rendering engine for ``opfunu.plot`` (requires ``matplotlib``)."""

import builtins
import typing

from opfunu.plot._core import sample_surface, validate_inputs

PLOT_EXTRA_MSG = "matplotlib is required for opfunu.plot.draw. Install it with: pip install opfunu-core[plot]"


def _require_matplotlib() -> typing.Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(PLOT_EXTRA_MSG) from exc
    return plt


def _in_notebook() -> bool:
    """Detect a Jupyter kernel (via the injected ``get_ipython``) for inline display."""
    get_ipython = getattr(builtins, "get_ipython", None)
    if not callable(get_ipython):
        return False
    shell = get_ipython()
    if shell is None:
        return False
    return bool(shell.__class__.__name__ == "ZMQInteractiveShell")


def draw(
    mode: str = "2d",
    target: typing.Callable[..., typing.Any] | None = None,
    lb: typing.Any = None,
    ub: typing.Any = None,
    selected_dims: tuple[int, int] | list[int] = (1, 2),
    n_points: int = 100,
    title: str | None = None,
    cmap: str = "viridis",
    show: bool = True,
) -> typing.Any:
    """Render a benchmark landscape through a single static entry point.

    Parameters
    ----------
    mode : str
        ``"2d"`` for a filled contour map, ``"3d"`` for a surface plot.
    target : callable
        Objective function ``f(x) -> float``, e.g. ``problem.evaluate``.
    lb : array-like
        Lower bounds, one per dimension (e.g. ``problem.lb``).
    ub : array-like
        Upper bounds, one per dimension (e.g. ``problem.ub``).
    selected_dims : tuple of two ints
        1-based indices of the rendered dimensions; the rest are fixed at
        bound midpoints. Defaults to ``(1, 2)``.
    n_points : int
        Grid resolution per rendered dimension. Defaults to 100.
    title : str or None
        Axes title; defaults to ``f"{mode} landscape dims {selected_dims}"``.
    cmap : str
        Matplotlib colormap name. Defaults to ``"viridis"``.
    show : bool
        Call ``plt.show()`` before returning. Skipped automatically inside
        Jupyter notebooks, where the returned figure displays inline.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure (its single axes holds the plot).
    """
    if mode not in ("2d", "3d"):
        raise ValueError(f'mode must be "2d" or "3d", got {mode!r}.')
    if target is None or lb is None or ub is None:
        raise TypeError("draw() requires target, lb and ub (e.g. target=f.evaluate, lb=f.lb, ub=f.ub).")
    lb_arr, ub_arr, dim0, dim1 = validate_inputs(target, lb, ub, selected_dims, n_points)
    plt = _require_matplotlib()
    x_grid, y_grid, z_grid = sample_surface(target, lb_arr, ub_arr, dim0, dim1, int(n_points))
    fig = plt.figure()
    label = title if title is not None else f"Landscape dims ({dim0 + 1}, {dim1 + 1}) [{mode}]"
    if mode == "2d":
        ax = fig.add_subplot(111)
        contour = ax.contourf(x_grid, y_grid, z_grid, levels=50, cmap=cmap)
        fig.colorbar(contour, ax=ax)
        ax.set_xlabel(f"x{dim0 + 1}")
        ax.set_ylabel(f"x{dim1 + 1}")
        ax.set_title(label)
    else:
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(x_grid, y_grid, z_grid, cmap=cmap, edgecolor="none", antialiased=True)
        ax.set_xlabel(f"x{dim0 + 1}")
        ax.set_ylabel(f"x{dim1 + 1}")
        ax.set_zlabel("f(x)")
        ax.set_title(label)
    if show and not _in_notebook():
        plt.show()
    return fig


__all__ = ["draw"]
