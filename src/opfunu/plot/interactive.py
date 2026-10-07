"""Interactive 3D mesh prototype for ``opfunu.plot`` (requires ``pyvista``)."""

import typing

from opfunu.plot._core import sample_surface, validate_inputs

INTERACTIVE_EXTRA_MSG = (
    "pyvista is required for opfunu.plot.interactive. Install it with: pip install opfunu-core[interactive]"
)


def _require_pyvista() -> typing.Any:
    try:
        import pyvista as pv
    except ImportError as exc:
        raise ImportError(INTERACTIVE_EXTRA_MSG) from exc
    return pv


def interactive(
    target: typing.Callable[..., typing.Any] | None = None,
    lb: typing.Any = None,
    ub: typing.Any = None,
    selected_dims: tuple[int, int] | list[int] = (1, 2),
    n_points: int = 50,
    title: str | None = None,
    show: bool = True,
) -> typing.Any:
    """Render the landscape as an interactive ``pyvista`` structured-grid mesh.

    This is an experimental proof-of-concept: the search bounds are sampled as
    a clean structured grid surface supporting panning, zooming and rotation.

    Parameters
    ----------
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
        Grid resolution per rendered dimension. Defaults to 50.
    title : str or None
        Plotter title; defaults to ``"opfunu landscape (dims ...)"``.
    show : bool
        Call ``plotter.show()`` before returning.

    Returns
    -------
    plotter : pyvista.Plotter
        The configured plotter with the landscape mesh added.
    """
    if target is None or lb is None or ub is None:
        raise TypeError("interactive() requires target, lb and ub (e.g. target=f.evaluate, lb=f.lb, ub=f.ub).")
    lb_arr, ub_arr, dim0, dim1 = validate_inputs(target, lb, ub, selected_dims, n_points)
    pv = _require_pyvista()
    x_grid, y_grid, z_grid = sample_surface(target, lb_arr, ub_arr, dim0, dim1, int(n_points))
    grid = pv.StructuredGrid(x_grid, y_grid, z_grid)
    label = title if title is not None else f"opfunu landscape (dims {dim0 + 1}, {dim1 + 1})"
    plotter = pv.Plotter(title=label)
    plotter.add_mesh(grid, scalars=z_grid.ravel(), show_edges=False)
    if show:
        plotter.show()
    return plotter


__all__ = ["interactive"]
