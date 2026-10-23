"""Decoupled visualization suite for benchmark landscapes (optional extras).

The core package stays free of heavy visualization dependencies: nothing is
imported here until :func:`draw` or :func:`interactive` is actually called.

* ``pip install opfunu-core[plot]`` — static 2D contour / 3D surface plots.
* ``pip install opfunu-core[interactive]`` — interactive 3D mesh (pyvista).
"""

import importlib.util
import typing

__all__ = [
    "HAS_MATPLOTLIB",
    "HAS_PYVISTA",
    "draw",
    "interactive",
]

HAS_MATPLOTLIB: bool = importlib.util.find_spec("matplotlib") is not None
HAS_PYVISTA: bool = importlib.util.find_spec("pyvista") is not None


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
    """Render a benchmark landscape (static matplotlib engine).

    See :func:`opfunu.plot.static.draw` for full documentation. Requires the
    ``plot`` extra; otherwise raises an ``ImportError`` guiding the install.
    """
    from opfunu.plot.static import draw as _draw

    return _draw(
        mode=mode,
        target=target,
        lb=lb,
        ub=ub,
        selected_dims=selected_dims,
        n_points=n_points,
        title=title,
        cmap=cmap,
        show=show,
    )


def interactive(
    target: typing.Callable[..., typing.Any] | None = None,
    lb: typing.Any = None,
    ub: typing.Any = None,
    selected_dims: tuple[int, int] | list[int] = (1, 2),
    n_points: int = 50,
    title: str | None = None,
    show: bool = True,
) -> typing.Any:
    """Render the landscape as an interactive pyvista mesh (experimental).

    See :func:`opfunu.plot.interactive.interactive` for full documentation.
    Requires the ``interactive`` extra; otherwise raises an ``ImportError``
    guiding the install.
    """
    from opfunu.plot.interactive import interactive as _interactive

    return _interactive(
        target=target,
        lb=lb,
        ub=ub,
        selected_dims=selected_dims,
        n_points=n_points,
        title=title,
        show=show,
    )
