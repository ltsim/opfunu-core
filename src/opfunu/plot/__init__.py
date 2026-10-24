"""Decoupled visualization suite for benchmark landscapes (optional extras).

The core package stays free of heavy visualization dependencies: importing
this package only pulls in the NumPy sampling math — ``matplotlib``,
``plotly`` and ``pyvista`` are imported by the :class:`Plotting` view when a
call actually renders.

* ``pip install opfunu-core[plot]``        — static 2D contour / 3D surface (matplotlib).
* ``pip install opfunu-core[plotly]``      — interactive 3D surface / 2D contour (plotly).
* ``pip install opfunu-core[pyvista]``     — interactive 3D mesh (pyvista).
* ``pip install opfunu-core[interactive]`` — legacy alias of the pyvista extra.

Without the respective extra, rendering raises a guiding ``ImportError``.
``HAS_MATPLOTLIB`` / ``HAS_PLOTLY`` / ``HAS_PYVISTA`` report availability
without importing the backends.
"""

import importlib.util

from opfunu.plot.plotting import Plotting

HAS_MATPLOTLIB: bool = importlib.util.find_spec("matplotlib") is not None
HAS_PLOTLY: bool = importlib.util.find_spec("plotly") is not None
HAS_PYVISTA: bool = importlib.util.find_spec("pyvista") is not None

__all__ = [
    "HAS_MATPLOTLIB",
    "HAS_PLOTLY",
    "HAS_PYVISTA",
    "Plotting",
]
