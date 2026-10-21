"""Decoupled visualization suite for benchmark landscapes (optional extras).

The core package stays free of heavy visualization dependencies: importing this
package only pulls in the NumPy sampling math — ``matplotlib`` / ``pyvista`` are
imported inside :func:`draw` / :func:`interactive` when the call is made.

* ``pip install opfunu-core[plot]`` — static 2D contour / 3D surface plots.
* ``pip install opfunu-core[interactive]`` — interactive 3D mesh (pyvista).

Without the respective extra, calling the entry point raises a guiding
``ImportError``. ``HAS_MATPLOTLIB`` / ``HAS_PYVISTA`` report availability
without importing the backends.
"""

import importlib.util

from opfunu.plot.interactive import interactive
from opfunu.plot.static import draw

HAS_MATPLOTLIB: bool = importlib.util.find_spec("matplotlib") is not None
HAS_PYVISTA: bool = importlib.util.find_spec("pyvista") is not None

__all__ = [
    "HAS_MATPLOTLIB",
    "HAS_PYVISTA",
    "draw",
    "interactive",
]
