#!/usr/bin/env python
"""Save benchmark landscapes with the ``opfunu.plot.Plotting`` class.

Requires the static extra::

    pip install opfunu-core[plot]

Run::

    python examples/plotting.py

Generated figures land in ``examples/figures/`` (git-ignored). The Jupyter
notebooks ``plotting_matplotlib.ipynb`` and ``plotting_plotly.ipynb`` show the
same workflow interactively.
"""

import pathlib

import matplotlib

matplotlib.use("Agg")

from opfunu.cec_based.cec2010 import F12010
from opfunu.plot import Plotting

f0 = F12010(ndim=10)
here = pathlib.Path(__file__).resolve().parent
outdir = here / "figures"
outdir.mkdir(exist_ok=True)

plot_2d = Plotting(
    f0.evaluate,
    backend="matplotlib",
    mode="2d",
    selected_dims=(2, 3),
    n_points=200,
)
plot_2d.save(outdir / "landscape_2d.png", dpi=600)
print("saved examples/figures/landscape_2d.png")

plot_3d = Plotting(
    f0.evaluate,
    backend="matplotlib",
    mode="3d",
    selected_dims=(1, 6),
    n_points=100,
    cmap="cividis",
    style=str(here / "styles" / "paper.mplstyle"),  # custom academic theme
)
plot_3d.save(outdir / "landscape_3d.pdf")
print("saved examples/figures/landscape_3d.pdf")
