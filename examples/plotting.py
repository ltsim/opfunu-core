#!/usr/bin/env python
"""Plot benchmark landscapes with the optional ``opfunu.plot`` suite.

Requires the static extra::

    pip install opfunu-core[plot]

Run::

    python examples/plotting.py
"""

import matplotlib

matplotlib.use("Agg")

from opfunu.cec_based.cec2010 import F12010
from opfunu.plot import draw

f0 = F12010(ndim=10)

fig_2d = draw(mode="2d", target=f0.evaluate, lb=f0.lb, ub=f0.ub, selected_dims=(2, 3), n_points=50, show=False)
fig_2d.savefig("landscape_2d.png")
print("saved landscape_2d.png")

fig_3d = draw(mode="3d", target=f0.evaluate, lb=f0.lb, ub=f0.ub, selected_dims=(1, 6), n_points=50, show=False)
fig_3d.savefig("landscape_3d.png")
print("saved landscape_3d.png")
