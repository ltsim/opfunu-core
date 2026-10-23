# Plotting

Landscape visualization lives in the decoupled `opfunu.plot` package so the core
stays free of heavy visualization dependencies. Nothing is imported until `draw`
or `interactive` is actually called — `import opfunu` never touches matplotlib.

Install the extras you need:

```bash
pip install opfunu-core[plot]         # static 2D contour / 3D surface (matplotlib)
pip install opfunu-core[interactive]  # interactive 3D mesh (pyvista, experimental)
```

Without the extra, calling the entry point raises a guiding `ImportError`
(e.g. ``pip install opfunu-core[plot]``). `opfunu.plot.HAS_MATPLOTLIB` and
`opfunu.plot.HAS_PYVISTA` report availability without importing the backends.

## Unified static engine

Everything flows through the singular `draw` entry point. The two rendered
dimensions are 1-based `selected_dims`; every other dimension is fixed at its
bound midpoint. When `target` is a bound `Benchmark.evaluate` method the
internal `_evaluate_batch` kernel is used; any `f(x) -> float` callable works.

```python
from opfunu.cec_based.cec2010 import F12010
from opfunu.plot import draw

f0 = F12010(ndim=10)

draw(mode="2d", target=f0.evaluate, lb=f0.lb, ub=f0.ub, selected_dims=(2, 3), n_points=300)
draw(mode="3d", target=f0.evaluate, lb=f0.lb, ub=f0.ub, selected_dims=(1, 6), n_points=100)
```

Inside Jupyter notebooks the figure displays inline automatically; in a
terminal `plt.show()` is called unless `show=False`. The `Figure` is always
returned, so callers can customize or save it. A runnable script lives at
`examples/plotting.py`.

## Interactive 3D prototype (experimental)

```python
from opfunu.plot import interactive

plotter = interactive(target=f0.evaluate, lb=f0.lb, ub=f0.ub, selected_dims=(1, 2), show=False)
plotter.show()  # pan / zoom / rotate the structured-grid mesh
```

## API

::: opfunu.plot
    options:
      members_order: alphabetical
      show_if_no_docstring: true
      filters: ["!^_"]

::: opfunu.plot.static
    options:
      members: [draw]

::: opfunu.plot.interactive
    options:
      members: [interactive]
