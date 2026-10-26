# Plotting

Landscape visualization lives in the decoupled `opfunu.plot` package so the core
stays free of heavy visualization dependencies. `import opfunu.plot` only pulls in
the NumPy sampling math — the backend itself is imported by the view when a call
actually renders.

Install the extras you need:

```bash
pip install opfunu-core[plot]         # static 2D contour / 3D surface (matplotlib)
pip install opfunu-core[plotly]       # interactive 3D surface / 2D contour (plotly)
pip install opfunu-core[pyvista]      # interactive 3D mesh (pyvista)
pip install opfunu-core[interactive]  # legacy alias of the pyvista extra
```

Without the extra, rendering raises a guiding `ImportError`.
`opfunu.plot.HAS_MATPLOTLIB`, `HAS_PLOTLY` and `HAS_PYVISTA` report availability
without importing the backends.

## The `Plotting` class

Everything flows through one class. The two rendered dimensions are 1-based
`selected_dims`; every other dimension is fixed at its bound midpoint. When
`target` is a bound `Benchmark.evaluate` method the search bounds are inferred
from the problem (sampling then runs through the internal `_evaluate_batch`
kernel); any `f(x) -> float` callable works when `lb`/`ub` are passed explicitly.

```python
from opfunu.cec_based.cec2010 import F12010
from opfunu.plot import Plotting

f0 = F12010(ndim=10)

plot = Plotting(f0.evaluate, backend="matplotlib", mode="3d", selected_dims=(1, 6))
plot.show()
plot.save("landscape.pdf", dpi=600)
```

Construction only validates its arguments: the grid is sampled lazily on the
first `render()` / `show()` / `save()` and cached, so several exports reuse one
sampling pass. `render()` and the `.native` property return the backend object
(`matplotlib`/`plotly` `Figure`, `pyvista` `Plotter`; also exposed as
`.figure` / `.plotter`).

| backend | modes | native object |
|---|---|---|
| `matplotlib` (default) | `2d`, `3d` | `matplotlib.figure.Figure` |
| `plotly` | `2d`, `3d` | `plotly.graph_objects.Figure` |
| `pyvista` | `3d` | `pyvista.Plotter` |

`mode` defaults to `"3d"` so the same constructor works for every backend.
Extra backend options are plain keywords: matplotlib `levels`, plotly `width`,
`height`, `colorscale`, `hovertemplate`, `renderer`, pyvista `window_size`,
`show_edges`, `show_scalar_bar`, `show_axes`, `off_screen`, `jupyter_backend`,
`theme`; unknown ones raise at construction. `close()` releases the native
object (the cached grid survives).

## Export for publication

`save()` infers the format from the file suffix and returns the written
`pathlib.Path`.

| backend | raster | vector | other |
|---|---|---|---|
| matplotlib | png, jpg/jpeg, tif/tiff, webp | pdf, svg, svgz, eps, ps, **pgf** | — |
| plotly | png, jpg/jpeg, webp (needs `kaleido`) | svg, pdf, eps (needs `kaleido`) | html |
| pyvista | png, jpg/jpeg, tif/tiff | svg, pdf, eps, ps | html (needs `trame`) |

Matplotlib keeps publication defaults — `dpi=300` and `bbox_inches="tight"`
unless overridden per call (`plot.save("fig.png", dpi=1200, transparent=True)`).
`.pgf` writes LaTeX-native figure source and requires a working TeX
installation; `.svg`/`.pdf` do not. Plotly HTML export always works
(`include_plotlyjs=True` produces a fully offline file); `pip install kaleido`
adds static image export. `close()` plus `save()` also work headless: pyvista
detects display-less Linux sessions and renders off-screen automatically.

## Custom themes

- **matplotlib** — `style` accepts a registered style name, a list of names, a
  path to a `.mplstyle` file or a dict of rc parameters, and `rc` layers
  further overrides on top. Both are applied with context managers around
  rendering *and* saving, so global `rcParams` stay untouched:

```python
Plotting(f.evaluate, style="ggplot", rc={"font.size": 9})
Plotting(f.evaluate, style="styles/paper.mplstyle")   # see examples/styles/
```

  Styles registered by third-party packages work the same way — import the
  package first so matplotlib knows the name; otherwise the error says exactly
  that.

- **plotly** — `template` takes a registered template name, a template dict or
  a `go.layout.Template`.
- **pyvista** — `theme` takes a `pyvista.Theme` applied via `plotter.set_theme`.

## Jupyter notebooks

- Evaluating `plot` as the last cell expression displays it inline: Plotly
  figures through their HTML repr, matplotlib figures as PNG.
- `plot.show()` works everywhere: `plt.show()` for matplotlib, `fig.show()`
  for plotly, and for pyvista `plotter.show(jupyter_backend="static")` inside
  notebooks (a dependency-free still image; choose `jupyter_backend="trame"`
  for the interactive viewer).
- Runnable notebooks (clean, no stored outputs):
  [`examples/plotting_matplotlib.ipynb`](https://github.com/ltsim/opfunu-core/blob/master/examples/plotting_matplotlib.ipynb)
  and
  [`examples/plotting_plotly.ipynb`](https://github.com/ltsim/opfunu-core/blob/master/examples/plotting_plotly.ipynb).
  A non-notebook script lives at [`examples/plotting.py`](https://github.com/ltsim/opfunu-core/blob/master/examples/plotting.py).
  From a repository checkout, `uv sync --extra plot --group notebook` (or
  `--extra plotly`) adds the Jupyter kernel; the package itself never depends on it.

## API

```{eval-rst}
.. automodule:: opfunu.plot
   :members:
   :undoc-members:
```

```{eval-rst}
.. automodule:: opfunu.plot.plotting
   :members: Plotting
```
