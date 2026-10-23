# Support

## Contributing

There are lots of ways how you can contribute to opfunu-core's development, and you are welcome to join in! For
example, you can report problems or make feature requests on the
[issues page](https://github.com/ltsim/opfunu-core/issues). To facilitate contributions, please check for the
guidelines in the [CONTRIBUTING.md](https://github.com/ltsim/opfunu-core/blob/master/CONTRIBUTING.md) file.

Please read the [CODE OF CONDUCT](https://github.com/ltsim/opfunu-core/blob/master/CODE_OF_CONDUCT.md) too.

## Development setup

The project is managed with [uv](https://github.com/astral-sh/uv):

```sh
$ uv sync                # install runtime (NumPy only) + dev dependencies
$ uv sync --group numba  # opt-in guvectorize vectorization (CPython only)
$ uv sync --group docs   # additionally install the docs tooling
```

CI runs three test jobs per pull request: a compiled CPython matrix (`uv sync --group numba`),
a no-Numba CPython job, and a PyPy job (both `uv sync` only), so every change is verified with
and without Numba.

### Ruff

[Ruff](https://docs.astral.sh/ruff/) is used for both linting and formatting. Its configuration lives in
`pyproject.toml` under `[tool.ruff]`.

```sh
$ uv run ruff check .            # lint
$ uv run ruff format .           # format
$ uv run ruff format --check .   # verify formatting
```

### Mypy

[Mypy](https://mypy.readthedocs.io/) is used for static type-checking in strict mode. Its configuration lives in
`pyproject.toml` under `[tool.mypy]` and checks the `src/` package.

```sh
$ uv run mypy           # type-check the package
$ uv run mypy-coverage --threshold 35  # report type-annotation coverage (enforced in CI)
```

### MkDocs

[MkDocs](https://www.mkdocs.org/) builds the documentation with the *Material* theme. Its configuration lives in
`mkdocs.yml`.

```sh
$ uv run mkdocs serve                # live preview at http://127.0.0.1:8000
$ uv run mkdocs build                # build the static site into site/
$ uv run mkdocs gh-deploy --force    # publish to GitHub Pages
```

### Tests

```sh
$ uv run pytest
```

## Citation

If you use this library in your research, please cite it as [@Van_Thieu_2024_Opfunu].

\bibliography

## Official channels

* [Official source code repository](https://github.com/ltsim/opfunu-core)
* [Official documentation](https://ltsim.github.io/opfunu-core/)
* [Download releases](https://pypi.org/project/opfunu-core/)
* [Issue tracker](https://github.com/ltsim/opfunu-core/issues)
