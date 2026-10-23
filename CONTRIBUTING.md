# Contributing

We appreciate your interest in contributing to Opfunu! This guide details how to contribute in a way that is efficient 
for everyone.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Branching Workflow](#branching-workflow)
- [Getting Started](#getting-started)
- [Git Hooks](#git-hooks)
- [How Can I Contribute?](#how-can-i-contribute)
- [Pull Requests](#pull-requests)
- [Issue Tracking](#issue-tracking)

## Code of Conduct

All contributors are expected to adhere to the project's [Code of Conduct](CODE_OF_CONDUCT.md). Please read the document before contributing. 

## Branching Workflow

This project uses a release-cadenced branching model. The long-lived branches are:

- **`master`** — the stable, default branch. It receives one merge per release cycle and carries
  the published version.
- **`release/v<year><edition><version>`** (e.g. `release/v2026b0`) — the current release branch,
  **reserved exclusively for versioning**: version bumps, changelog entries, citation updates,
  and the release tag. No feature work is committed directly to it.

Day-to-day work happens on short-lived **category branches** forked from the current release
branch. Pick the category that matches the change and open a pull request **into the release
branch**:

| Category         | Prefix                 | Use for                                          |
|------------------|------------------------|--------------------------------------------------|
| Development      | `development/<topic>`  | Work in progress, exploratory or multi-part work |
| Feature          | `feature/<name>`       | New functionality                                |
| Fixing           | `fixing/<name>`        | Bug fixes                                        |
| Refactoring      | `refactoring/<name>`   | Structural / cleanup changes with no behavior change |

### Versioning

Releases are tagged `v<year><edition><version>`, for example `v2026a0` or `v2026b9`:

- **year** — four-digit release year (`2026`).
- **edition** — a lowercase letter track within the year (`a`, `b`, `c`, …).
- **version** — a single digit `0`–`9` within the edition.

The sequence advances `v2026a0 → v2026a1 → … → v2026a9 → v2026b0 → …`.

The published package version is the cycle name itself, `v` prefix included:

```
__version__ = "v2026b0"
```

so branch, git tag, GitHub release and PyPI release share one spelling. Be aware of how PEP 440
reads that string:

- PyPI normalizes away the `v` prefix — `v2026b0` is published and resolved as `2026b0`.
- The edition letter is interpreted as a pre-release marker (`a` = alpha, `b` = beta, `c`/`rc` =
  release candidate), so installers do not pick it up by default: users pin it explicitly
  (`pip install "opfunu-core==v2026b0"`) or pass `--pre`.
- Letters beyond `c` are not valid PEP 440 at all (`v2026d0` cannot be published), so the letter
  track stops at edition `c`; continuing past it requires a different numeric form.

### Release cycle

1. Merge category-branch pull requests into `release/v<year><edition><version>` as work lands.
2. When the cycle opens, bump `__version__` in `src/opfunu/__init__.py` on the release branch and
   start the `CHANGELOG.md` / `docs/changelog.md` section for the new version; when the cycle is
   complete, refresh `CITATION.cff` (`version`, `date-released`) and finalize the changelog.
3. Open a pull request from the release branch into `master`; merge it once CI passes.
4. Tag the merge commit `v<year><edition><version>` and publish the GitHub release — this triggers
   `publish.yml` to build and upload to PyPI under the same name (`v2026b0`, resolved as `2026b0`).
5. Cut the next release branch from `master` and repeat.

## Getting Started

1. Fork the project repository and clone your fork:

    ```
    git clone https://github.com/ltsim/opfunu-core.git
    ```

2. Install the development tooling — Python dependencies with [uv](https://github.com/astral-sh/uv)
   and the git hooks with npm (Node >= 18, one-time setup):

    ```
    uv sync --group numba
    npm install
    ```

3. Create a new category branch for your changes, named after the work you are doing:

    ```
    git checkout -b feature/name-of-your-branch
    ```

4. Make your changes and commit them — the pre-commit hook checks them first (see
   [Git Hooks](#git-hooks)):

    ```
    git commit -m "Detailed commit message"
    ```

5. Push your changes to your fork — the pre-push hook runs the same checks again:

    ```
    git push origin feature/name-of-your-branch
    ```

6. Create a pull request from your branch to the current release branch
   (for example `release/v2026b0`).

## Git Hooks

[Husky](https://typicode.github.io/husky/) runs the same checks as CI before every commit and
every push. Both hooks source the shared script `.husky/checks.sh`:

1. `ruff check . --fix` — lint, auto-fixing safe issues (import sorting, unused imports);
   the files ruff rewrote are staged so the fix lands in the same commit. Files ruff did not
   touch are never staged, but if you hold unrelated unstaged edits in a file ruff *did*
   rewrite, stage granularly (`git add -p`) before committing
2. `ruff check .` — verify nothing remains
3. `mypy` — strict type check over `src/`
4. `pytest` — the full test suite

Notes:

- Hooks run through `uv run`, so they resolve against the project environment; if the environment
  is stale, `uv run` refreshes it automatically.
- Skip a run once with `HUSKY=0 git commit ...` / `HUSKY=0 git push ...`, or disable the hooks
  entirely with `git config --replace-all core.hooksPath ''` (re-enable with `npm install`).
- Committing from an IDE (a PATH without `uv` or `node`): add them to
  `~/.config/husky/init.sh`, which Husky sources before every hook, e.g.
  `export PATH="$HOME/.local/bin:$PATH"`.
- CI remains the gate for merging: the hooks only cover what you run locally.

## How Can I Contribute?

Here are some ways to contribute:

- Improve documentation
- Fix bugs or add new features
- Write tutorials or blog posts
- Review code submissions
- Test the application and report issues

However, before contributing, make sure that the unit tests pass and that new functionality is covered by unit tests. 
The unit tests can be run using `pytest`. Change working directory to opfunu and then use:

```python
# Test CEC-based functions
python -m pytest tests/cec_based

# Test Name-based functions
python -m pytest tests/name_based
```

Or you can test all files by:

```python
python -m pytest
```


## Pull Requests

[Pull requests](https://github.com/ltsim/opfunu-core/pulls) are the best way to propose changes to the codebase. We 
actively welcome your pull requests:

1. Fork the repo and create your category branch (see [Branching Workflow](#branching-workflow))
   from the current release branch, e.g. `release/v2026b0`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Issue that pull request!

## Issue Tracking

We use [GitHub issues](https://github.com/ltsim/opfunu-core/issues) to track public bugs and requests. Please ensure 
your description is clear and has sufficient instructions to be able to reproduce the issue.

## Any questions?

Don't hesitate to contact us if you have any questions. Contact [@ltsim](mailto:tsim@cucei.udg.mx)
or ask your question on issues.

Thank you for your contributions!
