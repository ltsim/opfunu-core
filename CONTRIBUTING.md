# Contributing

We appreciate your interest in contributing to Opfunu! This guide details how to contribute in a way that is efficient 
for everyone.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Branching Workflow](#branching-workflow)
- [Getting Started](#getting-started)
- [How Can I Contribute?](#how-can-i-contribute)
- [Pull Requests](#pull-requests)
- [Issue Tracking](#issue-tracking)

## Code of Conduct

All contributors are expected to adhere to the project's [Code of Conduct](CODE_OF_CONDUCT.md). Please read the document before contributing. 

## Branching Workflow

This project uses a release-cadenced branching model. The long-lived branches are:

- **`master`** — the stable, default branch. It receives one merge per release cycle.
- **`release/v<year><tag><version>`** (e.g. `release/v2026a0`) — the current release branch,
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

Releases are tagged `<year><tag><version>`, for example `v2026a0` or `v2026b9`:

- **year** — four-digit release year (`2026`).
- **tag** — a lowercase letter track within the year (`a`, `b`, `c`, …).
- **version** — a single digit `0`–`9` within the tag.

The sequence advances `v2026a0 → v2026a1 → … → v2026a9 → v2026b0 → …`. The `v` prefix is used
for branch names and git tags; the Python package version drops it (`__version__ = "2026a0"`).

### Release cycle

1. Merge category-branch pull requests into `release/v2026aN` as work lands.
2. When the cycle is complete, make the versioning commits on the release branch: bump
   `__version__` in `src/opfunu/__init__.py`, update `CHANGELOG.md` (and `docs/changelog.md`),
   refresh `CITATION.cff`, and commit.
3. Open a pull request from `release/v2026aN` into `master`; merge it once CI passes.
4. Tag the merge commit `v<year><tag><version>` and publish the GitHub release — this triggers
   `publish.yml` to build and upload to PyPI.
5. Cut the next release branch (e.g. `release/v2026a1`) from `master` and repeat.

## Getting Started

1. Fork the project repository and clone your fork:

    ```
    git clone https://github.com/ltsim/opfunu-core.git
    ```

2. Create a new category branch for your changes, named after the work you are doing:

    ```
    git checkout -b feature/name-of-your-branch
    ```

3. Make your changes and commit them:

    ```
    git commit -m "Detailed commit message"
    ```

4. Push your changes to your fork:

    ```
    git push origin feature/name-of-your-branch
    ```

5. Create a pull request from your branch to the current release branch
   (for example `release/v2026a0`).

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
   from the current release branch, e.g. `release/v2026a0`.
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
