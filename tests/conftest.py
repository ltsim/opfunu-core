import numpy as np
import pytest


@pytest.fixture
def assert_problem():
    """Return the standard set of sanity checks for a benchmark function object."""

    def _assert_problem(problem, ndim, base):
        x = np.ones(ndim)
        result = problem.evaluate(x)
        assert isinstance(result, np.float64)
        assert isinstance(problem, base)
        assert isinstance(problem.lb, np.ndarray)
        assert len(problem.lb) == ndim
        assert problem.bounds.shape[0] == ndim
        assert len(problem.x_global) == ndim

    return _assert_problem
