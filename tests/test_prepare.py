# Tests for configurable Numba preparation: the ``prepare`` constructor flag
# chooses eager compilation at construction, ``Benchmark.default_prepare`` is
# the package-wide policy, and ``Benchmark.prepare()`` compiles a deferred
# instance on demand.

import numpy as np
import pytest

from opfunu.benchmark import Benchmark
from opfunu.benchmark.func import FuncBenchmark
from opfunu.cec_based.cec2014 import F12014
from opfunu.name_based import Ackley01
from opfunu.utils.numba_compat import HAS_NUMBA

needs_numba = pytest.mark.skipif(not HAS_NUMBA, reason="requires numba")


class _Kernel(FuncBenchmark):
    """Minimal problem with a self-contained kernel, reused across these tests."""

    def __init__(self, ndim=None, bounds=None, **kwargs):
        def compute(x, shift, out):
            x = x - shift
            out[0] = float(np.sum(x * x))

        super().__init__(
            compute=compute,
            ndim=ndim,
            bounds=bounds,
            default_bounds=np.array([[-15.0, 15.0]] * 2),
            f_global=0.0,
            x_global=lambda n: np.zeros(n),
            dim_changeable=True,
            dim_default=2,
            **kwargs,
        )


def test_default_prepare_is_enabled():
    assert Benchmark.default_prepare is True


def test_prepare_is_a_method_not_a_property():
    problem = _Kernel(ndim=2)
    assert callable(problem.prepare)
    with pytest.raises(AttributeError):
        problem.precompile


@needs_numba
def test_default_prepares_at_construction():
    problem = _Kernel(ndim=3)
    assert problem.numba_compiled is True
    assert problem._compile_error is None


@needs_numba
def test_prepare_true_compiles_at_construction():
    problem = _Kernel(ndim=3, prepare=True)
    assert problem.numba_compiled is True


def test_prepare_false_binds_plain_kernel():
    """``prepare=False`` declines compilation at construction even with Numba available."""
    before = len(Benchmark._gufunc_cache)
    problem = _Kernel(ndim=3, prepare=False)
    assert problem.numba_compiled is False
    assert problem._compile_error is None
    assert len(Benchmark._gufunc_cache) == before
    x = np.array([1.0, 2.0, 3.0])
    assert problem.evaluate(x) == pytest.approx(14.0)


@needs_numba
def test_prepare_method_compiles_deferred_instance():
    problem = _Kernel(ndim=3, prepare=False)
    plain_value = float(problem.evaluate(np.array([1.0, 2.0, 3.0])))
    assert problem.prepare() is True
    assert problem.numba_compiled is True
    assert problem._compile_error is None
    assert float(problem.evaluate(np.array([1.0, 2.0, 3.0]))) == pytest.approx(plain_value)


@needs_numba
def test_prepare_method_is_idempotent():
    problem = _Kernel(ndim=2, prepare=False)
    assert problem.prepare() is True
    compiled = problem._compute
    assert problem.prepare() is True
    assert problem._compute is compiled


@needs_numba
def test_prepare_method_returns_true_for_eager_instance():
    problem = _Kernel(ndim=2, prepare=True)
    compiled = problem._compute
    assert problem.prepare() is True
    assert problem._compute is compiled


def test_prepare_false_batch_and_evaluate_match_compiled():
    plain = _Kernel(ndim=4, prepare=False)
    X = np.random.default_rng(7).uniform(-3, 3, (6, 4))
    batch = plain._evaluate_batch(X)
    assert batch.shape == (6,)
    assert batch.dtype == plain.dtype
    for i, row in enumerate(X):
        assert batch[i] == plain.evaluate(row)
    if not HAS_NUMBA:
        pytest.skip("compiled counterpart requires numba")
    prepared = _Kernel(ndim=4, prepare=False)
    assert prepared.prepare() is True
    assert np.allclose(batch, prepared._evaluate_batch(X), rtol=1e-9, atol=1e-12)


@needs_numba
def test_prepare_true_overrides_disabled_default(monkeypatch):
    monkeypatch.setattr(Benchmark, "default_prepare", False)
    problem = _Kernel(ndim=2, prepare=True)
    assert problem.numba_compiled is True


@needs_numba
def test_default_prepare_false_defers_until_prepare(monkeypatch):
    before = len(Benchmark._gufunc_cache)
    monkeypatch.setattr(Benchmark, "default_prepare", False)
    problem = _Kernel(ndim=2)
    assert problem.numba_compiled is False
    assert problem._compile_error is None
    assert len(Benchmark._gufunc_cache) == before
    assert problem.prepare() is True
    assert problem.numba_compiled is True


def test_default_prepare_is_inherited_by_subclasses(monkeypatch):
    monkeypatch.setattr(Benchmark, "default_prepare", False)
    assert Ackley01(ndim=2).numba_compiled is False


@needs_numba
def test_plain_binding_wins_over_prepare():
    """Kernels declared plain stay plain whatever ``prepare()`` says."""
    problem = Ackley01(ndim=2, prepare=True)
    problem._bind_kernel(problem.compute, [], plain=True)
    assert problem.numba_compiled is False
    assert problem.prepare() is False
    assert problem.numba_compiled is False
    assert problem._compile_error is None


@needs_numba
def test_prepare_method_compiles_late_bound_problem():
    """CEC problems bind their kernel after ``super().__init__``; ``prepare()`` still works."""
    problem = F12014(ndim=30, prepare=False)
    assert problem.numba_compiled is False
    value = float(problem.evaluate(problem.create_solution()))
    assert problem.prepare() is True
    assert problem.numba_compiled is True
    assert float(problem.evaluate(problem.x_global)) == pytest.approx(problem.f_global, abs=problem.epsilon)
    assert np.isfinite(value)


def test_prepare_true_without_numba_is_plain(monkeypatch):
    """Without Numba the flag is inert: plain kernel, no error recorded."""
    monkeypatch.setattr("opfunu.utils.numba_compat.HAS_NUMBA", False)
    problem = _Kernel(ndim=2, prepare=True)
    assert problem.numba_compiled is False
    assert problem._compile_error is None
    assert problem.evaluate(np.array([2.0, 3.0])) == np.float64(13.0)


def test_prepare_method_without_numba_is_noop(monkeypatch):
    """On PyPy/minimal installs ``prepare()`` declines quietly."""
    monkeypatch.setattr("opfunu.utils.numba_compat.HAS_NUMBA", False)
    problem = _Kernel(ndim=2, prepare=False)
    assert problem.prepare() is False
    assert problem.numba_compiled is False
    assert problem._compile_error is None
    assert problem.evaluate(np.array([2.0, 3.0])) == np.float64(13.0)


def test_prepare_false_still_validates_kernel_params():
    """Mistyped kernel parameters fail fast on the deferred path, as in the compiled one."""
    problem = Ackley01(ndim=2, prepare=False)
    problem.bad_param = object()
    with pytest.raises(TypeError):
        problem._bind_kernel(problem.compute, ["bad_param"])
