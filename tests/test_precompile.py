# Tests for configurable Numba precompilation: ``precompile`` per instance and
# ``Benchmark.default_precompile`` as the package-wide policy.

import numpy as np
import pytest

from opfunu.benchmark import Benchmark
from opfunu.benchmark.func import FuncBenchmark
from opfunu.name_based import Ackley01
from opfunu.utils.numba_compat import HAS_NUMBA

needs_numba = pytest.mark.skipif(not HAS_NUMBA, reason="requires numba")


class _Kernel(FuncBenchmark):
    """Minimal problem with a self-contained kernel, reused across these tests."""

    def __init__(self, ndim=None, bounds=None, **kwargs):
        def compute(x, out):
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


def test_precompile_is_read_only_property():
    problem = _Kernel(ndim=2)
    assert isinstance(getattr(type(problem), "precompile"), property)
    with pytest.raises(AttributeError):
        problem.precompile = False


def test_precompile_resolves_to_a_bool():
    assert _Kernel(ndim=2).precompile is Benchmark.default_precompile
    assert _Kernel(ndim=2, precompile=True).precompile is True
    assert _Kernel(ndim=2, precompile=False).precompile is False


@needs_numba
def test_default_precompiles_when_numba_is_installed():
    problem = _Kernel(ndim=3)
    assert problem.precompile is True
    assert problem.numba_compiled is True
    assert problem._compile_error is None


def test_precompile_false_binds_plain_kernel():
    """``precompile=False`` declines compilation even though Numba is available."""
    before = len(Benchmark._gufunc_cache)
    problem = _Kernel(ndim=3, precompile=False)
    assert problem.precompile is False
    assert problem.numba_compiled is False
    assert problem._compile_error is None
    assert len(Benchmark._gufunc_cache) == before
    x = np.array([1.0, 2.0, 3.0])
    assert problem.evaluate(x) == pytest.approx(14.0)


def test_precompile_false_batch_and_evaluate_match_compiled():
    plain = _Kernel(ndim=4, precompile=False)
    X = np.random.default_rng(7).uniform(-3, 3, (6, 4))
    batch = plain._evaluate_batch(X)
    assert batch.shape == (6,)
    assert batch.dtype == plain.dtype
    for i, row in enumerate(X):
        assert batch[i] == plain.evaluate(row)
    if not HAS_NUMBA:
        pytest.skip("compiled counterpart requires numba")
    compiled = _Kernel(ndim=4, precompile=True)
    assert np.allclose(batch, compiled._evaluate_batch(X), rtol=1e-9, atol=1e-12)


@needs_numba
def test_precompile_true_overrides_disabled_default(monkeypatch):
    monkeypatch.setattr(Benchmark, "default_precompile", False)
    problem = _Kernel(ndim=2, precompile=True)
    assert problem.precompile is True
    assert problem.numba_compiled is True


@needs_numba
def test_default_precompile_false_disables_compilation(monkeypatch):
    before = len(Benchmark._gufunc_cache)
    monkeypatch.setattr(Benchmark, "default_precompile", False)
    problem = _Kernel(ndim=2)
    assert problem.precompile is False
    assert problem.numba_compiled is False
    assert problem._compile_error is None
    assert len(Benchmark._gufunc_cache) == before


def test_default_precompile_is_inherited_by_subclasses(monkeypatch):
    monkeypatch.setattr(Benchmark, "default_precompile", False)
    assert Ackley01(ndim=2).precompile is False


def test_plain_binding_wins_over_precompile_true():
    """Kernels that cannot compile stay plain regardless of the precompile flag."""
    problem = Ackley01(ndim=2, precompile=True)
    problem._bind_kernel(problem.compute, [], plain=True)
    assert problem.precompile is True
    assert problem.numba_compiled is False


def test_precompile_true_without_numba_is_plain(monkeypatch):
    """Without Numba the flag is inert: plain kernel, no error recorded."""
    monkeypatch.setattr("opfunu.utils.numba_compat.HAS_NUMBA", False)
    problem = _Kernel(ndim=2, precompile=True)
    assert problem.numba_compiled is False
    assert problem._compile_error is None
    assert problem.evaluate(np.array([2.0, 3.0])) == np.float64(13.0)


def test_precompile_false_still_validates_kernel_params():
    """Mistyped kernel parameters fail fast on the plain path, as in the compiled one."""
    problem = Ackley01(ndim=2, precompile=False)
    problem.bad_param = object()
    with pytest.raises(TypeError):
        problem._bind_kernel(problem.compute, ["bad_param"])
