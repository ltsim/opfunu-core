#!/usr/bin/env python
# Created by "Thieu" at 20:21, 30/06/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
import pytest

from opfunu.benchmark import Benchmark
from opfunu.benchmark.func import FuncBenchmark
from opfunu.utils.numba_compat import HAS_NUMBA

needs_numba = pytest.mark.skipif(not HAS_NUMBA, reason="requires numba")


class _DummyProblem(FuncBenchmark):
    """Minimal concrete subclass using the declarative kernel + metadata style."""

    def __init__(self, ndim=None, bounds=None, **kwargs):
        def compute(x, out):
            out[0] = float(np.sum(x**2))

        super().__init__(
            compute=compute,
            ndim=ndim,
            bounds=bounds,
            default_bounds=np.array([[-15.0, 15.0] for _ in range(2)]),
            f_global=0.0,
            x_global=lambda n: np.zeros(n),
            dim_changeable=True,
            dim_default=2,
            **kwargs,
        )


class _InlineProblem(FuncBenchmark):
    """Compact declaration style: kernel and metadata are passed to ``super().__init__``."""

    def __init__(self, ndim=None, bounds=None, parallel=False, fastmath=True, dtype=np.float64):
        def compute(x, out):
            out[0] = float(np.sum(x**2))

        super().__init__(
            compute=compute,
            ndim=ndim,
            bounds=bounds,
            default_bounds=np.array([[-15.0, 15.0]] * 2),
            f_global=0.0,
            x_global=lambda n: np.zeros(n),
            dim_changeable=True,
            dim_default=2,
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
        )


def test_Benchmark_class():
    ndim = 10
    x = np.random.uniform(-15, 15, ndim)
    problem = _DummyProblem(ndim=ndim)

    assert isinstance(problem.lb, np.ndarray)
    assert len(problem.lb) == ndim
    assert isinstance(problem.bounds, np.ndarray)
    assert problem.bounds.shape[0] == ndim
    assert isinstance(problem.evaluate(x), np.float64)


def test_inline_compute_declaration_style():
    ndim = 5
    problem = _InlineProblem(ndim=ndim)
    x = np.random.uniform(-5, 5, ndim)
    # fastmath allows ~1ulp reassociation vs NumPy; compare with the documented tolerance.
    assert np.isclose(float(problem.evaluate(x)), float(np.sum(x**2)), rtol=1e-9, atol=1e-12)
    assert problem.dim_changeable is True
    assert problem.dim_default == 2
    assert len(problem.x_global) == ndim
    assert problem.x_global.tolist() == [0.0] * ndim


#: Every property defined on ``Benchmark`` must be reader-only: values are
#: supplied to ``__init__`` (or ``_bind_kernel``) at construction time.
READ_ONLY_PROPERTIES = [
    "parallel",
    "fastmath",
    "dtype",
    "verbose",
    "support_path",
    "paras",
    "numba_compiled",
    "f_global",
    "x_global",
    "dim_changeable",
    "dim_default",
    "_kernel",
    "_param_names",
    "_compute",
    "_compile_error",
    "_out",
]


def test_metadata_are_read_only_properties():
    problem = _DummyProblem(ndim=2)
    # Construction-time metadata keeps the values passed to the constructor.
    assert problem.f_global == 0.0
    assert np.array_equal(problem.x_global, np.zeros(2))
    assert problem.dim_default == 2
    assert problem.dim_changeable is True
    # Data-dependent metadata lives in protected storage; the rest is name-mangled.
    assert "_f_global" in vars(problem)
    assert "_x_global" in vars(problem)
    assert "_paras" in vars(problem)
    assert "_Benchmark__dim_default" in vars(problem)
    assert "_Benchmark__dim_changeable" in vars(problem)


@pytest.mark.parametrize("name", READ_ONLY_PROPERTIES)
def test_benchmark_properties_are_read_only(name):
    problem = _DummyProblem(ndim=2)
    assert isinstance(getattr(type(problem), name), property)
    with pytest.raises(AttributeError):
        setattr(problem, name, getattr(problem, name))


def test_kernel_state_is_encapsulated():
    problem = _DummyProblem(ndim=2)
    assert callable(problem.compute)
    assert problem.compute is problem._kernel
    assert "_Benchmark__kernel" in vars(problem)
    assert "_Benchmark__param_names" in vars(problem)
    assert "_Benchmark__compute" in vars(problem)
    with pytest.raises(AttributeError):
        problem.compute = problem.compute


def test_evaluate_is_inherited_not_reimplemented():
    """The base class provides ``evaluate``; subclasses only supply ``compute``."""
    assert "evaluate" in vars(Benchmark)
    assert "evaluate" not in vars(FuncBenchmark)
    assert "evaluate" not in vars(_DummyProblem)


def test_evaluate_casts_to_configured_dtype():
    ndim = 6
    x = np.linspace(-2.0, 2.0, ndim)
    f64 = _DummyProblem(ndim=ndim)
    f32 = _DummyProblem(ndim=ndim, dtype=np.float32)
    r64, r32 = f64.evaluate(x), f32.evaluate(x)
    assert isinstance(r64, np.float64)
    assert isinstance(r32, np.float32)
    assert np.isclose(float(r64), float(r32), rtol=1e-4, atol=1e-4)


def test_evaluate_batch_matches_elementwise():
    ndim = 4
    problem = _DummyProblem(ndim=ndim)
    X = np.random.uniform(-3, 3, (7, ndim))
    batch = problem._evaluate_batch(X)
    assert batch.shape == (7,)
    assert batch.dtype == problem.dtype
    for i, row in enumerate(X):
        assert batch[i] == problem.evaluate(row)


def test_evaluate_batch_matches_elementwise_without_numba():
    """Plain-Python fallback kernels are scalar writers; the batch path must loop."""
    ndim = 4
    problem = _DummyProblem(ndim=ndim)
    problem._bind_kernel(problem.compute, [], plain=True)
    assert problem.numba_compiled is False
    X = np.random.uniform(-3, 3, (7, ndim))
    batch = problem._evaluate_batch(X)
    assert batch.shape == (7,)
    assert batch.dtype == problem.dtype
    for i, row in enumerate(X):
        assert batch[i] == problem.evaluate(row)


def test_normalize_kernel_param_accepts_nested_list():
    problem = _DummyProblem(ndim=2)
    call_value, npdtype = problem._normalize_kernel_param([[1.0, 2.0], [3.0, 4.0]])
    assert call_value.ndim == 2
    assert call_value.shape == (2, 2)
    assert np.dtype(npdtype) == np.dtype(np.float64)


def test_normalize_kernel_param_unwraps_zero_dim_array():
    problem = _DummyProblem(ndim=2)
    call_value, _ = problem._normalize_kernel_param(np.array(5.0))
    assert float(call_value) == 5.0


def test_normalize_kernel_param_accepts_empty_list_and_bool_scalar():
    problem = _DummyProblem(ndim=2)
    call_value, npdtype = problem._normalize_kernel_param([])
    assert call_value.shape == (0,)
    assert np.dtype(npdtype) == np.dtype(np.float64)
    call_value, npdtype = problem._normalize_kernel_param(True)
    assert call_value is True
    assert np.dtype(npdtype) == np.dtype(bool)


def test_normalize_kernel_param_rejects_unsupported_types():
    problem = _DummyProblem(ndim=2)
    with pytest.raises(TypeError):
        problem._normalize_kernel_param(object())
    with pytest.raises(TypeError):
        problem._normalize_kernel_param(np.array([1j, 2j]))
    with pytest.raises(TypeError):
        problem._normalize_kernel_param(np.ones((2, 2, 2)))


@needs_numba
def test_build_compute_fallback_records_error():
    def bad_compute(x, out):
        out[0] = float(object())  # Numba cannot type this: forces a nopython failure

    problem = _DummyProblem(ndim=2)
    problem._bind_kernel(bad_compute, [])
    assert problem.numba_compiled is False
    assert problem._compile_error is not None
    assert problem._compute is bad_compute


def test_bind_kernel_bad_param_name_raises():
    problem = _DummyProblem(ndim=2)
    with pytest.raises(AttributeError):
        problem._bind_kernel(None, ["no_such_attr"])


def test_bind_kernel_unsupported_param_type_raises():
    problem = _DummyProblem(ndim=2)
    problem.bad_param = object()
    with pytest.raises(TypeError):
        problem._bind_kernel(None, ["bad_param"])


def test_bind_kernel_without_numba_uses_plain_python(monkeypatch):
    """With Numba absent every kernel binds plain Python and still evaluates."""
    monkeypatch.setattr("opfunu.utils.numba_compat.HAS_NUMBA", False)
    problem = _DummyProblem(ndim=2)
    assert problem.numba_compiled is False
    assert problem._compile_error is None
    x = np.array([1.0, 2.0])
    assert problem.evaluate(x) == np.float64(np.sum(x**2))


@needs_numba
def test_gufunc_cache_key_tracks_kernel_code():
    problem = _DummyProblem(ndim=3)
    assert problem.compute is not None
    assert any(key[0] is problem.compute.__code__ for key in Benchmark._gufunc_cache)


def test_abstract_func_benchmark_cannot_instantiate():
    with pytest.raises(TypeError):
        FuncBenchmark()  # abstract: compute() is mandatory since the Numba migration
