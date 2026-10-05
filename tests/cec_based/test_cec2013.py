#!/usr/bin/env python
# Created by "Thieu" at 19:57, 02/07/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import opfunu


def test_F12013_results(assert_problem):
    assert_problem(opfunu.cec_based.F12013(ndim=50), 50, opfunu.cec_based.CecBenchmark)


def test_F22013_results(assert_problem):
    assert_problem(opfunu.cec_based.F22013(ndim=50), 50, opfunu.cec_based.CecBenchmark)


def test_F32013_results(assert_problem):
    assert_problem(opfunu.cec_based.F32013(ndim=50), 50, opfunu.cec_based.CecBenchmark)


def test_F42013_results(assert_problem):
    assert_problem(opfunu.cec_based.F42013(ndim=50), 50, opfunu.cec_based.CecBenchmark)


def test_F52013_results(assert_problem):
    assert_problem(opfunu.cec_based.F52013(ndim=50), 50, opfunu.cec_based.CecBenchmark)


def test_F62013_results(assert_problem):
    assert_problem(opfunu.cec_based.F62013(ndim=50), 50, opfunu.cec_based.CecBenchmark)


def test_F72013_results(assert_problem):
    assert_problem(opfunu.cec_based.F72013(ndim=50), 50, opfunu.cec_based.CecBenchmark)


def test_F82013_results(assert_problem):
    assert_problem(opfunu.cec_based.F82013(ndim=50), 50, opfunu.cec_based.CecBenchmark)


def test_F92013_results(assert_problem):
    assert_problem(opfunu.cec_based.F92013(ndim=50), 50, opfunu.cec_based.CecBenchmark)


def test_F102013_results(assert_problem):
    assert_problem(opfunu.cec_based.F102013(ndim=50), 50, opfunu.cec_based.CecBenchmark)


def test_F112013_results(assert_problem):
    assert_problem(opfunu.cec_based.F112013(ndim=50), 50, opfunu.cec_based.CecBenchmark)


def test_F122013_results(assert_problem):
    assert_problem(opfunu.cec_based.F122013(ndim=50), 50, opfunu.cec_based.CecBenchmark)


def test_F132013_results(assert_problem):
    assert_problem(opfunu.cec_based.F132013(ndim=50), 50, opfunu.cec_based.CecBenchmark)


def test_F142013_results(assert_problem):
    assert_problem(opfunu.cec_based.F142013(ndim=50), 50, opfunu.cec_based.CecBenchmark)


def test_F152013_results(assert_problem):
    assert_problem(opfunu.cec_based.F152013(ndim=50), 50, opfunu.cec_based.CecBenchmark)


def test_F162013_results(assert_problem):
    assert_problem(opfunu.cec_based.F162013(ndim=50), 50, opfunu.cec_based.CecBenchmark)


def test_F172013_results(assert_problem):
    assert_problem(opfunu.cec_based.F172013(ndim=50), 50, opfunu.cec_based.CecBenchmark)


def test_F182013_results(assert_problem):
    assert_problem(opfunu.cec_based.F182013(ndim=50), 50, opfunu.cec_based.CecBenchmark)


def test_F192013_results(assert_problem):
    assert_problem(opfunu.cec_based.F192013(ndim=50), 50, opfunu.cec_based.CecBenchmark)


def test_F202013_results(assert_problem):
    assert_problem(opfunu.cec_based.F202013(ndim=50), 50, opfunu.cec_based.CecBenchmark)


def test_F212013_results(assert_problem):
    assert_problem(opfunu.cec_based.F212013(ndim=50), 50, opfunu.cec_based.CecBenchmark)


def test_F222013_results(assert_problem):
    assert_problem(opfunu.cec_based.F222013(ndim=50), 50, opfunu.cec_based.CecBenchmark)


def test_F232013_results(assert_problem):
    assert_problem(opfunu.cec_based.F232013(ndim=50), 50, opfunu.cec_based.CecBenchmark)


def test_F242013_results(assert_problem):
    assert_problem(opfunu.cec_based.F242013(ndim=50), 50, opfunu.cec_based.CecBenchmark)


def test_F252013_results(assert_problem):
    assert_problem(opfunu.cec_based.F252013(ndim=50), 50, opfunu.cec_based.CecBenchmark)


def test_F262013_results(assert_problem):
    assert_problem(opfunu.cec_based.F262013(ndim=50), 50, opfunu.cec_based.CecBenchmark)


def test_F272013_results(assert_problem):
    assert_problem(opfunu.cec_based.F272013(ndim=50), 50, opfunu.cec_based.CecBenchmark)


def test_F282013_results(assert_problem):
    assert_problem(opfunu.cec_based.F282013(ndim=50), 50, opfunu.cec_based.CecBenchmark)


def test_composite_functions_use_plain_compute_without_failed_compile():
    """Composite F21-F24 pass sub-instances as params: bind the plain kernel directly."""
    for cls in [
        opfunu.cec_based.F212013,
        opfunu.cec_based.F222013,
        opfunu.cec_based.F232013,
        opfunu.cec_based.F242013,
    ]:
        problem = cls(ndim=10)
        assert problem.numba_compiled is False
        assert problem._compile_error is None


def test_all_optimal_results(assert_problem):
    ndim = 30
    known_failing = []
    all_functions = [
        x for x in opfunu.get_all_cec_based_functions() if x.__name__[-4:] == "2013" and x.__name__ not in known_failing
    ]
    for function in all_functions:
        problem = function(ndim=ndim)
        x = problem.x_global
        result = problem.evaluate(x)
        assert abs(result - problem.f_global) <= problem.epsilon, f"{function.__name__} Failed Optimal Test"
