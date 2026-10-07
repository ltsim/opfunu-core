#!/usr/bin/env python
# Created by "Thieu" at 10:00, 13/07/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import opfunu


def test_F12022_results(assert_problem):
    assert_problem(opfunu.cec_based.F12022(ndim=10), 10, opfunu.cec_based.CecBenchmark)


def test_F22022_results(assert_problem):
    assert_problem(opfunu.cec_based.F22022(ndim=10), 10, opfunu.cec_based.CecBenchmark)


def test_F32022_results(assert_problem):
    assert_problem(opfunu.cec_based.F32022(ndim=10), 10, opfunu.cec_based.CecBenchmark)


def test_F42022_results(assert_problem):
    assert_problem(opfunu.cec_based.F42022(ndim=10), 10, opfunu.cec_based.CecBenchmark)


def test_F52022_results(assert_problem):
    assert_problem(opfunu.cec_based.F52022(ndim=10), 10, opfunu.cec_based.CecBenchmark)


def test_F62022_results(assert_problem):
    assert_problem(opfunu.cec_based.F62022(ndim=10), 10, opfunu.cec_based.CecBenchmark)


def test_F72022_results(assert_problem):
    assert_problem(opfunu.cec_based.F72022(ndim=10), 10, opfunu.cec_based.CecBenchmark)


def test_F82022_results(assert_problem):
    assert_problem(opfunu.cec_based.F82022(ndim=10), 10, opfunu.cec_based.CecBenchmark)


def test_F92022_results(assert_problem):
    assert_problem(opfunu.cec_based.F92022(ndim=10), 10, opfunu.cec_based.CecBenchmark)


def test_F102022_results(assert_problem):
    assert_problem(opfunu.cec_based.F102022(ndim=10), 10, opfunu.cec_based.CecBenchmark)


def test_F112022_results(assert_problem):
    assert_problem(opfunu.cec_based.F112022(ndim=10), 10, opfunu.cec_based.CecBenchmark)


def test_F122022_results(assert_problem):
    assert_problem(opfunu.cec_based.F122022(ndim=10), 10, opfunu.cec_based.CecBenchmark)


def test_all_optimal_results(assert_problem):
    ndim = 10
    known_failing = []
    all_functions = [
        x for x in opfunu.get_all_cec_based_functions() if x.__name__[-4:] == "2022" and x.__name__ not in known_failing
    ]
    for function in all_functions:
        problem = function(ndim=ndim)
        x = problem.x_global
        result = problem.evaluate(x)
        assert abs(result - problem.f_global) <= problem.epsilon, f"{function.__name__} Failed Optimal Test"
