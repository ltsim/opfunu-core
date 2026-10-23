#!/usr/bin/env python
# Created by "Thieu" at 16:37, 12/07/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import opfunu


def test_F12019_results(assert_problem):
    assert_problem(opfunu.cec_based.F12019(ndim=9), 9, opfunu.cec_based.CecBenchmark)


def test_F22019_results(assert_problem):
    assert_problem(opfunu.cec_based.F22019(ndim=16), 16, opfunu.cec_based.CecBenchmark)


def test_F32019_results(assert_problem):
    assert_problem(opfunu.cec_based.F32019(ndim=18), 18, opfunu.cec_based.CecBenchmark)


def test_F42019_results(assert_problem):
    assert_problem(opfunu.cec_based.F42019(ndim=10), 10, opfunu.cec_based.CecBenchmark)


def test_F52019_results(assert_problem):
    assert_problem(opfunu.cec_based.F52019(ndim=10), 10, opfunu.cec_based.CecBenchmark)


def test_F62019_results(assert_problem):
    assert_problem(opfunu.cec_based.F62019(ndim=10), 10, opfunu.cec_based.CecBenchmark)


def test_F72019_results(assert_problem):
    assert_problem(opfunu.cec_based.F72019(ndim=10), 10, opfunu.cec_based.CecBenchmark)


def test_F82019_results(assert_problem):
    assert_problem(opfunu.cec_based.F82019(ndim=10), 10, opfunu.cec_based.CecBenchmark)


def test_F92019_results(assert_problem):
    assert_problem(opfunu.cec_based.F92019(ndim=10), 10, opfunu.cec_based.CecBenchmark)


def test_F102019_results(assert_problem):
    assert_problem(opfunu.cec_based.F102019(ndim=10), 10, opfunu.cec_based.CecBenchmark)


def test_all_optimal_results(assert_problem):
    known_failing = []
    all_functions = [
        x for x in opfunu.get_all_cec_based_functions() if x.__name__[-4:] == "2019" and x.__name__ not in known_failing
    ]
    for function in all_functions:
        problem = function(10)
        x = problem.x_global
        result = problem.evaluate(x)
        assert abs(result - problem.f_global) <= problem.epsilon, f"{function.__name__} Failed Optimal Test"
