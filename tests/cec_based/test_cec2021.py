#!/usr/bin/env python
# Created by "Thieu" at 09:25, 13/07/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import opfunu


def test_F12021_results(assert_problem):
    assert_problem(opfunu.cec_based.F12021(ndim=10), 10, opfunu.cec_based.CecBenchmark)


def test_F22021_results(assert_problem):
    assert_problem(opfunu.cec_based.F22021(ndim=10), 10, opfunu.cec_based.CecBenchmark)


def test_F32021_results(assert_problem):
    assert_problem(opfunu.cec_based.F32021(ndim=10), 10, opfunu.cec_based.CecBenchmark)


def test_F42021_results(assert_problem):
    assert_problem(opfunu.cec_based.F42021(ndim=10), 10, opfunu.cec_based.CecBenchmark)


def test_F52021_results(assert_problem):
    assert_problem(opfunu.cec_based.F52021(ndim=10), 10, opfunu.cec_based.CecBenchmark)


def test_F62021_results(assert_problem):
    assert_problem(opfunu.cec_based.F62021(ndim=10), 10, opfunu.cec_based.CecBenchmark)


def test_F72021_results(assert_problem):
    assert_problem(opfunu.cec_based.F72021(ndim=10), 10, opfunu.cec_based.CecBenchmark)


def test_F82021_results(assert_problem):
    assert_problem(opfunu.cec_based.F82021(ndim=10), 10, opfunu.cec_based.CecBenchmark)


def test_F92021_results(assert_problem):
    assert_problem(opfunu.cec_based.F92021(ndim=10), 10, opfunu.cec_based.CecBenchmark)


def test_F102021_results(assert_problem):
    assert_problem(opfunu.cec_based.F102021(ndim=10), 10, opfunu.cec_based.CecBenchmark)


def test_all_optimal_results(assert_problem):
    ndim = 10
    known_failing = []
    all_functions = [
        x for x in opfunu.get_all_cec_based_functions() if x.__name__[-4:] == "2021" and x.__name__ not in known_failing
    ]
    for function in all_functions:
        problem = function(ndim=ndim)
        x = problem.x_global
        result = problem.evaluate(x)
        assert abs(result - problem.f_global) <= problem.epsilon, f"{function.__name__} Failed Optimal Test"
