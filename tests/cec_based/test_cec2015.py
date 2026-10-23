#!/usr/bin/env python
# Created by "Thieu" at 14:46, 07/07/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import opfunu


def test_F12015_results(assert_problem):
    assert_problem(opfunu.cec_based.F12015(ndim=30), 30, opfunu.cec_based.CecBenchmark)


def test_F22015_results(assert_problem):
    assert_problem(opfunu.cec_based.F22015(ndim=30), 30, opfunu.cec_based.CecBenchmark)


def test_F32015_results(assert_problem):
    assert_problem(opfunu.cec_based.F32015(ndim=30), 30, opfunu.cec_based.CecBenchmark)


def test_F42015_results(assert_problem):
    assert_problem(opfunu.cec_based.F42015(ndim=30), 30, opfunu.cec_based.CecBenchmark)


def test_F52015_results(assert_problem):
    assert_problem(opfunu.cec_based.F52015(ndim=30), 30, opfunu.cec_based.CecBenchmark)


def test_F62015_results(assert_problem):
    assert_problem(opfunu.cec_based.F62015(ndim=30), 30, opfunu.cec_based.CecBenchmark)


def test_F72015_results(assert_problem):
    assert_problem(opfunu.cec_based.F72015(ndim=30), 30, opfunu.cec_based.CecBenchmark)


def test_F82015_results(assert_problem):
    assert_problem(opfunu.cec_based.F82015(ndim=30), 30, opfunu.cec_based.CecBenchmark)


def test_F92015_results(assert_problem):
    assert_problem(opfunu.cec_based.F92015(ndim=30), 30, opfunu.cec_based.CecBenchmark)


def test_F102015_results(assert_problem):
    assert_problem(opfunu.cec_based.F102015(ndim=30), 30, opfunu.cec_based.CecBenchmark)


def test_F112015_results(assert_problem):
    assert_problem(opfunu.cec_based.F112015(ndim=30), 30, opfunu.cec_based.CecBenchmark)


def test_F122015_results(assert_problem):
    assert_problem(opfunu.cec_based.F122015(ndim=30), 30, opfunu.cec_based.CecBenchmark)


def test_F132015_results(assert_problem):
    assert_problem(opfunu.cec_based.F132015(ndim=30), 30, opfunu.cec_based.CecBenchmark)


def test_F142015_results(assert_problem):
    assert_problem(opfunu.cec_based.F142015(ndim=30), 30, opfunu.cec_based.CecBenchmark)


def test_F152015_results(assert_problem):
    assert_problem(opfunu.cec_based.F152015(ndim=30), 30, opfunu.cec_based.CecBenchmark)


def test_all_optimal_results(assert_problem):
    known_failing = []
    all_functions = [
        x for x in opfunu.get_all_cec_based_functions() if x.__name__[-4:] == "2015" and x.__name__ not in known_failing
    ]
    for function in all_functions:
        problem = function()
        x = problem.x_global
        result = problem.evaluate(x)
        assert abs(result - problem.f_global) <= problem.epsilon, f"{function.__name__} Failed Optimal Test"
