#!/usr/bin/env python
# Created by "Thieu" at 21:33, 12/07/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import opfunu


def test_F12020_results(assert_problem):
    assert_problem(opfunu.cec_based.F12020(ndim=30), 30, opfunu.cec_based.CecBenchmark)


def test_F22020_results(assert_problem):
    assert_problem(opfunu.cec_based.F22020(ndim=30), 30, opfunu.cec_based.CecBenchmark)


def test_F32020_results(assert_problem):
    assert_problem(opfunu.cec_based.F32020(ndim=30), 30, opfunu.cec_based.CecBenchmark)


def test_F42020_results(assert_problem):
    assert_problem(opfunu.cec_based.F42020(ndim=30), 30, opfunu.cec_based.CecBenchmark)


def test_F52020_results(assert_problem):
    assert_problem(opfunu.cec_based.F52020(ndim=30), 30, opfunu.cec_based.CecBenchmark)


def test_F62020_results(assert_problem):
    assert_problem(opfunu.cec_based.F62020(ndim=30), 30, opfunu.cec_based.CecBenchmark)


def test_F72020_results(assert_problem):
    assert_problem(opfunu.cec_based.F72020(ndim=30), 30, opfunu.cec_based.CecBenchmark)


def test_F82020_results(assert_problem):
    assert_problem(opfunu.cec_based.F82020(ndim=30), 30, opfunu.cec_based.CecBenchmark)


def test_F92020_results(assert_problem):
    assert_problem(opfunu.cec_based.F92020(ndim=30), 30, opfunu.cec_based.CecBenchmark)


def test_F102020_results(assert_problem):
    assert_problem(opfunu.cec_based.F102020(ndim=30), 30, opfunu.cec_based.CecBenchmark)


def test_all_optimal_results(assert_problem):
    ndim = 30
    known_failing = []
    all_functions = [
        x for x in opfunu.get_all_cec_based_functions() if x.__name__[-4:] == "2020" and x.__name__ not in known_failing
    ]
    for function in all_functions:
        problem = function(ndim=ndim)
        x = problem.x_global
        result = problem.evaluate(x)
        assert abs(result - problem.f_global) <= problem.epsilon, f"{function.__name__} Failed Optimal Test"
