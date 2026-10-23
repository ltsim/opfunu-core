#!/usr/bin/env python
# Created by "Thieu" at 17:54, 08/07/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import opfunu


def test_F12017_results(assert_problem):
    assert_problem(opfunu.cec_based.F12017(ndim=30), 30, opfunu.cec_based.CecBenchmark)


def test_F22017_results(assert_problem):
    assert_problem(opfunu.cec_based.F22017(ndim=30), 30, opfunu.cec_based.CecBenchmark)


def test_F32017_results(assert_problem):
    assert_problem(opfunu.cec_based.F32017(ndim=30), 30, opfunu.cec_based.CecBenchmark)


def test_F42017_results(assert_problem):
    assert_problem(opfunu.cec_based.F42017(ndim=30), 30, opfunu.cec_based.CecBenchmark)


def test_F52017_results(assert_problem):
    assert_problem(opfunu.cec_based.F52017(ndim=30), 30, opfunu.cec_based.CecBenchmark)


def test_F62017_results(assert_problem):
    assert_problem(opfunu.cec_based.F62017(ndim=30), 30, opfunu.cec_based.CecBenchmark)


def test_F72017_results(assert_problem):
    assert_problem(opfunu.cec_based.F72017(ndim=30), 30, opfunu.cec_based.CecBenchmark)


def test_F82017_results(assert_problem):
    assert_problem(opfunu.cec_based.F82017(ndim=30), 30, opfunu.cec_based.CecBenchmark)


def test_F92017_results(assert_problem):
    assert_problem(opfunu.cec_based.F92017(ndim=30), 30, opfunu.cec_based.CecBenchmark)


def test_F102017_results(assert_problem):
    assert_problem(opfunu.cec_based.F102017(ndim=30), 30, opfunu.cec_based.CecBenchmark)


def test_F112017_results(assert_problem):
    assert_problem(opfunu.cec_based.F112017(ndim=30), 30, opfunu.cec_based.CecBenchmark)


def test_F122017_results(assert_problem):
    assert_problem(opfunu.cec_based.F122017(ndim=30), 30, opfunu.cec_based.CecBenchmark)


def test_F132017_results(assert_problem):
    assert_problem(opfunu.cec_based.F132017(ndim=30), 30, opfunu.cec_based.CecBenchmark)


def test_F142017_results(assert_problem):
    assert_problem(opfunu.cec_based.F142017(ndim=30), 30, opfunu.cec_based.CecBenchmark)


def test_F152017_results(assert_problem):
    assert_problem(opfunu.cec_based.F152017(ndim=30), 30, opfunu.cec_based.CecBenchmark)


def test_F162017_results(assert_problem):
    assert_problem(opfunu.cec_based.F162017(ndim=30), 30, opfunu.cec_based.CecBenchmark)


def test_F172017_results(assert_problem):
    assert_problem(opfunu.cec_based.F172017(ndim=30), 30, opfunu.cec_based.CecBenchmark)


def test_F182017_results(assert_problem):
    assert_problem(opfunu.cec_based.F182017(ndim=30), 30, opfunu.cec_based.CecBenchmark)


def test_F192017_results(assert_problem):
    assert_problem(opfunu.cec_based.F192017(ndim=30), 30, opfunu.cec_based.CecBenchmark)


def test_F202017_results(assert_problem):
    assert_problem(opfunu.cec_based.F202017(ndim=30), 30, opfunu.cec_based.CecBenchmark)


def test_F212017_results(assert_problem):
    assert_problem(opfunu.cec_based.F212017(ndim=30), 30, opfunu.cec_based.CecBenchmark)


def test_F222017_results(assert_problem):
    assert_problem(opfunu.cec_based.F222017(ndim=30), 30, opfunu.cec_based.CecBenchmark)


def test_F232017_results(assert_problem):
    assert_problem(opfunu.cec_based.F232017(ndim=30), 30, opfunu.cec_based.CecBenchmark)


def test_F242017_results(assert_problem):
    assert_problem(opfunu.cec_based.F242017(ndim=30), 30, opfunu.cec_based.CecBenchmark)


def test_F252017_results(assert_problem):
    assert_problem(opfunu.cec_based.F252017(ndim=30), 30, opfunu.cec_based.CecBenchmark)


def test_F262017_results(assert_problem):
    assert_problem(opfunu.cec_based.F262017(ndim=30), 30, opfunu.cec_based.CecBenchmark)


def test_F272017_results(assert_problem):
    assert_problem(opfunu.cec_based.F272017(ndim=30), 30, opfunu.cec_based.CecBenchmark)


def test_F282017_results(assert_problem):
    assert_problem(opfunu.cec_based.F282017(ndim=30), 30, opfunu.cec_based.CecBenchmark)


def test_F292017_results(assert_problem):
    assert_problem(opfunu.cec_based.F292017(ndim=30), 30, opfunu.cec_based.CecBenchmark)


def test_all_optimal_results(assert_problem):
    ndim = 30
    known_failing = []
    all_functions = [
        x for x in opfunu.get_all_cec_based_functions() if x.__name__[-4:] == "2017" and x.__name__ not in known_failing
    ]
    for function in all_functions:
        problem = function(ndim=ndim)
        x = problem.x_global
        result = problem.evaluate(x)
        assert abs(result - problem.f_global) <= problem.epsilon, f"{function.__name__} Failed Optimal Test"
