#!/usr/bin/env python
# Created by "Thieu" at 15:46, 04/07/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import opfunu


def test_F12014_results(assert_problem):
    assert_problem(opfunu.cec_based.F12014(ndim=50), 50, opfunu.cec_based.CecBenchmark)


def test_F22014_results(assert_problem):
    assert_problem(opfunu.cec_based.F22014(ndim=50), 50, opfunu.cec_based.CecBenchmark)


def test_F32014_results(assert_problem):
    assert_problem(opfunu.cec_based.F32014(ndim=50), 50, opfunu.cec_based.CecBenchmark)


def test_F42014_results(assert_problem):
    assert_problem(opfunu.cec_based.F42014(ndim=50), 50, opfunu.cec_based.CecBenchmark)


def test_F52014_results(assert_problem):
    assert_problem(opfunu.cec_based.F52014(ndim=50), 50, opfunu.cec_based.CecBenchmark)


def test_F62014_results(assert_problem):
    assert_problem(opfunu.cec_based.F62014(ndim=50), 50, opfunu.cec_based.CecBenchmark)


def test_F72014_results(assert_problem):
    assert_problem(opfunu.cec_based.F72014(ndim=50), 50, opfunu.cec_based.CecBenchmark)


def test_F82014_results(assert_problem):
    assert_problem(opfunu.cec_based.F82014(ndim=50), 50, opfunu.cec_based.CecBenchmark)


def test_F92014_results(assert_problem):
    assert_problem(opfunu.cec_based.F92014(ndim=50), 50, opfunu.cec_based.CecBenchmark)


def test_F102014_results(assert_problem):
    assert_problem(opfunu.cec_based.F102014(ndim=50), 50, opfunu.cec_based.CecBenchmark)


def test_F112014_results(assert_problem):
    assert_problem(opfunu.cec_based.F112014(ndim=50), 50, opfunu.cec_based.CecBenchmark)


def test_F122014_results(assert_problem):
    assert_problem(opfunu.cec_based.F122014(ndim=50), 50, opfunu.cec_based.CecBenchmark)


def test_F132014_results(assert_problem):
    assert_problem(opfunu.cec_based.F132014(ndim=50), 50, opfunu.cec_based.CecBenchmark)


def test_F142014_results(assert_problem):
    assert_problem(opfunu.cec_based.F142014(ndim=50), 50, opfunu.cec_based.CecBenchmark)


def test_F152014_results(assert_problem):
    assert_problem(opfunu.cec_based.F152014(ndim=50), 50, opfunu.cec_based.CecBenchmark)


def test_F162014_results(assert_problem):
    assert_problem(opfunu.cec_based.F162014(ndim=50), 50, opfunu.cec_based.CecBenchmark)


def test_F172014_results(assert_problem):
    assert_problem(opfunu.cec_based.F172014(ndim=50), 50, opfunu.cec_based.CecBenchmark)


def test_F182014_results(assert_problem):
    assert_problem(opfunu.cec_based.F182014(ndim=50), 50, opfunu.cec_based.CecBenchmark)


def test_F192014_results(assert_problem):
    assert_problem(opfunu.cec_based.F192014(ndim=50), 50, opfunu.cec_based.CecBenchmark)


def test_F202014_results(assert_problem):
    assert_problem(opfunu.cec_based.F202014(ndim=50), 50, opfunu.cec_based.CecBenchmark)


def test_F212014_results(assert_problem):
    assert_problem(opfunu.cec_based.F212014(ndim=50), 50, opfunu.cec_based.CecBenchmark)


def test_F222014_results(assert_problem):
    assert_problem(opfunu.cec_based.F222014(ndim=50), 50, opfunu.cec_based.CecBenchmark)


def test_F232014_results(assert_problem):
    assert_problem(opfunu.cec_based.F232014(ndim=50), 50, opfunu.cec_based.CecBenchmark)


def test_F242014_results(assert_problem):
    assert_problem(opfunu.cec_based.F242014(ndim=50), 50, opfunu.cec_based.CecBenchmark)


def test_F252014_results(assert_problem):
    assert_problem(opfunu.cec_based.F252014(ndim=50), 50, opfunu.cec_based.CecBenchmark)


def test_F262014_results(assert_problem):
    assert_problem(opfunu.cec_based.F262014(ndim=50), 50, opfunu.cec_based.CecBenchmark)


def test_F272014_results(assert_problem):
    assert_problem(opfunu.cec_based.F272014(ndim=50), 50, opfunu.cec_based.CecBenchmark)


def test_F282014_results(assert_problem):
    assert_problem(opfunu.cec_based.F282014(ndim=50), 50, opfunu.cec_based.CecBenchmark)


def test_F292014_results(assert_problem):
    assert_problem(opfunu.cec_based.F292014(ndim=50), 50, opfunu.cec_based.CecBenchmark)


def test_F302014_results(assert_problem):
    assert_problem(opfunu.cec_based.F302014(ndim=50), 50, opfunu.cec_based.CecBenchmark)


def test_all_optimal_results(assert_problem):
    known_failing = []
    all_functions = [
        x for x in opfunu.get_all_cec_based_functions() if x.__name__[-4:] == "2014" and x.__name__ not in known_failing
    ]
    for function in all_functions:
        problem = function()
        x = problem.x_global
        result = problem.evaluate(x)
        assert abs(result - problem.f_global) <= problem.epsilon, f"{function.__name__} Failed Optimal Test"
