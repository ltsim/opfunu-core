#!/usr/bin/env python
# Created by "Thieu" at 20:37, 18/07/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import opfunu


def test_Ackley01_results(assert_problem):
    assert_problem(opfunu.name_based.Ackley01(ndim=10), 10, opfunu.name_based.FuncBenchmark)


def test_Ackley02_results(assert_problem):
    assert_problem(opfunu.name_based.Ackley02(ndim=2), 2, opfunu.name_based.FuncBenchmark)


def test_Ackley03_results(assert_problem):
    assert_problem(opfunu.name_based.Ackley03(ndim=2), 2, opfunu.name_based.FuncBenchmark)


def test_Adjiman_results(assert_problem):
    assert_problem(opfunu.name_based.Adjiman(ndim=2), 2, opfunu.name_based.FuncBenchmark)


def test_Alpine01_results(assert_problem):
    assert_problem(opfunu.name_based.Alpine01(ndim=18), 18, opfunu.name_based.FuncBenchmark)


def test_Alpine02_results(assert_problem):
    assert_problem(opfunu.name_based.Alpine02(ndim=18), 18, opfunu.name_based.FuncBenchmark)


def test_AMGM_results(assert_problem):
    assert_problem(opfunu.name_based.AMGM(ndim=11), 11, opfunu.name_based.FuncBenchmark)
