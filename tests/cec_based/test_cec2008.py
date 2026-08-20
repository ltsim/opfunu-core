#!/usr/bin/env python
# Created by "Thieu" at 22:27, 01/07/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import opfunu


def test_F12008_results(assert_problem):
    assert_problem(opfunu.cec_based.F12008(ndim=1000), 1000, opfunu.cec_based.CecBenchmark)


def test_F22008_results(assert_problem):
    assert_problem(opfunu.cec_based.F22008(ndim=1000), 1000, opfunu.cec_based.CecBenchmark)


def test_F32008_results(assert_problem):
    assert_problem(opfunu.cec_based.F32008(ndim=1000), 1000, opfunu.cec_based.CecBenchmark)


def test_F42008_results(assert_problem):
    assert_problem(opfunu.cec_based.F42008(ndim=1000), 1000, opfunu.cec_based.CecBenchmark)


def test_F52008_results(assert_problem):
    assert_problem(opfunu.cec_based.F52008(ndim=1000), 1000, opfunu.cec_based.CecBenchmark)


def test_F62008_results(assert_problem):
    assert_problem(opfunu.cec_based.F62008(ndim=1000), 1000, opfunu.cec_based.CecBenchmark)


def test_F72008_results(assert_problem):
    assert_problem(opfunu.cec_based.F72008(ndim=1000), 1000, opfunu.cec_based.CecBenchmark)
