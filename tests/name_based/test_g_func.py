#!/usr/bin/env python
# Created by "Thieu" at 17:27, 22/07/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import opfunu


def test_Giunta_results(assert_problem):
    assert_problem(opfunu.name_based.Giunta(ndim=2), 2, opfunu.name_based.FuncBenchmark)


def test_GoldsteinPrice_results(assert_problem):
    assert_problem(opfunu.name_based.GoldsteinPrice(ndim=2), 2, opfunu.name_based.FuncBenchmark)


def test_Griewank_results(assert_problem):
    assert_problem(opfunu.name_based.Griewank(ndim=17), 17, opfunu.name_based.FuncBenchmark)


def test_Gulf_results(assert_problem):
    assert_problem(opfunu.name_based.Gulf(ndim=3), 3, opfunu.name_based.FuncBenchmark)


def test_Gear_results(assert_problem):
    assert_problem(opfunu.name_based.Gear(ndim=4), 4, opfunu.name_based.FuncBenchmark)
