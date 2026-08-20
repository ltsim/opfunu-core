#!/usr/bin/env python
# Created by "Thieu" at 17:55, 22/07/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import opfunu


def test_Hansen_results(assert_problem):
    assert_problem(opfunu.name_based.Hansen(ndim=2), 2, opfunu.name_based.FuncBenchmark)


def test_Hartmann3_results(assert_problem):
    assert_problem(opfunu.name_based.Hartmann3(ndim=3), 3, opfunu.name_based.FuncBenchmark)


def test_Hartmann6_results(assert_problem):
    assert_problem(opfunu.name_based.Hartmann6(ndim=6), 6, opfunu.name_based.FuncBenchmark)


def test_HelicalValley_results(assert_problem):
    assert_problem(opfunu.name_based.HelicalValley(ndim=3), 3, opfunu.name_based.FuncBenchmark)


def test_Himmelblau_results(assert_problem):
    assert_problem(opfunu.name_based.Himmelblau(ndim=2), 2, opfunu.name_based.FuncBenchmark)


def test_Hosaki_results(assert_problem):
    assert_problem(opfunu.name_based.Hosaki(ndim=2), 2, opfunu.name_based.FuncBenchmark)


def test_HolderTable_results(assert_problem):
    assert_problem(opfunu.name_based.HolderTable(ndim=2), 2, opfunu.name_based.FuncBenchmark)
