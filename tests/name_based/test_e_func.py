#!/usr/bin/env python
# Created by "Thieu" at 11:04, 21/07/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import opfunu


def test_Easom_results(assert_problem):
    assert_problem(opfunu.name_based.Easom(ndim=2), 2, opfunu.name_based.FuncBenchmark)


def test_ElAttarVidyasagarDutta_results(assert_problem):
    assert_problem(opfunu.name_based.ElAttarVidyasagarDutta(ndim=2), 2, opfunu.name_based.FuncBenchmark)


def test_EggCrate(assert_problem):
    assert_problem(opfunu.name_based.EggCrate(ndim=2), 2, opfunu.name_based.FuncBenchmark)


def test_EggHolder(assert_problem):
    assert_problem(opfunu.name_based.EggHolder(ndim=7), 7, opfunu.name_based.FuncBenchmark)


def test_Exponential(assert_problem):
    assert_problem(opfunu.name_based.Exponential(ndim=7), 7, opfunu.name_based.FuncBenchmark)


def test_Exp2(assert_problem):
    assert_problem(opfunu.name_based.Exp2(ndim=2), 2, opfunu.name_based.FuncBenchmark)


def test_Eckerle4(assert_problem):
    assert_problem(opfunu.name_based.Eckerle4(ndim=3), 3, opfunu.name_based.FuncBenchmark)
