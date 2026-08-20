#!/usr/bin/env python
# Created by "Thieu" at 09:33, 21/07/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import opfunu


def test_Damavandi_results(assert_problem):
    assert_problem(opfunu.name_based.Damavandi(ndim=2), 2, opfunu.name_based.FuncBenchmark)


def test_Deb01_results(assert_problem):
    assert_problem(opfunu.name_based.Deb01(ndim=11), 11, opfunu.name_based.FuncBenchmark)


def test_Deb03_results(assert_problem):
    assert_problem(opfunu.name_based.Deb03(ndim=11), 11, opfunu.name_based.FuncBenchmark)


def test_Decanomial_results(assert_problem):
    assert_problem(opfunu.name_based.Decanomial(ndim=2), 2, opfunu.name_based.FuncBenchmark)


def test_Deceptive_results(assert_problem):
    assert_problem(opfunu.name_based.Deceptive(ndim=17), 17, opfunu.name_based.FuncBenchmark)


def test_DeckkersAarts_results(assert_problem):
    assert_problem(opfunu.name_based.DeckkersAarts(ndim=2), 2, opfunu.name_based.FuncBenchmark)


def test_DeflectedCorrugatedSpring_results(assert_problem):
    assert_problem(opfunu.name_based.DeflectedCorrugatedSpring(ndim=13), 13, opfunu.name_based.FuncBenchmark)


def test_DeVilliersGlasser01_results(assert_problem):
    assert_problem(opfunu.name_based.DeVilliersGlasser01(ndim=4), 4, opfunu.name_based.FuncBenchmark)


def test_DeVilliersGlasser02_results(assert_problem):
    assert_problem(opfunu.name_based.DeVilliersGlasser02(ndim=5), 5, opfunu.name_based.FuncBenchmark)


def test_DixonPrice(assert_problem):
    assert_problem(opfunu.name_based.DixonPrice(ndim=11), 11, opfunu.name_based.FuncBenchmark)


def test_Dolan(assert_problem):
    assert_problem(opfunu.name_based.Dolan(ndim=5), 5, opfunu.name_based.FuncBenchmark)


def test_DropWave(assert_problem):
    assert_problem(opfunu.name_based.DropWave(ndim=2), 2, opfunu.name_based.FuncBenchmark)
