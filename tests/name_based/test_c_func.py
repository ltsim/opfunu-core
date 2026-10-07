#!/usr/bin/env python
# Created by "Thieu" at 20:19, 19/07/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import opfunu


def test_CamelThreeHump_results(assert_problem):
    assert_problem(opfunu.name_based.CamelThreeHump(ndim=2), 2, opfunu.name_based.FuncBenchmark)


def test_CamelSixHump_results(assert_problem):
    assert_problem(opfunu.name_based.CamelSixHump(ndim=2), 2, opfunu.name_based.FuncBenchmark)


def test_ChenBird_results(assert_problem):
    assert_problem(opfunu.name_based.ChenBird(ndim=2), 2, opfunu.name_based.FuncBenchmark)


def test_ChenV_results(assert_problem):
    assert_problem(opfunu.name_based.ChenV(ndim=2), 2, opfunu.name_based.FuncBenchmark)


def test_Chichinadze_results(assert_problem):
    assert_problem(opfunu.name_based.Chichinadze(ndim=2), 2, opfunu.name_based.FuncBenchmark)


def test_ChungReynolds_results(assert_problem):
    assert_problem(opfunu.name_based.ChungReynolds(ndim=9), 9, opfunu.name_based.FuncBenchmark)


def test_Cigar_results(assert_problem):
    assert_problem(opfunu.name_based.Cigar(ndim=9), 9, opfunu.name_based.FuncBenchmark)


def test_Cola_results(assert_problem):
    assert_problem(opfunu.name_based.Cola(ndim=17), 17, opfunu.name_based.FuncBenchmark)


def test_Colville_results(assert_problem):
    assert_problem(opfunu.name_based.Colville(ndim=4), 4, opfunu.name_based.FuncBenchmark)


def test_Corana_results(assert_problem):
    assert_problem(opfunu.name_based.Corana(ndim=4), 4, opfunu.name_based.FuncBenchmark)


def test_CosineMixture_results(assert_problem):
    assert_problem(opfunu.name_based.CosineMixture(ndim=11), 11, opfunu.name_based.FuncBenchmark)


def test_CrossInTray_results(assert_problem):
    assert_problem(opfunu.name_based.CrossInTray(ndim=2), 2, opfunu.name_based.FuncBenchmark)


def test_CrossLegTable_results(assert_problem):
    assert_problem(opfunu.name_based.CrossLegTable(ndim=2), 2, opfunu.name_based.FuncBenchmark)


def test_CrownedCross_results(assert_problem):
    assert_problem(opfunu.name_based.CrownedCross(ndim=2), 2, opfunu.name_based.FuncBenchmark)


def test_Cube_results(assert_problem):
    assert_problem(opfunu.name_based.Cube(ndim=2), 2, opfunu.name_based.FuncBenchmark)
