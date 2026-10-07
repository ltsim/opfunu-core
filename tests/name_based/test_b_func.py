#!/usr/bin/env python
# Created by "Thieu" at 20:42, 18/07/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import opfunu


def test_BartelsConn_results(assert_problem):
    assert_problem(opfunu.name_based.BartelsConn(ndim=2), 2, opfunu.name_based.FuncBenchmark)


def test_Beale_results(assert_problem):
    assert_problem(opfunu.name_based.Beale(ndim=2), 2, opfunu.name_based.FuncBenchmark)


def test_BiggsExp02_results(assert_problem):
    assert_problem(opfunu.name_based.BiggsExp02(ndim=2), 2, opfunu.name_based.FuncBenchmark)


def test_BiggsExp03_results(assert_problem):
    assert_problem(opfunu.name_based.BiggsExp03(ndim=3), 3, opfunu.name_based.FuncBenchmark)


def test_BiggsExp04_results(assert_problem):
    assert_problem(opfunu.name_based.BiggsExp04(ndim=4), 4, opfunu.name_based.FuncBenchmark)


def test_BiggsExp05_results(assert_problem):
    assert_problem(opfunu.name_based.BiggsExp05(ndim=5), 5, opfunu.name_based.FuncBenchmark)


def test_Bird_results(assert_problem):
    assert_problem(opfunu.name_based.Bird(ndim=2), 2, opfunu.name_based.FuncBenchmark)


def test_Bohachevsky1_results(assert_problem):
    assert_problem(opfunu.name_based.Bohachevsky1(ndim=2), 2, opfunu.name_based.FuncBenchmark)


def test_Bohachevsky2_results(assert_problem):
    assert_problem(opfunu.name_based.Bohachevsky2(ndim=2), 2, opfunu.name_based.FuncBenchmark)


def test_Bohachevsky3_results(assert_problem):
    assert_problem(opfunu.name_based.Bohachevsky3(ndim=2), 2, opfunu.name_based.FuncBenchmark)


def test_Booth_results(assert_problem):
    assert_problem(opfunu.name_based.Booth(ndim=2), 2, opfunu.name_based.FuncBenchmark)


def test_BoxBetts_results(assert_problem):
    assert_problem(opfunu.name_based.BoxBetts(ndim=3), 3, opfunu.name_based.FuncBenchmark)


def test_Branin01_results(assert_problem):
    assert_problem(opfunu.name_based.Branin01(ndim=2), 2, opfunu.name_based.FuncBenchmark)


def test_Branin02_results(assert_problem):
    assert_problem(opfunu.name_based.Branin02(ndim=2), 2, opfunu.name_based.FuncBenchmark)


def test_Brent_results(assert_problem):
    assert_problem(opfunu.name_based.Brent(ndim=2), 2, opfunu.name_based.FuncBenchmark)


def test_Brown_results(assert_problem):
    assert_problem(opfunu.name_based.Brown(ndim=7), 7, opfunu.name_based.FuncBenchmark)


def test_Bukin02_results(assert_problem):
    assert_problem(opfunu.name_based.Bukin02(ndim=2), 2, opfunu.name_based.FuncBenchmark)


def test_Bukin04_results(assert_problem):
    assert_problem(opfunu.name_based.Bukin04(ndim=2), 2, opfunu.name_based.FuncBenchmark)


def test_Bukin06_results(assert_problem):
    assert_problem(opfunu.name_based.Bukin06(ndim=2), 2, opfunu.name_based.FuncBenchmark)
