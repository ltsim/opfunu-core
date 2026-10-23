#!/usr/bin/env python
# Created by "Thieu" at 18:47, 22/07/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import opfunu


def test_Infinity_results(assert_problem):
    assert_problem(opfunu.name_based.Infinity(ndim=7), 7, opfunu.name_based.FuncBenchmark)
