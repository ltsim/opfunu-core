#!/usr/bin/env python
# Created by "Thieu" at 17:23, 22/07/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import opfunu


def test_FreudensteinRoth_results(assert_problem):
    assert_problem(opfunu.name_based.FreudensteinRoth(ndim=2), 2, opfunu.name_based.FuncBenchmark)
