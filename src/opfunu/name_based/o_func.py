#!/usr/bin/env python
# Created by "Thieu" at 17:31, 30/07/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import typing

import numpy as np

from opfunu.benchmark.func import FuncBenchmark


class OddSquare(FuncBenchmark):
    """
    .. [1] Gavana, A. Global Optimization Benchmarks and AMPGO retrieved 2015

    .. math::
        f_{\text{OddSquare}}(x) = -e^{-\frac{d}{2\\pi}} \\cos(\\pi d) \\left( 1 + \frac{0.02h}{d + 0.01} \right )

    Where, in this exercise:

    .. math::

        \begin{cases}
        d = n \\cdot \\smash{\\displaystyle\\max_{1 \\leq i \\leq n}}
            \\left[ (x_i - b_i)^2 \right ] \\
        h = \\sum_{i=1}^{n} (x_i - b_i)^2
        \\end{cases}

    And :math:`b = [1, 1.3, 0.8, -0.4, -1.3, 1.6, -0.2, -0.6, 0.5, 1.4, 1, 1.3,
                    0.8, -0.4, -1.3, 1.6, -0.2, -0.6, 0.5, 1.4]`

    Here :math:`x_i \\in [-5 \\pi, 5 \\pi]` for :math:`i = 1, ..., n`. `n \\leq 20`.
    *Global optimum*: :math:`f(x) = -1.00846728102`for :math:`x \approx b`
    """

    name = "Odd Square Function"
    latex_formula = (
        r"f_{\text{OddSquare}}(x) = -e^{-\frac{d}{2\pi}} \cos(\pi d) \left( 1 + \frac{0.02h}{d + 0.01} \right )"
    )
    latex_formula_dimension = r"d = 20"
    latex_formula_bounds = r"x_i \in [-5 \pi, 5 \pi]"
    latex_formula_global_optimum = r"f(b) = -1.00846728102"
    continuous = False
    linear = False
    convex = True
    unimodal = False
    separable = False

    differentiable = False
    scalable = False
    randomized_term = False
    parametric = False

    modality = False  # Number of ambiguous peaks, unknown # peaks

    def __init__(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
    ) -> None:
        def compute(x: np.ndarray, b: np.ndarray, out: np.ndarray) -> None:
            b = b[0 : x.shape[0]]
            d = x.shape[0] * max((x - b) ** 2.0)
            h = np.sum((x - b) ** 2.0)
            out[0] = -np.exp(-d / (2.0 * np.pi)) * np.cos(np.pi * d) * (1.0 + 0.02 * h / (d + 0.01))

        self.b = np.array(
            [1, 1.3, 0.8, -0.4, -1.3, 1.6, -0.2, -0.6, 0.5, 1.4, 1, 1.3, 0.8, -0.4, -1.3, 1.6, -0.2, -0.6, 0.5, 1.4]
        )

        super().__init__(
            compute=compute,
            ndim=ndim,
            bounds=bounds,
            default_bounds=np.array([[-5 * np.pi, 5.0 * np.pi] for _ in range(2)]),
            f_global=-1.00846728102,
            x_global=lambda nd: self.b[:nd],
            dim_changeable=True,
            dim_default=2,
            param_names=["b"],
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
        )
