# Created by "LTSIM" at 10:47, 12/05/2026 ----------%
#       Email: tsim@cucei.udg.mx                    %
#       Github: https://github.com/ltsim            %
# --------------------------------------------------%
import abc
import dataclasses
import typing
from importlib.resources.abc import Traversable

import numpy as np


@dataclasses.dataclass
class Formula:
    latex: str

    def _repr_latex_(self) -> str:
        return f"$${self.latex}$$"


class Benchmark(abc.ABC):
    latex_formula: str = r'f(\mathbf{x})'
    latex_formula_dimension: str = r'd \in \mathbb{N}_{+}^{*}'
    latex_formula_bounds: str = r'x_i \in [-2\pi, 2\pi], \forall i \in \llbracket 1, d\rrbracket'
    latex_formula_global_optimum: str = r'f(0, ..., 0)=-1, \text{ for}, m=5, \beta=15'

    epsilon: typing.Final[float] = 1e-8

    def __init__(self) -> None:
        self.support_path: Traversable | None = None
        self.verbose: bool = False
        self.paras: dict[str, typing.Any] = {}

    def _repr_latex_(self) -> str:
        return f"$${self.latex_formula}$$"

    @property
    def formula(self) -> Formula:
        return Formula(self.latex_formula)

    @property
    def formula_dimension(self) -> Formula:
        return Formula(self.latex_formula_dimension)

    @property
    def formula_bounds(self) -> Formula:
        return Formula(self.latex_formula_bounds)

    @property
    def formula_global_optimum(self) -> Formula:
        return Formula(self.latex_formula_global_optimum)

    @abc.abstractmethod
    def check_ndim_and_bounds(self, *args: typing.Any, **kwargs: typing.Any) -> None:
        """
        Check the bounds when initializing the object.

        Parameters
        ----------
        ndim : int
            The number of dimensions (variables)
        bounds : list, tuple, np.ndarray
            List of lower bound and upper bound, should use default None value
        default_bounds : np.ndarray
            List of initial lower bound and upper bound values
        """
        ...

    @abc.abstractmethod
    def check_solution(self, x: np.ndarray) -> None:
        """
        Raise the error if the problem size is not equal to the solution length

        Parameters
        ----------
        x : np.ndarray
            The solution
        """
        ...

    def get_paras(self) -> dict[str, typing.Any]:
        """
        Return the parameters of the problem. Depended on function
        """
        default = {"bounds": self.bounds, "ndim": self.ndim}
        return {**default, **self.paras}

    @abc.abstractmethod
    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluation of the benchmark function.

        Parameters
        ----------
        x : np.ndarray
            The candidate vector for evaluating the benchmark problem. Must have ``len(x) == self.ndim``.

        Returns
        -------
        val : float
              the evaluated benchmark function
        """
        ...

    @abc.abstractmethod
    def is_ndim_compatible(self, ndim: int | None) -> bool:
        ...

    @abc.abstractmethod
    def is_succeed(self, x: np.ndarray, tol: float = 1.e-5) -> bool:
        """
        Check if a candidate solution at the global minimum.

        Parameters
        ----------
        x : np.ndarray
            The candidate vector for testing if the global minimum has been reached. Must have ``len(x) == self.ndim``
        tol : float
            The evaluated function and known global minimum must differ by less than this amount to be at a
            global minimum.

        Returns
        -------
        is_succeed : bool
            Answer the question: is the candidate vector at the global minimum?
        """
        ...

    @property
    @abc.abstractmethod
    def bounds(self) -> np.ndarray:
        """
        The lower/upper bounds to be used for optimization problem. This a 2D-matrix of [lower, upper] array that
        contain the lower and upper bounds for the problem. The problem should not be asked for evaluation outside
        these bounds. ``len(bounds) == ndim``.
        """
        ...

    @property
    @abc.abstractmethod
    def ndim(self) -> int:
        """
        The dimensionality of the problem.

        Returns
        -------
        ndim : int
            The dimensionality of the problem
        """
        ...

    @property
    @abc.abstractmethod
    def lb(self) -> np.ndarray:
        """
        The lower bounds for the problem

        Returns
        -------
        lb : 1D-vector
            The lower bounds for the problem
        """
        ...

    @property
    @abc.abstractmethod
    def ub(self) -> np.ndarray:
        """
        The upper bounds for the problem

        Returns
        -------
        ub : 1D-vector
            The upper bounds for the problem
        """
        ...

    def create_solution(self) -> np.ndarray:
        """
        Create a random solution for the current problem

        Returns
        -------
        solution: 1D-vector
            The random solution
        """
        return np.random.uniform(self.lb, self.ub)
