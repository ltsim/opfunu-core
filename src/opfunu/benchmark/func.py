#!/usr/bin/env python
# Created by "Thieu" at 16:47, 28/06/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import typing

import numpy as np

from opfunu.benchmark import Benchmark


class FuncBenchmark(Benchmark):
    """
    Defines an abstract class for optimization benchmark problem.

    All subclasses should implement the ``evaluate`` method for a particular optimization problem.

    Attributes
    ----------
    bounds : list
        The lower/upper bounds of the problem. This a 2D-matrix of [lower, upper] array that contain the lower and
        upper bounds.
        By default, each problem has its own bounds. But user can try to put different bounds to test the problem.
    ndim : int
        The dimensionality of the problem. It is calculated from bounds
    lb : np.ndarray
        The lower bounds for the problem
    ub : np.ndarray
        The upper bounds for the problem
    f_global : float
        The global optimum of the evaluated function.
    x_global : np.ndarray
        A list of vectors that provide the locations of the global minimum.
        Note that some problems have multiple global minima, not all of which may be listed.
        The number of function evaluations that the object has been asked to calculate.
    dim_changeable : bool
        Whether we can change the benchmark function `x` variable length (i.e., the dimensionality of the problem)
    """

    name: str = "Benchmark name"
    latex_formula: str = r"f(\mathbf{x})"
    latex_formula_dimension: str = r"d \in \mathbb{N}_{+}^{*}"
    latex_formula_bounds: str = r"x_i \in [-2\pi, 2\pi], \forall i \in \llbracket 1, d\rrbracket"
    latex_formula_global_optimum: str = r"f(0, ..., 0)=-1, \text{ for}, m=5, \beta=15"

    continuous: bool = True
    linear: bool = False
    convex: bool = True
    unimodal: bool = False
    separable: bool = False

    differentiable: bool = True
    scalable: bool = True
    randomized_term: bool = False
    parametric: bool = True

    modality: bool = True  # Number of ambiguous peaks, unknown # peaks

    # n_basins = 1
    # n_valleys = 1

    __ndim: int
    __bounds: np.ndarray
    __dim_supported: list[int]

    def __init__(
        self,
        parallel: bool = False,
        fastmath: bool = True,
        dtype: typing.Any = np.float64,
        compute: typing.Callable[..., None] | None = None,
        *,
        ndim: int | None = None,
        bounds: typing.Any = None,
        default_bounds: typing.Any = None,
        f_global: float | typing.Callable[[int], float] = 0.0,
        x_global: typing.Any = None,
        dim_changeable: bool | None = None,
        dim_default: int | None = None,
        dim_supported: list[int] | None = None,
        param_names: list[str] | None = None,
        plain: bool = False,
        params: dict[str, typing.Any] | None = None,
        paras: dict[str, typing.Any] | list[str] | tuple[str, ...] | None = None,
        verbose: bool = False,
        shift: typing.Any = None,
        rotate: typing.Any = None,
        rotate_bounds: bool = True,
    ) -> None:
        super().__init__(
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            compute=compute,
            # Callables need the resolved dimensionality; they are evaluated below,
            # after ``check_ndim_and_bounds``.
            f_global=f_global if not callable(f_global) else 0.0,
            x_global=x_global if (x_global is not None and not callable(x_global)) else None,
            dim_changeable=dim_changeable if dim_changeable is not None else False,
            dim_default=dim_default if dim_default is not None else 2,
            verbose=verbose,
            shift=shift,
            rotate=rotate,
            rotate_bounds=rotate_bounds,
        )

        self.__ndim = 0
        self.__bounds = np.array([])
        self.__dim_supported: list[int] = list(dim_supported) if dim_supported is not None else []

        if type(self) is FuncBenchmark:
            raise TypeError("FuncBenchmark is abstract; subclass it and provide compute()")
        if default_bounds is None:
            raise TypeError("FuncBenchmark requires default_bounds; pass it to super().__init__()")

        self.check_ndim_and_bounds(ndim, bounds, default_bounds)
        # ``f_global``/``x_global`` depend on the resolved dimensionality, so they
        # are (re)written through the protected storage after the bounds check.
        self._f_global = float(f_global(self.ndim) if callable(f_global) else f_global)
        if callable(x_global):
            self._x_global = np.asarray(x_global(self.ndim))
        elif x_global is None:
            self._x_global = np.asarray(np.zeros(self.ndim))
        else:
            self._x_global = np.asarray(x_global)
        self._bind_kernel(compute, param_names or [], plain=plain, params=params, paras=paras)

    @property
    def dim_supported(self) -> list[int]:
        """The list of supported dimensionalities (empty means any ``ndim > 1``)."""
        return self.__dim_supported

    @dim_supported.setter
    def dim_supported(self, value: list[int]) -> None:
        self.__dim_supported = list(value)

    def _set_bounds(self, bounds: np.ndarray) -> None:
        """Write the (ndim, 2) bounds matrix (used for coordinate shifts)."""
        self.__bounds = np.asarray(bounds, dtype=float)

    def check_ndim_and_bounds(
        self, ndim: int | None = None, bounds: typing.Any = None, default_bounds: typing.Any = None
    ) -> None:
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
        if ndim is None:
            self.__bounds = default_bounds if bounds is None else np.array(bounds).T
            self.__ndim = self.__bounds.shape[0]
        else:
            if bounds is None:
                if self.dim_changeable:
                    if isinstance(ndim, int) and ndim > 1:
                        self.__ndim = int(ndim)
                        self.__bounds = np.array([default_bounds[0] for _ in range(self.__ndim)])
                    else:
                        raise ValueError("ndim must be an integer and > 1!")
                else:
                    self.__ndim = self.dim_default
                    self.__bounds = default_bounds
                    print(f"{self.__class__.__name__} is fixed problem with {self.dim_default} variables!")
            else:
                if self.dim_changeable:
                    self.__bounds = np.array(bounds).T
                    self.__ndim = self.__bounds.shape[0]
                    print(f"{self.__class__.__name__} problem is set with {self.__ndim} variables!")
                else:
                    self.__bounds = np.array(bounds).T
                    if self.__bounds.shape[0] != self.dim_default:
                        raise ValueError(
                            f"{self.__class__.__name__} is fixed problem with {self.__ndim} variables. "
                            "Please setup the correct bounds!"
                        )
                    else:
                        self.__ndim = self.dim_default

    def check_solution(self, x: np.ndarray) -> None:
        """
        Raise the error if the problem size is not equal to the solution length

        Parameters
        ----------
        x : np.ndarray
            The solution
        """
        if not self.dim_changeable and (len(x) != self.__ndim):
            raise ValueError(f"The length of solution should have {self.__ndim} variables!")

    def is_ndim_compatible(self, ndim: int | None) -> bool:
        """
        Method to support searching the functions with input ndim

        Parameters
        ----------
        ndim : int
                The number of dimensions

        Returns
        -------
        val: bool
             Always true if dim_changeable = True, Else return ndim == self.ndim
        """
        assert (ndim is None) or (isinstance(ndim, int) and (not ndim < 0)), (
            "The dimension ndim must be None or a positive integer"
        )
        if ndim is None:
            return True
        else:
            if self.dim_changeable:
                return ndim > 0
            else:
                return ndim == self.ndim

    def is_succeed(self, x: np.ndarray, tol: float = 1.0e-5) -> bool:
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

        # the solution should still be in bounds, otherwise immediate fail.
        if np.any(x > self.ub) or np.any(x < self.lb):
            return False

        val = float(self.evaluate(np.squeeze(x)))
        if np.abs(val - self.f_global) < tol:
            return True

        # you found a lower global minimum.  This shouldn't happen.
        if val < self.f_global:
            raise ValueError("Found a lower global minimum", x, val, self.f_global)
        return False

    @property
    def bounds(self) -> np.ndarray:
        """
        The lower/upper bounds to be used for optimization problem. This a 2D-matrix of [lower, upper] array that
        contain the lower and upper bounds for the problem. The problem should not be asked for evaluation outside
        these bounds. ``len(bounds) == ndim``.
        """
        return self.__bounds

    @property
    def ndim(self) -> int:
        """
        The dimensionality of the problem.

        Returns
        -------
        ndim : int
            The dimensionality of the problem
        """
        return self.__ndim

    @property
    def lb(self) -> np.ndarray:
        """
        The lower bounds for the problem

        Returns
        -------
        lb : 1D-vector
            The lower bounds for the problem
        """
        return np.array([x[0] for x in self.bounds])

    @property
    def ub(self) -> np.ndarray:
        """
        The upper bounds for the problem

        Returns
        -------
        ub : 1D-vector
            The upper bounds for the problem
        """
        return np.array([x[1] for x in self.bounds])

    def create_solution(self) -> np.ndarray:
        """
        Create a random solution for the current problem

        Returns
        -------
        solution: 1D-vector
            The random solution
        """
        return np.random.uniform(self.lb, self.ub)
