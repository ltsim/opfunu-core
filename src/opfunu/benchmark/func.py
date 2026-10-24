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

    All subclasses should implement the ``compute`` method for a particular optimization problem; the shared
    ``evaluate``, shape (``bounds``/``ndim``/``lb``/``ub``) and ``is_succeed`` machinery comes from ``Benchmark``.

    Attributes
    ----------
    dim_supported : list[int]
        The dimensionalities the problem supports (empty means any ``ndim > 1``).
    """

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
        paras: dict[str, typing.Any] | list[str] | tuple[str, ...] | None = None,
        verbose: bool = False,
        shift: typing.Any = None,
        prepare: bool | None = None,
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
            prepare=prepare,
        )

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
        self._bind_kernel(compute, param_names or [], plain=plain, paras=paras)

    @property
    def dim_supported(self) -> list[int]:
        """The list of supported dimensionalities (empty means any ``ndim > 1``)."""
        return self.__dim_supported

    @dim_supported.setter
    def dim_supported(self, value: list[int]) -> None:
        self.__dim_supported = list(value)

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
            self._bounds = default_bounds if bounds is None else np.array(bounds).T
            self._ndim = self._bounds.shape[0]
        else:
            if bounds is None:
                if self.dim_changeable:
                    if isinstance(ndim, int) and ndim > 1:
                        self._ndim = int(ndim)
                        self._bounds = np.array([default_bounds[0] for _ in range(self._ndim)])
                    else:
                        raise ValueError("ndim must be an integer and > 1!")
                else:
                    self._ndim = self.dim_default
                    self._bounds = default_bounds
                    print(f"{self.__class__.__name__} is fixed problem with {self.dim_default} variables!")
            else:
                if self.dim_changeable:
                    self._bounds = np.array(bounds).T
                    self._ndim = self._bounds.shape[0]
                    print(f"{self.__class__.__name__} problem is set with {self._ndim} variables!")
                else:
                    self._bounds = np.array(bounds).T
                    if self._bounds.shape[0] != self.dim_default:
                        raise ValueError(
                            f"{self.__class__.__name__} is fixed problem with {self._ndim} variables. "
                            "Please setup the correct bounds!"
                        )
                    else:
                        self._ndim = self.dim_default

    def check_solution(self, x: np.ndarray) -> None:
        """
        Raise the error if the problem size is not equal to the solution length

        Parameters
        ----------
        x : np.ndarray
            The solution
        """
        if not self.dim_changeable and (len(x) != self._ndim):
            raise ValueError(f"The length of solution should have {self._ndim} variables!")
