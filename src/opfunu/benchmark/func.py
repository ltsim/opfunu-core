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
        paras: typing.Union[
            dict[str, typing.Any], list[str], tuple[str, ...], typing.Callable[..., dict[str, typing.Any]], None
        ] = None,
        verbose: bool = False,
        shift: typing.Any = None,
        prepare: bool | None = None,
    ) -> None:
        if type(self) is FuncBenchmark:
            raise TypeError("FuncBenchmark is abstract; subclass it and provide compute()")
        if default_bounds is None:
            raise TypeError("FuncBenchmark requires default_bounds; pass it to super().__init__()")

        changeable = dim_changeable if dim_changeable is not None else False
        default = dim_default if dim_default is not None else 2
        # Everything the base stores is resolved *first*, then handed to the constructor
        # as immutable state: the shape, the two global-optimum parts and the support data.
        resolved_ndim, resolved_bounds = self.check_ndim_and_bounds(
            ndim, bounds, default_bounds, dim_changeable=changeable, dim_default=default
        )
        # ``f_global``/``x_global`` may be callables of the resolved dimensionality.
        resolved_f_global = f_global(resolved_ndim) if callable(f_global) else f_global
        if callable(x_global):
            resolved_x_global = np.asarray(x_global(resolved_ndim))
        elif x_global is None:
            resolved_x_global = np.zeros(resolved_ndim)
        else:
            resolved_x_global = np.asarray(x_global)

        super().__init__(
            parallel=parallel,
            fastmath=fastmath,
            dtype=dtype,
            compute=compute,
            f_global=resolved_f_global,
            x_global=resolved_x_global,
            dim_changeable=changeable,
            dim_default=default,
            verbose=verbose,
            shift=shift,
            prepare=prepare,
            ndim=resolved_ndim,
            bounds=resolved_bounds,
            support_path=None,
        )
        self.__dim_supported: list[int] = list(dim_supported) if dim_supported is not None else []
        self._bind_kernel(compute, param_names or [], plain=plain, paras=paras)

    @property
    def dim_supported(self) -> list[int]:
        """The list of supported dimensionalities (empty means any ``ndim > 1``)."""
        return self.__dim_supported

    @dim_supported.setter
    def dim_supported(self, value: list[int]) -> None:
        self.__dim_supported = list(value)

    def check_ndim_and_bounds(
        self,
        ndim: int | None = None,
        bounds: typing.Any = None,
        default_bounds: typing.Any = None,
        *,
        dim_changeable: bool = False,
        dim_default: int = 2,
    ) -> tuple[int, np.ndarray]:
        """
        Resolve the dimensionality and bounds matrix for this problem (pure resolver).

        Validates the requested shape against the problem's configuration and returns
        ``(ndim, bounds)`` without touching any state; ``__init__`` installs the result
        as immutable state through ``super().__init__()``.

        Parameters
        ----------
        ndim : int
            The number of dimensions (variables)
        bounds : list, tuple, np.ndarray
            List of lower bound and upper bound, should use default None value
        default_bounds : np.ndarray
            List of initial lower bound and upper bound values
        dim_changeable : bool
            Whether the dimensionality can be changed at construction
        dim_default : int
            The fallback dimensionality for fixed-size problems
        """
        if ndim is None:
            resolved_bounds = default_bounds if bounds is None else np.array(bounds).T
            return len(resolved_bounds), np.asarray(resolved_bounds, dtype=float)
        if bounds is None:
            if dim_changeable:
                if isinstance(ndim, int) and ndim > 1:
                    return ndim, np.array([default_bounds[0] for _ in range(ndim)])
                raise ValueError("ndim must be an integer and > 1!")
            print(f"{self.__class__.__name__} is fixed problem with {dim_default} variables!")
            return dim_default, np.asarray(default_bounds, dtype=float)
        resolved_bounds = np.array(bounds).T
        if dim_changeable:
            print(f"{self.__class__.__name__} problem is set with {len(resolved_bounds)} variables!")
            return len(resolved_bounds), resolved_bounds
        if len(resolved_bounds) != dim_default:
            raise ValueError(
                f"{self.__class__.__name__} is fixed problem with {len(resolved_bounds)} variables. "
                "Please setup the correct bounds!"
            )
        return dim_default, resolved_bounds

    def check_solution(self, x: np.ndarray) -> None:
        """
        Raise the error if the problem size is not equal to the solution length

        Parameters
        ----------
        x : np.ndarray
            The solution
        """
        if not self.dim_changeable and (len(x) != self.ndim):
            raise ValueError(f"The length of solution should have {self.ndim} variables!")
