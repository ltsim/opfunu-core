# Created by "LTSIM" at 10:47, 12/05/2026 ----------%
#       Email: tsim@cucei.udg.mx                    %
#       Github: https://github.com/ltsim            %
# --------------------------------------------------%
"""Optional-Numba compatibility shim.

Numba enables ``guvectorize``-based vectorization, but it is only available on
CPython and must stay opt-in (``pip install opfunu-core[numba]``). This module
is import-safe when Numba is missing (PyPy, minimal installs): ``HAS_NUMBA``
reports availability and ``njit`` degrades to a transparent pass-through
decorator so all kernels and shared operators keep working as plain Python.
"""

import typing

try:
    import numba as nb

    HAS_NUMBA: bool = True
except ImportError:
    nb = typing.cast(typing.Any, None)
    HAS_NUMBA = False


_F = typing.TypeVar("_F", bound=typing.Callable[..., typing.Any])


@typing.overload
def njit(func: _F) -> _F: ...


@typing.overload
def njit(*args: typing.Any, **kwargs: typing.Any) -> typing.Callable[[_F], _F]: ...


def njit(__func: typing.Any = None, *njit_args: typing.Any, **njit_kwargs: typing.Any) -> typing.Any:
    """Drop-in replacement for ``numba.njit`` supporting both ``@njit`` and ``@njit(...)``.

    When Numba is installed the call is delegated to ``numba.njit`` unchanged;
    otherwise the decorated function is returned as-is.
    """
    if HAS_NUMBA:
        if __func is None:
            return nb.njit(*njit_args, **njit_kwargs)
        if not njit_args and not njit_kwargs:
            return nb.njit(__func)
        return nb.njit(__func, *njit_args, **njit_kwargs)
    if __func is not None and callable(__func) and not njit_args and not njit_kwargs:
        return __func

    def _decorator(func: _F) -> _F:
        return func

    return _decorator
