import functools
import typing

from importlib.util import find_spec

if typing.TYPE_CHECKING:
    from IPython.display import Latex

F = typing.TypeVar("F", bound=typing.Callable[..., typing.Any])
HAS_IPYTHON = find_spec("IPython") is not None


def requires_ipython(func: F) -> F:
    @functools.wraps(func)
    def wrapper(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        if not HAS_IPYTHON:
            raise RuntimeError(
                f"The function '{func.__name__}' requires IPython to be installed. "
                "Use: pip install 'opfunu-core[latex]'"
            )

        return func(*args, **kwargs)

    return wrapper


@requires_ipython
def render_latex(latex: str) -> "Latex":
    from IPython.display import Latex

    if HAS_IPYTHON and isinstance(latex, str):
        return Latex(latex)

    raise ValueError("Cannot render latex")
