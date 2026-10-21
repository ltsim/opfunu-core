"""Shared sampling math for the ``opfunu.plot`` visualization suite.

This module depends only on ``numpy`` so both the static (matplotlib) and
interactive (pyvista) engines can reuse it without pulling in heavy backends.
"""

import typing

import numpy as np


def validate_inputs(
    target: typing.Callable[..., typing.Any],
    lb: typing.Any,
    ub: typing.Any,
    selected_dims: tuple[int, int] | list[int],
    n_points: int,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Validate the common ``draw``/``interactive`` arguments.

    Parameters
    ----------
    target : callable
        Objective function ``f(x) -> float`` where ``x`` is a 1-D array.
    lb : array-like
        Lower bounds, one per dimension.
    ub : array-like
        Upper bounds, one per dimension.
    selected_dims : tuple of two ints
        1-based indices of the two dimensions to render, e.g. ``(1, 2)``.
    n_points : int
        Grid resolution per rendered dimension (must be >= 2).

    Returns
    -------
    lb_arr : np.ndarray
        Lower bounds as a 1-D float array.
    ub_arr : np.ndarray
        Upper bounds as a 1-D float array.
    dim0 : int
        First rendered dimension as a 0-based index.
    dim1 : int
        Second rendered dimension as a 0-based index.
    """
    if not callable(target):
        raise TypeError(f"target must be callable, got {type(target).__name__!r}.")
    lb_arr = np.asarray(lb, dtype=float).ravel()
    ub_arr = np.asarray(ub, dtype=float).ravel()
    if lb_arr.size == 0:
        raise ValueError("lb/ub must be non-empty.")
    if lb_arr.shape != ub_arr.shape:
        raise ValueError(f"lb shape {lb_arr.shape} != ub shape {ub_arr.shape}.")
    if bool(np.any(ub_arr <= lb_arr)):
        raise ValueError("Every ub entry must be strictly greater than the matching lb entry.")
    ndim = lb_arr.size
    dims = tuple(selected_dims)
    if len(dims) != 2:
        raise ValueError(f"selected_dims must hold exactly two 1-based indices, got {selected_dims!r}.")
    if any(not isinstance(d, (int, np.integer)) for d in dims):
        raise TypeError(f"selected_dims entries must be integers, got {selected_dims!r}.")
    if any(d < 1 or d > ndim for d in dims):
        raise ValueError(f"selected_dims {selected_dims!r} out of range for ndim={ndim} (1-based).")
    if dims[0] == dims[1]:
        raise ValueError(f"selected_dims entries must differ, got {selected_dims!r}.")
    if not isinstance(n_points, (int, np.integer)) or isinstance(n_points, bool):
        raise TypeError(f"n_points must be an integer, got {type(n_points).__name__!r}.")
    if int(n_points) < 2:
        raise ValueError(f"n_points must be >= 2, got {n_points!r}.")
    return lb_arr, ub_arr, int(dims[0]) - 1, int(dims[1]) - 1


def sample_surface(
    target: typing.Callable[..., typing.Any],
    lb: np.ndarray,
    ub: np.ndarray,
    dim0: int,
    dim1: int,
    n_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample ``target`` over a 2-D grid, fixing other dims at bound midpoints.

    When ``target`` is a bound method of a ``Benchmark`` instance its internal
    ``_evaluate_batch`` kernel is used; otherwise the callable is looped over.

    Parameters
    ----------
    target : callable
        Objective function ``f(x) -> float``.
    lb : np.ndarray
        1-D lower-bound vector (already validated).
    ub : np.ndarray
        1-D upper-bound vector (already validated).
    dim0 : int
        First rendered dimension (0-based).
    dim1 : int
        Second rendered dimension (0-based).
    n_points : int
        Grid resolution per rendered dimension.

    Returns
    -------
    X : np.ndarray
        ``(n_points, n_points)`` meshgrid over dimension ``dim0``.
    Y : np.ndarray
        ``(n_points, n_points)`` meshgrid over dimension ``dim1``.
    Z : np.ndarray
        ``(n_points, n_points)`` sampled objective values.
    """
    base = (lb + ub) / 2.0
    x_axis = np.linspace(float(lb[dim0]), float(ub[dim0]), int(n_points))
    y_axis = np.linspace(float(lb[dim1]), float(ub[dim1]), int(n_points))
    x_grid, y_grid = np.meshgrid(x_axis, y_axis)
    flat = np.repeat(base[np.newaxis, :], x_grid.size, axis=0)
    flat[:, dim0] = x_grid.ravel()
    flat[:, dim1] = y_grid.ravel()
    owner = getattr(target, "__self__", None)
    batch = getattr(owner, "_evaluate_batch", None)
    if callable(batch):
        values = np.asarray(batch(flat), dtype=float).ravel()
    else:
        values = np.asarray([float(target(row)) for row in flat], dtype=float)
    return x_grid, y_grid, values.reshape(x_grid.shape)
