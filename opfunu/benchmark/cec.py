#!/usr/bin/env python
# Created by "Thieu" at 06:43, 30/06/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import importlib.resources

import numpy as np

from opfunu.benchmark import Benchmark


class CecBenchmark(Benchmark):
    """
    Defines an abstract class for optimization benchmark problem.

    All subclasses should implement the ``evaluate`` method for a particular optimization problem.

    Attributes
    ----------
    bounds : list
        The lower/upper bounds of the problem. This a 2D-matrix of [lower, upper] array that contain the lower and upper bounds.
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
    n_fe : int
        The number of function evaluations that the object has been asked to calculate.
    dim_changeable : bool
        Whether we can change the benchmark function `x` variable length (i.e., the dimensionality of the problem)
    """

    name = "Benchmark name"
    latex_formula = r'f(\mathbf{x})'
    latex_formula_dimension = r'd \in \mathbb{N}_{+}^{*}'
    latex_formula_bounds = r'x_i \in [-2\pi, 2\pi], \forall i \in \llbracket 1, d\rrbracket'
    latex_formula_global_optimum = r'f(0, ..., 0)=-1, \text{ for}, m=5, \beta=15'
    
    continuous = True
    linear = False
    convex = True
    unimodal = False
    separable = False

    differentiable = True
    scalable = True
    randomized_term = False
    parametric = True
    shifted = True
    rotated = False

    modality = True  # Number of ambiguous peaks, unknown # peaks

    characteristics: list[str]

    # n_basins = 1
    # n_valleys = 1

    def __init__(
            self,
            ndim=None,
            bounds=None,
            f_shift=None,
            f_matrix=None,
            f_shuffle=None,
            f_bias=None,
            default_bounds=None,
            dim_changeable=True,
            dim_default=30,
            dim_max=100,
            dim_supported=None,
            data_name="",
            load_two_matrix=False,
            load_tow_matrix=False,
    ):
        super().__init__()

        self._ndim = ndim
        self._bounds = bounds

        self._dim_changeable = dim_changeable
        self._dim_default = dim_default
        self._dim_max = dim_max
        self._dim_supported = dim_supported

        self.check_ndim_and_bounds(ndim, dim_max, bounds, default_bounds)

        if not data_name:
            data_name = getattr(self, "data_name", "")
        if not data_name:
            import re
            match = re.search(r"20\d{2}", self.__class__.__name__) or re.search(r"20\d{2}", self.__module__)
            if match:
                data_name = f"data_{match.group(0)}"

        if data_name:
            self.make_support_data_path(data_name)

        self.f_matrix = None
        self.f_shuffle = None
        self.f_shift = None

        if load_two_matrix or load_tow_matrix:
            shift_data, a_matrix, b_matrix = self.load_two_matrix_and_shift_data(f_shift)
            self.f_shift = shift_data[:self.ndim]
            self.f_matrix_a = a_matrix[:self.ndim, :self.ndim]
            self.f_matrix_b = b_matrix[:self.ndim, :self.ndim]
        else:
            if f_shift is not None:
                self.f_shift = self.check_shift_data(f_shift)[:self.ndim]
            if f_matrix is not None:
                self.f_matrix = self.check_matrix_data(f_matrix)
            if f_shuffle is not None:
                self.f_shuffle = self.check_shuffle_data(f_shuffle)

        self.f_bias = f_bias
        self.f_global = f_bias if f_bias is not None else 0.0
        self.x_global = self.f_shift

        self.n_fe = 0

    @property
    def dim_max(self):
        return self._dim_max

    @dim_max.setter
    def dim_max(self, value):
        self._dim_max = value

    @property
    def dim_supported(self):
        return self._dim_supported

    @dim_supported.setter
    def dim_supported(self, value):
        self._dim_supported = value

    @property
    def dim_default(self):
        return self._dim_default

    @dim_default.setter
    def dim_default(self, value):
        self._dim_default = value

    @property
    def dim_changeable(self):
        return self._dim_changeable

    @dim_changeable.setter
    def dim_changeable(self, value):
        self._dim_changeable = value

    def make_support_data_path(self, data_name: str):
        self.support_path = importlib.resources.files("opfunu").joinpath(f"cec_based/{data_name}")

    def check_shift_data(self, f_shift):
        if type(f_shift) is str:
            return self.load_shift_data(f_shift)
        else:
            if type(f_shift) in [list, tuple, np.ndarray]:
                return np.squeeze(f_shift)
            else:
                raise ValueError(f"The shift data should be a list/tuple or np.array!")

    def check_shift_matrix(self, f_shift, selected_idx=None):
        if type(f_shift) is str:
            if selected_idx is None:
                return self.load_matrix_data(f_shift)
            else:
                return self.load_matrix_data(f_shift)[selected_idx, :self.ndim]
        else:
            if type(f_shift) in [list, tuple, np.ndarray]:
                return np.squeeze(f_shift)
            else:
                raise ValueError(f"The shift data should be a list/tuple or np.array!")

    def check_matrix_data(self, f_matrix, needed_dim=True):
        if type(f_matrix) is str:
            if needed_dim:
                return self.load_matrix_data(f"{f_matrix}{self.ndim}")
            else:
                return self.load_matrix_data(f_matrix)
        else:
            if type(f_matrix) is np.ndarray:
                return np.squeeze(f_matrix)
            else:
                raise ValueError(f"The matrix data should be an orthogonal matrix (2D np.array)!")

    def check_shuffle_data(self, f_shuffle, needed_dim=True):
        if type(f_shuffle) is str:
            if needed_dim:
                return self.load_shift_data(f"{f_shuffle}{self.ndim}")
            else:
                return self.load_shift_data(f_shuffle)
        else:
            if type(f_shuffle) in [list, tuple, np.ndarray]:
                return np.squeeze(f_shuffle)
            else:
                raise ValueError(f"The shuffle data should be a list/tuple or np.array!")

    def check_m_group(self, m_group=None):
        if type(m_group) is int:
            if int(self.ndim / m_group) > 1:
                return m_group
            else:
                raise ValueError(f"ndim is too small or m_group is too large!")
        else:
            raise ValueError(f"m_group is positive integer!")

    def load_shift_data(self, filename):
        fname = filename if filename.endswith('.txt') else f"{filename}.txt"
        filepath = self.support_path.joinpath(fname)
        if hasattr(filepath, "open"):
            with filepath.open("r") as f:
                data = np.genfromtxt(f, dtype=float)
        else:
            data = np.genfromtxt(f"{self.support_path}/{fname}", dtype=float)
        return data.reshape((-1))

    def load_matrix_data(self, filename=None):
        fname = filename if filename.endswith('.txt') else f"{filename}.txt"
        filepath = self.support_path.joinpath(fname)
        try:
            if hasattr(filepath, "open"):
                with filepath.open("r") as f:
                    data = np.genfromtxt(f, dtype=float)
            else:
                data = np.genfromtxt(f"{self.support_path}/{fname}", dtype=float)
            return data
        except (FileNotFoundError, OSError):
            print(f'The file named: {fname} is not found.')
            print(f"{self.__class__.__name__} problem is only supported ndim in {self.dim_supported}!")
            exit(1)

    def load_shift_and_matrix_data(self, filename=None):
        fname = filename if filename.endswith('.txt') else f"{filename}.txt"
        filepath = self.support_path.joinpath(fname)
        if hasattr(filepath, "open"):
            with filepath.open("r") as f:
                data = np.genfromtxt(f, dtype=float)
        else:
            data = np.genfromtxt(f"{self.support_path}/{fname}", dtype=float)
        shift_data = data[:1, :].ravel()
        matrix_data = data[1:, :]
        return shift_data, matrix_data

    def load_two_matrix_and_shift_data(self, filename=None):
        fname = filename if filename.endswith('.txt') else f"{filename}.txt"
        filepath = self.support_path.joinpath(fname)
        if hasattr(filepath, "open"):
            with filepath.open("r") as f:
                data = np.genfromtxt(f, dtype=float)
        else:
            data = np.genfromtxt(f"{self.support_path}/{fname}", dtype=float)
        a_matrix = data[:100, :]
        b_matrix = data[100:200, :]
        shift_data = data[200:, :].ravel()
        return shift_data, a_matrix, b_matrix

    def check_solution(self, x, dim_max=None, dim_support=None):
        """
        Raise the error if the problem size is not equal to the solution length

        Parameters
        ----------
        x : np.ndarray
            The solution
        dim_max : The maximum number of variables that the function is supported
        dim_support : List of the supported dimensions
        """

        if len(x) != self._ndim:
            raise ValueError(
                f"{self.__class__.__name__} problem, the length of solution should have {self._ndim} variables!")

        if (dim_max is not None) and (len(x) > dim_max):
            raise ValueError(f"{self.__class__.__name__} problem is not supported ndim > {dim_max}!")

        if (dim_support is not None) and (len(x) not in dim_support):
            raise ValueError(f"{self.__class__.__name__} problem is only supported ndim in {dim_support}!")

    def evaluate(self, x):
        raise NotImplementedError

    def is_ndim_compatible(self, ndim):
        assert (ndim is None) or (
                isinstance(ndim, int) and (not ndim < 0)), "The dimension ndim must be None or a positive integer"
        if ndim is None:
            return True
        else:
            if self.dim_changeable:
                return ndim > 0
            else:
                return ndim == self.ndim

    def is_succeed(self, x, tol=1.e-5):
        if np.any(x > self.ub) or np.any(x < self.lb):
            return False

        val = self.evaluate(np.squeeze(x))
        if np.abs(val - self.f_global) < tol:
            return True

        # you found a lower global minimum.  This shouldn't happen.
        if val < self.f_global:
            raise ValueError("Found a lower global minimum", x, val, self.f_global)
        return False

    def check_ndim_and_bounds(self, ndim=None, dim_max=None, bounds=None, default_bounds=None):
        """
        Check the bounds when initializing the object.

        Parameters
        ----------
        ndim : int
            The number of dimensions (variables)
        dim_max : int
            The maximum number of dimensions (variables) that the problem is supported
        bounds : list, tuple, np.ndarray
            List of lower bound and upper bound, should use default None value
        default_bounds : np.ndarray
            List of initial lower bound and upper bound values
        """

        if default_bounds is None:
            default_bounds = np.array([[-100., 100.] for _ in range(self.dim_default)])
        else:
            default_bounds = np.array(default_bounds)
            if default_bounds.ndim == 1 and len(default_bounds) == 2:
                default_bounds = np.array([default_bounds for _ in range(self.dim_default)])
            elif default_bounds.ndim == 2 and default_bounds.shape[0] == 1:
                default_bounds = np.array([default_bounds[0] for _ in range(self.dim_default)])

        if ndim is None:
            self._bounds = default_bounds if bounds is None else np.array(bounds).T
            self._ndim = self._bounds.shape[0]

            if dim_max is not None and self._ndim > dim_max:
                raise ValueError(f"{self.__class__.__name__} problem supports maximum {dim_max} variables!")
        else:
            if bounds is None:
                if self.dim_changeable:
                    if type(ndim) is int and ndim > 1:
                        if dim_max is None or ndim <= dim_max:
                            self._ndim = int(ndim)
                            self._bounds = np.array([default_bounds[0] for _ in range(self._ndim)])
                        else:
                            raise ValueError(f"{self.__class__.__name__} problem supports maximum {dim_max} variables!")
                    else:
                        raise ValueError('ndim must be an integer and > 1!')
                else:
                    self._ndim = self.dim_default
                    self._bounds = default_bounds
                    if self.verbose:
                        print(f"{self.__class__.__name__} is fixed problem with {self.dim_default} variables!")
            else:
                if self.dim_changeable:
                    self._bounds = np.array(bounds).T
                    self._ndim = self._bounds.shape[0]
                    if self._ndim > dim_max:
                        raise ValueError(f"{self.__class__.__name__} problem supports maximum {dim_max} variables!")
                    else:
                        print(f"{self.__class__.__name__} problem is set with {self._ndim} variables!")
                else:
                    self._bounds = np.array(bounds).T
                    if self._bounds.shape[0] == self.dim_default:
                        self._ndim = self.dim_default
                    else:
                        raise ValueError(
                            f"{self.__class__.__name__} is fixed problem with {self._ndim} variables. Please setup the correct bounds!")

    @property
    def bounds(self):
        """
        The lower/upper bounds to be used for optimization problem. This a 2D-matrix of [lower, upper] array that contain the lower and upper
        bounds for the problem. The problem should not be asked for evaluation outside these bounds. ``len(bounds) == ndim``.
        """
        return self._bounds

    @property
    def ndim(self):
        """
        The dimensionality of the problem.

        Returns
        -------
        ndim : int
            The dimensionality of the problem
        """
        return self._ndim

    @property
    def lb(self):
        """
        The lower bounds for the problem

        Returns
        -------
        lb : 1D-vector
            The lower bounds for the problem
        """
        return np.array([x[0] for x in self.bounds])

    @property
    def ub(self):
        """
        The upper bounds for the problem

        Returns
        -------
        ub : 1D-vector
            The upper bounds for the problem
        """
        return np.array([x[1] for x in self.bounds])
