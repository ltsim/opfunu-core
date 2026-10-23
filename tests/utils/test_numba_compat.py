import subprocess
import sys

from opfunu.utils import numba_compat
from opfunu.utils.numba_compat import njit


def test_njit_is_passthrough_without_numba(monkeypatch):
    monkeypatch.setattr(numba_compat, "HAS_NUMBA", False)

    def f(x):
        return x + 1

    assert njit(f) is f

    @njit(fastmath=True)
    def g(x):
        return x + 2

    assert g(1) == 3


def test_njit_decorated_function_computes():
    @njit(fastmath=True)
    def f(x):
        return x + 1

    assert f(1) == 2


def test_import_and_evaluate_without_numba():
    code = (
        "import sys; "
        "sys.modules['numba'] = None; "
        "import numpy as np; "
        "import opfunu; "
        "assert opfunu.HAS_NUMBA is False, 'HAS_NUMBA should be False without numba'; "
        "f = opfunu.name_based.Ackley01(ndim=2); "
        "assert f.numba_compiled is False; "
        "assert f._compile_error is None; "
        "print(f.evaluate(np.ones(2)))"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, f"no-numba smoke test failed:\n{proc.stdout}\n{proc.stderr}"
