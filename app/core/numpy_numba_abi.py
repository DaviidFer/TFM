"""Compatibilidad NumPy ↔ Numba/PyEventBT usada en backtests basados en PyEventBT."""

from __future__ import annotations


def numpy_too_new_for_numba_stack() -> bool:
    """Numba 0.62.x falla en JIT si NumPy >= 2.4 (mensaje: needs NumPy 2.3 or less)."""
    try:
        import numpy as np

        parts = np.__version__.split(".")
        maj, minor = int(parts[0]), int(parts[1])
        return maj > 2 or (maj == 2 and minor >= 4)
    except Exception:
        return False


def numpy_numba_abi_fail_message() -> str | None:
    if not numpy_too_new_for_numba_stack():
        return None
    try:
        import numpy as np

        ver = np.__version__
    except Exception:
        ver = "?"
    return (
        f"NumPy {ver} no es compatible con Numba/PyEventBT del backtest. "
        'Instale: python -m pip install "numpy>=1.24,<=2.3.5"'
    )
