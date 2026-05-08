"""Compatibilidad mantenida por compatibilidad hacia atras.

Usar `app.core.numpy_numba_abi` en nuevo codigo para evitar imports circulares.
"""

from app.core.numpy_numba_abi import numpy_numba_abi_fail_message, numpy_too_new_for_numba_stack

__all__ = ["numpy_too_new_for_numba_stack", "numpy_numba_abi_fail_message"]
