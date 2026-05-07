from __future__ import annotations


ALLOWED_EXECUTION_ACTORS = {
    "portfolio_manager_process",
    "portfolio_manager_agent",
    "portfolio_manager",
    "trader_agent",
}


def ensure_execution_access(actor: str) -> None:
    if actor not in ALLOWED_EXECUTION_ACTORS:
        raise PermissionError(
            f"Actor '{actor}' no autorizado para la capa de ejecucion. "
            f"Permitidos: {sorted(ALLOWED_EXECUTION_ACTORS)}"
        )
