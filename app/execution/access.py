from __future__ import annotations


ALLOWED_EXECUTION_ACTORS = {
    "portfolio_manager_agent",
    "portfolio_manager",
    "risk_agent",
    "trader_agent",
}


def ensure_execution_access(actor: str) -> None:
    if actor not in ALLOWED_EXECUTION_ACTORS:
        raise PermissionError(
            f"Actor '{actor}' no autorizado para la capa de ejecución. "
            f"Permitidos: {sorted(ALLOWED_EXECUTION_ACTORS)}"
        )

