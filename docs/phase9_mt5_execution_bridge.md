# Fase 9 - Integracion MT5 Execution Bridge

Se integra una capa de ejecución inspirada en `mt5-framework`, manteniendo aislamiento y control de acceso.

## Objetivo

- reutilizar lo útil del framework de operativa real sin acoplar el sistema al flujo forex original,
- sustituir la captura de datos MT5 por datos locales de `datos/Stocks` y `datos/ETFs`,
- restringir acceso a ejecución únicamente a `PortfolioManagerAgent`, `RiskAgent` y `TraderAgent`.

## Componentes añadidos

- `app/execution/models.py`
  - contratos de ejecución (`OrderIntent`, `OrderResult`, `ExecutionMode`).

- `app/execution/access.py`
  - guard de seguridad con actores permitidos.

- `app/execution/local_data_provider.py`
  - proveedor de mercado basado en CSV local (Stocks/ETFs), reemplaza data fetch forex de MT5.

- `app/execution/mt5_data_provider.py`
  - integración del patrón `data_provider` del framework MT5 para detectar nueva vela cerrada.
- `app/execution/mt5_events.py`
  - contratos de eventos de mercado/señal compatibles con el flujo de framework.

- `app/execution/mt5_connector.py`
  - adaptador mínimo de conexión y envío de órdenes MT5.
  - carga credenciales desde `.env`.

- `app/execution/router.py`
  - router único de ejecución:
    - modo `paper` (simulado),
    - modo `live_mt5` (real, conector MT5).

- `app/agents/base.py`
  - `AgentContext` amplía con `execution_router`.

- `app/agents/trader_agent.py`
  - `route_order(...)` para enrutado controlado al broker/router.

- `app/agents/portfolio_agent.py`
  - lectura de snapshot de ejecución y posiciones abiertas.

- `app/agents/risk_agent.py`
  - lectura de account info y posiciones del broker/router.

- `app/phase9_check.py`
  - escenario de validación de la integración:
    - genera/promueve/activa traders,
    - enruta órdenes en modo `paper`,
    - verifica control de acceso (data agent denegado),
    - comprueba trazabilidad en eventos.

## Seguridad de acceso

- Permitidos:
  - `portfolio_manager_agent`
  - `risk_agent`
  - `trader_agent`
- Denegados:
  - `data_agent`
  - `developer_agent`
  - `validation_agent`

## Uso

1. Revisar/ajustar `.env` (copiado del framework y editable por usuario).
2. Ejecutar validación:
   - `python -m app.phase9_check`
3. Dashboard:
   - `python -m streamlit run app/ui/dashboard.py`
   - botón `Lanzar integracion ejecución Fase 9`

