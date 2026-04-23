# Fase 10 - Live MVP sin Portfolio/Risk

Se integra el flujo operativo solicitado:

- `DataAgent` -> `DeveloperAgent` -> `ValidationAgent` generan traders,
- los traders promovidos pasan a `TraderAgent`,
- `PortfolioManagerAgent` y `RiskAgent` permanecen desactivados temporalmente,
- ejecución y datos en vivo se apoyan en componentes tomados de `mt5-framework`.

## Integración de mt5-framework utilizada

- concepto de `data_provider` de MT5 para detectar nueva vela cerrada:
  - `app/execution/mt5_data_provider.py`
- eventos de framework adaptados:
  - `app/execution/mt5_events.py`
- loop operacional:
  - `app/runtime/live_trading_runtime.py`

## Ejecución desde UI (actual)

Se usa `DevelopmentOperationalSupervisor` en `app/runtime/development_operational_supervisor.py`:

1. desarrollo continuo no bloqueante (`Data -> Developer -> Validation -> Trader`),
2. selección autónoma de activo (cualquier CSV de `datos/Stocks` o `datos/ETFs`),
3. selección autónoma de modelo/parámetros/split/validación,
4. conexión a MT5 solo cuando existen al menos 5 traders desplegados,
5. una vez conectado, runtime operativo D1 escucha `DataEvent` y ejecuta señales.

## Modo de ejecución

Nota:

- el `data_provider` operativo consume datos desde la API de MT5 (D1).
- `PortfolioManagerAgent` y `RiskAgent` quedan fuera del supervisor actual.

