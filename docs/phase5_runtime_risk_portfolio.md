# Fase 5 - Runtime, riesgo, portfolio y regeneracion

En esta fase se cierra el bucle operativo basico:

- decision de riesgo sobre traders live,
- solicitud de retraining al detectar degradacion,
- procesamiento de solicitudes y creacion de reemplazo,
- rebalanceo v1 con portfolio manager.

## Componentes añadidos

### Agentes nuevos

- `RiskAgent` (`app/agents/risk_agent.py`)
  - evalua metricas por trader,
  - decide `keep/suspend/retire`,
  - emite `risk_decision`,
  - si retira, emite `retrain_requested` y transita trader a `retraining`.

- `PortfolioManagerAgent` (`app/agents/portfolio_agent.py`)
  - consume metricas live y estados elegibles,
  - calcula score interpretable (sharpe/pnl vs drawdown/correlacion),
  - asigna pesos con cap (`max_weight`),
  - emite `portfolio_decision`.

### Orquestacion

- `RuntimeOrchestrator` (`app/orchestrator/runtime.py`)
  - detecta eventos `retrain_requested`,
  - ejecuta ciclo `Data -> Developer -> Validation -> Trader`,
  - emite `retrain_processed`.

### Logging estructurado de flujo

- `app/core/structured_logging.py`
  - emite JSONL con `ts_utc`, `component`, `event` y campos de contexto,
  - escribe en consola y en `app/.tmp/logs/runtime_flow.log`.
- Se instrumentan `DataAgent`, `DeveloperAgent`, `ValidationAgent`, `TraderAgent`,
  `RiskAgent`, `PortfolioManagerAgent` y `RuntimeOrchestrator`.
- `phase5_check` limpia el log al inicio y deja traza completa del ciclo.

### Estado compartido ampliado

En `app/storage/state_store.py`:

- nueva tabla `trader_metrics_latest`,
- métodos para `upsert/get/list` de métricas,
- listado de estado de agentes.

### Extensiones de eventos

En `app/contracts/enums.py`:

- `trader_metrics_updated`
- `portfolio_decision`
- `retrain_processed`

## Runner de comprobacion

`python -m app.phase5_check`

Escenario validado:

1. se crea trader live,
2. se inyectan métricas degradadas,
3. riesgo decide `retire` y solicita retraining,
4. orquestador procesa solicitud y crea nuevo trader live,
5. portfolio manager calcula pesos sobre el estado actualizado.

## Criterio de salida de Fase 5

- Existe ciclo cerrado básico con reemplazo automático de trader.
- Riesgo y portfolio emiten decisiones trazables.
- Orquestación consume eventos y reabre pipeline de desarrollo.

