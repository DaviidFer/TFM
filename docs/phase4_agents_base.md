# Fase 4 - Agentes base e integración

En esta fase se implementan los cuatro agentes iniciales y se valida su
interaccion secuencial.

## Agentes implementados

- `DataAgent` (`app/agents/data_agent.py`)
- `DeveloperAgent` (`app/agents/developer_agent.py`)
- `ValidationAgent` (`app/agents/validation_agent.py`)
- `TraderAgent` (`app/agents/trader_agent.py`)

Soporte comun:

- `AgentContext` (`app/agents/base.py`)
- export de agentes (`app/agents/__init__.py`)

## Flujo integrado de Fase 4

1. `DataAgent` prepara `DatasetContract` y emite `dataset_ready`.
2. `DeveloperAgent` genera candidatos multi-familia y emite `candidate_rules_ready`.
3. `ValidationAgent` valida, promueve trader y emite:
   - `validation_completed`
   - `trader_promoted`
4. `TraderAgent` activa el trader y emite `trader_state_changed` a `live`.

## Runner de comprobación

`python -m app.phase4_check`

Valida:

- dataset preparado,
- reglas candidatas no vacias,
- promocion de trader no vacia,
- estado final `LIVE`,
- eventos persistidos en `StateStore`.

## Que hace cada agente (v1)

- `DataAgent`: convierte una fuente de datos a contrato consumible por los demás.
- `DeveloperAgent`: explora familias y parametros para generar reglas candidatas.
- `ValidationAgent`: filtra y promociona especificaciones de trader.
- `TraderAgent`: activa especificaciones aprobadas y registra heartbeat inicial.

## Criterio de salida de Fase 4

- Los cuatro agentes funcionan juntos en flujo integrado.
- El estado de trader transita hasta `live`.
- Se persisten eventos y estados en el almacenamiento compartido.

