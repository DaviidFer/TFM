# Fase 2 - Contratos y estado compartido

Esta fase define el lenguaje comun entre agentes y un almacenamiento minimo
persistente para estados y eventos.

## Contratos definidos

En `app/contracts/models.py`:

- `DatasetContract`
- `ExperimentConfig`
- `CandidateRules`
- `ValidationReport`
- `PromotedTraderSpec`
- `TraderLiveMetrics`
- `PortfolioDecision`
- `RiskDecision`
- `RetrainRequest`
- `EventRecord`

En `app/contracts/enums.py`:

- `AgentKind`
- `AgentStatus`
- `TraderLifecycleState`
- `EventType`

## Estado compartido inicial

Se implementa `StateStore` en `app/storage/state_store.py` con SQLite:

- `trader_states`: estado actual del ciclo de vida por trader
- `agent_status`: estado operativo por agente
- `events`: eventos con payload JSON

Objetivo: ofrecer persistencia simple y trazable sin complejidad distribuida en v1.

## Simulaciones requeridas de Fase 2

`python -m app.phase2_check`

Valida:

1. serializacion de contratos,
2. transicion de ciclo:
   `candidate -> validated -> promoted -> live`,
3. transicion de retiro a regeneracion:
   `retired -> retraining`,
4. persistencia de un evento `retrain_requested`.

## Criterio de salida de Fase 2

- Contratos versionados y reutilizables por los siguientes agentes.
- Estados del ciclo de vida definidos y persistidos.
- Store de eventos funcional para orquestacion posterior.

