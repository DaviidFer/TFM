# Informe técnico: refactorización `RiskAgent` → `HumanResourcesProcess`

## 1. Nombre final elegido

**`HumanResourcesProcess`** (clase) / `human_resources_process` (módulo, atributo, identificador).

### Justificación

- **Ya no gobierna riesgo de cartera.** Toda la lógica de revisión de pesos, clipping, cash buffer, margen, exposición, drawdown de cuenta y emergency stop se ha eliminado.
- **Ya no es un gate de ejecución.** El runtime live ejecuta directamente la `PortfolioDecision` sin pasar por este componente.
- **No aprende.** Es un proceso determinista basado en métricas, umbrales y reglas. No tiene loop de aprendizaje propio.
- **Su responsabilidad real es supervisar el ciclo de vida de los traders promovidos**: comprobar si siguen funcionando como salieron de fábrica y, si no, mandarlos a reentrenamiento.

Por las mismas razones se descarta el sufijo `Agent` (que sugiere autonomía y aprendizaje) en favor de `Process`. La metáfora correcta es la de un *departamento de Recursos Humanos*: revisa desempeño, decide si el trabajador sigue válido, y si no, gestiona su sustitución.

---

## 2. Archivos eliminados

| Archivo | Motivo |
| --- | --- |
| `app/agents/risk_agent.py` | Sustituido por `app/agents/human_resources_process.py`. Toda la lógica de `review_portfolio_decision`, `assess_trader` y chequeos pre-trade desaparece. |
| `app/services/risk/__init__.py` | Carpeta renombrada a `app/services/trader_health/`. |
| `app/services/risk/risk_metrics.py` | Reemplazada por `app/services/trader_health/metrics.py`. |
| `app/services/risk/health_scoring.py` | Reemplazada por `app/services/trader_health/health_scoring.py`. |
| `app/services/risk/forward_backtest_service.py` | Reemplazada por `app/services/trader_health/forward_backtest_service.py`. |
| `app/tests/test_risk_portfolio_gate.py` | Test del gate de cartera (responsabilidad eliminada). |
| `app/tests/test_live_runtime_risk_integration.py` | Test de integración del gate con el runtime live. |
| `app/tests/test_risk_agent_force_evaluation.py` | Sustituido por `test_human_resources_force_evaluation.py`. |
| `app/tests/test_risk_health_scoring.py` | Sustituido por `test_trader_health_scoring.py`. |
| `app/tests/test_state_store_risk_tables.py` | Sustituido por `test_state_store_trader_review_tables.py`. |
| `app/tests/test_design_risk_profile_serialization.py` | Sustituido por `test_trader_design_profile_serialization.py`. |

---

## 3. Archivos renombrados / nuevos

| Antes | Ahora |
| --- | --- |
| `app/agents/risk_agent.py` | `app/agents/human_resources_process.py` |
| `app/services/risk/` | `app/services/trader_health/` |
| `app/services/risk/risk_metrics.py` | `app/services/trader_health/metrics.py` |
| `app/services/risk/health_scoring.py` | `app/services/trader_health/health_scoring.py` |
| `app/services/risk/forward_backtest_service.py` | `app/services/trader_health/forward_backtest_service.py` |
| `app/tests/test_risk_health_scoring.py` | `app/tests/test_trader_health_scoring.py` |
| `app/tests/test_risk_agent_force_evaluation.py` | `app/tests/test_human_resources_force_evaluation.py` |
| `app/tests/test_state_store_risk_tables.py` | `app/tests/test_state_store_trader_review_tables.py` |
| `app/tests/test_design_risk_profile_serialization.py` | `app/tests/test_trader_design_profile_serialization.py` |

---

## 4. Archivos modificados

- `app/contracts/enums.py`
- `app/contracts/models.py`
- `app/contracts/__init__.py`
- `app/agents/__init__.py`
- `app/storage/state_store.py`
- `app/runtime/live_trading_runtime.py`
- `app/runtime/development_operational_supervisor.py`
- `app/execution/access.py`
- `app/ui/dashboard.py`
- `app/ui/dashboard_data.py`
- `app/cloud_tasks/monthly_refresh.py`
- `app/phase5_check.py`
- `app/phase8_check.py`
- `app/phase9_check.py`
- `app/tests/test_forward_backtest_service_normalization.py`

---

## 5. Funciones / métodos eliminados

Del antiguo `RiskAgent`:

- `review_portfolio_decision(...)` — gate pre-trade sobre `PortfolioDecision`.
- `assess_trader(...)` — evaluación legacy basada en `trader_metrics_latest`.
- `get_broker_account_info(...)` — exposición de la cuenta del broker.
- Helpers internos: `_check_account_drawdown`, `_check_margin`, `_apply_emergency_stop`, `_apply_clipping`, `_apply_force_cash`, `_apply_scale_down`, `_compose_adjusted_decision`, etc.
- Toda referencia a `forced_cash_weight`, `blocked_traders`, `clipped_traders`, `account_info`, `open_positions`.

Del `LiveTradingRuntime`:

- Parámetro `risk_agent` en `__init__`.
- Llamadas a `self.risk_agent.review_portfolio_decision(...)`.
- Variables `training_run_id`, `model_version` propagadas a `_attempt_order` y `_audit_signal`.
- Snapshot intermedio `risk_review_result` y branches de `RiskAction` (`SCALE_DOWN`, `FORCE_CASH`, `EMERGENCY_STOP`, `REJECT_PORTFOLIO`, `APPROVE_WITH_CLIPPING`).

Del `DevelopmentOperationalSupervisor`:

- `run_risk_monthly_evaluation` → `run_trader_health_monthly_evaluation`.
- `force_risk_evaluation` → `force_trader_health_evaluation`.
- Atributo `self.risk_agent` → `self.human_resources_process`.
- Status fields `risk_last_*` → `trader_review_last_*`.

Del `state_store`:

- `save_risk_portfolio_check`, `list_risk_portfolio_checks`.
- `create_risk_evaluation_run`, `complete_risk_evaluation_run`, `list_risk_evaluation_runs` → renombrados a `create_trader_review_run`, `complete_trader_review_run`, `list_trader_review_runs`.
- `save_risk_evaluation_detail`, `list_risk_evaluation_details` → renombrados a `save_trader_review_detail`, `list_trader_review_details`.

---

## 6. Contratos eliminados o simplificados

### Eliminados

- `RiskDecision` (sustituido por `TraderHealthSnapshot`).
- `RiskAdjustedPortfolioDecision` (no existe gate pre-trade).
- Acciones de cartera del antiguo `RiskAction`: `APPROVE`, `APPROVE_WITH_CLIPPING`, `SCALE_DOWN`, `FORCE_CASH`, `REJECT_PORTFOLIO`, `EMERGENCY_STOP`.
- `RiskThresholds` (parametrización dispersa absorbida por `TraderHealthConfig`).

### Renombrados / simplificados

| Antes | Ahora |
| --- | --- |
| `AgentKind.RISK` | `AgentKind.HUMAN_RESOURCES` |
| `EventType.RISK_DECISION` | `EventType.TRADER_HEALTH_EVALUATED` |
| `RiskAction` (8 valores) | `TraderReviewAction` (2 valores: `KEEP`, `RETRAINING`) |
| `DesignRiskProfile` | `TraderDesignProfile` |
| `RiskLimitsConfig` | `TraderHealthConfig` (sin campos de cartera: solo `min_trades_for_evidence`, thresholds de health score, ratios vs baseline, multiplicadores de drawdown) |
| `TraderForwardMetrics.ppo_selected_count` | `TraderForwardMetrics.pm_selected_count` |
| `TraderForwardMetrics.ppo_blocked_count` | (eliminado) |
| `TraderForwardMetrics.risk_blocked_count` | (eliminado) |

`TraderHealthSnapshot` se mantiene como contrato de evaluación pero con campos coherentes con el nuevo alcance (`action ∈ {KEEP, RETRAINING}`, sin información sobre cartera).

---

## 7. Tablas SQLite

### Eliminadas

- `risk_portfolio_checks` (la tabla se borra de forma defensiva con `DROP TABLE IF EXISTS` durante el bootstrap del `StateStore` para BBDD antiguas).

### Renombradas in-place

| Antes | Ahora |
| --- | --- |
| `risk_evaluation_runs` | `trader_review_runs` (`ALTER TABLE ... RENAME TO`) |
| `risk_evaluation_details` | `trader_review_details` (`ALTER TABLE ... RENAME TO`) |

### Mantenidas

- `trader_design_profiles` — perfiles de diseño persistidos.
- `trader_forward_backtest_runs` — runs de backtest forward.
- `trader_forward_metrics` — métricas forward por trader.
- `retrain_requests` — cola de reentrenamiento.
- `trader_signal_audit` — auditoría por señal. Se añaden columnas `pm_selected` y `pm_weight`. Las columnas legacy `ppo_selected`, `ppo_weight` y `risk_approved` se conservan para no romper esquemas antiguos pero el código nuevo no las utiliza semánticamente (se rellenan con valores neutros).

---

## 8. Tests

### Eliminados

- `test_risk_portfolio_gate.py`
- `test_live_runtime_risk_integration.py`

### Reemplazados (mismo objetivo, nueva semántica)

- `test_risk_health_scoring.py` → `test_trader_health_scoring.py`
- `test_risk_agent_force_evaluation.py` → `test_human_resources_force_evaluation.py`
- `test_state_store_risk_tables.py` → `test_state_store_trader_review_tables.py`
- `test_design_risk_profile_serialization.py` → `test_trader_design_profile_serialization.py`

### Adaptados

- `test_forward_backtest_service_normalization.py` (cambia el path de import a `app.services.trader_health.forward_backtest_service`).

### Resultado

- 44 tests verdes en el árbol completo (`pytest app/tests/`).
- Sin referencias activas a `RiskAgent`, `RiskAction`, `RiskDecision`, `DesignRiskProfile`, `RiskLimitsConfig`, `RiskAdjustedPortfolioDecision` ni `app.services.risk` en el código vivo.

---

## 9. Resumen de la arquitectura final

```
PortfolioManagerProcess  ──► PortfolioDecision  ──► LiveTradingRuntime ──► Broker
                                                       (sin gate)

HumanResourcesProcess  (mensual / forzado por supervisor)
   │
   ├─► carga traders promovidos
   ├─► reconstruye o lee TraderDesignProfile
   ├─► ejecuta ForwardBacktestService → TraderForwardMetrics
   ├─► evaluate_trader_health(profile, metrics, config) → TraderHealthSnapshot
   │     └─► action ∈ {KEEP, RETRAINING}
   ├─► persiste trader_review_run + trader_review_details
   ├─► si RETRAINING: actualiza estado del trader y emite RetrainRequest
   └─► emite EventType.TRADER_HEALTH_EVALUATED
```

Una frase: *“`HumanResourcesProcess` compara el comportamiento forward del trader con su perfil de diseño y decide si sigue válido o si debe pasar a reentrenamiento.”* Nada más.

---

## 10. Sobrearquitectura eliminada

- Doble responsabilidad del antiguo `RiskAgent` (evaluación de traders + gate de cartera) descompuesta a una única responsabilidad clara.
- Enum `RiskAction` con 8 valores → `TraderReviewAction` con 2 valores. Los 6 valores eliminados (`APPROVE`, `APPROVE_WITH_CLIPPING`, `SCALE_DOWN`, `FORCE_CASH`, `REJECT_PORTFOLIO`, `EMERGENCY_STOP`) cubrían un dominio (cartera) que ya no es responsabilidad del componente.
- Contrato `RiskAdjustedPortfolioDecision` desaparece: era una capa intermedia entre `PortfolioDecision` y la ejecución que añadía complejidad sin valor real (el PM ya decide la cartera).
- `RiskLimitsConfig` mezclaba parámetros de cartera (max_weight_per_trader, max_weight_per_asset, min_cash_buffer, max_total_exposure, account_drawdown, broker margin) con parámetros de salud. Se ha reducido a `TraderHealthConfig` con solo los parámetros de salud del trader.
- Tabla `risk_portfolio_checks` y todo su CRUD eliminados.
- Bloque de UI de "Portfolio Risk Checks" eliminado del dashboard.
- Lógica de propagación de `training_run_id` y `model_version` por el runtime live (era residuo del modelo PPO y del gate Risk) eliminada.
- Métodos legacy del `RiskAgent` (`assess_trader`, `get_broker_account_info`) eliminados: ya no había una vía coherente que los justificase.
- Estados de ciclo de vida intermedios (`DEGRADED`, `SUSPENDED`, `RETIRED`) ya no son producidos por este proceso. El `HumanResourcesProcess` solo trabaja con la pareja `LIVE`/`RETRAINING` desde su punto de vista.

---

## 11. Comprobación de limpieza

Búsqueda en `app/**/*.py`:

```
RiskAgent | risk_agent | RiskAction | RiskDecision | RiskAdjustedPortfolioDecision
| DesignRiskProfile | RiskLimitsConfig | review_portfolio_decision | app\.services\.risk
```

Solo dos ocurrencias residuales, ambas en **comentarios** que documentan la migración (no son código vivo):

- `app/phase5_check.py` línea 34 — docstring del módulo: *"`HumanResourcesProcess`. Se simula la decision del HumanResourcesProcess… sin depender del antiguo `RiskAgent`"*.
- `app/storage/state_store.py` línea 327 — comentario sobre el `DROP TABLE IF EXISTS risk_portfolio_checks` defensivo.

Ambas referencias son pedagógicas y no implican lógica activa.

Resultados de pruebas: **44 passed**.

`access.py`: `ALLOWED_EXECUTION_ACTORS` ya no incluye `"risk_agent"`. El `LiveTradingRuntime` no recibe ni instancia ningún `RiskAgent`.

---

## 12. Deuda técnica residual

1. **Documentación heredada en `docs/risk_agent_memoria_tecnica.md`**: el archivo se mantiene tal cual porque es la memoria académica original entregada en el TFM. La nueva responsabilidad debería redactarse en un documento separado (`docs/human_resources_process_memoria_tecnica.md`) sin modificar el histórico. *Acción recomendada*: redactar la nueva memoria técnica del componente y dejar la antigua marcada como "estado anterior".

2. **Columnas legacy en `trader_signal_audit`**: `ppo_selected`, `ppo_weight` y `risk_approved` siguen existiendo a nivel SQL para no romper esquemas antiguos. El código nuevo las rellena con valores neutros (`pm_selected`/`pm_weight` reales y `risk_approved=1`). *Acción recomendada (opcional)*: una migración que las elimine en una versión mayor de BBDD; mientras tanto, el coste de mantenerlas es nulo.

3. **`infra/terraform/locals.tf`** sigue declarando una entrada `risk_agent = "/tfm/trading/risk-agent"` en SSM Parameters. No afecta al runtime Python pero conviene renombrarla a `human_resources_process = "/tfm/trading/human-resources-process"` en el siguiente despliegue de infraestructura.

4. **Métricas y enums sobre cartera en `TraderForwardMetrics`**: tras eliminar `ppo_blocked_count` y `risk_blocked_count`, `pm_selected_count` queda como única contribución de la cartera al perfil forward. Si en el futuro el PM decidiera ejecutar un esquema más rico (p. ej. distintos modos de selección), conviene revisar si este campo merece evolucionar a un sub-objeto en lugar de un escalar.

5. **No queda deuda técnica significativa fuera de lo anterior**. La superficie pública del componente es pequeña (`HumanResourcesProcess`, `evaluate_trader_health`, `ForwardBacktestService`, `build_trader_design_profile`), y todas las dependencias inversas (supervisor, dashboard, runtime, tests, cloud tasks) están alineadas con la nueva semántica.
