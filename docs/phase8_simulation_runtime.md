# Fase 8 - Simulacion funcional preproduccion

Se implementa una fase funcional donde el sistema ya no activa automaticamente cada trader promovido.

## Objetivo

- generar varios traders candidatos sobre un universo de activos,
- mantenerlos en cola en estado `promoted`,
- dejar que `PortfolioManagerAgent` decida cuales activar a `live`,
- conservar trazabilidad completa en eventos y logs.

## Componentes añadidos

- `app/orchestrator/simulation.py`
  - `SimulationRuntime`:
    - construye un pool de candidatos (`Data -> Developer -> Validation`) por activo,
    - hace que `DeveloperAgent` seleccione una sola familia de modelo por activo y sus parametros,
    - define reparto de datos IS/OOS por activo para desarrollo vs validacion,
    - hace que `ValidationAgent` seleccione perfil de validacion (monos IS/OOS, correlacion, forward, estabilidad),
    - publica metricas de scouting pre-live para cada candidato promovido,
    - ejecuta activacion selectiva de top candidatos segun decision del portfolio manager.

- `app/phase8_check.py`
  - runner de escenario funcional:
    - universo multi-activo (`GOOGL`, `AAPL`, `MSFT`),
    - verifica cola `promoted` antes de activar,
    - activa solo subset (`max_live_traders=2`),
    - confirma coexistencia de traders `live` y `promoted`.

- `app/ui/dashboard.py`
  - nuevo boton: `Lanzar simulacion funcional Fase 8`.
  - por defecto carga DB de fase 8 si existe.

## Uso

1. Ejecutar validacion funcional:
   - `python -m app.phase8_check`

2. Ver en dashboard:
   - `python -m streamlit run app/ui/dashboard.py`
   - pulsar `Lanzar simulacion funcional Fase 8`

## Criterio de salida de Fase 8

- se generan multiples traders promovidos en cola,
- portfolio activa solo un subconjunto a `live`,
- el resto permanece en `promoted` esperando decision posterior,
- flujo observable en eventos y logs estructurados.

