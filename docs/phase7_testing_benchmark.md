# Fase 7 - Bateria de pruebas y benchmark

Se incorpora una capa de validacion automatica para asegurar estabilidad del ciclo cerrado.

## Objetivo

- tener pruebas repetibles para contratos y estado compartido,
- validar integracion minima del flujo (fase 5 + dashboard snapshot),
- medir tiempo de ejecucion del ciclo base como benchmark inicial.

## Componentes añadidos

- `app/tests/test_contracts_state_store.py`
  - test unitario de serializacion de `PromotedTraderSpec`,
  - test roundtrip de `StateStore` (estado, metricas y eventos).

- `app/tests/test_dashboard_snapshot.py`
  - test de integracion de snapshot tras ejecutar `phase5_check`.

- `app/tests/test_runtime_logs.py`
  - verifica que el log estructurado contiene eventos clave del pipeline.

- `app/phase7_check.py`
  - runner de Fase 7,
  - ejecuta tests (unitarios + integracion),
  - ejecuta benchmark simple de `phase5_check`.

## Uso

Ejecutar:

- `python -m app.phase7_check`

Salida esperada:

- cada prueba con estado `OK/FAIL`,
- tiempo por prueba,
- tiempo del benchmark,
- contador total y fallos.

## Criterio de salida de Fase 7

- todos los tests pasan,
- benchmark del ciclo base se ejecuta sin errores,
- existe una base de control para detectar regresiones funcionales y de rendimiento.

