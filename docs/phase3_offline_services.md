# Fase 3 - Extraccion del pipeline offline a servicios

En esta fase se desacopla el pipeline del notebook para poder ejecutarlo
desde la aplicacion.

## Servicios creados

En `app/services/`:

- `data_service.py` -> carga y normalizacion OHLC
- `feature_service.py` -> wrapper de `build_feature_library` con libreria cerrada de 11 indicadores:
  `Momentum`, `ROC`, `RSI`, `Stoch`, `WPR`, `CCI`, `BullsPower`, `BearsPower`, `DeMarker`, `RVI`, `DPO`
- `split_service.py` -> wrapper de particion IS/OOS + holdout
- `target_service.py` -> wrapper de target por bloques
- `rule_generation_service.py` -> generacion de reglas candidatas por familias
- `validation_service.py` -> validacion IS/OOS, correlacion, forward, estabilidad
- `promotion_service.py` -> construccion de especificacion de trader promovido
- `pipeline_service.py` -> orquestacion offline completa sin notebook

## Runner de comprobacion de fase

Ejecutar:

`python -m app.phase3_check`

Este runner:

1. ejecuta pipeline offline completo sobre `AAPL`,
2. genera reglas candidatas,
3. aplica validaciones,
4. produce resumen de promocion,
5. exporta artefactos en `app/.tmp/phase3/<ASSET>/<EXPERIMENT_ID>/`.

## Artefactos esperados

- `features_tail.csv`
- `candidates_long_*.csv` y `candidates_short_*.csv`
- `decor_long.csv` y `decor_short.csv`
- `forward_long.csv` y `forward_short.csv`
- `stable_long.csv` y `stable_short.csv`
- `dataset_contract.csv`
- `experiment_config.csv`
- `validation_report.csv`
- `promoted_summary.csv`

## Criterio de salida de Fase 3

- Se puede ejecutar un pipeline completo sin notebook.
- Quedan trazados contrato de dataset, configuracion, validacion y promocion.
- Se generan artefactos reproducibles para comparar con el flujo legacy.

## Cobertura de modelos de generacion en Fase 3

La capa de servicios incluye explicitamente estas familias:

- `decision_tree`
- `rulefit`
- `genetico`
- `quantile`
- `subgroup`

Esto permite que el futuro `DeveloperAgent` tenga autonomia para explorar
parametros sobre todos los modelos principales del toolbox, no solo un subconjunto.

