# Fase 1 - Congelar toolbox y dominio

Este documento formaliza la primera fase de migracion del proyecto a aplicacion modular.

## Decisiones congeladas

- Dominio inicial: `acciones y ETFs` sobre datos en `datos/Stocks`.
- Temporalidad inicial: `D1`.
- El notebook `NOTEBOOK_CONSTRUCTION.ipynb` se mantiene como:
  - laboratorio de pruebas,
  - entorno de debugging,
  - referencia de comparacion.
- El notebook **no** es runtime oficial de la aplicacion.

## Toolbox congelado (fuente de verdad tecnica)

Las etapas congeladas en Fase 1 son:

1. features (`indicators.build_feature_library`) con 11 indicadores cerrados:
   `Momentum`, `ROC`, `RSI`, `Stoch`, `WPR`, `CCI`, `BullsPower`, `BearsPower`, `DeMarker`, `RVI`, `DPO`
2. split IS/OOS + holdout (`particion_IS_OOS.run_particion_is_oos`)
3. target (`definicion_target.run_target_para_bloques`)
4. reglas cuantiles (`ML_tools.build_quantile_bin_combinations`)
5. reglas arbol (`ML_tools.build_decision_tree_rules_multiseed`)
6. reglas rulefit (`ML_tools.build_rulefit_rules_multiseed`)
7. reglas genetico (`ML_tools.run_genetico_rules`)
8. reglas subgroup (`ML_tools.run_subgroup_discovery_rules`)
9. validacion monos (`validacion_monos.monkey_validate_oos_multi`)
10. pruning correlacion (`validacion_correlacion_pl.run_pl_correlation_pruning`)
11. validacion forward (`validacion_forward.validate_forward_year_profitability`)
12. estabilidad (`validacion_estabilidad_pl.run_pl_stability_selection`)
13. ejecucion reglas (`backtest_eventos.run_event_backtest`)

## Check automatizable de fase

Ejecutar:

`python -m app.phase1_check`

El check valida:

- que existan los archivos base del toolbox,
- que existan activos de muestra para pruebas de migracion (`AAPL`, `MSFT`, `NVDA`).

## Criterio de salida de Fase 1

- Dominio de la aplicacion fijado.
- Toolbox base inventariado y congelado.
- Checklist automatico de precondiciones en verde.

