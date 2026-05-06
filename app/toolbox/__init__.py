"""Toolbox interno reutilizable.

Agrupa los módulos cuantitativos de bajo nivel que el resto de servicios y
agentes consumen:

- ``particion_IS_OOS``: split temporal IS/OOS/holdout.
- ``definicion_target``: aplicación del target a los bloques.
- ``indicators``: librería de features técnicas.
- ``data_download``: mantenimiento del universo y descargas masivas.
- ``backtest_eventos``: motor de backtest histórico (envoltura sobre pyeventbt).
- ``ML_tools``: generadores de reglas (decision tree, rulefit, genético, quantile).

Estos paquetes vivían antes en la raíz del repositorio. Se han reubicado bajo
``app/toolbox`` para mantener una estructura coherente y autocontenida.
"""
