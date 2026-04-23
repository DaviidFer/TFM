# Fase 6 - Dashboard operativo

Se añade una interfaz de monitorización para visualizar el estado del sistema:

- estado de agentes,
- estado de traders,
- métricas live de traders,
- eventos recientes,
- resumen por estado y por tipo de evento.

## Componentes añadidos

- `app/ui/dashboard_data.py`
  - carga snapshot desde `StateStore`,
  - construye resumen para la UI.

- `app/ui/dashboard.py`
  - app Streamlit de monitorización en 2 pantallas (`Desarrollo`, `Operativa`),
  - ejecución no bloqueante con supervisor en segundo plano,
  - controles mínimos:
    - iniciar desarrollo de traders,
    - parar desarrollo,
    - borrar traders y reiniciar.

- `app/phase6_check.py`
  - smoke test de carga de snapshot para validar la fase.

## Uso

1. Generar base de ejemplo (si no existe):
   - `python -m app.phase5_check`

2. Verificar fase:
   - `python -m app.phase6_check`

3. Lanzar dashboard:
   - `streamlit run app/ui/dashboard.py`

## Criterio de salida de Fase 6

- La UI muestra correctamente agentes, traders, métricas y eventos.
- La UI permite comprobar visualmente el ciclo cerrado de la fase anterior.

