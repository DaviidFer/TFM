# Cloud scripts

Los scripts de `scripts/cloud` estan pensados para ejecutarse manualmente por RDP o de forma remota por SSM Run Command.

## Resolucion de la ruta del proyecto

Los scripts cargan `Resolve-TfmProjectDir.ps1` y usan `Get-TfmProjectDir`. Buscan el repo validando `requirements.txt` y `.git`, en este orden:

1. Variable de entorno `TFM_PROJECT_DIR` (si apunta a un directorio valido).
2. **`C:\tfm\tfm-project-gitpublic`** (si existe, tiene prioridad).
3. **`C:\tfm\tfm-project`** (valor clasico del bootstrap en EC2).
4. Variantes conocidas (`C:\tfm-trading`, etc.) y subcarpetas directas bajo `C:\tfm`.

Por tanto en EC2 puedes ejecutar los `.ps1` **desde cualquier directorio**, siempre que el repo exista en una de esas rutas. Ademas, al resolverlo exportan al proceso actual `TFM_PROJECT_DIR`, `TFM_ARTIFACTS_ROOT` y `TFM_DB_PATH` para que Python use exactamente la misma ruta.

## Scripts principales

- `bootstrap_windows_ec2.ps1`
  Bootstrap inicial de una EC2 Windows: instala herramientas base, clona el repo, crea `.env` y prepara `.venv`.
- `deploy_update.ps1`
  Hace `git pull`, actualiza dependencias y ejecuta `python -m app.cloud_tasks.smoke_test_cloud`.
- `run_streamlit.ps1`
  Arranca el dashboard Streamlit sobre `app/ui/dashboard.py`.
- `run_runtime.ps1`
  Ejecuta el wrapper `python -m app.cloud_tasks.run_runtime` para mantener el supervisor/runtime en marcha.
- `run_daily_update.ps1`
  Ejecuta `python -m app.cloud_tasks.daily_update` y despues `backup_to_s3.ps1`.
- `run_weekly_rebalance.ps1`
  Ejecuta `python -m app.cloud_tasks.weekly_rebalance` y despues `backup_to_s3.ps1`.
- `run_monthly_refresh.ps1`
  Ejecuta `python -m app.cloud_tasks.monthly_refresh` y despues `backup_to_s3.ps1`.
- `backup_to_s3.ps1`
  Sube SQLite, `datos/`, `app/.tmp/` y logs a S3.
- `sync_from_s3.ps1`
  Descarga datos y artefactos principales desde S3, util al recrear la EC2.
- `sync_to_s3.ps1`
  Sincroniza datos, artefactos y logs hacia S3.
- `healthcheck.ps1`
  Ejecuta `python -m app.cloud.heartbeat`.
- `install_cloudwatch_agent.ps1`
  Instala/configura Amazon CloudWatch Agent para logs y metricas basicas.

## Ejecucion

Ejemplos desde la raiz del repo:

```powershell
.\scripts\cloud\deploy_update.ps1
.\scripts\cloud\run_streamlit.ps1
.\scripts\cloud\run_runtime.ps1
.\scripts\cloud\run_monthly_refresh.ps1
.\scripts\cloud\backup_to_s3.ps1
.\scripts\cloud\healthcheck.ps1
```

## Logs

Todos los scripts escriben transcript local bajo `app/.tmp/logs`.

## Uso recomendado

- `bootstrap_windows_ec2.ps1`: una sola vez al crear o reconstruir la EC2.
- `deploy_update.ps1`: despues de cada `git push`.
- `run_streamlit.ps1`: cuando quieras exponer el dashboard.
- `run_runtime.ps1`: para mantener el runtime activo en segundo plano o en una sesion dedicada.
- `run_daily_update.ps1`, `run_weekly_rebalance.ps1`, `run_monthly_refresh.ps1`: para automatizacion programada.
- `backup_to_s3.ps1` y `sync_from_s3.ps1`: para resiliencia operativa y recuperacion.

