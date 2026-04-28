# Modulos cloud

`app/cloud` agrupa la capa opcional de integracion con AWS sin tocar la logica central de trading.

## Modulos

- `cloud_config.py`
  Carga configuracion desde variables de entorno y devuelve `CloudConfig`.
- `cloud_paths.py`
  Centraliza los prefijos S3 usados por backups, logs y artefactos.
- `s3_storage.py`
  Wrapper ligero sobre `boto3` para subir/bajar ficheros y directorios.
- `heartbeat.py`
  Genera un heartbeat JSON local y, si S3 esta activo, lo publica en S3.

## Activar S3

1. Define `TFM_ENABLE_S3=true`.
2. Define `TFM_S3_BUCKET=<bucket>`.
3. Opcionalmente define `TFM_S3_PREFIX=tfm-trading`.
4. En EC2 se recomienda usar IAM Role en lugar de access keys.

## Ejecutar heartbeat

```powershell
python -m app.cloud.heartbeat
```

## Ejecutar smoke test

```powershell
python -m app.cloud_tasks.smoke_test_cloud
```

## Compatibilidad local

Si AWS no esta configurado, los modulos cloud siguen funcionando en modo local:

- no fallan por ausencia de bucket salvo que se intente usar S3;
- mantienen las rutas locales del proyecto;
- permiten validar configuracion y heartbeat sin infraestructura cloud.

