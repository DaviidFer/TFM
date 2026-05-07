# Despliegue AWS

## Arquitectura

La capa cloud del proyecto esta pensada para ser minima y defendible academicamente:

- desarrollo principal en local con Cursor;
- GitHub como fuente oficial del codigo;
- Terraform para crear una EC2 Windows, S3, IAM, CloudWatch y automatizacion programada;
- EC2 Windows como servidor unico del sistema;
- SQLite local como estado operativo;
- S3 como repositorio externo de datos, modelos, logs y backups;
- EventBridge + SSM Run Command para lanzar scripts PowerShell dentro de la EC2.

MT5 no se automatiza: se instala manualmente por RDP.

## Flujo recomendado

1. Programar y probar en local con Cursor.
2. Hacer `git push`.
3. Ejecutar Terraform desde local.
4. Entrar por RDP en la EC2.
5. Instalar MT5 manualmente.
6. Ajustar `.env` dentro de la EC2.
7. Arrancar Streamlit y validar el dashboard.
8. Probar backups a S3 y healthcheck.

## Requisitos locales

- AWS CLI
- Terraform
- Git

## Key Pair en AWS

Antes de aplicar Terraform, crea o reutiliza un Key Pair en AWS EC2. Su nombre se pasa en `terraform.tfvars`.

## Preparar terraform.tfvars

1. Copia `infra/terraform/terraform.tfvars.example` a `infra/terraform/terraform.tfvars`.
2. Rellena:
   - region;
   - nombre del proyecto;
   - tipo de instancia;
   - key pair;
   - CIDR de RDP y Streamlit;
   - URL del repo GitHub;
   - bucket S3.

No subas `terraform.tfvars` a Git.

## Comandos Terraform

Desde `infra/terraform`:

```powershell
terraform init
terraform plan
terraform apply
```

## Obtener la IP publica

```powershell
terraform output ec2_public_ip
```

## Entrar por RDP

Usa la IP publica, el usuario `Administrator` y la contrasena desencriptada con tu Key Pair.

## Instalar MT5

Instala MT5 manualmente por RDP y completa en `.env`:

- `MT5_LOGIN`
- `MT5_PASSWORD`
- `MT5_SERVER`
- `MT5_PATH`

## Configurar .env en EC2

La plantilla `.env.example` se copia a `.env` en bootstrap si no existe. Despues ajusta los valores reales dentro de la EC2.

## Ejecutar scripts principales

```powershell
.\scripts\cloud\run_streamlit.ps1
.\scripts\cloud\backup_to_s3.ps1
.\scripts\cloud\healthcheck.ps1
```

## Actualizar el proyecto

Flujo recomendado:

1. commit y push desde local;
2. en la EC2, ejecutar:

```powershell
.\scripts\cloud\deploy_update.ps1
```

Tambien puedes hacer `git pull` manualmente.

## Destruir recursos

```powershell
terraform destroy
```

## Costes

Advertencias de coste:

- Windows en EC2 es sensiblemente mas caro que Linux;
- S3 versioning incrementa almacenamiento;
- CloudWatch Logs y metricas generan coste continuo;
- una instancia mayor puede ser necesaria si el entrenamiento PPO se vuelve pesado.

## Seguridad

- no abras RDP ni Streamlit a `0.0.0.0/0`;
- usa IAM Role en EC2 para S3 y CloudWatch;
- no subas `.env`, `tfstate`, `tfvars`, SQLite, datos ni artefactos a GitHub;
- revisa periodicamente el bucket S3 y las reglas programadas.

## Trabajo futuro: AWS Batch (no implementado)

AWS Batch no forma parte de esta iteracion. El modulo `app.cloud_tasks.develop_asset`
**no existe** todavia y no debe invocarse: queda mencionado aqui solo como
direccion futura. Si en una iteracion posterior se quisiera lanzar tareas
pesadas en paralelo, el patron seria del estilo:

```powershell
# (NO disponible hoy; ejemplo de patron futuro)
# python -m app.cloud_tasks.develop_asset --asset AAPL --family quantile --output-mode s3
# python -m app.cloud_tasks.develop_asset --asset MSFT --family genetic --output-mode s3
```

Ese patron seria util para:

- generacion masiva de reglas;
- validaciones IS/OOS;
- forward tests;
- backtests por activo;
- entrenamiento paralelo por familias.

