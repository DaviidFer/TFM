# Terraform AWS

## Que crea Terraform

La carpeta `infra/terraform` crea una capa cloud minima para el proyecto:

- bucket S3 privado con versioning y cifrado SSE-S3;
- IAM Role e Instance Profile para la EC2;
- Security Group con RDP y Streamlit limitados por CIDR;
- EC2 Windows Server 2022;
- CloudWatch Log Groups y alarmas basicas;
- reglas EventBridge que lanzan scripts PowerShell via SSM Run Command.

## Que no crea

- no instala ni configura MT5 dentro de Windows;
- no crea AWS Batch;
- no crea SNS ni pipelines CI/CD;
- no crea `terraform.tfvars` real ni gestiona secretos.

## Cambiar el tipo de instancia

Edita `instance_type` en `terraform.tfvars`.

## Cambiar la IP permitida

Edita `allowed_rdp_cidr` y `allowed_streamlit_cidr` en `terraform.tfvars`.

## Revisar el plan

```powershell
terraform init
terraform plan
```

## Destroy

```powershell
terraform destroy
```

## Por que no subir tfstate ni tfvars

- `terraform.tfstate` contiene identificadores y detalles reales de infraestructura;
- `terraform.tfvars` puede acabar guardando datos sensibles o configuracion privada;
- ambos deben quedar fuera de Git.

## MT5

MT5 se instala manualmente por RDP. Terraform solo prepara la EC2 Windows y la automatizacion base.

## Alternativa si EventBridge + SSM da problemas

Si el target nativo EventBridge -> SSM Run Command requiere ajustes adicionales en tu cuenta o provider, puedes mantener los mismos scripts PowerShell y programarlos con Windows Task Scheduler dentro de la EC2. La estructura de scripts ya queda preparada para ese fallback.

