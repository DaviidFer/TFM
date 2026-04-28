[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"

function Get-ProjectDir {
    if ($env:TFM_PROJECT_DIR) {
        return $env:TFM_PROJECT_DIR
    }
    return (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
}

function Get-S3Uri([string]$Bucket, [string]$Prefix, [string]$Suffix) {
    $cleanPrefix = $Prefix.Trim("/\")
    $cleanSuffix = $Suffix.Trim("/\")
    if ([string]::IsNullOrWhiteSpace($cleanPrefix)) {
        return "s3://$Bucket/$cleanSuffix"
    }
    return "s3://$Bucket/$cleanPrefix/$cleanSuffix"
}

$projectDir = Get-ProjectDir
$logDir = Join-Path $projectDir "app\.tmp\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$logFile = Join-Path $logDir "sync_to_s3.log"

Start-Transcript -Path $logFile -Append | Out-Null
try {
    $bucket = $env:TFM_S3_BUCKET
    if ([string]::IsNullOrWhiteSpace($bucket)) {
        throw "TFM_S3_BUCKET no esta definido."
    }
    $prefix = if ($env:TFM_S3_PREFIX) { $env:TFM_S3_PREFIX } else { "tfm-trading" }

    $datosDir = Join-Path $projectDir "datos"
    if (Test-Path $datosDir) {
        aws s3 sync "$datosDir" (Get-S3Uri -Bucket $bucket -Prefix $prefix -Suffix "datos")
    }

    $tmpDir = Join-Path $projectDir "app\.tmp"
    if (Test-Path $tmpDir) {
        aws s3 sync "$tmpDir" (Get-S3Uri -Bucket $bucket -Prefix $prefix -Suffix "artifacts")
    }

    $portfolioDir = Join-Path $tmpDir "portfolio_rl"
    if (Test-Path $portfolioDir) {
        aws s3 sync "$portfolioDir" (Get-S3Uri -Bucket $bucket -Prefix $prefix -Suffix "ppo_models")
    }

    $backtestsCsvDir = Join-Path $tmpDir "backtests_csv"
    if (Test-Path $backtestsCsvDir) {
        aws s3 sync "$backtestsCsvDir" (Get-S3Uri -Bucket $bucket -Prefix $prefix -Suffix "backtests")
    }

    $riskDir = Join-Path $tmpDir "risk_reports"
    if (Test-Path $riskDir) {
        aws s3 sync "$riskDir" (Get-S3Uri -Bucket $bucket -Prefix $prefix -Suffix "risk_reports")
    }

    if (Test-Path $logDir) {
        aws s3 sync "$logDir" (Get-S3Uri -Bucket $bucket -Prefix $prefix -Suffix "logs")
    }
}
finally {
    Stop-Transcript | Out-Null
}

