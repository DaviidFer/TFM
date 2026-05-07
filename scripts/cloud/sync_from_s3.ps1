[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"

. "$PSScriptRoot\Resolve-TfmProjectDir.ps1"

function Get-S3Uri([string]$Bucket, [string]$Prefix, [string]$Suffix) {
    $cleanPrefix = $Prefix.Trim("/\")
    $cleanSuffix = $Suffix.Trim("/\")
    if ([string]::IsNullOrWhiteSpace($cleanPrefix)) {
        return "s3://$Bucket/$cleanSuffix"
    }
    return "s3://$Bucket/$cleanPrefix/$cleanSuffix"
}

$projectDir = Get-TfmProjectDir
$logDir = Join-Path $projectDir "app\.tmp\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$logFile = Join-Path $logDir "sync_from_s3.log"

Start-Transcript -Path $logFile -Append | Out-Null
try {
    $bucket = $env:TFM_S3_BUCKET
    if ([string]::IsNullOrWhiteSpace($bucket)) {
        throw "TFM_S3_BUCKET no esta definido."
    }
    $prefix = if ($env:TFM_S3_PREFIX) { $env:TFM_S3_PREFIX } else { "tfm-trading" }

    $datosDir = Join-Path $projectDir "datos"
    New-Item -ItemType Directory -Force -Path $datosDir | Out-Null
    aws s3 sync (Get-S3Uri -Bucket $bucket -Prefix $prefix -Suffix "datos") "$datosDir"

    $tmpDir = Join-Path $projectDir "app\.tmp"
    New-Item -ItemType Directory -Force -Path $tmpDir | Out-Null
    aws s3 sync (Get-S3Uri -Bucket $bucket -Prefix $prefix -Suffix "artifacts") "$tmpDir"

    $sqliteBackupDir = Join-Path $tmpDir "sqlite_backups"
    New-Item -ItemType Directory -Force -Path $sqliteBackupDir | Out-Null
    aws s3 sync (Get-S3Uri -Bucket $bucket -Prefix $prefix -Suffix "sqlite_backups") "$sqliteBackupDir"
}
finally {
    Stop-Transcript | Out-Null
}

