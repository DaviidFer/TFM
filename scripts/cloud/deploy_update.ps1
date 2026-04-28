[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"

function Get-ProjectDir {
    if ($env:TFM_PROJECT_DIR) {
        return $env:TFM_PROJECT_DIR
    }
    return (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
}

function Get-PythonExe([string]$ProjectDir) {
    $venvPython = Join-Path $ProjectDir ".venv\Scripts\python.exe"
    if (Test-Path $venvPython) {
        return $venvPython
    }
    return "python"
}

$projectDir = Get-ProjectDir
$logDir = Join-Path $projectDir "app\.tmp\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$logFile = Join-Path $logDir "deploy_update.log"

Start-Transcript -Path $logFile -Append | Out-Null
try {
    Set-Location $projectDir

    if (Test-Path (Join-Path $projectDir ".git")) {
        git pull
    }
    else {
        throw "No se encontro un repositorio git en $projectDir"
    }

    $pythonExe = Get-PythonExe -ProjectDir $projectDir
    & $pythonExe -m pip install --upgrade pip
    if (Test-Path (Join-Path $projectDir "requirements.txt")) {
        & $pythonExe -m pip install -r "requirements.txt"
    }
    else {
        throw "No existe requirements.txt en $projectDir"
    }

    & $pythonExe -m app.cloud_tasks.smoke_test_cloud
}
finally {
    Stop-Transcript | Out-Null
}

