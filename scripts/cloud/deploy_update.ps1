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

function Install-ProjectRequirements([string]$PythonExe, [string]$ProjectDir) {
    $requirementsFile = Join-Path $ProjectDir "requirements.txt"
    if (-not (Test-Path $requirementsFile)) {
        throw "No existe requirements.txt en $ProjectDir"
    }

    & $PythonExe -m pip install -r $requirementsFile
    if ($LASTEXITCODE -ne 0) {
        throw "Fallo instalando requirements.txt"
    }

    $pyVersion = & $PythonExe -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
    if ($LASTEXITCODE -ne 0) {
        throw "No se pudo leer la version de Python"
    }
    $pyVersion = ($pyVersion | Select-Object -First 1).Trim()

    if ($pyVersion -eq "3.11") {
        & $PythonExe -m pip install --no-deps --ignore-requires-python "pyeventbt==0.0.9"
        if ($LASTEXITCODE -ne 0) {
            throw "Fallo instalando pyeventbt==0.0.9 para Python 3.11"
        }
    }
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
    Install-ProjectRequirements -PythonExe $pythonExe -ProjectDir $projectDir

    & $pythonExe -m app.cloud_tasks.smoke_test_cloud
}
finally {
    Stop-Transcript | Out-Null
}

