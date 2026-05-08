[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"

. "$PSScriptRoot\Resolve-TfmProjectDir.ps1"
. "$PSScriptRoot\Ensure-ProjectRequirements.ps1"

function Get-PythonExe([string]$ProjectDir) {
    $venvPython = Join-Path $ProjectDir ".venv\Scripts\python.exe"
    if (Test-Path $venvPython) {
        return $venvPython
    }
    return "python"
}

$projectDir = Get-TfmProjectDir
Set-TfmProjectEnvironment -ProjectDir $projectDir
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
    Ensure-ProjectRequirements -PythonExe $pythonExe -ProjectDir $projectDir -Force

    & $pythonExe -m app.cloud_tasks.smoke_test_cloud
}
finally {
    Stop-Transcript | Out-Null
}

