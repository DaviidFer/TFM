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
$logFile = Join-Path $logDir "runtime.log"

Start-Transcript -Path $logFile -Append | Out-Null
try {
    Set-Location $projectDir
    $pythonExe = Get-PythonExe -ProjectDir $projectDir
    Ensure-ProjectRequirements -PythonExe $pythonExe -ProjectDir $projectDir
    & $pythonExe -m app.cloud_tasks.run_runtime
}
finally {
    Stop-Transcript | Out-Null
}

