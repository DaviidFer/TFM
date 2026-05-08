[CmdletBinding()]
param(
    [string]$RepoUrl = $env:GITHUB_REPO_URL,
    [string]$Branch = $(if ($env:GITHUB_BRANCH) { $env:GITHUB_BRANCH } else { "main" }),
    [string]$ProjectDir = $(if ($env:TFM_PROJECT_DIR) { $env:TFM_PROJECT_DIR } else { "C:\tfm\tfm-project-gitpublic" })
)

$ErrorActionPreference = "Stop"
$bootstrapRoot = "C:\tfm"
$bootstrapLogDir = Join-Path $bootstrapRoot "logs"
$bootstrapLogFile = Join-Path $bootstrapLogDir "bootstrap.log"

. "$PSScriptRoot\Resolve-TfmProjectDir.ps1"
. "$PSScriptRoot\Ensure-ProjectRequirements.ps1"

function Install-ChocolateyIfNeeded {
    if (Get-Command choco -ErrorAction SilentlyContinue) {
        return
    }
    Set-ExecutionPolicy Bypass -Scope Process -Force
    [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
    Invoke-Expression ((New-Object System.Net.WebClient).DownloadString("https://community.chocolatey.org/install.ps1"))
}

function Install-ChocoPackageIfNeeded([string]$CommandName, [string]$PackageName, [string]$ExtraArgs = "") {
    if (Get-Command $CommandName -ErrorAction SilentlyContinue) {
        return
    }
    if ([string]::IsNullOrWhiteSpace($ExtraArgs)) {
        choco install $PackageName -y
    }
    else {
        choco install $PackageName -y --params $ExtraArgs
    }
}

New-Item -ItemType Directory -Force -Path $bootstrapRoot | Out-Null
New-Item -ItemType Directory -Force -Path $bootstrapLogDir | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $bootstrapRoot "bootstrap") | Out-Null

Start-Transcript -Path $bootstrapLogFile -Append | Out-Null
try {
    Install-ChocolateyIfNeeded
    Install-ChocoPackageIfNeeded -CommandName "git" -PackageName "git"
    Install-ChocoPackageIfNeeded -CommandName "python" -PackageName "python" -ExtraArgs "'/InstallDir:C:\Python311 /PrependPath:1'"
    Install-ChocoPackageIfNeeded -CommandName "aws" -PackageName "awscli"

    [Environment]::SetEnvironmentVariable("AWS_REGION", $(if ($env:AWS_REGION) { $env:AWS_REGION } else { "eu-west-2" }), "Machine")
    [Environment]::SetEnvironmentVariable("TFM_S3_BUCKET", $env:TFM_S3_BUCKET, "Machine")
    [Environment]::SetEnvironmentVariable("TFM_S3_PREFIX", $(if ($env:TFM_S3_PREFIX) { $env:TFM_S3_PREFIX } else { "tfm-trading" }), "Machine")
    [Environment]::SetEnvironmentVariable("TFM_PROJECT_DIR", $ProjectDir, "Machine")
    [Environment]::SetEnvironmentVariable("TFM_ARTIFACTS_ROOT", (Join-Path $ProjectDir "app\.tmp"), "Machine")
    [Environment]::SetEnvironmentVariable("TFM_DB_PATH", (Join-Path $ProjectDir "app\.tmp\supervisor\supervisor.sqlite"), "Machine")
    [Environment]::SetEnvironmentVariable("STREAMLIT_PORT", $(if ($env:STREAMLIT_PORT) { $env:STREAMLIT_PORT } else { "8501" }), "Machine")

    if (-not (Test-Path $ProjectDir)) {
        if ([string]::IsNullOrWhiteSpace($RepoUrl)) {
            throw "GITHUB_REPO_URL/RepoUrl no esta definido."
        }
        git clone --branch $Branch $RepoUrl $ProjectDir
    }
    else {
        Set-Location $ProjectDir
        if (Test-Path (Join-Path $ProjectDir ".git")) {
            git pull
        }
    }

    Set-Location $ProjectDir
    Set-TfmProjectEnvironment -ProjectDir $ProjectDir
    $envExample = Join-Path $ProjectDir ".env.example"
    $envFile = Join-Path $ProjectDir ".env"
    if ((Test-Path $envExample) -and (-not (Test-Path $envFile))) {
        Copy-Item $envExample $envFile
    }

    $venvPython = Join-Path $ProjectDir ".venv\Scripts\python.exe"
    $bootstrapPython = if (Test-Path "C:\Python311\python.exe") { "C:\Python311\python.exe" } else { "python" }
    if (-not (Test-Path $venvPython)) {
        & $bootstrapPython -m venv ".venv"
    }

    Ensure-ProjectRequirements -PythonExe $venvPython -ProjectDir $ProjectDir -Force
}
finally {
    Stop-Transcript | Out-Null
    if (Test-Path $ProjectDir) {
        $projectLogDir = Join-Path $ProjectDir "app\.tmp\logs"
        New-Item -ItemType Directory -Force -Path $projectLogDir | Out-Null
        if (Test-Path $bootstrapLogFile) {
            Copy-Item $bootstrapLogFile (Join-Path $projectLogDir "bootstrap_windows_ec2.log") -Force
        }
    }
}

