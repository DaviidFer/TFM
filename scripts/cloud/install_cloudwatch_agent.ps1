[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"

function Get-ProjectDir {
    if ($env:TFM_PROJECT_DIR) {
        return $env:TFM_PROJECT_DIR
    }
    return (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
}

$projectDir = Get-ProjectDir
$logDir = Join-Path $projectDir "app\.tmp\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$logFile = Join-Path $logDir "install_cloudwatch_agent.log"

Start-Transcript -Path $logFile -Append | Out-Null
try {
    $bootstrapDir = "C:\tfm\bootstrap"
    New-Item -ItemType Directory -Force -Path $bootstrapDir | Out-Null
    $agentRoot = "C:\Program Files\Amazon\AmazonCloudWatchAgent"
    $agentCtl = Join-Path $agentRoot "amazon-cloudwatch-agent-ctl.ps1"

    if (-not (Test-Path $agentCtl)) {
        $msiPath = Join-Path $env:TEMP "amazon-cloudwatch-agent.msi"
        Invoke-WebRequest -Uri "https://amazoncloudwatch-agent-windows.s3.amazonaws.com/amazon-cloudwatch-agent.msi" -OutFile $msiPath
        Start-Process "msiexec.exe" -ArgumentList "/i `"$msiPath`" /qn" -Wait -NoNewWindow
    }

    $configPath = Join-Path $bootstrapDir "cloudwatch-agent-config.json"
    $runtimeFlow = Join-Path $projectDir "app\.tmp\logs\runtime_flow.log"
    $streamlitLog = Join-Path $projectDir "app\.tmp\logs\streamlit.log"
    $runtimeLog = Join-Path $projectDir "app\.tmp\logs\runtime.log"
    $dailyLog = Join-Path $projectDir "app\.tmp\logs\daily_update.log"
    $monthlyLog = Join-Path $projectDir "app\.tmp\logs\monthly_refresh.log"
    $weeklyLog = Join-Path $projectDir "app\.tmp\logs\weekly_rebalance.log"
    $healthLog = Join-Path $projectDir "app\.tmp\logs\healthcheck.log"
    $bootstrapLog = "C:\tfm\logs\bootstrap.log"
    $region = if ($env:AWS_REGION) { $env:AWS_REGION } else { "eu-west-1" }

    @"
{
  "agent": {
    "region": "$region",
    "run_as_user": "Administrator"
  },
  "logs": {
    "logs_collected": {
      "files": {
        "collect_list": [
          {
            "file_path": "$runtimeLog",
            "log_group_name": "/tfm/trading/runtime",
            "log_stream_name": "{instance_id}-runtime"
          },
          {
            "file_path": "$streamlitLog",
            "log_group_name": "/tfm/trading/streamlit",
            "log_stream_name": "{instance_id}-streamlit"
          },
          {
            "file_path": "$runtimeFlow",
            "log_group_name": "/tfm/trading/supervisor",
            "log_stream_name": "{instance_id}-supervisor"
          },
          {
            "file_path": "$weeklyLog",
            "log_group_name": "/tfm/trading/portfolio",
            "log_stream_name": "{instance_id}-portfolio"
          },
          {
            "file_path": "$monthlyLog",
            "log_group_name": "/tfm/trading/risk-agent",
            "log_stream_name": "{instance_id}-risk"
          },
          {
            "file_path": "$dailyLog",
            "log_group_name": "/tfm/trading/runtime",
            "log_stream_name": "{instance_id}-daily-update"
          },
          {
            "file_path": "$healthLog",
            "log_group_name": "/tfm/trading/streamlit",
            "log_stream_name": "{instance_id}-healthcheck"
          },
          {
            "file_path": "$bootstrapLog",
            "log_group_name": "/tfm/trading/bootstrap",
            "log_stream_name": "{instance_id}-bootstrap"
          }
        ]
      }
    }
  },
  "metrics": {
    "namespace": "TFM/EC2",
    "append_dimensions": {
      "InstanceId": "{instance_id}"
    },
    "metrics_collected": {
      "LogicalDisk": {
        "measurement": [
          "% Free Space"
        ],
        "resources": [
          "*"
        ]
      },
      "Memory": {
        "measurement": [
          "% Committed Bytes In Use"
        ]
      }
    }
  }
}
"@ | Set-Content -Path $configPath -Encoding UTF8

    & $agentCtl -a fetch-config -m ec2 -s -c "file:$configPath"
}
finally {
    Stop-Transcript | Out-Null
}

