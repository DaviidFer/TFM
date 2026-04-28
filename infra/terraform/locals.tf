locals {
  name_prefix = "${var.project_name}-${var.environment}"

  common_tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "terraform"
  }

  log_groups = {
    runtime    = "/tfm/trading/runtime"
    streamlit  = "/tfm/trading/streamlit"
    supervisor = "/tfm/trading/supervisor"
    risk_agent = "/tfm/trading/risk-agent"
    portfolio  = "/tfm/trading/portfolio"
    bootstrap  = "/tfm/trading/bootstrap"
  }

  scheduled_tasks = {
    daily_backup = {
      schedule = "cron(0 21 * * ? *)"
      script   = "scripts\\cloud\\backup_to_s3.ps1"
    }
    daily_update = {
      schedule = "cron(0 18 * * ? *)"
      script   = "scripts\\cloud\\run_daily_update.ps1"
    }
    weekly_rebalance = {
      schedule = "cron(0 8 ? * MON *)"
      script   = "scripts\\cloud\\run_weekly_rebalance.ps1"
    }
    monthly_refresh = {
      schedule = "cron(0 7 1 * ? *)"
      script   = "scripts\\cloud\\run_monthly_refresh.ps1"
    }
    healthcheck = {
      schedule = "rate(30 minutes)"
      script   = "scripts\\cloud\\healthcheck.ps1"
    }
  }
}

