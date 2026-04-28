resource "aws_cloudwatch_log_group" "tfm" {
  for_each          = local.log_groups
  name              = each.value
  retention_in_days = 30
  tags              = local.common_tags
}

resource "aws_cloudwatch_metric_alarm" "status_check_failed" {
  alarm_name          = "${local.name_prefix}-status-check-failed"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "StatusCheckFailed"
  namespace           = "AWS/EC2"
  period              = 60
  statistic           = "Maximum"
  threshold           = 0
  alarm_description   = "EC2 instance status check failure."

  dimensions = {
    InstanceId = aws_instance.windows.id
  }

  tags = local.common_tags
}

resource "aws_cloudwatch_metric_alarm" "cpu_high" {
  alarm_name          = "${local.name_prefix}-cpu-high"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 3
  metric_name         = "CPUUtilization"
  namespace           = "AWS/EC2"
  period              = 300
  statistic           = "Average"
  threshold           = var.cpu_alarm_threshold
  alarm_description   = "EC2 CPU average above threshold."

  dimensions = {
    InstanceId = aws_instance.windows.id
  }

  tags = local.common_tags
}

resource "aws_cloudwatch_metric_alarm" "disk_free_low" {
  alarm_name          = "${local.name_prefix}-disk-free-low"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = 2
  metric_name         = "LogicalDisk % Free Space"
  namespace           = "TFM/EC2"
  period              = 300
  statistic           = "Average"
  threshold           = var.disk_free_alarm_threshold
  alarm_description   = "Low free disk space if CloudWatch Agent is publishing Windows disk metrics."
  treat_missing_data  = "notBreaching"

  dimensions = {
    InstanceId = aws_instance.windows.id
    instance   = "_Total"
    objectname = "LogicalDisk"
  }

  tags = local.common_tags
}

