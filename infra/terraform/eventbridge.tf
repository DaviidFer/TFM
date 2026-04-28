resource "aws_cloudwatch_event_rule" "scheduled" {
  for_each            = local.scheduled_tasks
  name                = "${local.name_prefix}-${each.key}"
  description         = "Scheduled task ${each.key} for TFM EC2 Windows."
  schedule_expression = each.value.schedule
  tags                = local.common_tags
}

resource "aws_cloudwatch_event_target" "ssm" {
  for_each = local.scheduled_tasks

  rule      = aws_cloudwatch_event_rule.scheduled[each.key].name
  target_id = "${each.key}-ssm"
  arn       = "arn:aws:ssm:${var.aws_region}::document/AWS-RunPowerShellScript"
  role_arn  = aws_iam_role.eventbridge_ssm.arn

  run_command_targets {
    key    = "InstanceIds"
    values = [aws_instance.windows.id]
  }

  input = jsonencode({
    commands = [
      "Set-Location 'C:\\tfm\\tfm-project'",
      "& 'C:\\tfm\\tfm-project\\${each.value.script}'"
    ]
  })
}

