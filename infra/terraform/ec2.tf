resource "aws_instance" "windows" {
  ami                         = data.aws_ssm_parameter.windows_2022_ami.value
  instance_type               = var.instance_type
  subnet_id                   = tolist(data.aws_subnets.default.ids)[0]
  key_name                    = var.key_pair_name
  associate_public_ip_address = true
  iam_instance_profile        = aws_iam_instance_profile.ec2.name
  vpc_security_group_ids      = [aws_security_group.ec2.id]
  monitoring                  = true
  user_data = templatefile("${path.module}/user_data_windows.ps1.tpl", {
    aws_region      = var.aws_region
    s3_bucket_name  = var.s3_bucket_name
    s3_prefix       = "tfm-trading"
    github_repo_url = var.github_repo_url
    github_branch   = var.github_branch
    streamlit_port  = var.streamlit_port
  })
  user_data_replace_on_change = true

  metadata_options {
    http_endpoint = "enabled"
    http_tokens   = "required"
  }

  root_block_device {
    volume_size           = var.root_volume_size_gb
    volume_type           = "gp3"
    encrypted             = true
    delete_on_termination = false
  }

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-windows"
  })
}

