resource "aws_security_group" "ec2" {
  name        = "${local.name_prefix}-ec2-sg"
  description = "Security group for TFM Windows EC2"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    description = "RDP"
    from_port   = 3389
    to_port     = 3389
    protocol    = "tcp"
    cidr_blocks = [var.allowed_rdp_cidr]
  }

  ingress {
    description = "Streamlit"
    from_port   = var.streamlit_port
    to_port     = var.streamlit_port
    protocol    = "tcp"
    cidr_blocks = [var.allowed_streamlit_cidr]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = local.common_tags
}

