variable "aws_region" {
  description = "AWS region for all resources."
  type        = string
}

variable "project_name" {
  description = "Logical project name used for tags and resource naming."
  type        = string
}

variable "environment" {
  description = "Deployment environment name."
  type        = string
}

variable "instance_type" {
  description = "EC2 Windows instance type."
  type        = string
  default     = "t3.large"
}

variable "key_pair_name" {
  description = "Existing AWS EC2 key pair name for RDP password retrieval."
  type        = string
}

variable "allowed_rdp_cidr" {
  description = "CIDR allowed to reach RDP 3389."
  type        = string
}

variable "allowed_streamlit_cidr" {
  description = "CIDR allowed to reach Streamlit."
  type        = string
}

variable "streamlit_port" {
  description = "Port exposed by Streamlit."
  type        = number
  default     = 8501
}

variable "root_volume_size_gb" {
  description = "Root volume size in GB."
  type        = number
  default     = 120
}

variable "github_repo_url" {
  description = "GitHub repository URL cloned by EC2."
  type        = string
}

variable "github_branch" {
  description = "Git branch pulled by EC2."
  type        = string
  default     = "main"
}

variable "s3_bucket_name" {
  description = "Private S3 bucket name for data and artifacts."
  type        = string
}

variable "cpu_alarm_threshold" {
  description = "CPU threshold for CloudWatch alarm."
  type        = number
  default     = 80
}

variable "disk_free_alarm_threshold" {
  description = "Disk free percentage threshold for CloudWatch alarm."
  type        = number
  default     = 15
}

