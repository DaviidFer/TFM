output "ec2_instance_id" {
  value       = aws_instance.windows.id
  description = "EC2 instance id."
}

output "ec2_public_ip" {
  value       = aws_instance.windows.public_ip
  description = "Public IP for RDP and Streamlit access."
}

output "s3_bucket_name" {
  value       = aws_s3_bucket.project.bucket
  description = "Project S3 bucket name."
}

output "security_group_id" {
  value       = aws_security_group.ec2.id
  description = "Security group attached to the EC2 instance."
}

output "cloudwatch_log_groups" {
  value       = values(aws_cloudwatch_log_group.tfm)[*].name
  description = "CloudWatch log groups created for the project."
}

