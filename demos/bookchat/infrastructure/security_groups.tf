resource "aws_security_group" "internal" {
  # format app name to replace hyphens with underscores
  name        = replace(local.app_name, "_", "-")
  description = "Allow internal traffic for ${local.app_name}"
  vpc_id      = local.vpc_id

  ingress {
    from_port = 0
    to_port   = 0
    protocol  = "-1"
    self      = true
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}