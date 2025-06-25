resource "aws_alb" "this" {
  name = replace(local.app_name, "_", "-")

  subnets = local.developer_vpc_public_subnets

  security_groups = [
    aws_security_group.internal.id,
    aws_security_group.web.id,
  ]
}

# redirect http -> https
resource "aws_lb_listener" "http_redirect" {
  load_balancer_arn = aws_alb.this.id
  port              = 80
  protocol          = "HTTP"

  default_action {
    type = "redirect"

    redirect {
      port        = "443"
      protocol    = "HTTPS"
      status_code = "HTTP_302"
    }
  }
}


resource "aws_lb_listener" "https" {
  load_balancer_arn = aws_alb.this.id
  port              = 443
  protocol          = "HTTPS"

  ssl_policy      = "ELBSecurityPolicy-2016-08"
  certificate_arn = data.aws_acm_certificate.app.arn


  default_action {
    target_group_arn = aws_alb_target_group.service.id
    type             = "forward"
  }
}

resource "aws_security_group" "web" {
  name        = local.app_name
  description = "Web access for ALB"
  vpc_id      = local.vpc_id

  ingress {
    protocol  = "tcp"
    from_port = 80
    to_port   = 80

    cidr_blocks = local.controlled_ingress_cidrs
  }

  ingress {
    protocol  = "tcp"
    from_port = 443
    to_port   = 443

    cidr_blocks = local.controlled_ingress_cidrs
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = local.app_name,
  }

  lifecycle {
    create_before_destroy = true
  }
}