resource "aws_ecr_repository" "app" {
  name = local.app_name
}

# ECS Cluster
resource "aws_ecs_cluster" "app" {
  name = local.app_name
}

# Logging container using fluentbit
module "log_router_container" {
  source    = "git::https://github.com/wellcomecollection/terraform-aws-ecs-service.git//modules/firelens?ref=v2.6.3"
  namespace = "moh"
}

module "log_router_permissions" {
  source    = "git::https://github.com/wellcomecollection/terraform-aws-ecs-service.git//modules/secrets?ref=v2.6.3"
  secrets   = module.log_router_container.shared_secrets_logging
  role_name = module.app_task_definition.task_execution_role_name
}

# Create container definitions
module "app_container_definition" {
  source = "git::https://github.com/wellcomecollection/terraform-aws-ecs-service.git//modules/container_definition?ref=v2.6.3"
  name   = local.app_name

  image = "${aws_ecr_repository.app.repository_url}:production"
  port_mappings = [{
    containerPort = 80
    hostPort      = 80
    protocol      = "tcp"
  }]

  secrets = local.secret_env_vars

  environment = local.environment_variables

  log_configuration = module.log_router_container.container_log_configuration
}

# Create task definition
module "app_task_definition" {
  source = "git::https://github.com/wellcomecollection/terraform-aws-ecs-service.git//modules/task_definition?ref=v2.6.3"

  cpu    = 1024
  memory = 4096

  container_definitions = [
    module.log_router_container.container_definition,
    module.app_container_definition.container_definition
  ]

  launch_types = ["FARGATE"]
  task_name    = local.app_name
}

# secrets
module "app_container_secrets_permissions" {
  source    = "git::github.com/wellcomecollection/terraform-aws-ecs-service.git//modules/secrets?ref=v2.6.3"
  secrets   = local.secret_env_vars
  role_name = module.app_task_definition.task_execution_role_name
}

# Create service
module "service" {
  source = "git::https://github.com/wellcomecollection/terraform-aws-ecs-service.git//modules/service?ref=v2.6.3"

  cluster_arn  = aws_ecs_cluster.app.arn
  service_name = local.app_name

  task_definition_arn = module.app_task_definition.arn

  subnets            = local.developer_vpc_private_subnets
  security_group_ids = [aws_security_group.internal.id, ]

  target_group_arn = aws_alb_target_group.service.arn
  container_port   = 80
  container_name   = local.app_name
}

resource "aws_alb_target_group" "service" {
  name        = replace(local.app_name, "_", "-")
  target_type = "ip"
  protocol    = "HTTP"

  deregistration_delay = 10
  port                 = 80
  vpc_id               = local.vpc_id

  health_check {
    path                = local.health_check_path
    port                = 80
    protocol            = "HTTP"
    matcher             = 200
    timeout             = 5
    healthy_threshold   = 3
    unhealthy_threshold = 3
  }

  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_alb_listener_rule" "https" {
  listener_arn = aws_lb_listener.https.arn
  priority     = 1

  action {
    type             = "forward"
    target_group_arn = aws_alb_target_group.service.arn
  }

  condition {
    path_pattern {
      values = ["/*"]
    }
  }
}