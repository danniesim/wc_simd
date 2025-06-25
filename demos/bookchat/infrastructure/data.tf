data "terraform_remote_state" "shared_infra" {
  backend = "s3"

  config = {
    assume_role = {
      role_arn = "arn:aws:iam::760097843905:role/platform-read_only"
    }
    bucket = "wellcomecollection-platform-infra"
    key    = "terraform/aws-account-infrastructure/platform.tfstate"
    region = "eu-west-1"
  }
}
variable "controlled_ingress_cidrs" {
  type = list(string)
  default = [
    "0.0.0.0/0"
  ]
}

locals {
  vpc_id                        = data.terraform_remote_state.shared_infra.outputs.developer_vpc_id
  developer_vpc_public_subnets  = data.terraform_remote_state.shared_infra.outputs.developer_vpc_public_subnets
  developer_vpc_private_subnets = data.terraform_remote_state.shared_infra.outputs.developer_vpc_private_subnets

  app_name = "wc_dsims_bookchat"

  secret_env_vars = {
    "PLACEHOLDER_SECRET" = "wc_dsims_bookchat/placeholder"
  }

  environment_variables = {
    "PLACEHOLDER" = "value"
  }

  domain = "bookchat.wellcomecollection.org"

  health_check_path        = "/_stcore/health"
  controlled_ingress_cidrs = var.controlled_ingress_cidrs
}

data "aws_acm_certificate" "app" {
  domain   = local.domain
  statuses = ["ISSUED"]
}
