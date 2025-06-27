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

data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

data "aws_iam_policy_document" "app_role_policy" {
  statement {
    actions = [
      "bedrock:InvokeModel",
      "bedrock:InvokeModelWithResponseStream"
    ]

    resources = [
      "arn:aws:bedrock:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:inference-profile/eu.anthropic.claude-3-7-sonnet-20250219-v1:0",
      "arn:aws:bedrock:*::foundation-model/anthropic.claude-3-7-sonnet-20250219-v1:0"
    ]
  }
}

locals {
  vpc_id                        = data.terraform_remote_state.shared_infra.outputs.developer_vpc_id
  developer_vpc_public_subnets  = data.terraform_remote_state.shared_infra.outputs.developer_vpc_public_subnets
  developer_vpc_private_subnets = data.terraform_remote_state.shared_infra.outputs.developer_vpc_private_subnets

  app_name = "wc_dsims_bookchat"

  secret_env_vars = {
    "PLACEHOLDER_SECRET" = "wc_dsims_bookchat/placeholder"
    "LANGSMITH_API_KEY"  = "wc_dsims_bookchat/langsmith_api_key"
    "ES_PASSWORD"        = "wc_dsims_bookchat/es_password"
    "ES_CLOUD_ID"        = "wc_dsims_bookchat/es_cloud_id"
    "COOKIE_PASSWORD"    = "wc_dsims_bookchat/cookie_password"
  }

  environment_variables = {
    "PLACEHOLDER"       = "value"
    "LANGSMITH_TRACING" = "true"
    "LANGSMITH_PROJECT" = "bookchat"
    "ES_USERNAME"       = "elastic"
  }

  domain = "bookchat.wellcomecollection.org"

  app_role_policy_json = data.aws_iam_policy_document.app_role_policy.json

  health_check_path        = "/_stcore/health"
  controlled_ingress_cidrs = var.controlled_ingress_cidrs
}

data "aws_acm_certificate" "app" {
  domain   = local.domain
  statuses = ["ISSUED"]
}
