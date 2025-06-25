provider "aws" {
  assume_role {
    role_arn = "arn:aws:iam::760097843905:role/platform-developer"
  }

  region = "eu-west-1"

  default_tags {
    tags = {
      Terraform = true
      Project   = "wc_dsim"
    }
  }
}

terraform {
  backend "s3" {
    bucket = "wellcomecollection-platform-infra"
    key    = "wc_dsim/terraform.tfstate"
    region = "eu-west-1"

    role_arn = "arn:aws:iam::760097843905:role/platform-developer"
  }

  required_providers {
    aws = {
      source = "hashicorp/aws"
    }
  }
}