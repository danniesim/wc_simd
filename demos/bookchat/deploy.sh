#!/usr/bin/env bash

aws ecr get-login-password \
    --profile platform-developer | \
    docker login \
        --username AWS \
        --password-stdin 760097843905.dkr.ecr.eu-west-1.amazonaws.com

cd app

# Build and push the Docker image
docker build -t bookchat . --platform linux/amd64
docker tag bookchat:latest 760097843905.dkr.ecr.eu-west-1.amazonaws.com/wc_dsims_bookchat:production
docker push 760097843905.dkr.ecr.eu-west-1.amazonaws.com/wc_dsims_bookchat:production

# Trigger the deployment
aws ecs update-service \
    --cluster wc_dsims_bookchat \
    --service wc_dsims_bookchat \
    --force-new-deployment \
    --profile platform-developer