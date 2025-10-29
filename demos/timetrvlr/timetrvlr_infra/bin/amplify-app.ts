#!/usr/bin/env node
import 'source-map-support/register';
import * as cdk from 'aws-cdk-lib';
import { AmplifyAppStack } from '../lib/amplify-app-stack';

const app = new cdk.App();

const stackName = `timetrvlr-amplify-${process.env.AMPLIFY_ENV || 'prod'}`;

new AmplifyAppStack(app, stackName, {
  env: {
    account: process.env.CDK_DEFAULT_ACCOUNT,
    region: process.env.CDK_DEFAULT_REGION,
  },
});

