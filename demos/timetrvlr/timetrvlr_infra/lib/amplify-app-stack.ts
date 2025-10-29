import * as cdk from 'aws-cdk-lib';
import { Construct } from 'constructs';
// Amplify higher-level (experimental) constructs live in the alpha module
import { App, GitHubSourceCodeProvider } from '@aws-cdk/aws-amplify-alpha';
import * as dotenv from 'dotenv';

// Load .env file early (non-fatal if missing, but variables may stay undefined)
dotenv.config();

export class AmplifyAppStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    const owner = process.env.GITHUB_OWNER;
    const repo = process.env.GITHUB_REPO;
    const appName = process.env.AMPLIFY_APP_NAME || 'timetrvlr';
    const oauthSecretName = process.env.GITHUB_TOKEN_SECRET_NAME;
    const oauthToken = process.env.GITHUB_TOKEN;

    if (!owner) throw new Error('GITHUB_OWNER is required');
    if (!repo) throw new Error('GITHUB_REPO is required');
    if (!oauthSecretName && !oauthToken) {
      throw new Error('Either GITHUB_TOKEN_SECRET_NAME or GITHUB_TOKEN is required');
    }

    const sourceProvider = new GitHubSourceCodeProvider({
      owner,
      repository: repo,
      oauthToken: oauthSecretName
        ? cdk.SecretValue.secretsManager(oauthSecretName)
        : cdk.SecretValue.unsafePlainText(oauthToken as string),
    });

    const app = new App(this, 'AmplifyApp', {
      appName,
      sourceCodeProvider: sourceProvider,
      environmentVariables: {
        NEXT_PUBLIC_TIMETRVLR_BACKEND_BASE: process.env.NEXT_PUBLIC_TIMETRVLR_BACKEND_BASE || '',
      },
      autoBranchDeletion: true,
    });

    // Main branch connection (change if your default branch differs)
    const branchName = process.env.BRANCH || 'main';
    const branch = app.addBranch(branchName);

  // (Optional) Additional configuration such as custom headers, rewrites, or domains can be added later.

    new cdk.CfnOutput(this, 'AmplifyAppId', { value: app.appId });
    new cdk.CfnOutput(this, 'AmplifyDefaultDomain', { value: app.defaultDomain });
    new cdk.CfnOutput(this, 'AmplifyBranchUrl', { value: `https://${branch.branchName}.${app.defaultDomain}` });
  }
}

