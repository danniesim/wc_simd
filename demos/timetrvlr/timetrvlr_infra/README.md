# CDK stack for AWS Amplify Hosting of the timetrvlr app

## Prerequisites

- Node 20.x
- AWS credentials with permissions for Amplify, IAM, CodeBuild, CloudFormation
- AWS CDK v2 (CLI optional): use `npx aws-cdk <command>` (or globally install with `yarn global add aws-cdk`)
- A GitHub Personal Access Token (classic) with `repo` scope (recommended: store in AWS Secrets Manager)
- This stack uses the experimental Amplify higher-level constructs from `@aws-cdk/aws-amplify-alpha`.
  - Alpha modules may introduce breaking changes; pin versions before promoting to production.

## Required environment variables

Export these before synth/deploy (examples assume macOS/Linux `bash`/`zsh`).

```bash
export GITHUB_OWNER=your-org
export GITHUB_REPO=wc_simd
# Preferred: store token in Secrets Manager beforehand, then reference the secret name
export GITHUB_TOKEN_SECRET_NAME=github/pat/timetrvlr

# Fallback (not recommended – exposes token in process env):
# export GITHUB_TOKEN=ghp_XXXXXXXXXXXXXXXXXXXX
```

Exactly one of `GITHUB_TOKEN_SECRET_NAME` or `GITHUB_TOKEN` must be set.

## Optional environment variables

- `AMPLIFY_APP_NAME` — Defaults to `timetrvlr`
- `BRANCH` — Repo branch to connect (defaults to `main`)
- `NEXT_PUBLIC_TIMETRVLR_BACKEND_BASE` — Passed to Amplify as an env var for the Next.js app
  (used in rewrites or runtime config). Leave empty if not needed.

## Deploy

1. Install dependencies:

  ```bash
  yarn install
  ```

1. (First time per account+region) bootstrap the environment:

  ```bash
  npx aws-cdk bootstrap
  ```

1. Synthesize (optional preview):

  ```bash
  yarn build && npx aws-cdk synth
  ```

1. Deploy:

  ```bash
  yarn build && npx aws-cdk deploy
  ```

1. Note the outputs:

- `AmplifyAppId` – Amplify App identifier
- `AmplifyDefaultDomain` – Base domain assigned by Amplify
- `AmplifyBranchUrl` – Full URL for the configured branch

## Destroy

Tear down the Amplify app and all branches:

```bash
npx aws-cdk destroy
```

## Notes

- Uses `amplify.yml` at the repository root; app root is `demos/timetrvlr`.
- The alpha module currently exposes `App`, `GitHubSourceCodeProvider`, and `Branch` constructs; some examples online reference properties (e.g. `Stage`, `BuildSpec`, `addWebhook`) not yet available or since removed in the version pinned here.
- Fail-fast pattern: the stack validates required env vars early and throws if misconfigured.
- To add rewrites, custom headers, or password protection, extend the `App` definition with additional props from the alpha module as they become available.
- Consider pinning `aws-cdk-lib` and `@aws-cdk/aws-amplify-alpha` to exact versions to avoid surprise breaking changes.

## Security

- Prefer `GITHUB_TOKEN_SECRET_NAME` over `GITHUB_TOKEN` to keep tokens out of local shells and CI logs.
- Rotate the GitHub token periodically and update the secret in Secrets Manager.
- Review Amplify app environment variables for secrets—treat Amplify console as part of your secret exposure surface.
