from ec2_control import is_sso_session_valid, perform_sso_login, DEFAULT_PROFILE
import sys
import os
import boto3
import argparse
from urllib.parse import urlparse

# Import SSO session helpers from ec2_control.py
sys.path.append(os.path.dirname(__file__))


def upload_file_to_s3(local_path, s3_root, repo_root):
    # Ensure SSO session is valid before proceeding
    if not is_sso_session_valid(DEFAULT_PROFILE):
        if not perform_sso_login():
            sys.exit(1)
    s3 = boto3.client('s3')
    abs_repo_root = os.path.abspath(repo_root)
    abs_local_path = os.path.abspath(local_path)
    if not abs_local_path.startswith(abs_repo_root):
        raise ValueError("File is not inside the repository root.")
    s3_key = os.path.relpath(abs_local_path, abs_repo_root)
    parsed = urlparse(s3_root)
    if parsed.scheme != "s3":
        raise ValueError("s3_root must be an S3 URI (s3://bucket/prefix)")
    bucket_name = parsed.netloc
    prefix = parsed.path.lstrip("/")
    full_key = f"{prefix}/{s3_key}" if prefix else s3_key
    s3.upload_file(abs_local_path, bucket_name, full_key)
    print(f"Uploaded {local_path} to s3://{bucket_name}/{full_key}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Upload a file to S3, preserving its repo-relative path.")
    parser.add_argument(
        "file_path",
        help="Path to the file to upload (relative or absolute)")
    parser.add_argument(
        "--s3-root",
        default="s3://wellcomecollection-dsim/wc_simd",
        help="S3 root URI (default: %(default)s)")
    parser.add_argument(
        "--repo-root", default=".",
        help="Path to the repo root (default: current directory)")
    args = parser.parse_args()
    upload_file_to_s3(args.file_path, args.s3_root, args.repo_root)
