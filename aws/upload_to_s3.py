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


def list_s3_under_root(
        s3_root: str, relative_prefix: str = "", show_sizes: bool = False,
        limit: int | None = None):
    """List objects in S3 under the provided root, optionally scoped by a repo-relative prefix.

    Parameters
    ----------
    s3_root : str
        The S3 URI root (e.g. s3://bucket/prefix) which corresponds to repo root.
    relative_prefix : str, optional
        A repo-relative path (folder or file prefix) to further filter objects.
    show_sizes : bool, optional
        If True include human readable sizes.
    limit : int | None, optional
        If set, stop after emitting this many keys.
    """
    if not is_sso_session_valid(DEFAULT_PROFILE):
        if not perform_sso_login():
            sys.exit(1)
    parsed = urlparse(s3_root)
    if parsed.scheme != "s3":
        raise ValueError("s3_root must be an S3 URI (s3://bucket/prefix)")
    bucket = parsed.netloc
    base_prefix = parsed.path.lstrip("/")
    # Normalise relative prefix
    rel = relative_prefix.lstrip("/")
    key_prefix = f"{base_prefix}/{rel}" if base_prefix and rel else (
        base_prefix
        or rel)
    key_prefix = key_prefix.rstrip("/")
    if key_prefix and not key_prefix.endswith("/"):
        # We want to list 'under' this prefix, so add trailing slash unless
        # user clearly wants exact file (contains a dot and exists) -> we just
        # attempt both.
        pass
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    pagination_kwargs = {"Bucket": bucket}
    if key_prefix:
        pagination_kwargs["Prefix"] = key_prefix
    count = 0
    for page in paginator.paginate(**pagination_kwargs):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            # Only output keys that start with our computed prefix (safety)
            if key_prefix and not key.startswith(key_prefix):
                continue
            size_part = ""
            if show_sizes:
                size_bytes = obj.get("Size", 0)
                for unit in ["B", "KB", "MB", "GB", "TB"]:
                    if size_bytes < 1024 or unit == "TB":
                        size_part = f"\t{size_bytes:.1f}{unit}" if unit != "B" else f"\t{size_bytes}B"
                        break
                    size_bytes /= 1024
            # Transform back to repo-relative path by stripping base_prefix
            rel_key = key[len(base_prefix) +
                          1:] if base_prefix and key.startswith(base_prefix +
                                                                "/") else key
            print(f"{rel_key}{size_part}")
            count += 1
            if limit and count >= limit:
                return


def download_from_s3(s3_root: str, relative_path: str, repo_root: str,
                     recursive: bool = False, overwrite: bool = False):
    """Download a file or directory (recursively) from S3 using repo-relative path mapping.

    Parameters
    ----------
    s3_root : str
        S3 URI root that maps to the local repo root (e.g. s3://bucket/prefix)
    relative_path : str
        Repo-relative file path or directory prefix to download.
    repo_root : str
        Local repository root to place downloaded files.
    recursive : bool, default False
        If True and relative_path is a directory/prefix, download all objects under it.
    overwrite : bool, default False
        If False, skip downloading files that already exist locally.
    """
    if not is_sso_session_valid(DEFAULT_PROFILE):
        if not perform_sso_login():
            sys.exit(1)
    parsed = urlparse(s3_root)
    if parsed.scheme != "s3":
        raise ValueError("s3_root must be an S3 URI (s3://bucket/prefix)")
    bucket = parsed.netloc
    base_prefix = parsed.path.lstrip("/")
    rel = relative_path.lstrip("/")
    s3_key_prefix = f"{base_prefix}/{rel}" if base_prefix and rel else (
        base_prefix
        or rel)
    s3 = boto3.client("s3")

    def ensure_parent(path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)

    repo_root_abs = os.path.abspath(repo_root)

    if not recursive:
        # Attempt single object download
        local_dest = os.path.join(repo_root_abs, rel)
        if os.path.exists(local_dest) and not overwrite:
            print(f"Skipping existing file: {local_dest}")
            return
        ensure_parent(local_dest)
        try:
            s3.download_file(bucket, s3_key_prefix, local_dest)
            print(f"Downloaded s3://{bucket}/{s3_key_prefix} -> {local_dest}")
        except s3.exceptions.NoSuchKey:  # type: ignore[attr-defined]
            print(
                f"No such key: s3://{bucket}/{s3_key_prefix}",
                file=sys.stderr)
            sys.exit(1)
        return

    # Recursive mode: paginate under prefix
    paginator = s3.get_paginator("list_objects_v2")
    pagination_kwargs = {
        "Bucket": bucket,
        "Prefix": s3_key_prefix.rstrip('/') + '/'}
    found_any = False
    for page in paginator.paginate(**pagination_kwargs):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.startswith(s3_key_prefix):
                continue
            found_any = True
            # derive relative portion after base_prefix/
            if base_prefix and key.startswith(base_prefix + "/"):
                rel_key = key[len(base_prefix) + 1:]
            else:
                rel_key = key
            local_path = os.path.join(repo_root_abs, rel_key)
            if os.path.exists(local_path) and not overwrite:
                print(f"Skipping existing file: {local_path}")
                continue
            ensure_parent(local_path)
            s3.download_file(bucket, key, local_path)
            print(f"Downloaded s3://{bucket}/{key} -> {local_path}")
    if not found_any:
        print(
            f"No objects found under prefix s3://{bucket}/{s3_key_prefix}",
            file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Upload to S3 (preserving repo-relative path), list objects, or download (single or recursive) using repo-relative paths.")
    parser.add_argument(
        "file_path", nargs="?",
        help="Path to the file to upload (if omitted and --list used, no upload is performed)")
    parser.add_argument(
        "--s3-root", default="s3://wellcomecollection-dsim/wc_simd",
        help="S3 root URI mapping to the repo root (default: %(default)s)")
    parser.add_argument(
        "--repo-root", default=".",
        help="Path to the repo root for computing relative paths (default: current directory)")
    parser.add_argument(
        "--list",
        nargs="?",
        const="",
        metavar="REL_PREFIX",
        help="List S3 objects under the root, optionally providing a repo-relative prefix (default: list entire root)")
    parser.add_argument(
        "--sizes", action="store_true",
        help="Show human-readable sizes when listing")
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of listed keys")
    parser.add_argument(
        "--download", metavar="REL_PATH",
        help="Download a repo-relative file or directory (with --recursive) from S3 root")
    parser.add_argument(
        "--recursive", action="store_true",
        help="When used with --download treat the path as a prefix and download all objects underneath")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing local files when downloading")
    args = parser.parse_args()

    # Mutual exclusivity basic checks
    if args.list is not None and args.download:
        parser.error("--list and --download cannot be used together")

    if args.download:
        download_from_s3(
            args.s3_root,
            args.download,
            args.repo_root,
            recursive=args.recursive,
            overwrite=args.overwrite)
    elif args.list is not None:
        list_s3_under_root(
            args.s3_root,
            args.list,
            show_sizes=args.sizes,
            limit=args.limit)
    else:
        if not args.file_path:
            parser.error("file_path is required unless --list is used")
        upload_file_to_s3(args.file_path, args.s3_root, args.repo_root)
