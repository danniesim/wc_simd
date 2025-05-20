#!/usr/bin/env python3
import argparse
import os
import sys
import subprocess

import boto3
from botocore.exceptions import ClientError, SSOTokenLoadError, TokenRetrievalError

import dotenv

dotenv.load_dotenv()

INSTANCE_IDS = [
    "i-053dc89605578305e",  # Example instance ID
    "i-0b9d4ff3cf046e312"]

DEFAULT_INSTANCE_ID = INSTANCE_IDS[0]
DEFAULT_REGION = "eu-west-2"
DEFAULT_PROFILE = os.getenv("AWS_PROFILE", "default")


def is_sso_session_valid(profile=DEFAULT_PROFILE):
    """Check if the AWS SSO session is valid."""
    try:
        # Try to load the SSO token - this will fail if not present or expired
        session = boto3.Session(profile_name=profile)
        # Try to make a simple API call to test credentials
        sts_client = session.client('sts')
        sts_client.get_caller_identity()
        return True
    except (SSOTokenLoadError, TokenRetrievalError, ClientError):
        return False


def perform_sso_login():
    """Perform AWS SSO login."""
    print(
        f"AWS SSO session is invalid or expired. Initiating login for profile 'default'...")
    try:
        result = subprocess.run(
            ["aws", "sso", "login", "--profile", "default"],
            check=True,
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("✅ AWS SSO login successful.")
            return True
        else:
            print(f"❌ AWS SSO login failed: {result.stderr}", file=sys.stderr)
            return False
    except subprocess.CalledProcessError as e:
        print(f"❌ AWS SSO login failed: {e}", file=sys.stderr)
        return False
    except FileNotFoundError:
        print(
            "❌ AWS CLI not found. Please install it to use SSO login.",
            file=sys.stderr)
        return False


def get_ec2_client(region: str, profile=DEFAULT_PROFILE):
    """Get an EC2 client with valid credentials."""
    if not is_sso_session_valid(profile):
        if not perform_sso_login():
            sys.exit(1)

    session = boto3.Session(profile_name=profile)
    return session.client("ec2", region_name=region)


def list_instances(client):
    """List all instances with their indices."""
    print("Available instances:")
    for idx, instance_id in enumerate(INSTANCE_IDS):
        try:
            response = client.describe_instances(InstanceIds=[instance_id])
            instance = response['Reservations'][0]['Instances'][0]
            state = instance['State']['Name']
            name = "Unnamed"
            for tag in instance.get('Tags', []):
                if tag['Key'] == 'Name':
                    name = tag['Value']
                    break
            print(f"[{idx}] {instance_id} - {name} ({state})")
        except ClientError as e:
            print(f"[{idx}] {instance_id} - Error: {e}", file=sys.stderr)
    return


def stop_instance(client, instance_id: str):
    print(f"Stopping instance {instance_id}…")
    try:
        client.stop_instances(InstanceIds=[instance_id])
        waiter = client.get_waiter("instance_stopped")
        waiter.wait(InstanceIds=[instance_id])
        print(f"✅ Instance {instance_id} is now stopped.")
    except ClientError as e:
        print(f"Error stopping instance: {e}", file=sys.stderr)
        sys.exit(1)


def start_instance(client, instance_id: str):
    print(f"Starting instance {instance_id}…")
    try:
        client.start_instances(InstanceIds=[instance_id])
        waiter = client.get_waiter("instance_running")
        waiter.wait(InstanceIds=[instance_id])
        print(f"✅ Instance {instance_id} is now running.")
    except ClientError as e:
        print(f"Error starting instance: {e}", file=sys.stderr)
        sys.exit(1)


def force_stop_instance(client, instance_id: str):
    print(f"Force stopping instance {instance_id}…")
    try:
        client.stop_instances(InstanceIds=[instance_id], Force=True)
        waiter = client.get_waiter("instance_stopped")
        waiter.wait(InstanceIds=[instance_id])
        print(f"✅ Instance {instance_id} is now forcibly stopped.")
    except ClientError as e:
        print(f"Error force stopping instance: {e}", file=sys.stderr)
        sys.exit(1)


def restart_instance(client, instance_id: str):
    print(f"Restarting instance {instance_id}…")
    try:
        stop_instance(client, instance_id)
        start_instance(client, instance_id)
        print(f"✅ Instance {instance_id} has been restarted.")
    except ClientError as e:
        print(f"Error restarting instance: {e}", file=sys.stderr)
        sys.exit(1)


def force_restart_instance(client, instance_id: str):
    print(f"Force restarting instance {instance_id}…")
    try:
        force_stop_instance(client, instance_id)
        start_instance(client, instance_id)
        print(f"✅ Instance {instance_id} has been force restarted.")
    except ClientError as e:
        print(f"Error force restarting instance: {e}", file=sys.stderr)
        sys.exit(1)


def parse_args():
    p = argparse.ArgumentParser(
        description="Start or stop a single EC2 instance and wait for it to change state."
    )
    p.add_argument(
        "action",
        choices=[
            "start",
            "stop",
            "force-stop",
            "restart",
            "force-restart",
            "list"],
        help="Whether to start, stop, force-stop, restart, force-restart the instance, or list available instances",
    )
    p.add_argument(
        "--instance-id",
        default=DEFAULT_INSTANCE_ID,
        help=f"EC2 instance ID (default: {DEFAULT_INSTANCE_ID})",
    )
    p.add_argument(
        "--select",
        type=int,
        help="Select instance by index from the predefined list",
    )
    p.add_argument(
        "--region",
        default=DEFAULT_REGION,
        help=f"AWS region (default: {DEFAULT_REGION})",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # Check if SSO session is valid, if not, perform login
    if not is_sso_session_valid(DEFAULT_PROFILE):
        if not perform_sso_login():
            sys.exit(1)

    ec2 = get_ec2_client(args.region, DEFAULT_PROFILE)

    # Handle --select option
    if args.select is not None:
        if 0 <= args.select < len(INSTANCE_IDS):
            args.instance_id = INSTANCE_IDS[args.select]
            print(f"Selected instance: {args.instance_id}")
        else:
            sys.exit(
                f"Invalid instance index. Choose between 0 and {
                    len(INSTANCE_IDS) - 1}")

    if args.action == "list":
        list_instances(ec2)
    elif args.action == "stop":
        stop_instance(ec2, args.instance_id)
    elif args.action == "start":
        start_instance(ec2, args.instance_id)
    elif args.action == "force-stop":
        force_stop_instance(ec2, args.instance_id)
    elif args.action == "restart":
        restart_instance(ec2, args.instance_id)
    elif args.action == "force-restart":
        force_restart_instance(ec2, args.instance_id)
    else:
        # argparse should prevent this
        sys.exit(
            "Invalid action; choose 'start', 'stop', 'force-stop', 'restart', 'force-restart', or 'list'.")


if __name__ == "__main__":
    main()
