# scripts/submit_optimization_batch_job.py

"""
Submits a Google Cloud Batch job to run optimization replicates.

This script performs the following steps:
1. Parses command-line arguments for GCP project, region, GCS paths, Docker image, etc.
2. Runs the `generate_optimization_configs.py` script to produce a list of experiment configurations.
3. Uploads the generated configuration list (JSON) to a specified GCS location.
4. Defines a Google Cloud Batch job using the `google-cloud-batch` library:
    - Each task in the job runs the `run_optimization_replicate.py` script inside the specified Docker container.
    - Common environment variables (`CONFIG_GCS_URI`, `BASE_OUTPUT_DIR_URI`) are passed to each task.
    - The target script (`run_optimization_replicate.py`) is responsible for reading its task index, downloading the config, selecting its specific configuration, determining the correct optimizer based on filter_type, and running the optimization.
5. Submits the defined job to Google Cloud Batch.
6. Prints the name and console URL of the created job.
"""

import argparse
import json
import subprocess
import uuid
import datetime
import os
import sys
from google.cloud import batch_v1 as batch
from google.cloud import storage
from google.protobuf import duration_pb2
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Dictionary mapping machine types to resources (milliCPU, MiB RAM)
# Add more types as needed
MACHINE_RESOURCES = {
    "e2-standard-4": (4000, 16 * 1024),
    "e2-standard-8": (8000, 32 * 1024),
    "n1-standard-4": (4000, 15 * 1024),
    "n1-standard-8": (8000, 30 * 1024),
    # Add other common types if you might use them
}


def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Submit a Google Cloud Batch job for optimization replicates.")

    # Required arguments
    parser.add_argument("--project_id", required=True, help="GCP Project ID.")
    parser.add_argument("--region", required=True, help="GCP Region for the Batch job (e.g., us-central1).")
    parser.add_argument("--gcs_bucket", required=True, help="GCS Bucket name (e.g., my-project-bucket).")
    parser.add_argument("--gcs_config_path", required=True, help="GCS path prefix within the bucket for the generated config file (e.g., batch_configs/optimization_study). A unique filename will be appended.")
    parser.add_argument("--gcs_output_path", required=True, help="GCS path prefix within the bucket for task outputs (e.g., batch_outputs/optimization_study_YYYYMMDD_HHMMSS). Task index/details will be appended by the target script.")
    parser.add_argument("--docker_image_uri", required=True, help="Full URI of the Docker image (e.g., us-central1-docker.pkg.dev/my-project/my-repo/my-image:latest).")

    # Optional arguments with defaults
    parser.add_argument("--machine_type", default="e2-standard-4", help="GCE machine type for tasks (Default: e2-standard-4).")
    parser.add_argument("--service_account", default=None, help="Email of the service account for tasks (Optional, defaults to Compute Engine default SA).")
    parser.add_argument("--network", default=None, help="VPC Network URI (Optional, defaults to project default 'global/networks/default').")
    parser.add_argument("--subnetwork", default=None, help="VPC Subnetwork URI (Optional).")
    parser.add_argument("--max_run_duration", default="10800", help="Maximum duration for each task in seconds (Default: 3600).")
    parser.add_argument("--job_name_prefix", default="optimization-study", help="Prefix for the Batch job name (Default: optimization-study).")
    parser.add_argument("--config_gen_script", default="scripts/generate_optimization_configs.py", help="Path to the config generation script (Default: scripts/generate_optimization_configs.py).")
    parser.add_argument("--target_script", default="scripts/run_optimization_replicate.py", help="Path to the target script run by tasks *inside the container* (Default: scripts/run_optimization_replicate.py).")
    parser.add_argument("--config_gen_args", default=None, help="Additional arguments to pass to the config generation script as a JSON string (Optional). Example: '{\"num_replicates\": 10}'")

    return parser.parse_args()

def generate_configurations(script_path, extra_args_json):
    """Runs the configuration generation script and returns the JSON output."""
    config_gen_command = [
        sys.executable, # Use the same python interpreter
        script_path
    ]
    if extra_args_json:
        try:
            extra_args = json.loads(extra_args_json)
            if not isinstance(extra_args, dict):
                raise ValueError("config_gen_args must be a JSON object (dictionary).")
            # Add as --key value pairs
            for key, value in extra_args.items():
                config_gen_command.extend([f"--{key}", str(value)])
        except json.JSONDecodeError:
            logging.error(f"Error: Invalid JSON provided for --config_gen_args: {extra_args_json}")
            sys.exit(1)
        except ValueError as e:
             logging.error(f"Error processing --config_gen_args: {e}")
             sys.exit(1)

    logging.info(f"Running config generation: {' '.join(config_gen_command)}")
    try:
        # Ensure the script path exists
        if not os.path.exists(script_path):
             raise FileNotFoundError(f"Config generation script not found at {script_path}")

        result = subprocess.run(config_gen_command, capture_output=True, text=True, check=True, encoding='utf-8')
        config_json_string = result.stdout
        logging.info("Config generation successful.")
        return config_json_string
    except FileNotFoundError as e:
        logging.error(e)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running config generation script (Return Code: {e.returncode}):")
        logging.error(f"Stderr:\n{e.stderr}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"An unexpected error occurred during config generation: {e}")
        sys.exit(1)


def upload_config_to_gcs(project_id, bucket_name, gcs_path_prefix, config_json_string):
    """Uploads the configuration JSON string to GCS and returns the GCS URI."""
    try:
        configs = json.loads(config_json_string)
        if not isinstance(configs, list):
            raise ValueError("Generated config is not a JSON list.")
        logging.info(f"Parsed {len(configs)} configurations.")
        if not configs:
            logging.warning("No configurations generated. Exiting.")
            sys.exit(0) # Exit gracefully if no configs
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from config generation script: {e}")
        sys.exit(1)
    except ValueError as e:
        logging.error(f"Error parsing configuration list: {e}")
        sys.exit(1)

    config_filename = f"config_list_{uuid.uuid4()}.json"
    # Ensure prefix doesn't start/end with / and join correctly
    gcs_blob_name = "/".join(part for part in [gcs_path_prefix.strip('/'), config_filename] if part)
    config_gcs_uri = f"gs://{bucket_name}/{gcs_blob_name}"

    try:
        storage_client = storage.Client(project=project_id)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(gcs_blob_name)
        blob.upload_from_string(config_json_string, content_type='application/json')
        logging.info(f"Uploaded configuration list to {config_gcs_uri}")
        return config_gcs_uri, configs, gcs_blob_name # Return blob name for potential cleanup
    except Exception as e:
        logging.error(f"Error uploading config list to GCS bucket '{bucket_name}' at '{gcs_blob_name}': {e}")
        sys.exit(1)

def define_batch_job(args, config_gcs_uri, num_tasks):
    """Defines the Google Cloud Batch job structure."""

    # 1. Compute Resource (Explicitly define based on machine type)
    if args.machine_type not in MACHINE_RESOURCES:
        logging.warning(f"Machine type '{args.machine_type}' not in predefined resources. Using defaults or potentially failing. Add it to MACHINE_RESOURCES dict.")
        # Defaulting to empty, might cause issues or use minimal resources
        cpu_milli, memory_mib = None, None # Should be integers or None
    else:
        cpu_milli, memory_mib = MACHINE_RESOURCES[args.machine_type]

    # Ensure cpu_milli and memory_mib are integers if not None
    compute_resource_args = {}
    if cpu_milli is not None:
        compute_resource_args['cpu_milli'] = int(cpu_milli)
    if memory_mib is not None:
        compute_resource_args['memory_mib'] = int(memory_mib)

    compute_resource = batch.ComputeResource(**compute_resource_args)
        # boot_disk_mib can be set if needed


    # 2. Allocation Policy
    # Specify machine_type within the InstancePolicy (defines the VM type)
    policy = batch.AllocationPolicy.InstancePolicy(
        machine_type=args.machine_type
        # Add other policy settings like provisioning_model if needed
    )
    instances = [batch.AllocationPolicy.InstancePolicyOrTemplate(policy=policy)]
    allocation_policy = batch.AllocationPolicy(instances=instances)

    # Add network/subnetwork if provided
    if args.network or args.subnetwork:
        network_policy = batch.AllocationPolicy.NetworkPolicy()
        network_interface = batch.AllocationPolicy.NetworkInterface(
            network=args.network if args.network else "global/networks/default",
            subnetwork=args.subnetwork if args.subnetwork else None,
            # no_external_ip_address=False # Set to True if needed
        )
        network_policy.network_interfaces = [network_interface]
        allocation_policy.network = network_policy

    # Add service account if specified
    if args.service_account:
       allocation_policy.service_account = batch.ServiceAccount(email=args.service_account)

    # 3. Task Specification (Template)
    common_env_vars = {
        "CONFIG_GCS_URI": config_gcs_uri,
        "BASE_OUTPUT_DIR_URI": f"gs://{args.gcs_bucket}/{args.gcs_output_path.strip('/')}"
        # Add other common env vars if needed by the target script
    }

    runnable = batch.Runnable()
    runnable.container = batch.Runnable.Container(
        image_uri=args.docker_image_uri,
        commands=["python", args.target_script], # Command inside container
        # entrypoint="/bin/bash", # Optional: Override entrypoint
        # volumes=["/mnt/data:/data"] # Optional: Mount host paths or PDs
    )
    runnable.environment = batch.Environment(variables=common_env_vars)
    # Add secret environment variables if needed:
    # runnable.environment.secret_variables = {"SECRET_KEY": "secret-name/versions/latest"}

    task_spec = batch.TaskSpec(
        runnables=[runnable],
        compute_resource=compute_resource, # Pass the explicit resource request
        max_run_duration=duration_pb2.Duration(seconds=int(args.max_run_duration)),
        max_retry_count=1, # Adjust as needed
        # lifecycle_policies=[...] # Optional: e.g., action on exit code
    )

    # 4. Task Group
    group = batch.TaskGroup(
        task_count=num_tasks,
        task_spec=task_spec,
        parallelism=min(num_tasks, 1000), # Run up to 1000 tasks in parallel (Batch limit)
        # require_hosts_file=True, # If tasks need to communicate
        # permissive_ssh=False,
    )

    # 5. Job Definition
    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    job_name = f"{args.job_name_prefix}-{timestamp}-{uuid.uuid4().hex[:8]}"
    job = batch.Job(
        task_groups=[group],
        allocation_policy=allocation_policy,
        labels={"env": "production", "study": "optimization", "submitter": "script"}, # Example labels
        logs_policy=batch.LogsPolicy(
            destination=batch.LogsPolicy.Destination.CLOUD_LOGGING
            # Optional: logs_policy.logs_path = "/mnt/share/logs" # If saving to GCS/NFS
        )
    )
    return job, job_name


def submit_batch_job(project_id, region, job, job_name):
    """Submits the defined job to Google Cloud Batch."""
    batch_client = batch.BatchServiceClient()
    parent = f"projects/{project_id}/locations/{region}"

    try:
        logging.info(f"Submitting Batch job '{job_name}' to {parent}...")
        created_job = batch_client.create_job(
            parent=parent,
            job=job,
            job_id=job_name # Use the generated name as the Job ID
        )
        logging.info(f"Job created successfully: {created_job.name}")
        # Provide a clickable link to the Google Cloud Console
        job_url = f"https://console.cloud.google.com/batch/jobs/detail/{region}/{job_name}?project={project_id}"
        print(f"View Job: {job_url}") # Print URL clearly for user
        return created_job
    except Exception as e:
        logging.error(f"Error creating Batch job '{job_name}': {e}")
        # Re-raise the exception to trigger cleanup in the main block
        raise

def cleanup_gcs_config(project_id, bucket_name, blob_name):
    """Attempts to delete the uploaded GCS config file."""
    try:
        logging.info(f"Attempting to clean up config file: gs://{bucket_name}/{blob_name}")
        storage_client = storage.Client(project=project_id)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.delete()
        logging.info(f"Cleaned up config file: gs://{bucket_name}/{blob_name}")
    except Exception as cleanup_e:
        logging.warning(f"Warning: Failed to cleanup config file gs://{bucket_name}/{blob_name}: {cleanup_e}")


def main():
    """Main execution flow."""
    args = parse_arguments()

    # 1. Generate Configurations
    config_json_string = generate_configurations(args.config_gen_script, args.config_gen_args)

    # 2. Upload Configuration to GCS
    config_gcs_uri, configs, gcs_blob_name = upload_config_to_gcs(
        args.project_id, args.gcs_bucket, args.gcs_config_path, config_json_string
    )

    # 3. Define Batch Job
    job, job_name = define_batch_job(args, config_gcs_uri, len(configs))

    # 4. Submit Job
    try:
        submit_batch_job(args.project_id, args.region, job, job_name)
    except Exception:
        # Attempt cleanup if submission fails
        cleanup_gcs_config(args.project_id, args.gcs_bucket, gcs_blob_name)
        sys.exit(1) # Exit with error after cleanup attempt

    logging.info("Script finished successfully.")


if __name__ == "__main__":
    main()