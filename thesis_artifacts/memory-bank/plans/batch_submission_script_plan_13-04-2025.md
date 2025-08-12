# Plan: Google Cloud Batch Job Submission Script (`submit_optimization_batch_job.py`)

**Version:** 1.0
**Date:** 13-04-2025
**Author:** Architect Mode (Roo)

## 1. Goal

Create a Python script (`scripts/submit_optimization_batch_job.py`) that uses the `google-cloud-batch` library to:
1.  Generate experiment configurations by running `scripts/generate_optimization_configs.py`.
2.  Parse the generated JSON configuration list.
3.  Define a Google Cloud Batch job where each task executes `scripts/run_optimization_replicate.py` for one configuration.
4.  Dynamically assign the correct optimizer (`ArmijoBFGS` for PF, `DampedTrustRegionBFGS` for BIF) to each task based on the `filter_type` in the configuration (handled within the target script).
5.  Submit the defined job to Google Cloud Batch.

## 2. Prerequisites

*   **Google Cloud Project:** A GCP project with the Batch API enabled.
*   **Authentication:** The environment where the script runs needs authentication credentials for GCP (e.g., Application Default Credentials via `gcloud auth application-default login`).
*   **Permissions:** The service account used by Batch (or the user running the script) needs permissions for Batch (e.g., `roles/batch.jobsEditor`), Cloud Storage (e.g., `roles/storage.objectAdmin` for the specified bucket), Logging (e.g., `roles/logging.logWriter`), and potentially Artifact Registry if using a private Docker image (e.g., `roles/artifactregistry.reader`).
*   **Docker Image:** A container image containing the project code, dependencies (`requirements.txt`), and the target script (`scripts/run_optimization_replicate.py`), pushed to a registry accessible by Google Cloud Batch (e.g., Google Artifact Registry).
*   **GCS Bucket:** A Google Cloud Storage bucket for storing:
    *   The generated configuration JSON file.
    *   The output results from each `run_optimization_replicate.py` task.
*   **Python Environment:** Python 3.x with `google-cloud-batch` and `google-cloud-storage` libraries installed (`pip install google-cloud-batch google-cloud-storage`).

## 3. Script Structure (`scripts/submit_optimization_batch_job.py`)

The script will perform the following steps:

```mermaid
graph TD
    A[Start submit_optimization_batch_job.py] --> B(Parse Arguments: Project ID, Region, GCS Path, Docker URI, etc.);
    B --> C{Run generate_optimization_configs.py};
    C --> D[Parse JSON Output (List of Config Dicts)];
    D --> E[Upload Config JSON to GCS];
    E --> F[Initialize Batch & Storage Clients];
    F --> G[Define Task Runnable (Docker, Entrypoint, Common Env Vars)];
    G --> H[Define TaskSpec (using Runnable)];
    H --> I[Create TaskGroup (using TaskSpec, Task Count)];
    I --> J[Define ComputeResource & AllocationPolicy];
    J --> K[Create Job Definition (Name, TaskGroup, Policy, Logs)];
    K --> L[Submit Job via Batch Client];
    L --> M[Print Job Name/ID];
    M --> N[End];

    subgraph "Config Generation (C)"
        C1[subprocess.run('python scripts/generate_optimization_configs.py ...')] --> C2[Capture stdout (JSON string)];
    end

    subgraph "Task Runnable Definition (G)"
        direction LR
        G1[Define Container Image URI] --> G2[Define Command (e.g., python target_script.py)];
        G2 --> G3[Define Common Env Vars (CONFIG_GCS_URI, BASE_OUTPUT_DIR)];
        G3 --> G4[Create Runnable Object];
    end

    subgraph "Target Script Logic (run_optimization_replicate.py - Conceptual)"
        direction TD
        TS1[Start run_optimization_replicate.py] --> TS2[Read Env Vars (BATCH_TASK_INDEX, CONFIG_GCS_URI, BASE_OUTPUT_DIR)];
        TS2 --> TS3[Download Config JSON from GCS];
        TS3 --> TS4[Parse JSON & Select Config[BATCH_TASK_INDEX]];
        TS4 --> TS5{Determine Optimizer based on filter_type};
        TS5 -- PF --> TS6[optimizer = 'ArmijoBFGS'];
        TS5 -- BIF --> TS7[optimizer = 'DampedTrustRegionBFGS'];
        TS6 --> TS8[Construct Output Path];
        TS7 --> TS8;
        TS8 --> TS9[Run Optimization with Config + Optimizer];
        TS9 --> TS10[Upload Results to GCS Output Path];
        TS10 --> TS11[End Task];
    end

```

### 3.1. Imports

```python
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
```

### 3.2. Configuration (Command-Line Arguments)

Use `argparse` to define command-line arguments for user-configurable parameters:

*   `--project_id`: GCP Project ID (Required).
*   `--region`: GCP Region for the Batch job (Required).
*   `--gcs_bucket`: GCS Bucket name (e.g., `my-project-bucket`) (Required).
*   `--gcs_config_path`: GCS path prefix *within the bucket* for the generated config file (e.g., `batch_configs/optimization_study`) (Required). The script will append a unique filename.
*   `--gcs_output_path`: GCS path prefix *within the bucket* for task outputs (e.g., `batch_outputs/optimization_study_{timestamp}`) (Required). Task index/details will be appended by the target script.
*   `--docker_image_uri`: Full URI of the Docker image (e.g., `us-central1-docker.pkg.dev/my-project/my-repo/my-image:latest`) (Required).
*   `--machine_type`: GCE machine type for tasks (e.g., `n1-standard-4`) (Default: `n1-standard-4`).
*   `--service_account`: Email of the service account for tasks (Optional, defaults to Compute Engine default SA).
*   `--network`: VPC Network URI (Optional, defaults to project default).
*   `--subnetwork`: VPC Subnetwork URI (Optional).
*   `--max_run_duration`: Maximum duration for each task in seconds (e.g., "3600") (Default: "3600").
*   `--job_name_prefix`: Prefix for the Batch job name (Default: "optimization-study").
*   `--config_gen_script`: Path to the config generation script (Default: `scripts/generate_optimization_configs.py`).
*   `--target_script`: Path to the target script run by tasks *inside the container* (Default: `scripts/run_optimization_replicate.py`).
*   `--config_gen_args`: Additional arguments to pass to the config generation script as a JSON string (Optional).

### 3.3. Generate Configurations

1.  Construct the command to run `scripts/generate_optimization_configs.py`.
2.  Use `subprocess.run` to execute the script, capturing its standard output.
3.  Check the return code and handle potential errors during generation.

```python
# Example
config_gen_command = [
    sys.executable, # Use the same python interpreter
    args.config_gen_script
]
if args.config_gen_args:
    try:
        extra_args = json.loads(args.config_gen_args)
        # Example: Add as --key value pairs
        for key, value in extra_args.items():
            config_gen_command.extend([f"--{key}", str(value)])
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON provided for --config_gen_args: {args.config_gen_args}", file=sys.stderr)
        sys.exit(1)

print(f"Running config generation: {' '.join(config_gen_command)}")
try:
    result = subprocess.run(config_gen_command, capture_output=True, text=True, check=True, encoding='utf-8')
    config_json_string = result.stdout
    print("Config generation successful.")
except FileNotFoundError:
    print(f"Error: Config generation script not found at {args.config_gen_script}", file=sys.stderr)
    sys.exit(1)
except subprocess.CalledProcessError as e:
    print(f"Error running config generation script (Return Code: {e.returncode}):", file=sys.stderr)
    print(e.stderr, file=sys.stderr)
    sys.exit(1)
```

### 3.4. Parse & Upload Configuration

1.  Use `json.loads()` to parse the captured JSON string into a Python list of dictionaries. Handle potential `JSONDecodeError`.
2.  Generate a unique filename for the config list.
3.  Construct the full GCS URI.
4.  Use the `google-cloud-storage` client to upload the `config_json_string` to the GCS URI.

```python
# Example
try:
    configs = json.loads(config_json_string)
    if not isinstance(configs, list):
        raise ValueError("Generated config is not a JSON list.")
    print(f"Parsed {len(configs)} configurations.")
except json.JSONDecodeError as e:
    print(f"Error decoding JSON from config generation script: {e}", file=sys.stderr)
    sys.exit(1)
except ValueError as e:
    print(f"Error parsing configuration list: {e}", file=sys.stderr)
    sys.exit(1)

if not configs:
    print("Warning: No configurations generated. Exiting.", file=sys.stderr)
    sys.exit(0)

# Upload to GCS
config_filename = f"config_list_{uuid.uuid4()}.json"
config_gcs_blob_name = f"{args.gcs_config_path.strip('/')}/{config_filename}"
config_gcs_uri = f"gs://{args.gcs_bucket}/{config_gcs_blob_name}"

try:
    storage_client = storage.Client(project=args.project_id)
    bucket = storage_client.bucket(args.gcs_bucket)
    blob = bucket.blob(config_gcs_blob_name)
    blob.upload_from_string(config_json_string, content_type='application/json')
    print(f"Uploaded configuration list to {config_gcs_uri}")
except Exception as e:
    print(f"Error uploading config list to GCS: {e}", file=sys.stderr)
    sys.exit(1)

```

### 3.5. Define Batch Job (Python Library)

1.  **Client Instantiation:**
    ```python
    batch_client = batch.BatchServiceClient()
    ```

2.  **Compute Resource:** Define the VM resources for each task. (Uniform as per user request).
    ```python
    compute_resource = batch.ComputeResource(
        machine_type=args.machine_type,
        # cpu_milli: Optional[int] = None, # Can override machine type defaults
        # memory_mib: Optional[int] = None,
        # boot_disk_mib: Optional[int] = None,
    )
    ```

3.  **Allocation Policy:** Define how VMs are allocated.
    ```python
    policy = batch.AllocationPolicy.InstancePolicy(compute_resource=compute_resource)
    instances = [batch.AllocationPolicy.InstancePolicyOrTemplate(policy=policy)]
    allocation_policy = batch.AllocationPolicy(instances=instances)

    # Add network/subnetwork if provided
    if args.network or args.subnetwork:
        network_policy = batch.AllocationPolicy.NetworkPolicy()
        network_interface = batch.AllocationPolicy.NetworkInterface(
            network=args.network if args.network else "global/networks/default",
            subnetwork=args.subnetwork if args.subnetwork else None,
        )
        network_policy.network_interfaces = [network_interface]
        allocation_policy.network = network_policy

    # Add service account if specified
    if args.service_account:
       allocation_policy.service_account = batch.ServiceAccount(email=args.service_account)
    ```

4.  **Task Specification (Template):** Define the template task that all instances will run.
    *   **Define Common Environment Variables:** Pass the GCS config URI and base output path.
        ```python
        common_env_vars = {
            "CONFIG_GCS_URI": config_gcs_uri,
            "BASE_OUTPUT_DIR_URI": f"gs://{args.gcs_bucket}/{args.gcs_output_path.strip('/')}"
            # Add any other common env vars needed by the target script
        }
        ```
    *   **Create Runnable:** Define the container execution.
        ```python
        runnable = batch.Runnable()
        runnable.container = batch.Runnable.Container(
            image_uri=args.docker_image_uri,
            commands=["python", args.target_script] # Command to run inside container
        )
        runnable.environment = batch.Environment(variables=common_env_vars)
        ```
    *   **Create TaskSpec:** Define the task template.
        ```python
        task_spec = batch.TaskSpec(
            runnables=[runnable],
            compute_resource=compute_resource, # Reuse compute resource definition
            max_run_duration=duration_pb2.Duration(seconds=int(args.max_run_duration)),
            # max_retry_count=1, # Optional
        )
        ```

5.  **Task Group:** Group tasks based on the template spec. (Only one group needed as resources are uniform).
    ```python
    group = batch.TaskGroup(
        task_count=len(configs),
        task_spec=task_spec,
        parallelism=len(configs) # Attempt to run all tasks in parallel (subject to quotas)
    )
    ```

6.  **Job Definition:** Combine TaskGroups, AllocationPolicy, and Logs configuration.
    ```python
    job_name = f"{args.job_name_prefix}-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    job = batch.Job(
        task_groups=[group],
        allocation_policy=allocation_policy,
        labels={"env": "production", "study": "optimization"}, # Optional labels
        logs_policy=batch.LogsPolicy(
            destination=batch.LogsPolicy.Destination.CLOUD_LOGGING
        )
    )
    ```

### 3.6. Submit Job

1.  Use the client's `create_job` method.
2.  Handle potential exceptions during submission.

```python
try:
    created_job = batch_client.create_job(
        parent=f"projects/{args.project_id}/locations/{args.region}",
        job=job,
        job_id=job_name # Use the generated name as the Job ID
    )
    print(f"Job created successfully: {created_job.name}")
    # Provide a clickable link to the Google Cloud Console
    job_url = f"https://console.cloud.google.com/batch/jobs/detail/{args.region}/{job_name}?project={args.project_id}"
    print(f"View Job: {job_url}")
except Exception as e:
    print(f"Error creating Batch job: {e}", file=sys.stderr)
    # Consider cleaning up the uploaded config file on failure
    try:
        storage_client = storage.Client(project=args.project_id)
        bucket = storage_client.bucket(args.gcs_bucket)
        blob = bucket.blob(config_gcs_blob_name)
        blob.delete()
        print(f"Cleaned up config file: {config_gcs_uri}", file=sys.stderr)
    except Exception as cleanup_e:
        print(f"Warning: Failed to cleanup config file {config_gcs_uri}: {cleanup_e}", file=sys.stderr)
    sys.exit(1)
```

### 3.7. Error Handling

*   Use `try...except` blocks around `subprocess.run`, `json.loads`, GCS upload, and `client.create_job`.
*   Provide informative error messages to `stderr`.
*   Exit with a non-zero status code on failure.
*   Attempt to clean up uploaded GCS config file if job submission fails.

## 4. Recommendation: Modify `generate_optimization_configs.py`

**Action:** Modify `scripts/generate_optimization_configs.py` to **remove** the `optimizer_name` field from its output configurations. (This was completed in Subtask 2.2.1).

## 5. Parameter Passing Strategy (Finalized)

1.  Submission script (`submit_optimization_batch_job.py`) generates configurations (without optimizer) and uploads the list as a single JSON file to GCS.
2.  Submission script passes `CONFIG_GCS_URI` and `BASE_OUTPUT_DIR_URI` as common environment variables to all tasks.
3.  Target script (`run_optimization_replicate.py`) reads `BATCH_TASK_INDEX`, `CONFIG_GCS_URI`, `BASE_OUTPUT_DIR_URI`.
4.  Target script downloads and parses the config JSON.
5.  Target script selects its specific config using `BATCH_TASK_INDEX`.
6.  Target script determines `optimizer_name` based on `config['filter_type']` (PF -> `ArmijoBFGS`, BIF -> `DampedTrustRegionBFGS`).
7.  Target script constructs its unique GCS output path using `BASE_OUTPUT_DIR_URI` and task details.
8.  Target script runs optimization using its config, determined optimizer, and output path.

## 6. Example Usage (Conceptual)

```bash
python scripts/submit_optimization_batch_job.py \
    --project_id "my-gcp-project" \
    --region "us-central1" \
    --gcs_bucket "my-project-bucket" \
    --gcs_config_path "batch_configs/optimization_study" \
    --gcs_output_path "batch_outputs/optimization_study_$(date +%Y%m%d_%H%M%S)" \
    --docker_image_uri "us-central1-docker.pkg.dev/my-gcp-project/my-repo/bellman-filter-dfsv:latest" \
    --machine_type "n1-standard-8" \
    --max_run_duration "7200" \
    --job_name_prefix "bf-pf-opt-study" \
    # --service_account "my-batch-sa@my-gcp-project.iam.gserviceaccount.com" # Optional
    # --config_gen_args '{"some_arg": "value"}' # Optional
```

## 7. Next Steps (Post-Planning)

1.  Implement the submission script `scripts/submit_optimization_batch_job.py` according to this plan.
2.  Ensure the Docker image is up-to-date with the latest `run_optimization_replicate.py`.
3.  Test the submission script with a small number of configurations.
4.  Run the full batch job.
5.  Implement `scripts/aggregate_optimization_results.py` (Phase 3).
6.  Document the workflow (Phase 4).