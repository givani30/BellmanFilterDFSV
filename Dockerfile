# 1. Base Image (CUDA 12.1, cuDNN 8, Ubuntu 22.04 - Runtime version)
# Check JAX documentation for recommended CUDA/cuDNN versions if issues arise
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Set environment variables to prevent interactive prompts during installs
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# 2. Install Python 3.12, pip, and common tools
# Using deadsnakes PPA for Python 3.12 on Ubuntu 22.04
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-distutils \
    python3-pip \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Make python3.12 the default python3 and pip3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/pip3 pip3 /usr/bin/pip 1
# Ensure pip is up-to-date for python3
RUN python3 -m pip install --no-cache-dir --upgrade pip

# 3. Install uv
RUN pip3 install --no-cache-dir uv

# Set the working directory
WORKDIR /app

# Copy dependency definition files
COPY pyproject.toml uv.lock* ./
# uv.lock* handles cases where the lock file might not exist initially

# 4. Install JAX with CUDA support (adjust cuda version if base image changes)
# Ensure this matches the CUDA version in the base image (12.x)
RUN uv pip install --system --no-cache-dir "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# 5. Install the rest of the project dependencies (editable mode)
# This will install packages from pyproject.toml, skipping jax/jaxlib if already installed by the previous step
# Using --system to avoid creating a virtual env inside the container, as recommended by uv docs for containers
RUN uv pip install --system --no-cache-dir -e .

# 6. Copy the application source code and scripts
COPY src/ ./src/
COPY scripts/ ./scripts/

# 7. Define the command to run the batch script
# Arguments will be passed to this command by Cloud Batch via environment variables in batch_job.json
CMD ["python3", "scripts/run_config_batch.py"]