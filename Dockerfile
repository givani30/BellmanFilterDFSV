# 1. Base Image (Python 3.12 Slim)
FROM python:3.12-slim-bookworm

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# 2. Install common tools (git is needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 3. Upgrade pip (Optional but recommended)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# 4. Install uv
RUN pip install --no-cache-dir uv

# Set the working directory
WORKDIR /app

# Copy dependency definition files
COPY pyproject.toml uv.lock* ./

# 5. Install JAX with CPU support
RUN uv pip install --system --no-cache-dir "jax[cpu]"

# 6. Copy the application source code, scripts, and data
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY Data/ ./Data/

# 7. Install the rest of the project dependencies (editable mode)
RUN uv pip install --system --no-cache-dir -e .

# 8. Define the command to run the batch script
CMD ["python3", "scripts/run_config_batch.py"]