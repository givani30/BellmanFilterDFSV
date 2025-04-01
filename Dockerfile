# Use the Python version specified in pyproject.toml
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install uv for package management
RUN pip install uv

# Set the working directory
WORKDIR /app

# Copy dependency definition files
COPY pyproject.toml uv.lock* ./
# uv.lock* handles cases where the lock file might not exist initially,
# though it's best practice to commit it.

# Install dependencies using uv
# Using uv sync is generally preferred if uv.lock exists and is up-to-date.
# Fallback to pip install if needed or if lock file is absent/outdated.
# We install the project in editable mode to ensure scripts can import the src package.
RUN uv pip install --system -e .

# Copy the application source code and scripts
COPY src/ ./src/
COPY scripts/ ./scripts/

# Define the command to run the batch script
# Arguments will be passed to this command by Cloud Batch
CMD ["python", "scripts/run_config_batch.py"]