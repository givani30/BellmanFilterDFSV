import jax
import jax.numpy as jnp
import time

# Ensure JAX is configured to use the GPU if available
# JAX automatically uses the fastest available device by default
print(f"JAX devices: {jax.devices()}")


# Define a simple JAX function
@jax.jit
def matrix_multiply(x, y):
    return jnp.dot(x, y)


# Define matrix dimensions
size = 1000
key = jax.random.PRNGKey(0)
x = jax.random.normal(key, (size, size))
y = jax.random.normal(key, (size, size))

# --- Benchmark on GPU (default device) ---
print("\nBenchmarking on GPU...")

# Run once to compile
_ = matrix_multiply(x, y).block_until_ready()

# Measure execution time
start_time_gpu = time.time()
result_gpu = matrix_multiply(x, y).block_until_ready()
end_time_gpu = time.time()
gpu_time = end_time_gpu - start_time_gpu

print(f"GPU execution time: {gpu_time:.6f} seconds")

# --- Benchmark on CPU ---
print("\nBenchmarking on CPU...")

# Switch to CPU device
cpu_device = jax.devices("cpu")[0]
with jax.default_device(cpu_device):
    # Re-create inputs on CPU device
    # Create inputs and move to CPU device
    x_cpu = jax.device_put(
        jax.random.normal(key, (size, size), dtype=x.dtype), cpu_device
    )
    y_cpu = jax.device_put(
        jax.random.normal(key, (size, size), dtype=y.dtype), cpu_device
    )

    # Run once to compile on CPU
    _ = matrix_multiply(x_cpu, y_cpu).block_until_ready()

    # Measure execution time on CPU
    start_time_cpu = time.time()
    result_cpu = matrix_multiply(x_cpu, y_cpu).block_until_ready()
    end_time_cpu = time.time()
    cpu_time = end_time_cpu - start_time_cpu

print(f"CPU execution time: {cpu_time:.6f} seconds")

# Optional: Verify results are close
# print("\nVerifying results...")
# print(f"Results are close: {jnp.allclose(result_gpu, result_cpu)}")
