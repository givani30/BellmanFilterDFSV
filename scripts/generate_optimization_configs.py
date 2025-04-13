import itertools
import json
import hashlib # Using hashlib for more stable hashing

# --- Parameter Grid Definition ---
nk_pairs = [(5, 2), (10, 3), (15, 5)]
t_values = [500, 1000, 2000]
filter_types = ['BIF', 'PF']
# optimizer_names = ['DampedTrustRegionBFGS'] # Removed as per plan
num_particles_list = [1000, 5000] # Only for PF
fix_mu_values = [True, False]
num_replicates = 5

# --- Default/Placeholder Values ---
DEFAULT_STABILITY_PENALTY = 1e4
DEFAULT_MAX_STEPS = 2000
DEFAULT_SAVE_FORMAT = 'pkl'
PLACEHOLDER_OUTPUT_DIR = "PLACEHOLDER_GCS_OUTPUT_DIR"

config_list = []

# Iterate through core configurations
core_params_product = itertools.product(
    nk_pairs,
    t_values,
    filter_types,
    # optimizer_names, # Removed
    fix_mu_values
)

for (N, K), T, filter_type, fix_mu in core_params_product: # Removed optimizer_name

    # Handle conditional parameter: num_particles
    if filter_type == 'PF':
        particle_iterator = num_particles_list
    else:
        # For BIF, we don't need particles, so iterate once with a placeholder
        particle_iterator = [None] # Use None to signify no particle count

    for num_particles in particle_iterator:

        # Generate a base seed from the core configuration + particles (if applicable)
        # Using hashlib for more stable hashing across runs/platforms
        hasher = hashlib.sha256()
        # Removed optimizer_name from config string
        config_str = f"{N}-{K}-{T}-{filter_type}-{fix_mu}"
        if num_particles is not None:
             config_str += f"-{num_particles}"
        hasher.update(config_str.encode('utf-8'))
        # Use part of the hash as an integer base seed
        base_seed = int(hasher.hexdigest()[:8], 16) # Use first 8 hex digits

        # Generate replicates for this specific configuration
        for rep_index in range(num_replicates):
            replicate_seed = base_seed + rep_index

            config_dict = {
                'N': N,
                'K': K,
                'T': T,
                'filter_type': filter_type,
                # 'optimizer_name': optimizer_name, # Removed
                'fix_mu': fix_mu,
                'stability_penalty': DEFAULT_STABILITY_PENALTY,
                'max_steps': DEFAULT_MAX_STEPS,
                'replicate_seed': replicate_seed,
                'output_dir': PLACEHOLDER_OUTPUT_DIR,
                'save_format': DEFAULT_SAVE_FORMAT,
            }

            # Conditionally add num_particles for PF
            if filter_type == 'PF':
                config_dict['num_particles'] = num_particles

            config_list.append(config_dict)

# Output the list of configurations as JSON
print(json.dumps(config_list, indent=2))