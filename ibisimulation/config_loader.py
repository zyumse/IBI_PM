import yaml
import os

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    config['n_cpus'] = config.get("n_cpus", int(os.getenv("SLURM_CPUS_ON_NODE", 1)))
    return config
