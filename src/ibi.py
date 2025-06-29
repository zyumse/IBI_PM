import argparse
import os
import subprocess
import time
from copy import copy
import joblib
import matplotlib.pyplot as plt
import numpy as np
import yaml
from joblib import Parallel, delayed
from scipy.constants import Avogadro as NA
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d

from zy_md import tools_CG_polymer as tcp
from zy_md import tools_lammps as tl_lmp
from potential import *
from utils import *

class IBISimulation:
    def __init__(self, config_path):
        """Initialize the simulation with the given configuration file."""
        self.load_config(config_path)
        self.initialize_parameters()
        self.all_potentials = {}

    def load_config(self, config_path):
        """Load configuration from a YAML file."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

    def initialize_parameters(self):
        """Initialize simulation parameters and attributes."""
        # General parameters
        self.alpha = self.config.get("alpha", 0.01)
        self.n_iter = self.config.get("n_iter", 100)
        self.run_lammps = self.config.get("run_lammps_command", "lmp_mpi")
        self.n_cpus = self.config.get("n_cpus", int(os.getenv("SLURM_CPUS_ON_NODE", 1)))
        self.temp = self.config.get("temp", 300)

        # RDF-related parameters
        self.RDF_cutoff = self.config.get("RDF_cutoff", 10.0)
        self.RDF_delta_r = self.config.get("RDF_delta_r", 0.1)
        self.r_pot = self._generate_r_pot(self.RDF_cutoff, self.RDF_delta_r)
        self.pair_types = self.config.get("pair_types", [])
        self.pdf_ref = self._load_rdf_reference(self.config.get("RDF_ref"))

        # Bond-related parameters
        self.bond_types = self.config.get("bond_types", [])
        self.alpha_bond = self.config.get("alpha_bond", 0.01)
        self.bond_length_ref = self._load_bond_reference(self.config.get("bond_ref"))

        # Angle-related parameters
        self.angle_types = self.config.get("angle_types", [])
        self.alpha_angle = self.config.get("alpha_angle", 0.01)
        self.angle_dist_ref = self._load_angle_reference(self.config.get("angle_ref"))

        # Density correction
        self.density_ref = self.config.get("density_ref", 0.8)
        self.turn_on_LC = False
        self.rho = []

        # Smoothing parameters
        self.smooth_sigma = self.config.get("smooth_sigma", 3)

    def _generate_r_pot(self, cutoff, delta_r):
        """Generate r_pot array based on cutoff and delta_r."""
        r_tmp = np.arange(0, cutoff, delta_r)
        return (r_tmp[1:] + r_tmp[:-1]) / 2

    def _load_rdf_reference(self, rdf_path):
        """Load RDF reference data."""
        if not rdf_path or not os.path.exists(rdf_path):
            raise FileNotFoundError(f"RDF reference file not found: {rdf_path}")
        rdf_data = np.loadtxt(rdf_path)
        rdf_data = rdf_data[:len(self.r_pot), :]
        return {f"pair_{key}": rdf_data[:, i + 1] for i, key in enumerate(self.pair_types)}

    def _load_bond_reference(self, bond_path):
        """Load bond reference data."""
        if not bond_path or not os.path.exists(bond_path):
            return {}
        bond_data = np.loadtxt(bond_path)
        return {f"bond_{key}": bond_data[:, i + 1] for i, key in enumerate(self.bond_types)}

    def _load_angle_reference(self, angle_path):
        """Load angle reference data."""
        if not angle_path or not os.path.exists(angle_path):
            return {}
        angle_data = np.loadtxt(angle_path)
        return {f"angle_{key}": angle_data[:, i + 1] for i, key in enumerate(self.angle_types)}

    def run_simulation(self, start=1):
        """Run the simulation for the specified number of iterations."""
        for i in range(start, self.n_iter + start):
            start_time = time.time()
            self.i_iter = i
            self.run_iteration()
            end_time = time.time()
            print(f"Iteration {i}, Time taken: {end_time - start_time:.2f} seconds", flush=True)
            self._apply_lr_decay(i)

    def _apply_lr_decay(self, iteration):
        """Apply learning rate decay if enabled."""
        if self.config.get("is_lr_decay") and iteration % self.config.get("decay_freq", 10) == 0:
            min_alpha = self.config.get("min_alpha", 0.001)
            if abs(self.alpha) > abs(min_alpha):
                self.alpha *= self.config.get("decay_rate", 0.9)

    def run_iteration(self):
        """Run a single iteration of the simulation."""
        self.directory = f"CG{self.i_iter}"
        os.makedirs(self.directory, exist_ok=True)
        self._prepare_potentials()
        self._run_lammps_simulation()
        self.process_results_bonded()
        self.update_potentials()

    def _prepare_potentials(self):
        """Prepare potential files for the simulation."""
        r_pot_write = copy(self.r_pot)
        r_pot_write[0] = 1e-8
        tmp_args = [(r_pot_write, self.e_pot[f"pair_{key}"], self.f_pot[f"pair_{key}"], f"pair_{key}") for key in self.pair_types]
        tcp.write_pot_table(f"{self.directory}/pot.table", tmp_args)

    def _run_lammps_simulation(self):
        """Run the LAMMPS simulation."""
        subprocess.run(f"{self.run_lammps} -in {self.config['lmp_input']} > log", shell=True, cwd=self.directory)

    def process_results_bonded(self):
        """Process bonded results from the simulation."""
        dump_file = os.path.join(self.directory, "dump.xyz")
        output_file = os.path.join(self.directory, "out.dat")
        log_file = os.path.join(self.directory, "log.lammps")
        # Process results (simplified for brevity)
        self.property = self._compute_properties(dump_file, output_file, log_file)

    def update_potentials(self):
        """Update potentials based on the computed properties."""
        # Update logic (simplified for brevity)
        pass

    def plot_results(self):
        """Plot simulation results."""
        # Plotting logic (simplified for brevity)
        pass

# Usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run IBI simulation.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Configuration file path")
    parser.add_argument("--start", type=int, default=1, help="Start iteration")
    args = parser.parse_args()

    simulation = IBISimulation(args.config)
    simulation.run_simulation(start=args.start)
    joblib.dump(simulation, "simulation.pkl")