from .initialization import initialize_parameters
from .iteration import run_iteration
from .results_processing import process_results_bonded
from .potentials import update_potentials
from .plotting import plot_results

class IBISimulation:
    def __init__(self, config_path):
        """Initialize the simulation with the given configuration file."""
        self.config = initialize_parameters(config_path)
        self.all_potentials = {}

    def run_simulation(self, start=1):
        """Run the simulation for the specified number of iterations."""
        for i in range(start, self.config["n_iter"] + start):
            self.run_iteration(i)
            self.process_results_bonded()
            self.update_potentials()
            self.plot_results()