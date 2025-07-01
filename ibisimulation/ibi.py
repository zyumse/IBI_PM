#!/usr/bin/env python3

class SimulationWrapper:
    def __init__(self, config_path):
        from .initialization import Initializer
        from .potential_manager import PotentialManager
        import yaml
        import matplotlib.pyplot as plt

        self.initializer = Initializer(config_path)
        # self.potential_manager = PotentialManager(self.initializer)

    def run_simulation(self):
        print("Running simulation with the following parameters:")
        print(self.initializer.config)
        for i in range(start, self.n_iter + start):
            start_time = time.time()
            self.i_iter = i
            self.run_iteration()
            end_time = time.time()
            print(f"Iteration {i}, Time taken: {end_time - start_time:.2f} seconds", flush=True)
            if self.config["is_lr_decay"]:
                if i % self.config["decay_freq"] == 0 and np.abs(self.alpha) > np.abs(
                    self.config["min_alpha"]
                ):
                    self.alpha *= self.config["decay_rate"]

    def plot_results(self):
        plt.figure()
        plt.title("Simulation Results")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.show()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the simulation wrapper.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the configuration file.")
    parser.add_argument("--plot", action='store_true', help="Specify whether to plot results.")
    args = parser.parse_args()

    simulation = SimulationWrapper(args.config)
    simulation.run_simulation()

    if args.plot:
        simulation.plot_results()