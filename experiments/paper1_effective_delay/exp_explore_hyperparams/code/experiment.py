"""
Experiment: Exploring the effects of hyperparameters alpha and delta on effective delay regression.
"""

import os
import json
from datetime import datetime

# Import necessary libraries
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from copy import deepcopy
import os.path as op
import h5py

from src import regmod
from src import solver

from src.utils import load_json, load, save, annotate_heatmap, add_cbar, remove_diagonal_entries, add_diagonal_entries

# Get the directory of this script for relative paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(EXPERIMENT_DIR, "data")
RESULTS_DIR = os.path.join(EXPERIMENT_DIR, "results")


def run(save_results: bool = True, verbose: bool = True) -> dict:
    """
    Run the basic experiment.

    Parameters
    ----------
    save_results : bool
        Whether to save results to the results directory.
    verbose : bool
        Whether to print progress information.

    Returns
    -------
    dict
        Dictionary containing experiment results.
    """
    if verbose:
        print("=" * 60)
        print("Experiment: Exploring the effects of hyperparameters alpha and delta on effective delay regression")
        print("=" * 60)

    # Configuration
    config = load_json(os.path.join(EXPERIMENT_DIR, "config.json"))

    if verbose:
        print(f"\nConfiguration: {config}")

    experiments = Experiments(config, verbose=verbose)

    fig1, fig2 = None, None  # Placeholder for figures from experiment 1

    print("\nRunning Experiment 1: Alpha exploration on F-TRACT 2018 data")
    fig1 = experiments.run_experiment1()

    print("\nRunning Experiment 2: Delta exploration on F-TRACT 2018 data")
    fig2 = experiments.run_experiment2()

    results = {
        "config": config,
        "timestamp": datetime.now().isoformat(),
    }

    # Save results
    if save_results:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        if fig1 is not None:
            fig1.savefig(
                os.path.join(RESULTS_DIR, f"alpha_exploration.png"),
                dpi=300,
                bbox_inches="tight",
            )
        
        if fig2 is not None:
            fig2.savefig(
                os.path.join(RESULTS_DIR, f"delta_exploration.png"),
                dpi=300,
                bbox_inches="tight",
            )

        results_file = os.path.join(RESULTS_DIR, "experiment_results.json")
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        if verbose:
            print(f"\nResults saved to: {RESULTS_DIR}")

    if verbose:
        print("\n" + "=" * 60)
        print("Experiment completed successfully!")
        print("=" * 60)


class Experiments:
    def __init__(self, config: dict, verbose: bool = True):
        self.config = config
        self.verbose = verbose
        self.path_to_resources = './resources/'
        self.title_fontsize = 14
        self.ticks_labels_fontsize = 12
        
        self.scale = self.config["scale"]
        self.age_range = self.config["age_range"]
        self.delay_max = self.config["delay"]
        self.feature = self.config["feature"]
        self.n_iter = self.config["optimization_parameters"]["n_iter"]
        self.step_size = self.config["optimization_parameters"]["step_size"]
        self.early_stop = self.config["optimization_parameters"]["early_stop"]
        self.l2_penalty = self.config["optimization_parameters"]["l2_penalty"]

        # F-TRACT 2018 data
        self.ftracts = load("/Users/mikichan/Desktop/mywork/data_resources/atlas_data/fiber_atlas_2018/Lausanne2018_FTRACT_NEW/agg_ftract_dict_allscales_age_ranges_delays_features.pkl")

        self.path_to_bundle_atlas = f"/Users/mikichan/Desktop/mywork/data_resources/atlas_data/fiber_atlas_2018/probconnatlas/wm.connatlas.scale{self.scale}.h5"

    def get_aggprop(self, h5dict: h5py._hl.files.File, property: str):
        """
        Get the bundles statistics on whole brain level from the HDF5 file.

        Parameters
        ----------
        h5dict : h5py._hl.files.File 
            The opened HDF5 file.
        property : str
            The property to extract from the HDF5 file.

        Returns
        -------
        ret : np.array
            The array containing the requested property values.
        """

        try:
            ret = np.array(h5dict.get("matrices").get(property))
        except:
            print("Not valid property OR h5 not opened")
        return ret

    def run_experiment1(self):
        # Experiment on Real Conductance Delays with increasing alpha
        hf = h5py.File(self.path_to_bundle_atlas, 'r')
        self.gmregions_names = hf.get('header').get('gmregions')[()]

        consistency_view = self.get_aggprop(hf, 'consistency')
        consistency_view = consistency_view.astype(float) / float(consistency_view.max()) # Normalize to [0,1]
        n = consistency_view.shape[0]
        adj = consistency_view[:n-1, :n-1]

        adj = (adj > self.config['bundle_prob_thresh']).astype(int)

        # Conductance delays from F-TRACT 2018
        prob_thresh = 0.0

        dict_key = f"scale{self.scale}__{self.age_range}__{self.delay_max}__{self.feature}"

        y_ground_mat = self.ftracts[dict_key]
        y_ground_mat = y_ground_mat[:n-1, :n-1]
        y_ground_mat *= (y_ground_mat > prob_thresh)
        y_ground = solver.torch.tensor(remove_diagonal_entries(y_ground_mat).flatten())

        alpha_range = np.linspace(0, 1, 6)
        delta = 0
        if os.path.exists(op.join(DATA_DIR, f"ftract_delay_regress_alpha_range_{alpha_range}_{delta}_{self.scale}_{self.age_range}_{self.delay_max}.pkl")):
            x_opts, losses = load(op.join(DATA_DIR, f"ftract_delay_regress_alpha_range_{alpha_range}_{delta}_{self.scale}_{self.age_range}_{self.delay_max}.pkl"))
        else:
            design_matrices = regmod.get_shortest_matrices(adj, self.config['max_path_depth'], progress=True)

            x_opts, losses = [], []
            for alpha in alpha_range:
                design_shortest = regmod.apply_alpha_to_design(design_matrix=design_matrices, n_subopt=self.config['max_path_depth'], alpha=alpha)
                design_model = solver.torch.tensor(design_shortest)

                np.random.seed(99)
                x_init = solver.torch.tensor(np.random.rand(len(y_ground)))

                x = deepcopy(x_init)
                x_opt, loss = solver.gradient_descent_solver(x, y_ground, design_model,
                                                            n_iter=self.n_iter, verbose=False, 
                                                            early_stop=self.early_stop, step_size=self.step_size, delta=delta,
                                                            l2_penalty=self.l2_penalty)
                x_opts.append(x_opt)
                losses.append(loss)

            save(op.join(DATA_DIR, f"ftract_delay_regress_alpha_range_{alpha_range}_{delta}_{self.scale}_{self.age_range}_{self.delay_max}.pkl"), (x_opts, losses))

        # plot the mapping curve and see what it looks like
        fig, ax = plt.subplots(1,1, figsize=(8,4))

        import matplotlib.cm as cm
        colors = cm.rainbow(np.linspace(0, 1, len(alpha_range)))

        for aidx, (alpha, x_opt) in enumerate(zip(alpha_range, x_opts)):

            x1_mask = x_opt > 1
            y_mask = y_ground != 0
            xy_mask1 = np.logical_and(x1_mask, y_mask).numpy().astype(bool)

            ax.scatter(y_ground[xy_mask1], x_opt[xy_mask1], s=20, alpha=.25, edgecolors="black", color=colors[aidx], label=r'$\alpha=$' + f'{alpha: .1f}')
            x_vals = y_ground[xy_mask1].detach().cpu().numpy()
            y_vals = x_opt[xy_mask1]

            if x_vals.size > 2:
                # Bin the x values and compute means
                n_bins = 20
                x_bins = np.linspace(x_vals.min(), x_vals.max(), n_bins)
                bin_indices = np.digitize(x_vals, x_bins)
                
                bin_means_x = []
                bin_means_y = []
                for bin_idx in range(1, n_bins):
                    mask = bin_indices == bin_idx
                    if mask.sum() > 0:
                        bin_means_x.append(x_vals[mask].mean())
                        bin_means_y.append(y_vals[mask].mean())
                
                if bin_means_x:
                    ax.plot(bin_means_x, bin_means_y, color=colors[aidx], linewidth=2, alpha=0.9, marker='o', markersize=6)

        ax.plot(np.linspace(0,x_opt.max()), np.linspace(0,x_opt.max()), linestyle='--', color="black", linewidth=2, label="1:1")

        ax.set_xlabel("Delays (C)", fontsize=16)
        ax.set_ylabel("Estimated Delays (E)", fontsize=16)
        ax.tick_params(labelsize=14)
        ax.legend(fontsize=12)
        ax.grid(axis='both', 
                linestyle='--', 
                alpha=0.7,
                color='gray',
                linewidth=0.5)
        
        fig.tight_layout()
        if not self.verbose:
            plt.close()
        plt.show()

        return fig
    
    def run_experiment2(self):
        # Experiment on Real Conductance Delays with increasing delta
        hf = h5py.File(self.path_to_bundle_atlas, 'r')
        self.gmregions_names = hf.get('header').get('gmregions')[()]

        consistency_view = self.get_aggprop(hf, 'consistency')
        consistency_view = consistency_view.astype(float) / float(consistency_view.max()) # Normalize to [0,1]
        n = consistency_view.shape[0]
        adj = consistency_view[:n-1, :n-1]

        adj = (adj > self.config['bundle_prob_thresh']).astype(int)

        # Conductance delays from F-TRACT 2018
        prob_thresh = 0.0

        dict_key = f"scale{self.scale}__{self.age_range}__{self.delay_max}__{self.feature}"

        y_ground_mat = self.ftracts[dict_key]
        y_ground_mat = y_ground_mat[:n-1, :n-1]
        y_ground_mat *= (y_ground_mat > prob_thresh)
        y_ground = solver.torch.tensor(remove_diagonal_entries(y_ground_mat).flatten())

        delta_range = np.arange(0, 30, 3)
        if os.path.exists(op.join(DATA_DIR, f"ftract_delay_regress_delta_range_{delta_range}_{self.scale}_{self.age_range}_{self.delay_max}.pkl")):
            x_opts, losses = load(op.join(DATA_DIR, f"ftract_delay_regress_delta_range_{delta_range}_{self.scale}_{self.age_range}_{self.delay_max}.pkl"))
        else:
            alpha = 0.5
            design_matrices = regmod.get_shortest_matrices(adj, self.config['max_path_depth'], progress=True)

            x_opts, losses = [], []
            for delta in delta_range:
                design_shortest = regmod.apply_alpha_to_design(design_matrix=design_matrices, n_subopt=self.config['max_path_depth'], alpha=alpha)
                design_model = solver.torch.tensor(design_shortest)

                np.random.seed(99)
                x_init = solver.torch.tensor(np.random.rand(len(y_ground)))

                x = deepcopy(x_init)
                x_opt, loss = solver.gradient_descent_solver(x, y_ground, design_model,
                                                            n_iter=self.n_iter, verbose=False, 
                                                            early_stop=self.early_stop, step_size=self.step_size, delta=delta,
                                                            l2_penalty=self.l2_penalty)
                x_opts.append(x_opt)
                losses.append(loss)

            save(op.join(DATA_DIR, f"ftract_delay_regress_delta_range_{delta_range}_{self.scale}_{self.age_range}_{self.delay_max}.pkl"), (x_opts, losses))

        # plot the mapping curve and see what it looks like
        fig, ax = plt.subplots(1,1, figsize=(8,4))

        import matplotlib.cm as cm
        colors = cm.rainbow(np.linspace(0, 1, len(delta_range)))

        for didx, (delta, x_opt) in enumerate(zip(delta_range, x_opts)):
            x1_mask = x_opt > 1
            y_mask = y_ground != 0
            xy_mask1 = np.logical_and(x1_mask, y_mask).numpy().astype(bool)

            ax.scatter(y_ground[xy_mask1], x_opt[xy_mask1], s=20, alpha=.25, edgecolors="black", color=colors[didx], label=r'$\delta=$' + f'{delta}')
            x_vals = y_ground[xy_mask1].detach().cpu().numpy()
            y_vals = x_opt[xy_mask1]

            if x_vals.size > 2:
                # Bin the x values and compute means
                n_bins = 20
                x_bins = np.linspace(x_vals.min(), x_vals.max(), n_bins)
                bin_indices = np.digitize(x_vals, x_bins)
                
                bin_means_x = []
                bin_means_y = []
                for bin_idx in range(1, n_bins):
                    mask = bin_indices == bin_idx
                    if mask.sum() > 0:
                        bin_means_x.append(x_vals[mask].mean())
                        bin_means_y.append(y_vals[mask].mean())
                
                if bin_means_x:
                    ax.plot(bin_means_x, bin_means_y, color=colors[didx], linewidth=2, alpha=0.9, marker='o', markersize=6)

        ax.plot(np.linspace(0,x_opt.max()), np.linspace(0,x_opt.max()), linestyle='--', color="black", linewidth=2, label="1:1")

        ax.set_xlabel("Delays (C)", fontsize=16)
        ax.set_ylabel("Estimated Delays (E)", fontsize=16)
        ax.tick_params(labelsize=14)
        ax.legend(fontsize=12)
        ax.grid(axis='both', 
                linestyle='--', 
                alpha=0.7,
                color='gray',
                linewidth=0.5)
        ax.set_xlim(-20, y_ground.max() + 60)

        fig.tight_layout()
        if not self.verbose:
            plt.close()
        plt.show()

        return fig

if __name__ == "__main__":
    run()
