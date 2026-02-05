"""
Experiment: Hop Depth Effect on Effective Delay Regression
"""

import os
import json
from datetime import datetime

# Import necessary libraries
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import os.path as op
from scipy.ndimage import gaussian_filter1d
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
    import warnings
    import logging

    if not verbose:
        # Suppress Python warnings
        warnings.filterwarnings("ignore")
        # Quiet common noisy libraries
        for logger_name in ("matplotlib", "networkx", "numba", "flowgsp", "urllib3"):
            logging.getLogger(logger_name).setLevel(logging.ERROR)

    if verbose:
        print("=" * 60)
        print("Experiment: Hop Depth Effect on Effective Delay Regression")
        print("=" * 60)

    # Configuration
    config = load_json(os.path.join(EXPERIMENT_DIR, "config.json"))

    if verbose:
        print(f"\nConfiguration: {config}")

    fig1 = None
    experiments = Experiments(config, verbose=verbose)
    print("\nRunning Experiment 1: Effect of Hop Depth on Delay Regression")
    fig1 = experiments.run_experiment1()

    results = {
        "config": config,
        "timestamp": datetime.now().isoformat(),
    }

    # Save results
    if save_results:
        os.makedirs(RESULTS_DIR, exist_ok=True)

        if fig1 is not None:
            fig1_file = os.path.join(RESULTS_DIR, "experiment_figure1.png")
            fig1.savefig(fig1_file, dpi=300)

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

        self.scale = self.config["scale"]
        self.age_range = self.config["age_range"]
        self.delay_max = self.config["delay"]
        self.feature = self.config["feature"]
        self.bundle_prob_thresh = self.config["bundle_prob_thresh"]
        self.n_iter = self.config["optimization_parameters"]["n_iter"]
        self.step_size = self.config["optimization_parameters"]["step_size"]
        self.early_stop = self.config["optimization_parameters"]["early_stop"]
        self.l2_penalty = self.config["optimization_parameters"]["l2_penalty"]

        self.ftracts = load("/Users/mikichan/Desktop/mywork/data_resources/atlas_data/fiber_atlas_2018/Lausanne2018_FTRACT_NEW/agg_ftract_dict_allscales_age_ranges_delays_features.pkl")
        self.path_to_bundle_atlas = f"/Users/mikichan/Desktop/mywork/data_resources/atlas_data/fiber_atlas_2018/probconnatlas/wm.connatlas.scale{self.scale}.h5"

        hf = h5py.File(self.path_to_bundle_atlas, 'r')
        self.gmregions_names = hf.get('header').get('gmregions')[()]

        consistency_view = self.get_aggprop(hf, 'consistency')
        n = consistency_view.shape[0]
        adj = consistency_view[:n-1, :n-1]
        adj -= np.diag(np.diag(adj))

        self.adj = (adj > self.config['bundle_prob_thresh']).astype(int)

        self.depths = [1, 2]
        self.design_matrices_depths = []
        
        for depth in self.depths:
            if op.exists(op.join(DATA_DIR, f"design_matrices_depth_{self.scale}_{self.age_range}_{self.delay_max}_{depth}.pkl")):
                design_matrices = load(op.join(DATA_DIR, f"design_matrices_depth_{self.scale}_{self.age_range}_{self.delay_max}_{depth}.pkl"))
                self.design_matrices_depths.append(design_matrices)
            else:
                design_matrices = regmod.get_shortest_matrices(self.adj, depth, progress=True)
                save(op.join(DATA_DIR, f"design_matrices_depth_{self.scale}_{self.age_range}_{self.delay_max}_{depth}.pkl"), design_matrices)
                self.design_matrices_depths.append(design_matrices)

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
        # Regression for different max path depths
        if os.path.exists(op.join(DATA_DIR, f"effective_delays_maxdepth_sets.pkl")):
            x_opts, losses = load(op.join(DATA_DIR, f"effective_delays_maxdepth_sets.pkl"))
        else:
            x_opts = []
            losses = []
            for d, depth in enumerate(self.depths):
                a = 0.8

                design_model = regmod.apply_alpha_to_design(self.design_matrices_depths[d], n_subopt=depth, alpha=a)
                design_model = solver.torch.tensor(design_model)

                np.random.seed(99)
                x_init = solver.torch.tensor(np.random.rand(len(self.y_ground))).requires_grad_(True)
                x = deepcopy(x_init)
                x_opt, loss = solver.gradient_descent_solver(x, self.y_ground, design_model,
                                                            n_iter=self.n_iter, verbose=False, 
                                                            early_stop=self.early_stop, step_size=self.step_size, delta=0,
                                                            l2_penalty=self.l2_penalty)
                x_opts.append(x_opt)
                losses.append(loss)
            save(op.join(DATA_DIR, f"effective_delays_maxdepth_sets.pkl"), (x_opts, losses))

        y = self.y_ground

        x_opt1, x_opt2 = x_opts
        x_mask1 = x_opt1 > 1
        x_mask2 = x_opt2 > 1
        y_mask = y != 0
        xy_mask1 = np.logical_and(x_mask1, y_mask).numpy().astype(bool)
        xy_mask2 = np.logical_and(x_mask2, y_mask).numpy().astype(bool)

        nb_bins = 16
        bins = np.linspace(10, 210, nb_bins + 1)
        bins_center = (bins[:-1] + bins[1:]) /  2
        avg1 = np.zeros(nb_bins)
        avg2 = np.zeros(nb_bins)
        for bidx in range(len(bins)-1):
            bin_flag1 = (bins[bidx] < y[xy_mask1]) & (y[xy_mask1] < bins[bidx+1])
            bin_flag2 = (bins[bidx] < y[xy_mask2]) & (y[xy_mask2] < bins[bidx+1])
            
            avg1[bidx] = x_opt1[xy_mask1][bin_flag1].mean()
            avg2[bidx] = x_opt2[xy_mask2][bin_flag2].mean()

        y = self.y_ground

        x_mask1 = x_opt1 > 1
        x_mask2 = x_opt2 > 1
        y_mask = y != 0
        xy_mask1 = np.logical_and(x_mask1, y_mask).numpy().astype(bool)
        xy_mask2 = np.logical_and(x_mask2, y_mask).numpy().astype(bool)

        fig, ax = plt.subplots(1, 1, figsize=(12,8))

        ax.scatter(y[xy_mask1], x_opt1[xy_mask1], s=30, alpha=.1, 
                edgecolors="none", color='red')
        ax.scatter(y[xy_mask2], x_opt2[xy_mask2], s=30, alpha=.1, 
                edgecolors="none", color='blue')

        ax.plot(np.linspace(0,100), np.linspace(0,100), linestyle='--', 
                color="gray", linewidth=2, alpha=0.5, label="1:1")

        ksize = 1
        ax.plot(bins_center, gaussian_filter1d(avg1, ksize), linestyle='--', 
                color="red", linewidth=2, label='Depth: 2')
        ax.plot(bins_center, gaussian_filter1d(avg2, ksize), linestyle='--', 
                color="blue", linewidth=2, label='Depth: 3')

        ax.set_xlim(0, 250)
        ax.set_ylim(0, 120)
        ax.set_xlabel("Conductance delays", fontsize=20)
        ax.set_ylabel("Effective delays", fontsize=20)
        ax.tick_params(labelsize=20)

        ax.legend(fontsize=20)

        fig.tight_layout()
        if not self.verbose:
            plt.close()
        plt.show()

        return fig

if __name__ == "__main__":
    run()
