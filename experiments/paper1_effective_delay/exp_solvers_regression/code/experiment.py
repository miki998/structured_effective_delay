"""
Experiment: Compare Solvers for Regression of Effective Delays
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
        print("Experiment: Compare Solvers for Regression of Effective Delays")
        print("=" * 60)

    # Configuration
    config = load_json(os.path.join(EXPERIMENT_DIR, "config.json"))

    if verbose:
        print(f"\nConfiguration: {config}")

    experiments = Experiments(config, verbose=verbose)

    figs1 = [None, None]  # Placeholder for figures from experiment 1
    figs2 = [None, None]  # Placeholder for figures from experiment 2
    print("\nRunning Experiment 1: Bundle Probability Atlas with Synthetic Delays")
    figs1 = experiments.run_experiment1()

    print("\nRunning Experiment 2: Bundle Probability Atlas with Real Delays")
    figs2 = experiments.run_experiment2()

    results = {
        "config": config,
        "timestamp": datetime.now().isoformat(),
    }

    # Save results
    if save_results:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        for i, fig in enumerate(figs1):
            if fig is not None:
                fig.savefig(
                    os.path.join(RESULTS_DIR, f"synthetic_graph_{i}.png"),
                    dpi=300,
                    bbox_inches="tight",
                )
        
        for i, fig in enumerate(figs2):
            if fig is not None:
                fig.savefig(
                    os.path.join(RESULTS_DIR, f"bundle_probability_atlas_{i}.png"),
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
        # Synthetic Conductance Delays with Bundle Probability Atlas
        hf = h5py.File(self.path_to_bundle_atlas, 'r')
        self.gmregions_names = hf.get('header').get('gmregions')[()]

        consistency_view = self.get_aggprop(hf, 'consistency')
        n = consistency_view.shape[0]
        adj = consistency_view[:n-1, :n-1]
        adj -= np.diag(np.diag(adj))

        adj = (adj > self.config['bundle_prob_thresh']).astype(int)
        x_ground = remove_diagonal_entries(adj).flatten()

        true_a, true_delta = 0.5, 0
        guess_a, guess_delta = 0.0, 1.0

        if op.exists(op.join(DATA_DIR, f"design_matrices_{self.config['scale']}.pkl")):
            design_matrices = load(op.join(DATA_DIR, f"design_matrices_{self.config['scale']}.pkl"))
        else:
            design_matrices = regmod.get_shortest_matrices(adjacency=adj, n_subopt=self.config['max_path_depth'])
            save(op.join(DATA_DIR, f"design_matrices_{self.config['scale']}.pkl"), design_matrices)
        
        if op.exists(op.join(DATA_DIR, f"compare_syn_ftract_exp_{true_a}_{true_delta}_{self.config['scale']}.pkl")):
            compare_solvers = load(op.join(DATA_DIR, f"compare_syn_ftract_exp_{true_a}_{true_delta}_{self.config['scale']}.pkl"))
        else:
            design_model = solver.torch.tensor(regmod.apply_alpha_to_design(design_matrices, n_subopt=self.config['max_path_depth'], alpha=true_a))

            guess_design_model = solver.torch.tensor(regmod.apply_alpha_to_design(design_matrices, n_subopt=self.config['max_path_depth'], alpha=guess_a))

            y_ground = solver.forward(design_model.float(), solver.torch.tensor(x_ground).float() + true_delta * (solver.torch.tensor(x_ground).float() > 0))

            np.random.seed(99)
            x_init = np.random.rand(len(x_ground))
            
            x = solver.torch.tensor(x_init).requires_grad_(True)
            x_opt_gd, loss_gd = solver.gradient_descent_solver(x, y_ground, guess_design_model, delta=guess_delta,
                                                n_iter=self.n_iter, verbose=self.verbose, 
                                                early_stop=self.early_stop, step_size=self.step_size,
                                                l2_penalty=self.l2_penalty)
            
            x = solver.torch.tensor(x_init).float().requires_grad_(True)
            x_opt_alpha, a_est, loss_alpha = solver.gradient_descent_solver_alpha(x, y_ground, solver.torch.tensor(design_matrices),
                                                               alpha=solver.torch.tensor(guess_a), delta=guess_delta,
                                                               n_iter=self.n_iter, verbose=self.verbose,early_stop=self.early_stop, step_size=self.step_size,l2_penalty=self.l2_penalty)
            
            x = solver.torch.tensor(x_init).requires_grad_(True)
            x_opt_delta, delta_est, loss_delta = solver.gradient_descent_solver_delta(x, y_ground, guess_design_model,
                                                               delta=solver.torch.tensor(guess_delta),
                                                               n_iter=self.n_iter, verbose=self.verbose,early_stop=self.early_stop, step_size=self.step_size,l2_penalty=self.l2_penalty)
            
            x = solver.torch.tensor(x_init).float().requires_grad_(True)
            x_opt_joint, (a_joint_est, delta_joint_est), loss_joint = solver.effective_delay_solver(x, y_ground, solver.torch.tensor(design_matrices),
                                                               alpha=solver.torch.tensor(guess_a), delta=solver.torch.tensor(guess_delta),
                                                               n_iter=self.n_iter, verbose=self.verbose,early_stop=self.early_stop, step_size=self.step_size,l2_penalty=self.l2_penalty)

            compare_solvers = {"vanilla": (x_opt_gd, loss_gd), "alpha": (x_opt_alpha, a_est, loss_alpha), "delta": (x_opt_delta, delta_est, loss_delta), "joint": (x_opt_joint, (a_joint_est, delta_joint_est), loss_joint), "design_matrices": design_matrices}

            save(op.join(DATA_DIR, f"compare_syn_ftract_exp_{true_a}_{true_delta}_{self.config['scale']}.pkl"), compare_solvers)

        design_model = solver.torch.tensor(regmod.apply_alpha_to_design(compare_solvers["design_matrices"], n_subopt=self.config['max_path_depth'], alpha=true_a))
        x_ground_mat = add_diagonal_entries(x_ground.reshape(adj.shape[0], adj.shape[1]-1))
        y_ground = solver.forward(design_model.float(), solver.torch.tensor(x_ground).float() + true_delta * (solver.torch.tensor(x_ground).float() > 0))
        y_ground_mat = add_diagonal_entries(y_ground.numpy().reshape(adj.shape[0], adj.shape[1]-1))

        colors = ['blue', 'orange', 'green', 'red']
        fig1, axes = plt.subplots(nrows=2, ncols=3, figsize=(11, 6))
        axes[0, 0].imshow(x_ground_mat, cmap='gray')
        axes[0, 0].set_title("Effective delays $x=\mathbf{1}$\n (if bundle)")
        add_cbar(fig1, axes[0, 0])
        axes[1, 2].plot(np.linspace(x_ground.min(), x_ground.max()), np.linspace(x_ground.min(), x_ground.max()), linestyle='--', color='gray', linewidth=2, label="1:1")
        axes[1, 2].set_xlabel("Effective delays Ground", fontsize=12)
        axes[1, 2].set_ylabel("Effective delays Predicted", fontsize=12)
        for i, key in enumerate(["vanilla", "alpha", "delta", "joint"]):
            j = i + 1
            x_opt = compare_solvers[key][0]
            loss = compare_solvers[key][-1]
            x_pred_mat = add_diagonal_entries(x_opt.reshape(adj.shape[0], adj.shape[1]-1))
            axes[j//3, j%3].imshow(x_pred_mat, cmap='gray')#, vmax=y_pred_mat.max())
            axes[j//3, j%3].set_title(f"Method: {key}, \nloss={np.round(loss,4)}")
            add_cbar(fig1, axes[j//3, j%3])
            axes[1, 2].scatter(x_ground, x_opt, s=20, alpha=.5, edgecolors="black", color=colors[i], label=key)

        axes[1, 2].legend()
        fig1.tight_layout()
        if not self.verbose:
            plt.close()
        plt.show()

        fig2, ax = plt.subplots(2, 3, figsize=(11, 6))
        ax[0, 0].set_title('Ground Truth')
        ax[0, 0].imshow(y_ground_mat, cmap='gray')
        add_cbar(fig2, ax[0, 0])
        ax[1, 2].plot(np.linspace(y_ground.numpy().min(), y_ground.numpy().max()), np.linspace(y_ground.numpy().min(), y_ground.numpy().max()), linestyle='--', color='gray', linewidth=2, label="1:1")
        ax[1, 2].set_xlabel("Conductance Estimated", fontsize=12)
        ax[1, 2].set_ylabel("Conductance Predicted", fontsize=12)

        for i, key in enumerate(["vanilla", "alpha", "delta", "joint"]):
            j = i + 1
            x_opt = compare_solvers[key][0]
            if key in ["vanilla", "delta"]:
                alpha = guess_a
            else:
                alpha = compare_solvers[key][1] if key == "alpha" else compare_solvers[key][1][0]
            if key in ["vanilla", "alpha"]:
                delta = guess_delta
            else:
                delta = compare_solvers[key][1] if key == "delta" else compare_solvers[key][1][1]

            design_model = solver.torch.tensor(regmod.apply_alpha_to_design(compare_solvers["design_matrices"], n_subopt=self.config['max_path_depth'], alpha=alpha))

            y_est = solver.forward(design_model.float(), solver.torch.tensor(x_ground).float() + delta * (solver.torch.tensor(x_ground).float() > 0))
            y_est_mat = add_diagonal_entries(y_est.numpy().reshape(adj.shape[0], adj.shape[1]-1))

            ax[j//3, j%3].set_title(f"Method: {key}, \n" + rf"$\alpha={np.round(alpha,4)}$, $\delta={np.round(delta,4)}$")
            ax[j//3, j%3].imshow(y_est_mat, cmap='gray')#, vmax=y_pred_mat.max())
            add_cbar(fig2, ax[j//3, j%3])

            ax[1, 2].scatter(y_ground.numpy(), y_est.numpy(), s=20, alpha=.5, edgecolors="black", color=colors[i], label=key)

        ax[1, 2].legend()
        fig2.tight_layout()
        if not self.verbose:
            plt.close()
        plt.show()

        return fig1, fig2

    def run_experiment2(self):
        # Experiment on Real Conductance Delays
        hf = h5py.File(self.path_to_bundle_atlas, 'r')
        self.gmregions_names = hf.get('header').get('gmregions')[()]

        consistency_view = self.get_aggprop(hf, 'consistency')
        n = consistency_view.shape[0]
        adj = consistency_view[:n-1, :n-1]
        adj = (adj > self.config['bundle_prob_thresh']).astype(int)

        # Conductance delays from F-TRACT 2018
        prob_thresh = 0.0
        delay_to_compare = 100
        dict_key = f"scale{self.scale}__{self.age_range}__{self.delay_max}__{self.feature}"
        dict_key_compare = f"scale{self.scale}__{self.age_range}__{delay_to_compare}__{self.feature}"

        y_ground_mat = self.ftracts[dict_key]
        y_ground_mat = y_ground_mat[:n-1, :n-1]
        y_ground_mat *= (y_ground_mat > prob_thresh)
        y_ground = solver.torch.tensor(remove_diagonal_entries(y_ground_mat).flatten())

        y_ground_mat_compare = self.ftracts[dict_key_compare]
        y_ground_mat_compare = y_ground_mat_compare[:n-1, :n-1]
        y_ground_mat_compare *= (y_ground_mat_compare > prob_thresh)
        y_ground_compare = solver.torch.tensor(remove_diagonal_entries(y_ground_mat_compare).flatten())

        guess_a, guess_delta = 0.5, 0.0

        if op.exists(op.join(DATA_DIR, f"design_matrices_{self.config['scale']}.pkl")):
            design_matrices = load(op.join(DATA_DIR, f"design_matrices_{self.config['scale']}.pkl"))
        else:
            design_matrices = regmod.get_shortest_matrices(adjacency=adj, n_subopt=self.config['max_path_depth'])
            save(op.join(DATA_DIR, f"design_matrices_{self.config['scale']}.pkl"), design_matrices)

        if os.path.exists(op.join(DATA_DIR, f"compare_ftract_delay_regress_{self.scale}_{self.age_range}_{self.delay_max}.pkl")):
            compare_solvers = load(op.join(DATA_DIR, f"compare_ftract_delay_regress_{self.scale}_{self.age_range}_{self.delay_max}.pkl"))
        else:
            design_model = solver.torch.tensor(regmod.apply_alpha_to_design(design_matrix=design_matrices, n_subopt=self.config['max_path_depth'], alpha=guess_a))

            np.random.seed(99)
            x_init = np.random.rand(len(y_ground))
            
            x = solver.torch.tensor(x_init).requires_grad_(True)
            x_opt_gd, loss_gd = solver.gradient_descent_solver(x, y_ground, design_model, delta=guess_delta,
                                                n_iter=self.n_iter, verbose=self.verbose, 
                                                early_stop=self.early_stop, step_size=self.step_size,
                                                l2_penalty=self.l2_penalty)
            
            x = solver.torch.tensor(x_init).float().requires_grad_(True)
            x_opt_alpha, a_est, loss_alpha = solver.gradient_descent_solver_alpha(x, y_ground, solver.torch.tensor(design_matrices),
                                                               alpha=solver.torch.tensor(guess_a), delta=guess_delta,
                                                               n_iter=self.n_iter, verbose=self.verbose,early_stop=self.early_stop, step_size=self.step_size,l2_penalty=self.l2_penalty)
            
            x = solver.torch.tensor(x_init).requires_grad_(True)
            x_opt_delta, delta_est, loss_delta = solver.gradient_descent_solver_delta(x, y_ground, design_model,
                                                               delta=solver.torch.tensor(guess_delta),
                                                               n_iter=self.n_iter, verbose=self.verbose,early_stop=self.early_stop, step_size=self.step_size,l2_penalty=self.l2_penalty)
            
            x = solver.torch.tensor(x_init).float().requires_grad_(True)
            x_opt_joint, (a_joint_est, delta_joint_est), loss_joint = solver.effective_delay_solver(x, y_ground, solver.torch.tensor(design_matrices).float(),
                                                               alpha=solver.torch.tensor(guess_a), delta=solver.torch.tensor(guess_delta),
                                                               n_iter=self.n_iter, verbose=self.verbose,early_stop=self.early_stop, step_size=self.step_size,l2_penalty=self.l2_penalty)

            compare_solvers = {"vanilla": (x_opt_gd, loss_gd), "alpha": (x_opt_alpha, a_est, loss_alpha), "delta": (x_opt_delta, delta_est, loss_delta), "joint": (x_opt_joint, (a_joint_est, delta_joint_est), loss_joint), "design_matrices": design_matrices}

            save(op.join(DATA_DIR, f"compare_ftract_delay_regress_{self.scale}_{self.age_range}_{self.delay_max}.pkl"), compare_solvers)

        fig1, axes = plt.subplots(ncols=5, figsize=(13, 6))

        axes[0].imshow(y_ground_mat, cmap='gray')
        axes[0].set_title(f"Conduction delays $y$", fontsize=12)
        axes[0].set_xlabel("Region", fontsize=12)
        axes[0].set_ylabel("Region", fontsize=12)
        add_cbar(fig1, axes[0])

        for i, key in enumerate(["vanilla", "alpha", "delta", "joint"]):
            x_opt = compare_solvers[key][0]
            loss = compare_solvers[key][-1]
            x_pred_mat = add_diagonal_entries(x_opt.reshape(adj.shape[0], adj.shape[1]-1))
            axes[i+1].imshow(x_pred_mat, cmap='gray')#, vmax=y_pred_mat.max())
            axes[i+1].set_title(f"Method: {key},\n loss={np.round(loss,4)}")
            add_cbar(fig1, axes[i+1])

        fig1.tight_layout()
        if not self.verbose:
            plt.close()
        plt.show()

        fig2, ax = plt.subplots(1, 1, figsize=(10,6))

        x2 = y_ground_compare
        x2_mask = x2 > 1
        y = y_ground
        y_mask = y != 0
        xy_mask2 = np.logical_and(x2_mask, y_mask).numpy().astype(bool)
        ax.scatter(y[xy_mask2], x2[xy_mask2], s=20, alpha=.25, edgecolors="none", color='purple', label=r'peak-delay $100$')

        colors = ['blue', 'orange', 'green', 'red']
        for i, key in enumerate(["vanilla", "alpha", "delta", "joint"]):
            x_opt = compare_solvers[key][0]
            x1 = x_opt
            x1_mask = x1 > 1
            xy_mask1 = np.logical_and(x1_mask, y_mask).numpy().astype(bool)

            ax.scatter(y[xy_mask1], x1[xy_mask1], s=20, alpha=.25, edgecolors="none", color=colors[i], label=r'Method: ' + key)

        ax.plot(np.linspace(0,150), np.linspace(0,150), linestyle='--', color="gray", linewidth=2, label="1:1")

        ax.set_xlabel("Conductance delays", fontsize=16)
        ax.set_ylabel("Effective delays", fontsize=16)
        ax.tick_params(labelsize=14)
        ax.legend(fontsize=16)
        fig2.tight_layout()

        if not self.verbose:
            plt.close()
        plt.show()

        return fig1, fig2


if __name__ == "__main__":
    run()
