"""
Experiment: Basic Regression of Effective Delays
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
        print("Experiment: Regression of Effective Delays")
        print("=" * 60)

    # Configuration
    config = load_json(os.path.join(EXPERIMENT_DIR, "config.json"))

    if verbose:
        print(f"\nConfiguration: {config}")

    experiments = Experiments(config, verbose=verbose)

    figs1 = [None, None]  # Placeholder for figures from experiment 1
    figs2 = [None, None]  # Placeholder for figures from experiment 2
    figs3 = [None, None]  # Placeholder for figures from experiment 3

    # print("\nRunning Experiment 1: Synthetic Toy Graph")
    # figs1 = experiments.run_experiment1()

    print("\nRunning Experiment 2: Synthetic Bundle Probability Atlas")
    figs2 = experiments.run_experiment2()

    print("\nRunning Experiment 3: Bundle Probability Atlas + F-TRACT")
    figs3 = experiments.run_experiment3()

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

        for i, fig in enumerate(figs3):
            if fig is not None:
                fig.savefig(
                    os.path.join(RESULTS_DIR, f"bundle_probability_atlas_ftract_{i}.png"),
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
        # Synthetic Conductance Delays with Toy Graph
        adj = load(op.join(self.path_to_resources, "adjacency_synthetic.pkl"))
        adj -= np.diag(np.diag(adj)) # Remove self-loops if any
        toy_graph = nx.Graph(adj)

        a, delta = 0, 0 # true hyperparameters
        design_model = regmod.build_design_shortest(adj, n_subopt=self.config['max_path_depth'], alpha=a)

        # regression of effective delays
        y_pred_mat = regmod.predict_conduction_delays(design_model, adj, invert_weights=False)

        design_model = solver.torch.tensor(design_model)
        x_ground = remove_diagonal_entries(adj).flatten()
        y_ground = solver.forward(design_model.float(), solver.torch.tensor(x_ground).float() + delta * (solver.torch.tensor(x_ground).float() > 0))

        if os.path.exists(op.join(DATA_DIR, f"syn_delay_{a}_{delta}_regress.pkl")):
            x_opt, loss = load(op.join(DATA_DIR, f"syn_delay_{a}_{delta}_regress.pkl"))
        else:
            # Here solve while knowing the true hyperparameter a=0
            np.random.seed(99)
            x_init = solver.torch.tensor(np.random.rand(len(x_ground)))

            x = deepcopy(x_init)
            x_opt, loss = solver.gradient_descent_solver(x, y_ground, design_model,
                                                        delta=delta, n_iter=self.n_iter, verbose=False, early_stop=self.early_stop, step_size=self.step_size,
                                                        l2_penalty=self.l2_penalty)
            save(op.join(DATA_DIR, f"syn_delay_{a}_{delta}_regress.pkl"), (x_opt, loss))

        x_ground_mat = add_diagonal_entries(x_ground.reshape(adj.shape[0], adj.shape[1]-1))
        x_pred_mat = add_diagonal_entries(x_opt.reshape(adj.shape[0], adj.shape[1]-1))

        # Fig 1: Plot the design matrix
        fig1, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))
        axes[0].imshow(y_pred_mat, cmap='hot')
        axes[0].set_title("Conductance $\hat{y}$")
        add_cbar(fig1, axes[0])
        annotate_heatmap(fig1, axes[0], y_pred_mat, adapt_color=0.6)
        axes[1].imshow(design_model, cmap='gray')#, vmax=y_pred_mat.max())
        axes[1].set_title("Design matrix")
        add_cbar(fig1, axes[1])

        axes[2].imshow(adj, cmap='gray')#, vmax=y_pred_mat.max())
        axes[2].set_title("Effective $x=\mathbf{1}$ (if bundle)")
        add_cbar(fig1, axes[2])

        nx.draw(toy_graph, ax=axes[3], with_labels=True, node_color='lightblue', edge_color='black')
        axes[3].set_title("Toy Graph")

        fig1.tight_layout()
        
        if not self.verbose:
            plt.close()
        plt.show()

        # Fig 2: Plot the estimated effective delays vs ground truth
        fig2, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
        axes[0].imshow(x_ground_mat, cmap='gray')
        axes[0].set_title("Effective $x=\mathbf{1}$\n (if bundle)")
        add_cbar(fig2, axes[0])
        annotate_heatmap(fig2, axes[0], x_ground_mat, adapt_color=0.6)

        axes[1].imshow(x_pred_mat, cmap='gray')#, vmax=y_pred_mat.max())
        axes[1].set_title(f"Estimated Effective \n loss={np.round(loss,4)}")
        add_cbar(fig2, axes[1])
        # NOTE: we need to rechek this, it seems that the colors are flipped? (transposed?)
        annotate_heatmap(fig2, axes[1], x_pred_mat.T, adapt_color=0.6)
        fig2.tight_layout()
        if not self.verbose:
            plt.close()
        plt.show()

        return fig1, fig2

    def run_experiment2(self):
        # Synthetic Conductance Delays with Bundle Probability Atlas
        hf = h5py.File(self.path_to_bundle_atlas, 'r')
        self.gmregions_names = hf.get('header').get('gmregions')[()]

        consistency_view = self.get_aggprop(hf, 'consistency')
        consistency_view = consistency_view.astype(float) / float(consistency_view.max()) # Normalize to [0,1]

        adj = consistency_view
        adj -= np.diag(np.diag(adj))

        adj = (adj > self.config['bundle_prob_thresh']).astype(int)
        a, delta = 0.5, 0.5 # true hyperparameters

        if op.exists(op.join(DATA_DIR, f"syn_bundle_exp_{a}_{delta}_{self.config['scale']}_{self.config['bundle_prob_thresh']}.pkl")):
            design_model, x_ground, x_opt, loss = load(op.join(DATA_DIR, f"syn_bundle_exp_{a}_{delta}_{self.config['scale']}_{self.config['bundle_prob_thresh']}.pkl"))
        else:
            design_model = regmod.build_design_shortest(adj, n_subopt=self.config['max_path_depth'], alpha=a)
            design_model = solver.torch.tensor(design_model)
            x_ground = remove_diagonal_entries(adj).flatten()

            y_ground = solver.forward(design_model.float(), solver.torch.tensor(x_ground).float() + delta * (solver.torch.tensor(x_ground).float() > 0))

            np.random.seed(99)
            x_init = solver.torch.tensor(np.random.rand(len(x_ground)))

            x = deepcopy(x_init)
            x_opt, loss = solver.gradient_descent_solver(x, y_ground, design_model,
                                                n_iter=self.n_iter, verbose=self.verbose, 
                                                early_stop=self.early_stop, step_size=self.step_size,
                                                l2_penalty=self.l2_penalty, delta=delta)

            save(op.join(DATA_DIR, f"syn_bundle_exp_{a}_{delta}_{self.config['scale']}_{self.config['bundle_prob_thresh']}.pkl"), (design_model, x_ground, x_opt, loss))
        
        x_ground_mat = add_diagonal_entries(x_ground.reshape(adj.shape[0], adj.shape[1]-1))
        x_pred_mat = add_diagonal_entries(x_opt.reshape(adj.shape[0], adj.shape[1]-1))
        y_ground = solver.forward(design_model.float(), solver.torch.tensor(x_ground).float() + delta * (solver.torch.tensor(x_ground).float() > 0))
        y_est = solver.forward(design_model.float(), solver.torch.tensor(x_opt).float() + delta * (solver.torch.tensor(x_opt).float() > 0))

        y_ground_mat = add_diagonal_entries(y_ground.numpy().reshape(adj.shape[0], adj.shape[1]-1))
        y_est_mat = add_diagonal_entries(y_est.numpy().reshape(adj.shape[0], adj.shape[1]-1))


        fig1, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
        axes[0].imshow(y_ground_mat, cmap='gray')
        axes[0].set_title("Conduction", fontsize=self.title_fontsize)
        add_cbar(fig1, axes[0], ticksize=self.ticks_labels_fontsize)
        axes[1].imshow(x_pred_mat, cmap='gray')#, vmax=y_pred_mat.max())
        axes[1].set_title(f"Estimated Effective", fontsize=self.title_fontsize)
        add_cbar(fig1, axes[1], ticksize=self.ticks_labels_fontsize)
        axes[2].imshow(x_ground_mat, cmap='gray')#, vmax=y_pred_mat.max())
        axes[2].set_title(f"Real Effective", fontsize=self.title_fontsize)
        add_cbar(fig1, axes[2], ticksize=self.ticks_labels_fontsize)

        for ax in axes:
            ax.tick_params(axis="both", labelsize=self.ticks_labels_fontsize)

        fig1.tight_layout()
        if not self.verbose:
            plt.close()
        plt.show()

        fig2, ax = plt.subplots(1, 3, figsize=(12, 4))
        ax[0].set_title('Predicted Conductance')
        ax[0].imshow(y_est_mat, cmap='gray')
        add_cbar(fig2, ax[0])
        ax[1].set_title('Ground Truth Conductance')
        ax[1].imshow(y_ground_mat, cmap='gray')
        add_cbar(fig2, ax[1])
        ax[2].scatter(y_ground.numpy(), y_est.numpy(), s=20, alpha=.5, edgecolors="black", color='blue')
        ax[2].plot(np.linspace(y_ground.numpy().min(), y_ground.numpy().max()), np.linspace(y_ground.numpy().min(), y_ground.numpy().max()), linestyle='--', color='gray', linewidth=2, label="1:1")
        ax[2].set_xlabel("Conductance Estimated", fontsize=self.title_fontsize)
        ax[2].set_ylabel("Conductance Predicted", fontsize=self.title_fontsize)
        ax[2].legend(fontsize=self.ticks_labels_fontsize)
        fig2.tight_layout()
        if not self.verbose:
            plt.close()
        plt.show()

        return fig1, fig2

    def run_experiment3(self):
        # Experiment on Real Conductance Delays
        hf = h5py.File(self.path_to_bundle_atlas, 'r')
        self.gmregions_names = hf.get('header').get('gmregions')[()]

        consistency_view = self.get_aggprop(hf, 'consistency')
        consistency_view = consistency_view.astype(float) / float(consistency_view.max()) # Normalize to [0,1]

        n = consistency_view.shape[0]
        adj = consistency_view
        adj = (adj > self.config['bundle_prob_thresh']).astype(int)

        # Conductance delays from F-TRACT 2018
        dict_key = f"scale{self.scale}__{self.age_range}__{self.delay_max}__{self.feature}"
        dict_key_compare = f"scale{self.scale}__{self.age_range}__{100}__{self.feature}"

        prob_dict_key = f"scale{self.scale}__{self.age_range}__{self.delay_max}__probability"
        prob_dict_key_compare = f"scale{self.scale}__{self.age_range}__{100}__probability"

        prob_y_ground = self.ftracts[prob_dict_key][:n, :n]
        prob_y_ground_compare = self.ftracts[prob_dict_key_compare][:n, :n]

        prob_thresh = self.config['ftract_prob_thresh'] # Threshold for F-TRACT probabilities to consider a connection as present
        y_ground_mat = self.ftracts[dict_key][:n, :n]
        y_ground_mat *= (prob_y_ground > prob_thresh)
        y_ground = solver.torch.tensor(remove_diagonal_entries(y_ground_mat).flatten())

        y_ground_mat_compare = self.ftracts[dict_key_compare][:n, :n]
        y_ground_mat_compare *= (prob_y_ground_compare > prob_thresh)
        y_ground_compare = solver.torch.tensor(remove_diagonal_entries(y_ground_mat_compare).flatten())

        guess_a, guess_delta = 0.8, 10. # assumed hyperparameters

        if os.path.exists(op.join(DATA_DIR, f"design_matrix_{self.scale}.pkl")):
            design_matrices, design_model = load(op.join(DATA_DIR, f"design_matrix_{self.scale}.pkl"))
        else:
            design_matrices = regmod.get_shortest_matrices(adj, self.config['max_path_depth'], progress=True)
            design_shortest = regmod.apply_alpha_to_design(design_matrix=design_matrices, n_subopt=self.config['max_path_depth'], alpha=guess_a)
            design_model = solver.torch.tensor(design_shortest)
            save(op.join(DATA_DIR, f"design_matrix_{self.scale}.pkl"), (design_matrices, design_model))

        if os.path.exists(op.join(DATA_DIR, f"ftract_delay_regress_alpha_{guess_a}_{guess_delta}_{self.scale}_{self.age_range}_{self.delay_max}.pkl")):
            x_opt, loss, a_est, delta_est = load(op.join(DATA_DIR, f"ftract_delay_regress_alpha_{guess_a}_{guess_delta}_{self.scale}_{self.age_range}_{self.delay_max}.pkl"))
        else:
            np.random.seed(99)
            x_init = solver.torch.tensor(np.random.rand(len(y_ground)))

            x = solver.torch.tensor(x_init).float().requires_grad_(True)
            x_opt, (a_est, delta_est), loss = solver.effective_delay_solver(x, y_ground, solver.torch.tensor(design_matrices).float(),
                                                               alpha=solver.torch.tensor(guess_a), delta=solver.torch.tensor(guess_delta),
                                                               n_iter=self.n_iter, verbose=self.verbose,early_stop=self.early_stop, step_size=self.step_size,l2_penalty=self.l2_penalty)
            
            save(op.join(DATA_DIR, f"ftract_delay_regress_alpha_{guess_a}_{guess_delta}_{self.scale}_{self.age_range}_{self.delay_max}.pkl"), (x_opt, loss, a_est, delta_est))
        x_pred_mat = add_diagonal_entries(x_opt.reshape(adj.shape[0], adj.shape[1]-1))
        # plot the mapping curve and see what it looks like
        x1 = x_opt
        x2 = y_ground_compare
        y = y_ground

        x1_mask = x1 > 1
        x2_mask = x2 > 1
        y_mask = y != 0
        xy_mask1 = np.logical_and(x1_mask, y_mask).numpy().astype(bool)
        xy_mask2 = np.logical_and(x2_mask, y_mask).numpy().astype(bool)

        y_est = solver.forward(design_model.float(), solver.torch.tensor(x_opt).float() + delta_est * (solver.torch.tensor(x_opt).float() > 0))

        fig1, ax = plt.subplots(1, 3, figsize=(10, 4))
        ax[0].imshow(y_ground_mat, cmap='gray')
        ax[0].set_title(f"Input Cond.", fontsize=12)
        ax[1].imshow(y_ground_mat_compare, cmap='gray')
        ax[1].set_title(f"Cond. (peak-delay=100)", fontsize=12)
        ax[2].imshow(x_pred_mat, cmap='gray')
        ax[2].set_title(f"Est. Eff.", fontsize=12)
        add_cbar(fig1, ax[0])
        add_cbar(fig1, ax[1])
        add_cbar(fig1, ax[2])

        fig1.tight_layout()
        if not self.verbose:
            plt.close()
        plt.show()

        fig2, axes = plt.subplots(ncols=3, figsize=(13, 5))
        fig2.suptitle(rf"$\alpha={a_est:2f}$ $\delta={delta_est:2f}$", fontsize=14)

        axes[0].imshow(y_ground_mat, cmap='gray')
        axes[0].set_title(f"Cond. delays", fontsize=12)
        axes[0].set_xlabel("Region", fontsize=12)
        axes[0].set_ylabel("Region", fontsize=12)
        add_cbar(fig2, axes[0])
        # utils.annotate_heatmap(fig, axes, np.round(x_pred_mat,4), adapt_color=0.6)

        prop_loss = loss / np.sum(y_ground_mat != 0)
        axes[1].imshow(x_pred_mat, cmap='gray')
        axes[1].set_title(f"Eff. delays)", fontsize=12)
        add_cbar(fig2, axes[1])

        axes[2].scatter(y_ground.numpy(), y_est.numpy(), s=20, alpha=.5, edgecolors="black", color='blue')
        axes[2].plot(np.linspace(y_ground.numpy().min(), y_ground.numpy().max()), np.linspace(y_ground.numpy().min(), y_ground.numpy().max()), linestyle='--', color='gray', linewidth=2, label="1:1")
        axes[2].set_xlabel("Conduction Estimated", fontsize=12)
        axes[2].set_ylabel("Conduction Predicted", fontsize=12)

        fig2.tight_layout()
        if not self.verbose:
            plt.close()
        plt.show()

        fig3, ax = plt.subplots(1,1, figsize=(10,6))

        ax.scatter(y[xy_mask1], x1[xy_mask1], s=20, alpha=.25, edgecolors="black", color='red', label=r'$\alpha=$' + f"{a_est:.2f}")
        ax.scatter(y[xy_mask2], x2[xy_mask2], s=20, alpha=.25, edgecolors="black", color='blue', label=r'peak-delay $100$')
        # ax.scatter(y[~y_mask], x[~y_mask], s=100, alpha=.4, color="tab:brown", edgecolors="none")
        # ax.scatter(y[~x_mask], x[~x_mask], s=100, alpha=.4, color="tab:purple", edgecolors="none")
        ax.plot(np.linspace(0,150), np.linspace(0,150), linestyle='--', color="gray", linewidth=2, label="1:1")

        ax.set_xlabel("Conductance delays", fontsize=16)
        ax.set_ylabel("Effective delays", fontsize=16)
        ax.tick_params(labelsize=14)
        ax.legend(fontsize=16)
        fig3.tight_layout()
        if not self.verbose:
            plt.close()
        plt.show()

        return fig1, fig2, fig3


if __name__ == "__main__":
    run()
