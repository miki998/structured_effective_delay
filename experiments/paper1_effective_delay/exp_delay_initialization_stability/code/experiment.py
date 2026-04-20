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

from tqdm import tqdm

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

    print("\nRunning Experiment 1: On Real Data, Stability of initial guesses")
    fig = experiments.run_experiment1()

    results = {
        "config": config,
        "timestamp": datetime.now().isoformat(),
    }

    # Save results
    if save_results:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        if fig is not None:
            fig.savefig(
                os.path.join(RESULTS_DIR, f"initialization_stability_fitting_results.png"),
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
        """
        This experiment tests the stability of the effective delay regression to different initializations on real data, looking at how the conduction error changes.
        """
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
        dict_key_dcm = f"scale{self.scale}__{self.age_range}__{100}__feature_dcm_axonal_delay"

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

        y_ground_mat_dcm = self.ftracts[dict_key_dcm][:n, :n]
        y_ground_mat_dcm *= (prob_y_ground_compare > prob_thresh)
        y_ground_dcm = solver.torch.tensor(remove_diagonal_entries(y_ground_mat_dcm).flatten())

        if os.path.exists(op.join(DATA_DIR, f"design_matrices_{self.scale}.pkl")):
            design_matrices = load(op.join(DATA_DIR, f"design_matrices_{self.scale}.pkl"))
        else:
            design_matrices = regmod.get_shortest_matrices(adj, self.config['max_path_depth'], progress=True)
            save(op.join(DATA_DIR, f"design_matrices_{self.scale}.pkl"), design_matrices)


        a_inits = np.linspace(0, 1, self.config['granule_init_a'])
        delta_inits = np.linspace(0, 50, self.config['granule_init_delta'])

        if os.path.exists(op.join(DATA_DIR, f"ftract_delay_regress_alpha_{self.config['granule_init_a']}_delta_{self.config['granule_init_delta']}_{self.scale}_{self.age_range}_{self.delay_max}.pkl")):
            regression_results = load(op.join(DATA_DIR, f"ftract_delay_regress_alpha_{self.config['granule_init_a']}_delta_{self.config['granule_init_delta']}_{self.scale}_{self.age_range}_{self.delay_max}.pkl"))
        else:
            regression_results = {}
            for a_init in tqdm(a_inits, desc="Running initializations for alpha", disable=not self.verbose):
                for delta_init in delta_inits:
                    guess_a, guess_delta = a_init, delta_init # assumed hyperparameters

                    np.random.seed(99)
                    x_init = solver.torch.tensor(np.random.rand(len(y_ground)))

                    x = solver.torch.tensor(x_init).float().requires_grad_(True)
                    x_opt, (a_est, delta_est), loss, full_loss_logs, datafit_loss_logs = solver.effective_delay_solver(x, y_ground, solver.torch.tensor(design_matrices).float(),
                                                                    alpha=solver.torch.tensor(guess_a).float(), delta=solver.torch.tensor(guess_delta).float(),
                                                                    n_iter=self.n_iter, verbose=False, early_stop=self.early_stop, step_size=self.step_size, l2_penalty=self.l2_penalty,
                                                                    return_logs=True)
                    
                    design_shortest = regmod.apply_alpha_to_design(design_matrix=design_matrices, n_subopt=self.config['max_path_depth'], alpha=a_est)
                    design_model = solver.torch.tensor(design_shortest)

                    y_est = solver.forward(design_model.float(), solver.torch.tensor(x_opt).float() + delta_est * (solver.torch.tensor(x_opt).float() > 0)).numpy()
                    
                    regression_results[(a_init, delta_init)] = (design_model, x_opt, loss, a_est, delta_est, y_est, full_loss_logs, datafit_loss_logs)

            save(op.join(DATA_DIR, f"ftract_delay_regress_alpha_{self.config['granule_init_a']}_delta_{self.config['granule_init_delta']}_{self.scale}_{self.age_range}_{self.delay_max}.pkl"), regression_results)

        scores_corr = np.zeros((len(a_inits), len(delta_inits)))
        scores_loss = np.zeros((len(a_inits), len(delta_inits)))
        scores_a_est = np.zeros((len(a_inits), len(delta_inits)))
        scores_delta_est = np.zeros((len(a_inits), len(delta_inits)))
        for (a_init, delta_init), (design_model, x_opt, loss, a_est, delta_est, y_est, full_loss_logs, datafit_loss_logs) in regression_results.items():
            corr = np.corrcoef(y_est, y_ground.numpy())[0, 1]
            idx_a = np.where(a_inits == a_init)[0][0]
            idx_delta = np.where(delta_inits == delta_init)[0][0]

            scores_corr[idx_a, idx_delta] = corr
            scores_loss[idx_a, idx_delta] = loss
            scores_a_est[idx_a, idx_delta] = a_est
            scores_delta_est[idx_a, idx_delta] = delta_est

        fig, ax = plt.subplots(1, 4, figsize=(11, 3))
        ax[0].imshow(scores_corr, cmap='gray')
        ax[0].set_title(f"Correlation", fontsize=self.ticks_labels_fontsize)
        ax[1].imshow(scores_loss, cmap='gray')
        ax[1].set_title(f"Loss", fontsize=self.ticks_labels_fontsize)
        ax[2].imshow(scores_a_est, cmap='gray')
        ax[2].set_title(f"Estimated Alpha", fontsize=self.ticks_labels_fontsize)
        ax[3].imshow(scores_delta_est, cmap='gray')
        ax[3].set_title(f"Estimated Delay (D)", fontsize=self.ticks_labels_fontsize)
        add_cbar(fig, ax[0], ticksize=self.ticks_labels_fontsize)
        add_cbar(fig, ax[1], ticksize=self.ticks_labels_fontsize)
        add_cbar(fig, ax[2], ticksize=self.ticks_labels_fontsize)
        add_cbar(fig, ax[3], ticksize=self.ticks_labels_fontsize)
        for axis in ax:
            axis.tick_params(axis="both", labelsize=self.ticks_labels_fontsize)

        fig.tight_layout()
        if not self.verbose:
            plt.close()
        plt.show()

        return fig


    def run_experiment2(self):
            """
            This experiment tests the stability of the effective delay regression to different initializations on real data, looking at how fiber length correlation and conduction speed changes.
            """
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
            dict_key_dcm = f"scale{self.scale}__{self.age_range}__{100}__feature_dcm_axonal_delay"

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

            y_ground_mat_dcm = self.ftracts[dict_key_dcm][:n, :n]
            y_ground_mat_dcm *= (prob_y_ground_compare > prob_thresh)
            y_ground_dcm = solver.torch.tensor(remove_diagonal_entries(y_ground_mat_dcm).flatten())

            if os.path.exists(op.join(DATA_DIR, f"design_matrices_{self.scale}.pkl")):
                design_matrices = load(op.join(DATA_DIR, f"design_matrices_{self.scale}.pkl"))
            else:
                design_matrices = regmod.get_shortest_matrices(adj, self.config['max_path_depth'], progress=True)
                save(op.join(DATA_DIR, f"design_matrices_{self.scale}.pkl"), design_matrices)


            guess_a, guess_delta = 0.8, 10. # assumed hyperparameters


            if os.path.exists(op.join(DATA_DIR, f"ftract_delay_regress_alpha_{guess_a}_{guess_delta}_{self.scale}_{self.age_range}_{self.delay_max}.pkl")):
                design_model, x_opt, loss, a_est, delta_est, full_loss_logs, datafit_loss_logs = load(op.join(DATA_DIR, f"ftract_delay_regress_alpha_{guess_a}_{guess_delta}_{self.scale}_{self.age_range}_{self.delay_max}.pkl"))
            else:
                np.random.seed(99)
                x_init = solver.torch.tensor(np.random.rand(len(y_ground)))

                x = solver.torch.tensor(x_init).float().requires_grad_(True)
                x_opt, (a_est, delta_est), loss, full_loss_logs, datafit_loss_logs = solver.effective_delay_solver(x, y_ground, solver.torch.tensor(design_matrices).float(),
                                                                alpha=solver.torch.tensor(guess_a), delta=solver.torch.tensor(guess_delta),
                                                                n_iter=self.n_iter, verbose=self.verbose,early_stop=self.early_stop, step_size=self.step_size,l2_penalty=self.l2_penalty,
                                                                return_logs=True)
                
                design_shortest = regmod.apply_alpha_to_design(design_matrix=design_matrices, n_subopt=self.config['max_path_depth'], alpha=a_est)
                design_model = solver.torch.tensor(design_shortest)

                save(op.join(DATA_DIR, f"ftract_delay_regress_alpha_{guess_a}_{guess_delta}_{self.scale}_{self.age_range}_{self.delay_max}.pkl"), (design_model, x_opt, loss, a_est, delta_est, full_loss_logs, datafit_loss_logs))

            x_pred_mat = add_diagonal_entries(x_opt.reshape(adj.shape[0], adj.shape[1]-1))
            # plot the mapping curve and see what it looks like
            x1 = x_opt
            x2 = y_ground_compare
            x3 = y_ground_dcm
            y = y_ground

            x1_mask = x1 > 1
            x2_mask = x2 > 1
            x3_mask = x3 > 1
            y_mask = y != 0
            xy_mask1 = np.logical_and(x1_mask, y_mask).numpy().astype(bool)
            xy_mask2 = np.logical_and(x2_mask, y_mask).numpy().astype(bool)
            xy_mask3 = np.logical_and(x3_mask, y_mask).numpy().astype(bool)

            y_est = solver.forward(design_model.float(), solver.torch.tensor(x_opt).float() + delta_est * (solver.torch.tensor(x_opt).float() > 0)).numpy()
            y_pred_mat = add_diagonal_entries(y_est.reshape(adj.shape[0], adj.shape[1]-1))

            fig1, ax = plt.subplots(1, 3, figsize=(10, 4))
            ax[0].imshow(y_ground_mat, cmap='gray')
            ax[0].set_title(f"F-Tract (C)", fontsize=self.ticks_labels_fontsize)
            ax[1].imshow(y_pred_mat, cmap='gray')
            ax[1].set_title(f"Estimated Delay (C)", fontsize=self.ticks_labels_fontsize)
            ax[2].imshow(x_pred_mat, cmap='gray')
            ax[2].set_title(f"Estimated Delay (E)", fontsize=self.ticks_labels_fontsize)
            add_cbar(fig1, ax[0], ticksize=self.ticks_labels_fontsize)
            add_cbar(fig1, ax[1], ticksize=self.ticks_labels_fontsize)
            add_cbar(fig1, ax[2], ticksize=self.ticks_labels_fontsize)
            for axis in ax:
                axis.tick_params(axis="both", labelsize=self.ticks_labels_fontsize)

            fig1.tight_layout()
            if not self.verbose:
                plt.close()
            plt.show()

            fig2, axes = plt.subplots(1, 1, figsize=(6, 4))
            fig2.suptitle(rf"$\alpha={a_est:2f}$ $\delta={delta_est:2f}$", fontsize=14)

            axes.scatter(y_ground.numpy(), y_est, s=20, alpha=.5, edgecolors="black", color='blue')
            axes.plot(np.linspace(y_ground.numpy().min(), y_ground.numpy().max()), np.linspace(y_ground.numpy().min(), y_ground.numpy().max()), linestyle='--', color='black', linewidth=2, label="1:1")
            axes.set_xlabel("F-Tract (C)", fontsize=self.ticks_labels_fontsize)
            axes.set_ylabel("Estimated Delays (C)", fontsize=self.ticks_labels_fontsize)

            fig2.tight_layout()
            if not self.verbose:
                plt.close()
            plt.show()

            fig3, ax = plt.subplots(1,1, figsize=(6, 4))

            alpha = 0.7
            ax.scatter(y[xy_mask1], x1[xy_mask1], s=20, alpha=alpha, edgecolors="black", color='darkorange', label=r'Estimated delay (E)')
            ax.scatter(y[xy_mask2], x2[xy_mask2], s=20, alpha=alpha, edgecolors="black", color='yellowgreen', label=r'Thresholded delay (C) $100$ ms')
            ax.scatter(y[xy_mask3], x3[xy_mask3], s=20, alpha=alpha, edgecolors="black", color='royalblue', label=r'DCM (C) $100$ ms')
            # ax.scatter(y[~y_mask], x[~y_mask], s=100, alpha=.4, color="tab:brown", edgecolors="none")
            # ax.scatter(y[~x_mask], x[~x_mask], s=100, alpha=.4, color="tab:purple", edgecolors="none")
            ax.plot(np.linspace(0,100), np.linspace(0,100), linestyle='--', color="black", linewidth=2, label="1:1")

            ax.set_xlabel("Delays (C)", fontsize=16)
            ax.set_ylabel("Delays (E)", fontsize=16)
            ax.tick_params(labelsize=self.ticks_labels_fontsize)
            ax.legend(fontsize=14)

            ax.grid(axis='both', 
                    linestyle='--', 
                    alpha=0.7,
                    color='gray',
                    linewidth=0.5)
            
            fig3.tight_layout()
            if not self.verbose:
                plt.close()
            plt.show()

            return fig1, fig2, fig3


if __name__ == "__main__":
    run()
