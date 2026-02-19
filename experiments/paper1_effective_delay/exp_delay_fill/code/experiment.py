"""
Experiment: Delay Fill Results of Regression
"""

import os
import json
from datetime import datetime

# Import necessary libraries
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import os.path as op
from tqdm import tqdm
import h5py

from src import regmod
from src import solver

from src.utils import load_json, load, save, remove_diagonal_entries, add_diagonal_entries, add_cbar

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
        for logger_name in ("matplotlib", "networkx", "numba", "urllib3"):
            logging.getLogger(logger_name).setLevel(logging.ERROR)

    if verbose:
        print("=" * 60)
        print("Experiment: Delay Fill Results of Regression")
        print("=" * 60)

    # Configuration
    config = load_json(os.path.join(EXPERIMENT_DIR, "config.json"))

    if verbose:
        print(f"\nConfiguration: {config}")

    fig1, fig2, fig3 = None, None, None
    experiments = Experiments(config, verbose=verbose)
    print("\nRunning Experiment 1: Illustrate Delay Fill on synthetic delays")
    fig1, fig2 = experiments.run_experiment1()

    print("\nRunning Experiment 2: Results as a function of missing data percentage")
    # fig2 = experiments.run_experiment2()

    results = {
        "config": config,
        "timestamp": datetime.now().isoformat(),
    }

    # Save results
    if save_results:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        if fig1 is not None:
            fig1.savefig(
                os.path.join(RESULTS_DIR, "illustrative_fill_in.png"),
                dpi=300,
                bbox_inches="tight",
            )
        if fig2 is not None:
            fig2.savefig(
                os.path.join(RESULTS_DIR, "illustrative_fill_in_2.png"),
                dpi=300,
                bbox_inches="tight",
            )
        if fig3 is not None:
            fig3.savefig(
                os.path.join(RESULTS_DIR, "increasing_missing_data.png"),
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
        self.bundle_prob_thresh = self.config["bundle_prob_thresh"]
        self.max_path_depth = self.config["max_path_depth"]
        self.missing_perc = self.config["missing_perc"]
        
        # F-TRACT 2018 data
        self.ftracts = load("/Users/mikichan/Desktop/mywork/data_resources/atlas_data/fiber_atlas_2018/Lausanne2018_FTRACT_NEW/agg_ftract_dict_allscales_age_ranges_delays_features.pkl")

        self.path_to_ftract = f"/Users/mikichan/Desktop/mywork/data_resources/atlas_data/f-tract_v2112/F-TRACT-090624/{self.delay_max}"

        self.path_to_bundle_atlas = f"/Users/mikichan/Desktop/mywork/data_resources/atlas_data/fiber_atlas_2018/probconnatlas/wm.connatlas.scale{self.scale}.h5"

        # Synthetic Conductance Delays with Bundle Probability Atlas
        hf = h5py.File(self.path_to_bundle_atlas, 'r')
        self.gmregions_names = hf.get('header').get('gmregions')[()]

        consistency_view = self.get_aggprop(hf, 'consistency')
        consistency_view = consistency_view.astype(float) / float(consistency_view.max()) # Normalize to [0,1]
        self.n = consistency_view.shape[0]
        adj = consistency_view[:self.n-1, :self.n-1]
        adj -= np.diag(np.diag(adj))

        self.adj = (adj > self.bundle_prob_thresh).astype(int)

        if op.exists(op.join(DATA_DIR, f"design_matrices_{self.scale}_{self.age_range}_{self.delay_max}_maxdepth_{self.max_path_depth}_thresh_{self.bundle_prob_thresh}.pkl")):
            self.design_matrices = load(op.join(DATA_DIR, f"design_matrices_{self.scale}_{self.age_range}_{self.delay_max}_maxdepth_{self.max_path_depth}_thresh_{self.bundle_prob_thresh}.pkl"))
        else:
            self.design_matrices = regmod.get_shortest_matrices(self.adj, self.max_path_depth, progress=True)
            save(op.join(DATA_DIR, f"design_matrices_{self.scale}_{self.age_range}_{self.delay_max}_maxdepth_{self.max_path_depth}_thresh_{self.bundle_prob_thresh}.pkl"), self.design_matrices)

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

    def relative_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
        """
        Compute the relative error between true and predicted values.
        """

        div = np.abs(y_true)
        div[div == 0] = 1  # to avoid division by zero
        distrib = np.abs(y_true - y_pred) / div
        return distrib.mean(), distrib.std()

    def run_experiment1(self):
        a, delta = 0.5, 0.1

        # Build design matrix
        design_shortest = regmod.apply_alpha_to_design(self.design_matrices, n_subopt=self.max_path_depth, alpha=a)
        design_model = solver.torch.tensor(design_shortest)
        x_ground = remove_diagonal_entries(self.adj).flatten().astype(float)

        y_ground = solver.forward(design_model.float(), solver.torch.tensor(x_ground).float() + delta * (solver.torch.tensor(x_ground).float() > 0))

        np.random.seed(99)
        y_non_complete = deepcopy(y_ground)
        n_remove = int(self.missing_perc * len(y_ground))
        remove_indices = np.random.choice(len(y_ground), size=n_remove, replace=False)
        y_non_complete[remove_indices] = 0.0  # Set removed observations to zero

        y_incomplete = solver.torch.tensor(deepcopy(y_non_complete))

        if os.path.exists(op.join(DATA_DIR, f"illustrate_delay_fill_results_{self.missing_perc}.pkl")):
            x_opt, loss, loss_logs, df_loss = load(op.join(DATA_DIR, f"illustrate_delay_fill_results_{self.missing_perc}.pkl"))
        else:
            np.random.seed(99)
            x_init = solver.torch.tensor(np.random.rand(len(x_ground))).requires_grad_(True)

            x = deepcopy(x_init)
            x_opt, loss, loss_logs, df_loss = solver.gradient_descent_solver(x, y_incomplete, design_model, delta=delta,
                                                        n_iter=self.n_iter, verbose=False, 
                                                        early_stop=self.early_stop, step_size=self.step_size,
                                                        l2_penalty=self.l2_penalty, return_logs=True)
            save(op.join(DATA_DIR, f"illustrate_delay_fill_results_{self.missing_perc}.pkl"), (x_opt, loss, loss_logs, df_loss))

        y_est_opt = solver.forward(design_model.float(), solver.torch.tensor(x_opt).float() + delta * (solver.torch.tensor(x_opt).float() > 0)).numpy()

        y_ground_mat = add_diagonal_entries(y_ground.numpy().reshape(self.adj.shape[0], self.adj.shape[1]-1))
        y_masked_mat = add_diagonal_entries(y_incomplete.numpy().reshape(self.adj.shape[0], self.adj.shape[1]-1))
        y_est_opt_mat = add_diagonal_entries(y_est_opt.reshape(self.adj.shape[0], self.adj.shape[1]-1))

        fig1, ax = plt.subplots(1, 3, figsize=(13, 4))

        ax[0].imshow(y_ground_mat, cmap='gray')
        add_cbar(fig1, ax[0], ticksize=self.ticks_labels_fontsize)
        ax[0].set_title("Complete Conduction", fontsize=self.title_fontsize)
        
        ax[1].imshow(y_masked_mat, cmap='gray')
        add_cbar(fig1, ax[1], ticksize=self.ticks_labels_fontsize)
        ax[1].set_title(f"Masked \n({int(self.missing_perc*100)}% amiss)", fontsize=self.title_fontsize)

        ax[2].imshow(y_est_opt_mat, cmap='gray')
        add_cbar(fig1, ax[2], ticksize=self.ticks_labels_fontsize)
        ax[2].set_title(f"Forward Conduction \n(After Fill-In)", fontsize=self.title_fontsize)

        for axe in ax:
            axe.tick_params(axis="both", labelsize=self.ticks_labels_fontsize)

        fig1.tight_layout()
        if not self.verbose:
            plt.close()
        plt.show()
    
        fig2, ax = plt.subplots(1, figsize=(6, 3))

        ax.scatter(y_ground.numpy(), y_est_opt, color='blue', alpha=0.6, edgecolor='k')
        ax.plot([y_ground.min(), y_ground.max()], [y_ground.min(), y_ground.max()], 'r--', label='Ideal Fit')
        ax.set_xlabel('Forward Conduction', fontsize=self.title_fontsize)
        ax.set_ylabel('Grd. Conduction', fontsize=self.title_fontsize)

        ax.tick_params(axis="both", labelsize=self.ticks_labels_fontsize)
        ax.grid(alpha=0.7, linestyle='--', color='gray', linewidth=0.5)

        fig2.tight_layout()
        if not self.verbose:
            plt.close()
        plt.show()

        return fig1, fig2

    def run_experiment2(self):
        a, delta = 0.5, 0.1
        
        # Build design matrix
        design_shortest = regmod.apply_alpha_to_design(self.design_matrices, n_subopt=self.max_path_depth, alpha=a)
        design_model = solver.torch.tensor(design_shortest)
        x_ground = remove_diagonal_entries(self.adj).flatten().astype(float)

        y_ground = solver.forward(design_model.float(), solver.torch.tensor(x_ground).float() + delta * (solver.torch.tensor(x_ground).float() > 0))

        n_samples = 1000
        np.random.seed(99)
        percentages = np.linspace(0, 0.9, 6)

        x_ground = remove_diagonal_entries(self.adj).flatten().astype(float)
        deviations = []
        deviations_masked = []

        if os.path.exists(op.join(DATA_DIR, "synthetic_delay_fill_results.pkl")):
            deviations, deviations_masked = load(op.join(DATA_DIR, "synthetic_delay_fill_results.pkl"))
        else:
            for p in tqdm(percentages, total=len(percentages)):
                # Randomly remove p% of the observations
                y_ground = solver.forward(design_model.float(), solver.torch.tensor(x_ground).float() + delta * (solver.torch.tensor(x_ground).float() > 0))
                y_non_complete = deepcopy(y_ground)
                n_remove = int(p * len(y_ground))
                remove_indices = np.random.choice(len(y_ground), size=n_remove, replace=False)
                y_non_complete[remove_indices] = 0.0  # Set removed observations to zero

                y_ground = solver.torch.tensor(deepcopy(y_non_complete))

                x_init = solver.torch.tensor(np.random.rand(len(x_ground))).requires_grad_(True)

                non_zero_mask = y_ground > 0

                x_opt, _ = solver.gradient_descent_solver(x_init, y_ground, design_model, delta=delta,
                                                        n_iter=self.n_iter, verbose=False, 
                                                        early_stop=self.early_stop, step_size=self.step_size,
                                                        l2_penalty=self.l2_penalty)
                
                deviation_percent_masked = self.relative_error(x_ground[non_zero_mask], x_opt[non_zero_mask])
                deviation_percent = self.relative_error(x_ground[~non_zero_mask], x_opt[~non_zero_mask])
                
                deviations.append(deviation_percent)
                deviations_masked.append(deviation_percent_masked)
                save(op.join(DATA_DIR, "synthetic_delay_fill_results.pkl"), (deviations, deviations_masked))

        # Adapted: create synthetic samples for lists of deviations and plot grouped boxplots
        np.random.seed(0)

        # extract means and stds from the lists, handle NaNs
        means_masked = [float(m[0]) if not np.isnan(m[0]) else 0.0 for m in deviations_masked]
        stds_masked  = [float(m[1]) if not np.isnan(m[1]) else 0.0 for m in deviations_masked]

        means_unmasked = [float(m[0]) if not np.isnan(m[0]) else 0.0 for m in deviations]
        stds_unmasked  = [float(m[1]) if not np.isnan(m[1]) else 0.0 for m in deviations]

        # generate samples (ensure non-negative)
        samples_masked = [np.clip(np.random.normal(loc=mu, scale=sigma if sigma>0 else 0.0, size=n_samples),
                                a_min=0, a_max=None)
                        for mu, sigma in zip(means_masked, stds_masked)]
        samples_unmasked = [np.clip(np.random.normal(loc=mu, scale=sigma if sigma>0 else 0.0, size=n_samples),
                                    a_min=0, a_max=None)
                            for mu, sigma in zip(means_unmasked, stds_unmasked)]

        # prepare grouped boxplot positions
        n_groups = len(samples_masked)
        ind = np.arange(n_groups)
        width = 0.35

        fig, ax = plt.subplots(1, figsize=(9, 4))

        ax.cla()
        bp1 = ax.boxplot([samples_masked[i] for i in range(n_groups)],
                        positions=ind - width/2, widths=width, patch_artist=True, boxprops=dict(facecolor='blue'))
        bp2 = ax.boxplot([samples_unmasked[i] for i in range(n_groups)],
                            positions=ind + width/2, widths=width, patch_artist=True, boxprops=dict(facecolor='red'))

        # plot means and annotate
        for i, (m_s, m_u, s_s, s_u) in enumerate(zip(means_masked, means_unmasked, stds_masked, stds_unmasked)):
            ax.scatter(i - width/2, m_s, color='white', edgecolor='k', zorder=3)
            # ax.text(i - width/2 + 0.04, m_s, f"μ={m_s:.2f}\nσ={s_s:.2f}", va='center', fontsize=8)
            ax.scatter(i + width/2, m_u, color='white', edgecolor='k', zorder=3)
            # ax.text(i + width/2 + 0.04, m_u, f"μ={m_u:.2f}\nσ={s_u:.2f}", va='center', fontsize=8)

        # x labels using percentages
        ax.set_xticks(ind)
        ax.set_xticklabels([f"{int(p*100)}%" for p in percentages])
        ax.set_xlabel('% missing')
        ax.set_ylabel('Relative Error (Eff. delay)')

        # enlarge fonts and tick sizes for the figure and boxplots
        fontsize = 18
        tick_labelsize = 18
        text_fontsize = 12

        # Apply to all axes in the current figure
        for a in fig.axes:
            # axis labels and title
            if a.title:
                a.title.set_fontsize(fontsize + 2)
            a.xaxis.label.set_fontsize(fontsize)
            a.yaxis.label.set_fontsize(fontsize)
            # ticks
            a.tick_params(axis='both', which='major', labelsize=tick_labelsize, width=1.2, length=6)
            a.tick_params(axis='both', which='minor', labelsize=tick_labelsize-2, width=1.0, length=4)
            # any manual text annotations
            for t in a.texts:
                t.set_fontsize(text_fontsize)
            # legend (if present)
            leg = a.get_legend()
            if leg is not None:
                for txt in leg.get_texts():
                    txt.set_fontsize(text_fontsize)
                if leg.get_title():
                    leg.get_title().set_fontsize(text_fontsize)

        # Ensure xtick labels use the desired fontsize
        ax.set_xticks(ind)
        ax.set_xticklabels([f"{int(p*100)}%" for p in percentages], fontsize=fontsize)
        ax.grid(axis='both', 
                linestyle='--', 
                alpha=0.7,
                color='gray',
                linewidth=0.5)

        ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ['w/ conduction', 'w/o conduction'], loc='upper left', prop={'size': 18})

        # increase overall figure title / layout if any
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        if not self.verbose:
            plt.close()
        plt.show()

        return fig
    
    def run_experiment3(self):
        a, delta = 0.5, 0.0
        # Build design matrix
        design_shortest = regmod.apply_alpha_to_design(self.design_matrices, n_subopt=self.max_path_depth, alpha=a)
        design_model = solver.torch.tensor(design_shortest)
        x_ground = remove_diagonal_entries(self.adj).flatten().astype(float)

        np.random.seed(99)
        percentages = np.linspace(0, 0.9, 6)

        x_ground = remove_diagonal_entries(self.adj).flatten().astype(float)
        deviations = []
        deviations_masked = []

        dict_key = f"scale{self.scale}__{self.age_range}__{self.delay_max}__{self.feature}"

        if os.path.exists(op.join(DATA_DIR, "real_delay_fill_results.pkl")):
            deviations, deviations_masked = load(op.join(DATA_DIR, "real_delay_fill_results.pkl"))
        else:
            for p in tqdm(percentages, total=len(percentages)):
                np.random.seed(99)
                # Randomly remove p% of the observations
                prob_thresh = 0.0
                y_ground_mat = self.ftracts[dict_key]
                y_ground_mat = y_ground_mat[:self.n-1, :self.n-1]
                y_ground_mat *= (y_ground_mat > prob_thresh)
                y_ground = remove_diagonal_entries(y_ground_mat).flatten()
                y_non_complete = deepcopy(y_ground)
                n_remove = int(p * len(y_ground))
                remove_indices = np.random.choice(len(y_ground), size=n_remove, replace=False)
                y_non_complete[remove_indices] = 0.0  # Set removed observations to zero

                y_ground = solver.torch.tensor(deepcopy(y_non_complete))

                x_init = solver.torch.tensor(np.random.rand(len(x_ground))).requires_grad_(True)

                non_zero_mask = y_ground > 0

                x_opt, _ = solver.gradient_descent_solver(x_init, y_ground, design_model, delta=delta,
                                                        n_iter=self.n_iter, verbose=False, 
                                                        early_stop=self.early_stop, step_size=self.step_size,
                                                        l2_penalty=self.l2_penalty)
                
                deviation_percent_masked = self.relative_error(x_ground[non_zero_mask], x_opt[non_zero_mask])
                deviation_percent = self.relative_error(x_ground[~non_zero_mask], x_opt[~non_zero_mask])
                
                deviations.append(deviation_percent)
                deviations_masked.append(deviation_percent_masked)
                save(op.join(DATA_DIR, "real_delay_fill_results.pkl"), (deviations, deviations_masked))

        # Adapted: create synthetic samples for lists of deviations and plot grouped boxplots
        np.random.seed(0)

        # extract means and stds from the lists, handle NaNs
        means_masked = [float(m[0]) if not np.isnan(m[0]) else 0.0 for m in deviations_masked]
        stds_masked  = [float(m[1]) if not np.isnan(m[1]) else 0.0 for m in deviations_masked]

        means_unmasked = [float(m[0]) if not np.isnan(m[0]) else 0.0 for m in deviations]
        stds_unmasked  = [float(m[1]) if not np.isnan(m[1]) else 0.0 for m in deviations]

        # generate samples (ensure non-negative)
        n_samples = 1000
        samples_masked = [np.clip(np.random.normal(loc=mu, scale=sigma if sigma>0 else 0.0, size=n_samples),
                                a_min=0, a_max=None)
                        for mu, sigma in zip(means_masked, stds_masked)]
        samples_unmasked = [np.clip(np.random.normal(loc=mu, scale=sigma if sigma>0 else 0.0, size=n_samples),
                                    a_min=0, a_max=None)
                            for mu, sigma in zip(means_unmasked, stds_unmasked)]

        # prepare grouped boxplot positions
        n_groups = len(samples_masked)
        ind = np.arange(n_groups)
        width = 0.35

        fig, ax = plt.subplots(1, figsize=(9, 4))

        ax.cla()
        bp1 = ax.boxplot([samples_masked[i] for i in range(n_groups)],
                        positions=ind - width/2, widths=width, patch_artist=True, boxprops=dict(facecolor='blue'))
        bp2 = ax.boxplot([samples_unmasked[i] for i in range(n_groups)],
                            positions=ind + width/2, widths=width, patch_artist=True, boxprops=dict(facecolor='red'))

        # plot means and annotate
        for i, (m_s, m_u, s_s, s_u) in enumerate(zip(means_masked, means_unmasked, stds_masked, stds_unmasked)):
            ax.scatter(i - width/2, m_s, color='white', edgecolor='k', zorder=3)
            # ax.text(i - width/2 + 0.04, m_s, f"μ={m_s:.2f}\nσ={s_s:.2f}", va='center', fontsize=8)
            ax.scatter(i + width/2, m_u, color='white', edgecolor='k', zorder=3)
            # ax.text(i + width/2 + 0.04, m_u, f"μ={m_u:.2f}\nσ={s_u:.2f}", va='center', fontsize=8)

        # x labels using percentages
        ax.set_xticks(ind)
        ax.set_xticklabels([f"{int(p*100)}%" for p in percentages])
        ax.set_xlabel('% missing')
        ax.set_ylabel('Relative Error (Eff. delay)')

        # enlarge fonts and tick sizes for the figure and boxplots
        fontsize = 18
        tick_labelsize = 18
        text_fontsize = 12

        # Apply to all axes in the current figure
        for a in fig.axes:
            # axis labels and title
            if a.title:
                a.title.set_fontsize(fontsize + 2)
            a.xaxis.label.set_fontsize(fontsize)
            a.yaxis.label.set_fontsize(fontsize)
            # ticks
            a.tick_params(axis='both', which='major', labelsize=tick_labelsize, width=1.2, length=6)
            a.tick_params(axis='both', which='minor', labelsize=tick_labelsize-2, width=1.0, length=4)
            # any manual text annotations
            for t in a.texts:
                t.set_fontsize(text_fontsize)
            # legend (if present)
            leg = a.get_legend()
            if leg is not None:
                for txt in leg.get_texts():
                    txt.set_fontsize(text_fontsize)
                if leg.get_title():
                    leg.get_title().set_fontsize(text_fontsize)

        # Ensure xtick labels use the desired fontsize
        ax.set_xticks(ind)
        ax.set_xticklabels([f"{int(p*100)}%" for p in percentages], fontsize=fontsize)
        ax.grid(axis='both', 
                linestyle='--', 
                alpha=0.7,
                color='gray',
                linewidth=0.5)

        ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ['w/ conduction', 'w/o conduction'], loc='upper left', prop={'size': 18})

        # increase overall figure title / layout if any
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        if not self.verbose:
            plt.close()
        plt.show()

        return fig

if __name__ == "__main__":
    run()
