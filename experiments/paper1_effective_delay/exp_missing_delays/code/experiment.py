"""
Experiment: Missing Delays effect on Regression
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
        print("Experiment: Missing Delays effect on Regression")
        print("=" * 60)

    # Configuration
    config = load_json(os.path.join(EXPERIMENT_DIR, "config.json"))

    if verbose:
        print(f"\nConfiguration: {config}")

    fig1, fig2, fig3 = None, None, None
    experiments = Experiments(config, verbose=verbose)
    
    print("\nRunning Experiment 1: Effect of missing data percentage on delay estimation")
    fig1, fig2 = experiments.run_experiment1()

    print("\nRunning Experiment 2: Increasing missing data percentage on synthetic bundle probability atlas")
    fig3 = experiments.run_experiment2()

    results = {
        "config": config,
        "timestamp": datetime.now().isoformat(),
    }

    # Save results
    if save_results:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        if fig1 is not None:
            fig1.savefig(
                os.path.join(RESULTS_DIR, "illustrative_missing_percentage.png"),
                dpi=300,
                bbox_inches="tight",
            )
        if fig2 is not None:
            fig2.savefig(
                os.path.join(RESULTS_DIR, "illustrative_missing_percentage2.png"),
                dpi=300,
                bbox_inches="tight",
            )
        if fig3 is not None:
            fig3.savefig(
                os.path.join(RESULTS_DIR, "synthetic_conduction_missing_percentages.png"),
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
        self.ticks_fontsize = 12

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

        # Randomly remove  of the observations
        np.random.seed(99)
        y_non_complete = deepcopy(y_ground)
        n_remove = int(self.missing_perc * (y_ground > 0).sum())

        remove_indices = np.random.choice(np.where(y_ground > 0)[0], size=n_remove, replace=False)
        y_non_complete[remove_indices] = 0.0  # Set removed observations to zero

        y_ground_incomplete = solver.torch.tensor(deepcopy(y_non_complete))

        if os.path.exists(op.join(DATA_DIR, f"illustrate_delay_fill_results_{self.missing_perc}.pkl")):
            x_opt_c, loss_c, loss_logs_c, df_loss_c, x_opt, loss, loss_logs, df_loss = load(op.join(DATA_DIR, f"illustrate_delay_fill_results_{self.missing_perc}.pkl"))
        else:
            np.random.seed(99)
            x_init = solver.torch.tensor(np.random.rand(len(x_ground))).requires_grad_(True)

            x = deepcopy(x_init)
            x_opt_c, loss_c, loss_logs_c, df_loss_c = solver.gradient_descent_solver(x, y_ground_incomplete, design_model, delta=delta,
                                                        n_iter=self.n_iter, verbose=False, 
                                                        early_stop=self.early_stop, step_size=self.step_size,
                                                        l2_penalty=self.l2_penalty, return_logs=True)
            x = deepcopy(x_init)
            x_opt, loss, loss_logs, df_loss = solver.gradient_descent_solver(x, y_ground, design_model, delta=delta,
                                                        n_iter=self.n_iter, verbose=False, 
                                                        early_stop=self.early_stop, step_size=self.step_size,
                                                        l2_penalty=self.l2_penalty, return_logs=True)
            save(op.join(DATA_DIR, f"illustrate_delay_fill_results_{self.missing_perc}.pkl"), (x_opt_c, loss_c, loss_logs_c, df_loss_c, x_opt, loss, loss_logs, df_loss))


        # Plotting
        fig1, ax = plt.subplots(1, 3, figsize=(10, 3.3))

        x_ground_mat = add_diagonal_entries(x_ground.reshape(self.adj.shape[0], self.adj.shape[1]-1))
        x_opt_c_mat = add_diagonal_entries(x_opt_c.reshape(self.adj.shape[0], self.adj.shape[1]-1))
        x_opt_mat = add_diagonal_entries(x_opt.reshape(self.adj.shape[0], self.adj.shape[1]-1))
        diff = (x_ground_mat - x_opt_c_mat)
        diff2 = (x_ground_mat - x_opt_mat)
        vmax = max(np.abs(diff.min()), np.abs(diff2.min()), diff.max(), diff2.max())

        ax[0].imshow(x_ground_mat, cmap='gray')
        ax[0].set_title("Grd. Eff.", fontsize=self.title_fontsize)
        add_cbar(fig1, ax[0], ticksize=self.ticks_fontsize)
        ax[1].imshow(diff, cmap='bwr', vmin=-vmax, vmax=vmax)
        ax[1].set_title("(Grd. Eff. - .) w/ missing", fontsize=self.title_fontsize)
        add_cbar(fig1, ax[1], ticksize=self.ticks_fontsize)
        ax[2].imshow(diff2, cmap='bwr', vmin=-vmax, vmax=vmax)
        ax[2].set_title("(Grd. Eff. - .) w/o missing", fontsize=self.title_fontsize)
        add_cbar(fig1, ax[2], ticksize=self.ticks_fontsize)
        
        fig1.tight_layout()
        if not self.verbose:
            plt.close()
        plt.show()

        fig2, ax = plt.subplots(1, figsize=(5, 4))
        bp = ax.boxplot(
            [diff.flatten(), diff2.flatten()],
            labels=["w/ missing", "w/o missing"],
            patch_artist=True,
            notch=True,
            widths=0.6,
        )
        colors = ["skyblue", "lightcoral"]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
        ax.set_title("Distribution of differences", fontsize=self.title_fontsize)
        ax.tick_params(axis="both", labelsize=self.ticks_fontsize)
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
        percentages = np.linspace(0, 0.9, 10)

        x_ground = remove_diagonal_entries(self.adj).flatten().astype(float)
        deviations = []
        deviations_c = []

        if os.path.exists(op.join(DATA_DIR, "synthetic_delay_fill_results.pkl")):
            deviations, deviations_c = load(op.join(DATA_DIR, "synthetic_delay_fill_results.pkl"))
        else:
            for p in tqdm(percentages, total=len(percentages)):
                # Randomly remove p% of the observations
                y_ground = solver.forward(design_model.float(), solver.torch.tensor(x_ground).float() + delta * (solver.torch.tensor(x_ground).float() > 0))
                y_non_complete = deepcopy(y_ground)
                n_remove = int(p * (y_ground > 0).sum())

                remove_indices = np.random.choice(np.where(y_ground > 0)[0], size=n_remove, replace=False)
                y_non_complete[remove_indices] = 0.0  # Set removed observations to zero

                y_ground_incomplete = solver.torch.tensor(deepcopy(y_non_complete))

                x_init = solver.torch.tensor(np.random.rand(len(x_ground))).requires_grad_(True)

                x = deepcopy(x_init)
                x_opt_c, _ = solver.gradient_descent_solver(x, y_ground_incomplete, design_model, delta=delta,
                                                        n_iter=self.n_iter, verbose=False, 
                                                        early_stop=self.early_stop, step_size=self.step_size,
                                                        l2_penalty=self.l2_penalty)
                
                x = deepcopy(x_init)
                x_opt, _ = solver.gradient_descent_solver(x, y_ground, design_model, delta=delta,
                                                        n_iter=self.n_iter, verbose=False, 
                                                        early_stop=self.early_stop, step_size=self.step_size,
                                                        l2_penalty=self.l2_penalty)
                
                deviation_percent = self.relative_error(x_ground, x_opt)
                deviation_percent_c = self.relative_error(x_ground, x_opt_c)
                
                deviations.append(deviation_percent)
                deviations_c.append(deviation_percent_c)
            save(op.join(DATA_DIR, "synthetic_delay_fill_results.pkl"), (deviations, deviations_c))

        # Adapted: create synthetic samples for lists of deviations and plot grouped boxplots
        np.random.seed(0)

        # extract means and stds from the lists, handle NaNs
        means_c = [float(m[0]) if not np.isnan(m[0]) else 0.0 for m in deviations_c]
        stds_c  = [float(m[1]) if not np.isnan(m[1]) else 0.0 for m in deviations_c]

        means = [float(m[0]) if not np.isnan(m[0]) else 0.0 for m in deviations]
        stds  = [float(m[1]) if not np.isnan(m[1]) else 0.0 for m in deviations]

        # generate samples (ensure non-negative)
        samples_masked = [np.clip(np.random.normal(loc=mu, scale=sigma if sigma>0 else 0.0, size=n_samples),
                                a_min=0, a_max=None)
                        for mu, sigma in zip(means_c, stds_c)]
        samples_unmasked = [np.clip(np.random.normal(loc=mu, scale=sigma if sigma>0 else 0.0, size=n_samples),
                                    a_min=0, a_max=None)
                            for mu, sigma in zip(means, stds)]

        # prepare grouped boxplot positions
        n_groups = len(samples_masked)
        ind = np.arange(n_groups)
        width = 0.35

        fig, ax = plt.subplots(1, figsize=(9, 4))

        ax.cla()
        bp1 = ax.boxplot([samples_masked[i] for i in range(1, n_groups)],
                        positions=ind[1:], widths=width, patch_artist=True, boxprops=dict(facecolor='blue'))
        
        bp2 = ax.boxplot([samples_masked[0]],
                 positions=[ind[0]],
                 widths=width,
                 patch_artist=True,
                 boxprops=dict(facecolor='red'))

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
        
        ax.set_xlabel('% missing')
        ax.set_ylabel('Eff. Relative Error')

        ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ['missing', 'complete'], loc='upper left', prop={'size': 18})

        # increase overall figure title / layout if any
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        if not self.verbose:
            plt.close()
        plt.show()

        return fig

if __name__ == "__main__":
    run()
