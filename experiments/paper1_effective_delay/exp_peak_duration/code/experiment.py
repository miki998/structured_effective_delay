"""
Experiment: Relating filled delays and peak duration
"""

import os
import json
from datetime import datetime

# Import necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import os.path as op

from sklearn.linear_model import LinearRegression
from scipy import stats
import h5py

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
        for logger_name in ("matplotlib", "networkx", "numba", "urllib3"):
            logging.getLogger(logger_name).setLevel(logging.ERROR)

    if verbose:
        print("=" * 60)
        print("Experiment: Relating filled delays and peak duration")
        print("=" * 60)

    # Configuration
    config = load_json(os.path.join(EXPERIMENT_DIR, "config.json"))

    if verbose:
        print(f"\nConfiguration: {config}")

    experiments = Experiments(config, verbose=verbose)
    fig1 = experiments.run_experiment1()

    results = {
        "config": config,
        "timestamp": datetime.now().isoformat(),
    }

    # Save results
    if save_results:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        fig1.savefig(
            os.path.join(RESULTS_DIR, ".png"),
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

        # F-TRACT 2018 data
        self.ftracts = load("/Users/mikichan/Desktop/mywork/data_resources/atlas_data/fiber_atlas_2018/Lausanne2018_FTRACT_NEW/agg_ftract_dict_allscales_age_ranges_delays_features.pkl")

        self.path_to_bundle_atlas = f"/Users/mikichan/Desktop/mywork/data_resources/atlas_data/fiber_atlas_2018/probconnatlas/wm.connatlas.scale{self.scale}.h5"

        # Synthetic Conductance Delays with Bundle Probability Atlas
        hf = h5py.File(self.path_to_bundle_atlas, 'r')
        self.gmregions_names = hf.get('header').get('gmregions')[()]

        self.consistency = self.get_aggprop(hf, 'consistency')
        self.n = self.consistency.shape[0]
        adj = self.consistency[:self.n-1, :self.n-1]
        adj -= np.diag(np.diag(adj))

        self.adj = (adj > self.config['bundle_prob_thresh']).astype(int)

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
        # Loading effective delays
        if os.path.exists(op.join(DATA_DIR, f"../../exp_solvers_regression/data/compare_ftract_delay_regress_{self.scale}_{self.age_range}_{self.delay_max}.pkl")):
            compare_solvers = load(op.join(DATA_DIR, f"../../exp_solvers_regression/data/compare_ftract_delay_regress_{self.scale}_{self.age_range}_{self.delay_max}.pkl"))
        else:
            raise FileNotFoundError(f"Required data file not found: {op.join(DATA_DIR, f'../../exp_solvers_regression/data/compare_ftract_delay_regress_{self.scale}_{self.age_range}_{self.delay_max}.pkl')}")
        
        vanilla_delay = compare_solvers["vanilla"][0]
        vanilla_delay = add_diagonal_entries(vanilla_delay.reshape(self.n-1, self.n-2))
        alpha_delay = compare_solvers["alpha"][0]
        alpha_delay = add_diagonal_entries(alpha_delay.reshape(self.n-1, self.n-2))
        delta_delay = compare_solvers["delta"][0]
        delta_delay = add_diagonal_entries(delta_delay.reshape(self.n-1, self.n-2))
        joint_delay = compare_solvers["joint"][0]
        joint_delay = add_diagonal_entries(joint_delay.reshape(self.n-1, self.n-2))


        display_choice = np.array([False, False, False, True])
        eff_delay = [vanilla_delay, alpha_delay, delta_delay, joint_delay]
        eff_delay = [d for d, display in zip(eff_delay, display_choice) if display]
        labels = ['vanilla', 'alpha', 'delta', 'joint']
        labels = [label for label, display in zip(labels, display_choice) if display]
        shapes = ["o", "s", "D", "^"]
        shapes = [shape for shape, display in zip(shapes, display_choice) if display]

        # Loading conduction delays
        prob_thresh = 0.0
        dict_key = f"scale{self.scale}__{self.age_range}__{self.delay_max}__{self.feature}"
        conduction = self.ftracts[dict_key]
        conduction = conduction[:self.n-1, :self.n-1]
        conduction *= (conduction > prob_thresh)
        
        # Loading peak durations
        dict_key = f"scale{self.scale}__{self.age_range}__{self.delay_max}__feature_duration"
        peak_durations = self.ftracts[dict_key]
        peak_durations = peak_durations[:self.n-1, :self.n-1]
        peak_durations *= (conduction > prob_thresh)

        fig, ax = plt.subplots(figsize=(8, 6))

        ax.set_title(f"Peak Duration vs Regression Effect \n(scale {self.scale}, age {self.age_range}, delay {self.delay_max}ms)", fontsize=14)

        ax.set_xlabel("Regression Effect (Conduction Delay - Effective Delay) [ms]", fontsize=12)
        ax.set_ylabel("Peak Duration [ms]", fontsize=12)

        wo_conn = self.adj == 0
        w_conn = self.adj == 1
        conn = [wo_conn, w_conn]
        conn_labels = ['without structure', 'with structure']
        for i, effective in enumerate(eff_delay):
            regression_effect = (conduction - effective)
            # Remove effective > conduction entries
            mask = regression_effect > 0

            for c, clabel, color in zip(conn, conn_labels, ['red', 'blue']):
                x = regression_effect[c][mask[c]]
                y = peak_durations[c][mask[c]]

                model = LinearRegression()
                model.fit(x.flatten().reshape(-1,1), y.flatten())

                r_squared = model.score(x.flatten().reshape(-1,1), y.flatten())

                model_est = LinearRegression()
                model_est.fit(y.flatten().reshape(-1,1), x.flatten())
                
                corr, _ = stats.pearsonr(x, y)

                ax.plot([0, x.max()], [model.intercept_, model.intercept_ + x.max() * model.coef_[0]], 
                            linewidth=2, color=color, linestyle='--', 
                            label=rf'$R^2={np.round(r_squared, 3)}$ | $\rho={np.round(corr,3)}$ | $v={np.round(model_est.coef_[0],3)}$')

                # Plot with and without structure connections
                ax.scatter(x.flatten(), y.flatten(), alpha=0.5, edgecolors='black', linewidths=0.2, label=labels[i] + f" ({clabel})", color=color, marker=shapes[i])

        ax.legend(title="Solver Type", fontsize=12)
        ax.set_yscale('log')

        fig.tight_layout()
        if not self.verbose:
            plt.close()
        plt.show()

        return fig



        

if __name__ == "__main__":
    run()
