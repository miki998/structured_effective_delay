"""
Experiment: Effective Delays Validations
"""

import os
import json
from datetime import datetime

# Import necessary libraries
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import os.path as op
import h5py
from sklearn.linear_model import LinearRegression
from scipy import stats

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
        print("Experiment: Effective Delays Validations")
        print("=" * 60)

    # Configuration
    config = load_json(os.path.join(EXPERIMENT_DIR, "config.json"))

    if verbose:
        print(f"\nConfiguration: {config}")

    fig1, fig2, fig3 = None, None, None
    experiments = Experiments(config, verbose=verbose)
    print("\nRunning Experiment 1: Comparison vs other F-Tract delays")
    fig1 = experiments.run_experiment1()

    print("\nRunning Experiment 2: Comparison vs DCM delays")
    fig2 = experiments.run_experiment2()

    print("\nRunning Experiment 3: Comparison vs Effective delays")
    fig3 = experiments.run_experiment3()

    results = {
        "config": config,
        "timestamp": datetime.now().isoformat(),
    }

    # Save results
    if save_results:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        if fig1 is not None:
            fig1.savefig(os.path.join(RESULTS_DIR, "comparison_ftract_delays.png"), dpi=300)
        if fig2 is not None:
            fig2.savefig(os.path.join(RESULTS_DIR, "comparison_dcm_delays.png"), dpi=300)
        if fig3 is not None:
            fig3.savefig(os.path.join(RESULTS_DIR, "comparison_effective_delays.png"), dpi=300)

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
        self.length = self.get_aggprop(hf, 'length')[:self.n-1, :self.n-1]
    
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
        ret : np.arrasy
            The array containing the requested property values.
        """

        try:
            ret = np.array(h5dict.get("matrices").get(property))
        except:
            print("Not valid property OR h5 not opened")
        return ret
    
    def run_experiment1(self):
        # Loading other F-Tract delays
        prob_thresh = 0.0
        delays = [50, 100, 200, 400]
        
        ftract_delays = {}
        for delay in delays:
            dict_key = f"scale{self.scale}__{self.age_range}__{delay}__{self.feature}"
            ftract_delays[delay] = self.ftracts[dict_key]
            ftract_delays[delay] = ftract_delays[delay][:self.n-1, :self.n-1]
            ftract_delays[delay] *= (ftract_delays[delay] > prob_thresh)
            # ftract_delays[delay] = remove_diagonal_entries(ftract_delays[delay])

        speed4cond_delay = []
        fig, ax = plt.subplots(2,2, figsize=(12, 8))
        for k in range(4):
            r, c = k//2, k%2
            mask = np.logical_and((ftract_delays[delays[k]] > 0), (self.length > 0))

            y = ftract_delays[delays[k]][mask]
            x = self.length[mask]

            model = LinearRegression()
            model.fit(x.reshape(-1,1), y)

            r_squared = model.score(x.reshape(-1,1), y)

            model_est_speed = LinearRegression()
            model_est_speed.fit(y.reshape(-1,1)/1e3, x/1e3)
            
            speed4cond_delay.append(model_est_speed.coef_[0])
            
            corr = stats.pearsonr(x, y)[0]
            
            ax[r, c].scatter(x, y, color='gray', alpha=0.7, edgecolors='k', s=10)
            ax[r, c].plot([0, x.max()], [model.intercept_, model.intercept_ + x.max() * model.coef_[0]], 
                        linewidth=2, color='r', linestyle='--', 
                        label=rf'$R^2={np.round(r_squared, 3)}$ | $\rho={np.round(corr,3)}$ | $v={np.round(model_est_speed.coef_[0],3)}$')
            
            ax[r, c].legend(fontsize=16)
            ax[r, c].set_title(f'delay<{delays[k]} ms', fontsize=16)
            if (r == 0) and (c == 0):
                ax[r, c].set_xlabel('fiber length (mm)', fontsize=16)
                ax[r, c].set_ylabel('delay (ms)', fontsize=16)

            ax[r, c].tick_params(labelsize=14)
    
        fig.tight_layout()
        if not self.verbose:
            plt.close()
        plt.show()

        return fig

    def run_experiment2(self):
        # Loading DCM
        delays = [50, 100, 200, 400]

        ftract_dcms = {}
        for delay in delays:
            dict_key = f"scale{self.scale}__{self.age_range}__{delay}__feature_dcm_axonal_delay"
            ftract_dcms[delay] = self.ftracts[dict_key]
            ftract_dcms[delay] = ftract_dcms[delay][:self.n-1, :self.n-1]

        fig, ax = plt.subplots(2,2, figsize=(12, 8))
        for k in range(4):
            r, c = k//2, k%2
            mask = np.logical_and((ftract_dcms[delays[k]] > 0), (self.length > 0))

            y = ftract_dcms[delays[k]][mask]
            x = self.length[mask]

            model = LinearRegression()
            model.fit(x.reshape(-1,1), y)
            r_squared = model.score(x.reshape(-1,1), y)
            corr = stats.pearsonr(x, y)[0]

            model_est_speed = LinearRegression()
            model_est_speed.fit(y.reshape(-1,1)/1e3, x/1e3)
            speed4dcm = np.round(model_est_speed.coef_[0], 3)

            ax[r, c].scatter(x, y, color='gray', alpha=0.7, edgecolors='k', s=10)
            ax[r, c].plot([0, x.max()], [model.intercept_, model.intercept_ + x.max() * model.coef_[0]], 
                            linewidth=2, color='r', linestyle='--', 
                            label=rf'$R^2={np.round(r_squared, 3)}$ | $\rho={np.round(corr,3)}$ | $v={np.round(model_est_speed.coef_[0],3)}$')

            ax[r, c].legend(fontsize=16)
            ax[r, c].set_title(f'DCM (<{delays[k]} ms), v={speed4dcm}', fontsize=16)
            ax[r, c].set_xlabel('fiber length', fontsize=16)
            ax[r, c].set_ylabel('delay (ms)', fontsize=16)

            ax[r, c].tick_params(labelsize=14)

        fig.tight_layout()
        if not self.verbose:
            plt.close()
        plt.show()

        return fig

    def run_experiment3(self):
        # Loading effective delays
        if os.path.exists(op.join(DATA_DIR, f"../../exp_solvers_regression/data/compare_ftract_delay_regress_{self.scale}_{self.age_range}_{self.delay_max}.pkl")):
            compare_solvers = load(op.join(DATA_DIR, f"../../exp_solvers_regression/data/compare_ftract_delay_regress_{self.scale}_{self.age_range}_{self.delay_max}.pkl"))
        else:
            raise FileNotFoundError(f"Required data file not found: {op.join(DATA_DIR, f'../../exp_solvers_regression/data/compare_ftract_delay_regress_{self.scale}_{self.age_range}_{self.delay_max}.pkl')}")

        vanilla_delay = compare_solvers["vanilla"][0]
        vanilla_delay = add_diagonal_entries(vanilla_delay.reshape(self.length.shape[0], self.length.shape[1]-1))
        alpha_delay = compare_solvers["alpha"][0]
        alpha_delay = add_diagonal_entries(alpha_delay.reshape(self.length.shape[0], self.length.shape[1]-1))
        delta_delay = compare_solvers["delta"][0]
        delta_delay = add_diagonal_entries(delta_delay.reshape(self.length.shape[0], self.length.shape[1]-1))
        joint_delay = compare_solvers["joint"][0]
        joint_delay = add_diagonal_entries(joint_delay.reshape(self.length.shape[0], self.length.shape[1]-1))

        # Some examples to first visualize
        eff_delay = [vanilla_delay, alpha_delay, delta_delay, joint_delay]
        labels = ['vanilla', 'alpha', 'delta', 'joint']

        min_delay = 0.1
        fig, ax = plt.subplots(2,2, figsize=(12, 8))
        speed4eff_delay = []
        for k in range(4):
            r, c = k//2, k%2
            mask = np.logical_and((eff_delay[k] > min_delay), (self.length > 0))
            y = eff_delay[k][mask]
            x = self.length[mask]
            if len(x) == 0 or len(y) == 0:
                continue

            model = LinearRegression()
            model.fit(x.reshape(-1,1), y)
            r_squared = model.score(x.reshape(-1,1), y)

            model_est_speed = LinearRegression()
            model_est_speed.fit(y.reshape(-1,1)/1e3, x/1e3)
            
            speed4eff_delay.append(model_est_speed.coef_[0])

            corr = stats.pearsonr(x, y)[0]
            
            speed = np.round((np.round(model_est_speed.coef_[0], 3)  * 100 // 1) / 100, 2)
            ax[r, c].scatter(x, y, color='k', alpha=0.7, edgecolors='k', s=10)
            ax[r, c].plot([0, x.max()], [model.intercept_, model.intercept_ + x.max() * model.coef_[0]], 
                        linewidth=2, color='r', linestyle='--', 
                        label=rf'$R^2={np.round(r_squared, 3)}$ | $\rho={np.round(corr,3)}$ | $v={speed}$')
            
            ax[r, c].legend(fontsize=16)
            ax[r, c].set_title(f'effect: {labels[k]}', fontsize=16)

            # if (r == 0) and (c == 0):
            #     ax[r, c].set_xlabel('fiber length', fontsize=16)
            #     ax[r, c].set_ylabel('delay (ms)', fontsize=16)

            ax[r, c].set_xlabel('fiber length (mm)', fontsize=16)
            ax[r, c].set_ylabel('delay (ms)', fontsize=16)

            ax[r, c].tick_params(labelsize=14)

        fig.tight_layout()
        if not self.verbose:
            plt.close()
        plt.show()
        
        return fig

if __name__ == "__main__":
    run()
