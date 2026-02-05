"""
Experiment: Regression of Effective Delays
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
from tqdm import tqdm
import nibabel as nib

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
        print("Experiment: Regression of Effective Delays")
        print("=" * 60)

    # Configuration
    config = load_json(os.path.join(EXPERIMENT_DIR, "config.json"))

    if verbose:
        print(f"\nConfiguration: {config}")

    experiments = Experiments(config, verbose=verbose)
    # Plot 1: Synthetic Toy Graph
    print("\nRunning Experiment 1: Varying Alpha, varying Delta")
    experiments.run_experiment1()

    # Plot 2: Synthetic Signal Graph
    print("\nRunning Experiment 2: Varying Alpha and Delta Landscape")
    experiments.run_experiment2()

    results = {
        "config": config,
        "timestamp": datetime.now().isoformat(),
    }

    # Save results
    if save_results:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        fig1.savefig(
            os.path.join(RESULTS_DIR, "torus_graph.png"),
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
        delay_max = 400
        self.path_to_ftract = f"/Users/mikichan/Desktop/mywork/data_resources/atlas_data/f-tract_v2112/F-TRACT-090624/{delay_max}"

        img = nib.load('/Users/mikichan/Desktop/mywork/data_resources/atlas_data/f-tract_v2112/lausanne2008_33-meta/region_masks.nii')
        masks_volume = img.get_fdata()
        regions_centroid = np.array([np.array(np.where(masks_volume == r)).T.mean(axis=0) for r in range(1, int(masks_volume.max()) + 1)])

        filename = "bundle_probability_atlas.pkl"

        self.adj = load(op.join(self.path_to_resources, filename))

        self.adj = self.adj[:83, :83]
        self.adj -= np.diag(np.diag(self.adj))
        self.N = self.adj.shape[0]
        bundle_prob_thresh = 0.9

        self.adj = (self.adj > bundle_prob_thresh).astype(int)

        bundle_prob = load(op.join(self.path_to_resources, "bundle_probability_atlas.pkl"))
        bundle_prob = bundle_prob[:83, :83]
        bundle_prob -= np.diag(np.diag(bundle_prob))
        ftract_prob = load(op.join(self.path_to_resources, "adj_probability_ftract.pkl"))
        ftract_prob = ftract_prob[:83, :83]

        _, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

        axes[0].imshow(self.adj, cmap='gray')
        axes[0].set_title('Structural connectivity')
        toy_graph = nx.Graph(self.adj)
        pos = {i: regions_centroid[i][:2] for i in range(len(regions_centroid))}
        # draw nodes, edges with transparency, and labels
        nx.draw_networkx_nodes(toy_graph, pos=pos, ax=axes[1], node_size=100, edgecolors='purple')
        nx.draw_networkx_edges(toy_graph, pos=pos, ax=axes[1], alpha=0.3, edge_color='black')
        nx.draw_networkx_labels(toy_graph, pos=pos, ax=axes[1], font_size=6)
        axes[1].set_axis_off()

        if not self.verbose:
            plt.close()
        plt.show()

        with open(op.join(self.path_to_ftract, 'peak_delay__median.txt')) as f:
            text = f.readlines()

        y_ground_mat = []
        for t in text[8:]:
            y_ground_mat.append(t.split(' '))
        y_ground_mat = np.array(y_ground_mat).astype(float)
        y_ground_mat = np.nan_to_num(y_ground_mat[:-1, :-1])

        prob_thresh = 0
        y_ground_mat *= ftract_prob > prob_thresh
        self.y_ground = remove_diagonal_entries(y_ground_mat).flatten()

        if os.path.exists(f'{DATA_DIR}/dmax{delay_max}-bthresh{bundle_prob_thresh}-fthresh{0}.pkl'):
            self.design_matrices_subopts = load(f'{DATA_DIR}/dmax{delay_max}-bthresh{bundle_prob_thresh}-fthresh{0}.pkl')
        else:
            self.design_matrices_subopts = []
            for k in tqdm(range(2)):
                self.design_matrices_subopts.append(regmod.get_shortest_matrices(adjacency=self.adj, n_subopt=k))

            save(f'{DATA_DIR}/dmax{delay_max}-bthresh{bundle_prob_thresh}-fthresh{0}.pkl', self.design_matrices_subopts)

    def run_experiment1(self):
        max_path_depth = 1
        alpha_space = np.linspace(0, 2, 21)

        if os.path.exists(op.join(DATA_DIR, f"effective_delays_maxdepth{max_path_depth}_alphas_results.pkl")):
            alphas_results = load(op.join(DATA_DIR, f"effective_delays_maxdepth{max_path_depth}_alphas_results.pkl"))
        else:
            alphas_results = {}
            for alpha in tqdm(alpha_space):
                design_shortest = regmod.apply_alpha_to_design(self.design_matrices_subopts[max_path_depth], n_subopt=max_path_depth, alpha=alpha)
                design_model = solver.torch.tensor(design_shortest)

                # Define mapping area
                non_zero_mask = self.y_ground > 0
                y_masked = self.y_ground[non_zero_mask]
                design_model_masked = design_model[non_zero_mask]

                np.random.seed(99)
                x_init = solver.torch.tensor(np.random.rand(len(self.y_ground))).requires_grad_(True)
                x = deepcopy(x_init)
                x_opt, loss = solver.gradient_descent_solver(x, y_masked, design_model_masked,
                                                        n_iter=10000, verbose=False, 
                                                        early_stop=1e-10, step_size=5e-1, delta=0,
                                                        #early_stop=1e-10, step_size=5e-1,
                                                        l2_penalty=1)
                
                alphas_results[alpha] = (x_opt,loss)

            max_path_depth = 1

            delta_space = np.linspace(0, 60, 21)
            alpha = 0.8

            design_shortest = regmod.apply_alpha_to_design(self.design_matrices_subopts[max_path_depth], n_subopt=max_path_depth, alpha=alpha)
            design_model = solver.torch.tensor(design_shortest)

            delta_results = {}
            for delta in tqdm(delta_space):

                # Define mapping area
                non_zero_mask = self.y_ground > 0
                y_masked = self.y_ground[non_zero_mask]
                design_model_masked = design_model[non_zero_mask]

                np.random.seed(99)
                x_init = solver.torch.tensor(np.random.rand(len(self.y_ground))).requires_grad_(True)

                x = deepcopy(x_init)
                x_opt, loss = solver.gradient_descent_solver(x, y_masked, design_model_masked,
                                                        n_iter=10000, verbose=False, 
                                                        early_stop=1e-10, step_size=5e-1, delta=delta,
                                                        l2_penalty=1, verbose=False)
                
                delta_results[delta] = (x_opt,loss)

            save(op.join(DATA_DIR, f"effective_delays_maxdepth{max_path_depth}_alphas_results.pkl"), alphas_results)
            save(op.join(DATA_DIR, f"effective_delays_maxdepth{max_path_depth}_delta_results.pkl"), delta_results)


        # ax[0] & ax[1]: varying alpha / ax[2] & ax[3]: scatter plots varying delta
        fig, ax = plt.subplots(1, 4, figsize=(17, 3))
        losses = [alphas_results[alpha][1] for alpha in alpha_space]
        ax[0].set_ylabel('Loss', fontsize=16)
        ax[0].set_xlabel(r'Search space - $\alpha$', fontsize=16)
        ax[0].tick_params(labelsize=14)
        ax[0].plot(alpha_space, losses, color='k')

        x1 = alphas_results[alpha_space[8]][0]
        x2 = alphas_results[alpha_space[-1]][0]
        y = self.y_ground

        x1_mask = x1 > 1
        x2_mask = x2 > 1
        y_mask = y != 0
        xy_mask1 = np.logical_and(x1_mask, y_mask).numpy().astype(bool)
        xy_mask2 = np.logical_and(x2_mask, y_mask).numpy().astype(bool)

        ax[1].scatter(y[xy_mask1], x1[xy_mask1], s=100, alpha=.4, edgecolors="none", color='red', label=rf'$\alpha={alpha_space[8]:.2f}$')
        ax[1].scatter(y[xy_mask2], x2[xy_mask2], s=100, alpha=.4, edgecolors="none", color='blue', label=rf'$\alpha={alpha_space[-1]:.2f}$')
        # ax.scatter(y[~y_mask], x[~y_mask], s=100, alpha=.4, color="tab:brown", edgecolors="none")
        # ax.scatter(y[~x_mask], x[~x_mask], s=100, alpha=.4, color="tab:purple", edgecolors="none")
        ax[1].plot(np.linspace(0,150), np.linspace(0,150), linestyle='--', color="gray", linewidth=2, label="1:1")

        ax[1].set_xlabel("Conductance delays", fontsize=16)
        ax[1].set_ylabel("Effective delays", fontsize=16)
        ax[1].tick_params(labelsize=14)
        ax[1].legend(fontsize=16)

        losses = [delta_results[delta][1] for delta in delta_space]
        ax[2].set_ylabel('Loss', fontsize=16)
        ax[2].set_xlabel(r'Search space - $\delta$ (ms)', fontsize=16)
        ax[2].tick_params(labelsize=14)
        ax[2].plot(delta_space, losses, color='k')
        ax[2].scatter([delta_space[1], delta_space[17]], [losses[1], losses[17]], 100, c='r', marker='x', label='local minima')
        ax[2].legend(prop={'size': 16})

        # plot the mapping curve and see what it looks like
        x1 = delta_results[delta_space[1]][0]
        x2 = delta_results[delta_space[17]][0]
        y = self.y_ground

        x_mask1 = x1 > 1e-2
        x_mask2 = x2 > 1e-2
        y_mask = y != 0
        xy_mask1 = np.logical_and(x_mask1, y_mask).numpy().astype(bool)
        xy_mask2 = np.logical_and(x_mask2, y_mask).numpy().astype(bool)

        ax[3].scatter(y[xy_mask1], x1[xy_mask1], s=100, alpha=.4, edgecolors="none", color='red', label=r'$\delta=5$')
        ax[3].scatter(y[xy_mask2], x2[xy_mask2], s=100, alpha=.4, edgecolors="none", color='blue', label=r'$\delta=50$')
        # ax.scatter(y[~y_mask], x[~y_mask], s=100, alpha=.4, color="tab:brown", edgecolors="none")
        # ax.scatter(y[~x_mask], x[~x_mask], s=100, alpha=.4, color="tab:purple", edgecolors="none")
        ax[3].plot(np.linspace(0,250), np.linspace(0,250), linestyle='--', color="gray", linewidth=2, label="1:1")
        ax[3].set_xlabel("Conductance delays", fontsize=16)
        ax[3].set_ylabel("Effective delays", fontsize=16)
        ax[3].set_yscale('log')
        ax[3].tick_params(labelsize=14)

        ax[3].legend(fontsize=16)
        if not self.verbose:
            plt.close()
        plt.show()

        return

    def run_experiment2(self):
        max_path_depth = 1

        alpha_space = np.linspace(0, 10, 21)
        delta_space = np.linspace(0, 60, 21)

        if os.path.exists(op.join(DATA_DIR, f"effective_delays_maxdepth{max_path_depth}_ad_results.pkl")):
            ad_results = load(op.join(DATA_DIR, f"effective_delays_maxdepth{max_path_depth}_ad_results.pkl"))
        else:
            ad_results = {}
            for aidx, alpha in enumerate(tqdm(alpha_space)):
                for didx, delta in enumerate(delta_space):
                    
                    design_shortest = regmod.apply_alpha_to_design(self.design_matrices_subopts[max_path_depth], n_subopt=max_path_depth, alpha=alpha)

                    # Define mapping area
                    non_zero_mask = self.y_ground > 0
                    y_masked = self.y_ground[non_zero_mask]
                    design_model_masked = design_shortest[non_zero_mask]

                    non_zero_mask = solver.torch.from_numpy(non_zero_mask)
                    y_masked = solver.torch.from_numpy(y_masked)
                    design_model_masked = solver.torch.from_numpy(design_model_masked)

                    np.random.seed(99)
                    x_init = solver.torch.tensor(np.random.rand(len(self.y_ground))).requires_grad_(True)

                    x_opt, loss = solver.gradient_descent_solver(x_init, y_masked, design_model_masked,
                                                            n_iter=5000, verbose=False, 
                                                            early_stop=1e-10, step_size=5e-1, delta=delta,
                                                            l2_penalty=1)
                    
                    ad_results[(aidx, didx)] = (x_opt,loss)
            save(op.join(DATA_DIR, f"effective_delays_maxdepth{max_path_depth}_ad_results.pkl"), ad_results)

        # Loss / Speed / fiber length correlation Landscape
        from sklearn.linear_model import LinearRegression
        from scipy import stats

        fibers_array = load(os.path.join(self.path_to_resources, 'fiber_lengths.pkl'))

        speed_land = np.zeros((len(alpha_space), len(delta_space)))
        corr_land = np.zeros((len(alpha_space), len(delta_space)))
        loss_land = np.zeros((len(alpha_space), len(delta_space)))

        for aidx, alpha in enumerate(alpha_space):
            for didx, delta in enumerate(delta_space):
                opt, loss = ad_results[(aidx,didx)]
                loss_land[aidx, didx] = loss
                opt = add_diagonal_entries(opt.reshape(fibers_array.shape[0], fibers_array.shape[1]-1))

                mask = np.logical_and((opt > 0.1), (fibers_array > 0))

                y = opt[mask]
                x = fibers_array[mask]

                model = LinearRegression()
                model.fit(x.reshape(-1,1), y)

                # r_squared = model.score(x.reshape(-1,1), y)

                model_est_speed = LinearRegression()
                model_est_speed.fit(y.reshape(-1,1)/1e3, x/1e3)
                
                speed_land[aidx, didx] = model_est_speed.coef_[0]
                
                corr = stats.pearsonr(x, y)[0]
                corr_land[aidx, didx] = corr

        x, y = np.meshgrid(delta_space, alpha_space)

        lands = [corr_land, speed_land]
        # lands_label = ['Correlation w/ Fiber Length', r'Axonal Speed (m.s$^{-1}$)']
        # crosses_pairs = [(11, 2), (10, 2), (9, 2), (12, 1), (11, 1), (10, 1)]

        fig, ax = plt.subplots(1, 2, figsize=(12, 3))
        for k in range(2):
            cs = ax[k].contourf(x / 3, y * 2, lands[k], 40, cmap='viridis')
            
            
            # ax.axhline(y=2, xmin=0, xmax=0.5,color='red', linestyle=':', linewidth=2)
            # ax.axvline(x=10, ymin=0, ymax=0.1, color='red', linestyle=':', linewidth=2)
            ax[k].axhline(y=2, color='red', linestyle=':', linewidth=2)
            ax[k].axvline(x=10, color='red', linestyle=':', linewidth=2)
            # ax.text(-1, coord[0], str(coord[0]) + 'th', va='center', ha='right', 
            #         color='black', fontsize=8, fontname='Helvetica')
            ax[k].text(-1.6, 1.5,  r'$\rho_{max}$', va='bottom', ha='center', 
                    color='red', fontsize=14, fontname='Helvetica')
            
            ax[k].set_xticks(np.arange(len(delta_space))[::5])
            ax[k].set_xticklabels(delta_space.astype(int)[::5], rotation=0, fontsize=16)  # Increase xticks size
            # ax.set_title(lands_label[k], fontsize=20)  # Increase title size
            cbar = plt.colorbar(cs, ax=ax[k])
            cbar.ax.tick_params(labelsize=16)  # Increase color bar font size

            ax[k].set_yticks(np.arange(len(alpha_space))[::-1][::4])
            ax[k].set_yticklabels(alpha_space.astype(float)[::-1].astype(int)[::4], rotation=0, fontsize=16)  # Increase yticks size
            if k == 0:
                ax[k].set_xlabel(r'$\delta$ (ms)', size=16)
                ax[k].set_ylabel(r'$\alpha$', size=16)
        
        if not self.verbose:
            plt.close()
        plt.show()
        return

if __name__ == "__main__":
    run()
