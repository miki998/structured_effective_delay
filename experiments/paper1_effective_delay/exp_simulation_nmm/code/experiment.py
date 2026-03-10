"""
Experiment: Simulation of Neural Mass Model
"""

import os
import json
from datetime import datetime

# Import necessary libraries
import matplotlib.pyplot as plt
import numpy as np

from src.utils import load_json

import numpy as np
from dataclasses import dataclass
from typing import Callable, Dict, Tuple, List, Optional

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
        print("Experiment: Simulation of Neural Mass Model")
        print("=" * 60)

    # Configuration
    config = load_json(os.path.join(EXPERIMENT_DIR, "config.json"))

    if verbose:
        print(f"\nConfiguration: {config}")

    experiments = Experiments(config, verbose=verbose)

    fig1 = None  # Placeholder for figures from experiment 1

    print("\nRunning Experiment 1: Simple graph simulation")
    fig1 = experiments.run_experiment1()

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
                os.path.join(RESULTS_DIR, f"simulated_activity.png"),
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

        self.d = np.array(self.config["delays"])

    # ----------------------------
    # Part A: Direct DDE simulation
    # ----------------------------
    def simulate_triangle_dde(self,
        d: np.ndarray,
        t_end: float = 10.0,
        dt: float = 1e-3,
        V0_at_0: float = 1.0,
    ):
        """
        Simulate the 3-node cycle (triangle) DDE with symmetric binary weights:

            dV0/dt = -V0(t) + V1(t-d01) + V2(t-d02)
            dV1/dt = -V1(t) + V0(t-d01) + V2(t-d12)
            dV2/dt = -V2(t) + V0(t-d02) + V1(t-d12)

        History: V(t)=0 for t<0 and V(0)=[V0_at_0,0,0]^T.

        Method: explicit RK4 with delayed terms obtained by linear interpolation
        of past samples on a uniform grid.
        """
        if np.min(d) < 0:
            raise ValueError("Delays must be nonnegative.")
        if dt <= 0 or t_end <= 0:
            raise ValueError("dt and t_end must be positive.")

        t = np.arange(0.0, t_end + dt, dt)
        T = len(t)
        V = np.zeros((T, 3), dtype=float)
        V[0, :] = np.array([V0_at_0, 0.0, 0.0], dtype=float)

        # helper: delayed value V_j(t_query) for t_query in R
        # using the current discrete history V[0:k+1] at times t[0:k+1]
        def delayed_value(k: int, j: int, t_query: float) -> float:
            if t_query < 0.0:
                return 0.0  # history is zero for t<0
            # Convert t_query to index in grid: between t[m] and t[m+1]
            x = t_query / dt
            m = int(np.floor(x))
            if m >= k:  # t_query is at or beyond current time -> clamp
                return V[k, j]
            if m < 0:
                return 0.0
            frac = x - m
            # linear interpolation between samples m and m+1
            return (1.0 - frac) * V[m, j] + frac * V[m + 1, j]

        # define RHS f(t, V(t), delayed terms)
        def rhs(k: int, tt: float, state: np.ndarray) -> np.ndarray:
            # We evaluate delays based on already computed history V up to index k
            V0, V1, V2 = state
            V1_d01 = delayed_value(k, 1, tt - d[1, 0])
            V2_d02 = delayed_value(k, 2, tt - d[2, 0])
            V0_d01 = delayed_value(k, 0, tt - d[0, 1])
            V2_d12 = delayed_value(k, 2, tt - d[2, 1])
            V0_d02 = delayed_value(k, 0, tt - d[0, 2])
            V1_d12 = delayed_value(k, 1, tt - d[1, 2])

            dV0 = -V0 + V1_d01 + V2_d02
            dV1 = -V1 + V0_d01 + V2_d12
            dV2 = -V2 + V0_d02 + V1_d12
            return np.array([dV0, dV1, dV2], dtype=float)

        # time stepping with RK4 (still explicit; stable enough for small dt)
        for k in range(T - 1):
            tk = t[k]
            yk = V[k].copy()

            k1 = rhs(k, tk, yk)
            k2 = rhs(k, tk + 0.5 * dt, yk + 0.5 * dt * k1)
            k3 = rhs(k, tk + 0.5 * dt, yk + 0.5 * dt * k2)
            k4 = rhs(k, tk + dt, yk + dt * k3)

            V[k + 1] = yk + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        return t, V


    # -------------------------------------------------------
    # Part B: Modal (Laplace/residue) approximation (optional)
    # -------------------------------------------------------
    # This uses your formulas:
    #
    #   Delta(s) = (s+1)^3 - (s+1)(e^{-2sd01}+e^{-2sd02}+e^{-2sd12}) - 2 e^{-s(d01+d02+d12)}
    #   N0(s) = (s+1)^2 - e^{-2sd12}
    #   N1(s) = e^{-sd01}(s+1) + e^{-s(d02+d12)}
    #   N2(s) = e^{-sd02}(s+1) + e^{-s(d01+d12)}
    #   c_{k,i} = N_i(s_k)/Delta'(s_k)
    #
    # Roots s_k are complex and must be found numerically.
    #
    # Requires: mpmath
    #
    # NOTE: This is a *truncated* approximation. The DDE simulator above is the reference.

    def modal_approx_triangle(self,
        d: np.ndarray,
        roots_initial_guesses: List[complex],
        t_eval: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute a K-mode modal approximation for V(t) on t_eval using user-derived Laplace/residue formulas.

        Parameters
        ----------
        roots_initial_guesses : list of complex
            Initial guesses for roots of Delta(s)=0 (one guess per desired root).
            If you include a complex root, also include its conjugate guess if you want real output.

        Returns
        -------
        s_roots : complex ndarray shape (K,)
        V_modal : complex ndarray shape (len(t_eval), 3)
            If you include conjugate pairs, take np.real(V_modal) to get real signals.
        """
        import mpmath as mp

        mp.mp.dps = 50  # precision; increase if needed

        def M(s):
            return (s + 1)**3 - (s + 1) * mp.e**(-s*(d[1, 2] + d[2, 1]))

        def M_prime(s):
            # derivative of M(s) with respect to s
            dsum = d[1, 2] + d[2, 1]
            return 3*(s+1)**2 - mp.e**(-s * dsum) + (s + 1) * dsum * mp.e**(-s * dsum)

        def N0(s):
            return (s + 1)**2 - mp.e**(-s*(d[1, 2] + d[2, 1]))

        def N1(s):
            return mp.e**(-s*d[0, 1])*(s + 1) + mp.e**(-s*(d[0, 2] + d[2, 1]))

        def N2(s):
            return mp.e**(-s*d[0, 2])*(s + 1) + mp.e**(-s*(d[0, 1] + d[1, 2]))

        # Find roots
        s_roots = []
        for guess in roots_initial_guesses:
            # mp.findroot works best with one guess for simple roots
            root = mp.findroot(M, guess, solver='newton', tol=1e-20, maxsteps=10000)
            s_roots.append(complex(root))
        
        # Remove duplicates (roots that are very close to each other)
        tolerance = 1e-6
        unique_roots = []
        for root in s_roots:
            is_duplicate = False
            for unique_root in unique_roots:
                if abs(root - unique_root) < tolerance:
                    is_duplicate = True
                    break
            if not is_duplicate:
                if self.config["complex_only"] and np.abs(root.imag) < tolerance:
                    continue
                unique_roots.append(root)

        s_roots = unique_roots
        s_roots = sorted(s_roots, key=lambda x: abs(x))  # sort by magnitude
        s_roots = np.array(s_roots, dtype=np.complex128)

        # Residue coefficients for each mode and component
        c = np.zeros((len(s_roots), 3), dtype=np.complex128)
        for k, sk in enumerate(s_roots):
            sk_mp = mp.mpc(sk.real, sk.imag)
            denom = M_prime(sk_mp)
            c[k, 0] = complex(N0(sk_mp) / denom)
            c[k, 1] = complex(N1(sk_mp) / denom)
            c[k, 2] = complex(N2(sk_mp) / denom)

        # Evaluate modal sum
        t_eval = np.asarray(t_eval, dtype=float)
        V_modal = np.zeros((len(t_eval), 3), dtype=np.complex128)
        for k, sk in enumerate(s_roots):
            V_modal += np.exp(sk * t_eval[:, None]) * c[k, :][None, :]

        return s_roots, V_modal
    
    def modal_approx_triangle_2_root(self,
        d: np.ndarray,
        roots_initial_guesses: List[complex],
        t_eval: np.ndarray, simulated_timecourse: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute a K-mode modal approximation for V(t) on t_eval using user-derived Laplace/residue formulas.

        Parameters
        ----------
        roots_initial_guesses : list of complex
            Initial guesses for roots of Delta(s)=0 (one guess per desired root).
            If you include a complex root, also include its conjugate guess if you want real output.

        Returns
        -------
        s_roots : complex ndarray shape (K,)
        V_modal : complex ndarray shape (len(t_eval), 3)
            If you include conjugate pairs, take np.real(V_modal) to get real signals.
        """
        import mpmath as mp

        mp.mp.dps = 50  # precision; increase if needed

        def M(s):
            return (s + 1)**3 - (s + 1) * mp.e**(-s*(d[1, 2] + d[2, 1]))

        def M_prime(s):
            # derivative of M(s) with respect to s
            dsum = d[1, 2] + d[2, 1]
            return 3*(s+1)**2 - mp.e**(-s * dsum) + (s + 1) * dsum * mp.e**(-s * dsum)

        def N0(s):
            return (s + 1)**2 - mp.e**(-s*(d[1, 2] + d[2, 1]))

        def N1(s):
            return mp.e**(-s*d[0, 1])*(s + 1) + mp.e**(-s*(d[0, 2] + d[2, 1]))

        def N2(s):
            return mp.e**(-s*d[0, 2])*(s + 1) + mp.e**(-s*(d[0, 1] + d[1, 2]))

        # Find roots
        s_roots = []
        for guess in roots_initial_guesses:
            # mp.findroot works best with one guess for simple roots
            root = mp.findroot(M, guess, solver='newton', tol=1e-20, maxsteps=10000)
            s_roots.append(complex(root))
        
        # Remove duplicates (roots that are very close to each other)
        tolerance = 1e-6
        unique_roots = []
        for root in s_roots:
            is_duplicate = False
            for unique_root in unique_roots:
                if abs(root - unique_root) < tolerance:
                    is_duplicate = True
                    break
            if not is_duplicate:
                if self.config["complex_only"] and np.abs(root.imag) < tolerance:
                    continue
                unique_roots.append(root)

        s_roots = unique_roots
        s_roots = np.array(s_roots, dtype=np.complex128)

        tolerance = 1e-6

        combinations = []
        for root in s_roots:
            if np.abs(root.imag) < tolerance:
                continue  # skip real roots
            nearby_roots = [s for s in s_roots if (np.abs(np.abs(s) - np.abs(root)) < 5e-1) and s != root and s != np.conjugate(root)]
            print(nearby_roots)
            s_root_candidate = np.array([root, np.conjugate(root)] + nearby_roots, dtype=np.complex128)

            # Residue coefficients for each mode and component
            c = np.zeros((len(s_root_candidate), 3), dtype=np.complex128)
            for k, sk in enumerate(s_root_candidate):
                sk_mp = mp.mpc(sk.real, sk.imag)
                denom = M_prime(sk_mp)
                c[k, 0] = complex(N0(sk_mp) / denom)
                c[k, 1] = complex(N1(sk_mp) / denom)
                c[k, 2] = complex(N2(sk_mp) / denom)

            # Evaluate modal sum
            t_eval = np.asarray(t_eval, dtype=float)
            V_modal = np.zeros((len(t_eval), 3), dtype=np.complex128)
            for k, sk in enumerate(s_root_candidate):
                V_modal += np.exp(sk * t_eval[:, None]) * c[k, :][None, :]

            error = np.linalg.norm(simulated_timecourse[:, 1] - V_modal.real[:, 1]) + np.linalg.norm(simulated_timecourse[:, 2] - V_modal.real[:, 2])

            combinations.append((s_root_candidate, V_modal, error))

        return combinations

    def run_experiment1(self):
        t, V = self.simulate_triangle_dde(self.d, t_end=10.0)

        vmin, vmax = -self.config["range_roots"], self.config["range_roots"]
        density_guesses = (vmax - vmin) / self.config["density_roots"]
        
        real_root_guesses = list(np.arange(vmin, vmax, density_guesses))
        complex_root_guesses = [r1 + 1.0j * r2 for r1 in real_root_guesses for r2 in real_root_guesses]
        guesses = real_root_guesses + complex_root_guesses  # Initial guesses for roots (real and complex)

        rescaled_t = t * 1  # Rescale time for root finding (optional, can help with convergence)
        s_roots, V_modal = self.modal_approx_triangle(self.d, guesses, t_eval=rescaled_t)
        print("Roots:", s_roots)
        print("Number of distinct roots:", s_roots.shape[0])

        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(t, V[:, 0], label=r'$V_0$')
        ax[0].plot(t, V[:, 1], label=r'$V_1$')
        ax[0].plot(t, V[:, 2], label=r'$V_2$')
        ax[0].set_xlabel("Time (s)", fontsize=self.ticks_labels_fontsize)
        ax[0].set_ylabel("Activity", fontsize=self.ticks_labels_fontsize)
        ax[0].set_title(f"DDE Simulation", fontsize=self.title_fontsize)
        ax[0].legend(prop={'size': self.ticks_labels_fontsize})
        ax[0].grid(
            True, which='both', linestyle='--', linewidth=0.5, alpha=0.7
        )

        # Enforce history function on all modes
        V_modal[:, 1] -= V_modal[:, 1].min()
        V_modal[:, 2] -= V_modal[:, 2].min()

        ax[1].plot(t, V_modal.real[:, 0], label=r'$V_0$')
        ax[1].plot(t, V_modal.real[:, 1], label=r'$V_1$')
        ax[1].plot(t, V_modal.real[:, 2], label=r'$V_2$')
        ax[1].set_xlabel("Time (s)", fontsize=self.ticks_labels_fontsize)
        # ax[1].set_ylabel("Activity", fontsize=self.ticks_labels_fontsize)
        ax[1].set_title(f"DDE Modal Approximation ({s_roots.shape[0]} roots)", fontsize=self.title_fontsize)
        ax[1].legend(prop={'size': self.ticks_labels_fontsize})
        ax[1].grid(
            True, which='both', linestyle='--', linewidth=0.5, alpha=0.7
        )

        ax[1].set_ylim(-0.05, 1.05)

        ax[0].tick_params(labelsize=self.ticks_labels_fontsize)
        ax[1].tick_params(labelsize=self.ticks_labels_fontsize)
        if not self.verbose:
            plt.close()
        plt.show()

        return fig
    
    def run_experiment2(self):
        from matplotlib.widgets import Slider

        # Find the best 2 roots that approximate the DDE solution and plot the modal approximation using only those 2 roots.
        
        t, V = self.simulate_triangle_dde(self.d, t_end=10.0)

        vmin, vmax = -self.config["range_roots"], self.config["range_roots"]
        density_guesses = (vmax - vmin) / self.config["density_roots"]
        
        real_root_guesses = list(np.arange(vmin, vmax, density_guesses))
        complex_root_guesses = [r1 + 1.0j * r2 for r1 in real_root_guesses for r2 in real_root_guesses]
        guesses = real_root_guesses + complex_root_guesses  # Initial guesses for roots (real and complex)

        rescaled_t = t * 1  # Rescale time for root finding (optional, can help with convergence)
        combinations = self.modal_approx_triangle_2_root(self.d, guesses, t_eval=rescaled_t, simulated_timecourse=V)
        
        # Interactive plotting with slider
        fig_interactive, ax_plot = plt.subplots(1, figsize=(10, 8))
        plt.subplots_adjust(bottom=0.25)

        ax_slider = plt.axes([0.25, 0.1, 0.5, 0.03])
        slider = Slider(ax_slider, 'Combination', 0, len(combinations) - 1, valinit=0, valstep=1)
        
        def update(val):
            idx = int(slider.val)
            roots, modal, err = combinations[idx]
            modal_plot = modal.copy()
            modal_plot[:, 1] -= modal_plot[:, 1].min()
            modal_plot[:, 2] -= modal_plot[:, 2].min()
            
            ax_plot.clear()
            ax_plot.plot(t, V[:, 1], label=r'$V_1$ (DDE)', linewidth=2)
            ax_plot.plot(t, V[:, 2], label=r'$V_2$ (DDE)', linewidth=2)
            ax_plot.plot(t, modal_plot.real[:, 1], label=r'$V_1$ (Modal)', linestyle='--', linewidth=1.5)
            ax_plot.plot(t, modal_plot.real[:, 2], label=r'$V_2$ (Modal)', linestyle='--', linewidth=1.5)
            ax_plot.set_title(f"Combination {idx}/{len(combinations)-1}: error={err:.4f}", fontsize=self.title_fontsize)
            ax_plot.legend(fontsize=self.ticks_labels_fontsize)
            ax_plot.grid(True, alpha=0.3)
            ax_plot.tick_params(labelsize=self.ticks_labels_fontsize)
            ax_plot.set_xlabel("Time (s)", fontsize=self.ticks_labels_fontsize)
            ax_plot.set_ylabel("Activity", fontsize=self.ticks_labels_fontsize)
            fig_interactive.canvas.draw_idle()
        
        slider.on_changed(update)
        update(0)
        
        if not self.verbose:
            plt.close()
        plt.show()


if __name__ == "__main__":
    run()
