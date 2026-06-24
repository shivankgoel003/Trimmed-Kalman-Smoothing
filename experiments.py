"""
experiments.py
--------------
Monte Carlo trial runner for the constant-velocity model.

Runs one trial (single seed) that pits the standard Gaussian smoother
against the measurement-trimmed smoother and returns a summary dict
with MSE, outlier detection rate, and false-trim rate.

Typical usage: call run_one_trial in a loop and collect results into
a pandas DataFrame for aggregate analysis.

Usage
-----
    import pandas as pd
    from experiments import run_one_trial

    results = [run_one_trial(seed=s) for s in range(30)]
    df = pd.DataFrame(results)
    print(df.mean())
"""

import numpy as np

from simulation       import simulate_constant_velocity_data
from smoother         import weighted_batch_smoother
from measurement_trim import trimmed_kalman_smoother


def run_one_trial(seed, N=100, outlier_fraction=0.10, gamma=0.5):
    """
    Run one simulation trial and return summary statistics.

    Parameters
    ----------
    seed : int
        Random seed for data generation.
    N : int
        Number of time steps.
    outlier_fraction : float
        Fraction of measurements that are outliers.
    gamma : float
        Projected-gradient stepsize for the trimmed smoother.

    Returns
    -------
    dict with keys:
        seed, mse_gaussian, mse_trimmed, improvement_factor,
        detection_rate, false_trim_rate, iterations
    """
    G, H, Q, R, x_true, z, outlier_idx = simulate_constant_velocity_data(
        N=N,
        outlier_fraction=outlier_fraction,
        seed=seed,
    )

    # Gaussian smoother (uniform weights)
    x_gauss = weighted_batch_smoother(
        G, H, Q, R, z,
        omega=np.ones(N),
        x0_prior=np.array([1.0, 0.0]),
        P0=1.0 * np.eye(2),
    )

    # Trimmed smoother
    h = int((1.0 - outlier_fraction) * N)

    x_trim, omega_trim, _, history = trimmed_kalman_smoother(
        G, H, Q, R, z,
        h=h,
        gamma=gamma,
        max_iter=50,
        tol=1e-8,
        x0_prior=np.array([1.0, 0.0]),
        P0=1.0 * np.eye(2),
        verbose=False,
    )

    # MSE on position (column 1)
    mse_gauss = np.mean((x_gauss[:, 1] - x_true[:, 1]) ** 2)
    mse_trim  = np.mean((x_trim[:, 1]  - x_true[:, 1]) ** 2)

    # Outlier detection metrics
    q_trim       = N - h
    trimmed_idx  = np.argsort(omega_trim)[:q_trim]

    true_outliers     = set(outlier_idx.tolist())
    detected_trimmed  = set(trimmed_idx.tolist())

    correctly_trimmed = true_outliers.intersection(detected_trimmed)
    false_trimmed     = detected_trimmed.difference(true_outliers)

    detection_rate   = len(correctly_trimmed) / len(true_outliers)
    false_trim_rate  = len(false_trimmed) / q_trim

    return {
        "seed":              seed,
        "mse_gaussian":      mse_gauss,
        "mse_trimmed":       mse_trim,
        "improvement_factor": mse_gauss / mse_trim,
        "detection_rate":    detection_rate,
        "false_trim_rate":   false_trim_rate,
        "iterations":        len(history),
    }
