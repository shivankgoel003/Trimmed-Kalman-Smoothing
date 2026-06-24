"""
measurement_trim.py
-------------------
Alternating projected-gradient smoother for measurement outliers.

Algorithm (per iteration)
--------------------------
1. Fix omega -> solve the weighted batch smoother for x_hat.
2. Compute per-measurement losses  f_k(x_hat).
3. Project-gradient update: omega <- proj_{Delta_h}(omega - gamma * f).

Convergence is declared when ||omega_new - omega|| < tol.

Usage
-----
    from measurement_trim import trimmed_kalman_smoother

    x_hat, omega, losses, history = trimmed_kalman_smoother(
        G, H, Q, R, z, h=h_keep
    )
"""

import numpy as np

from projection import project_capped_simplex
from smoother   import weighted_batch_smoother
from losses     import measurement_losses


def trimmed_kalman_smoother(
    G,
    H,
    Q,
    R,
    z,
    h,
    gamma=0.5,
    max_iter=50,
    tol=1e-8,
    x0_prior=None,
    P0=None,
    verbose=True,
):
    """
    Alternating measurement-trimmed Kalman smoother.

    Parameters
    ----------
    G, H, Q, R : np.ndarray
        State-space matrices.
    z : np.ndarray, shape (N,)
        Measurements.
    h : int or float
        Number of measurements to *keep*.
        E.g. N=100, h=90 trims the 10 highest-loss measurements.
    gamma : float
        Projected-gradient stepsize for the weight update.
    max_iter : int
        Maximum number of alternating iterations.
    tol : float
        Convergence tolerance on ||omega_new - omega||.
    x0_prior : np.ndarray, shape (n,), optional
        Prior mean on the initial state.
    P0 : np.ndarray, shape (n, n), optional
        Prior covariance on the initial state.
    verbose : bool
        Print convergence info each iteration.

    Returns
    -------
    x_hat : np.ndarray, shape (N, n)
        Final smoothed state estimate.
    omega : np.ndarray, shape (N,)
        Final measurement weights.
    losses : np.ndarray, shape (N,)
        Measurement losses at the final x_hat.
    history : list of dict
        Per-iteration diagnostics.
    """
    N = len(z)

    # Initialise with uniform weights summing to h
    omega = np.full(N, h / N)
    history = []

    for it in range(max_iter):
        # Step 1: weighted smoother
        x_hat = weighted_batch_smoother(
            G, H, Q, R, z,
            omega=omega,
            x0_prior=x0_prior,
            P0=P0,
        )

        # Step 2: measurement losses
        f = measurement_losses(H, R, z, x_hat)

        # Step 3: projected-gradient weight update
        omega_new = project_capped_simplex(omega - gamma * f, h)

        omega_change = np.linalg.norm(omega_new - omega)

        history.append({
            "iteration":   it,
            "omega_change": omega_change,
            "min_weight":  np.min(omega_new),
            "max_weight":  np.max(omega_new),
            "max_loss":    np.max(f),
            "mean_loss":   np.mean(f),
        })

        if verbose:
            print(
                f"iter {it:02d} | "
                f"omega change = {omega_change:.3e} | "
                f"min omega = {np.min(omega_new):.3f} | "
                f"max omega = {np.max(omega_new):.3f}"
            )

        omega = omega_new

        if omega_change < tol:
            break

    # Final estimate at converged weights
    x_hat = weighted_batch_smoother(
        G, H, Q, R, z,
        omega=omega,
        x0_prior=x0_prior,
        P0=P0,
    )
    losses = measurement_losses(H, R, z, x_hat)

    return x_hat, omega, losses, history
