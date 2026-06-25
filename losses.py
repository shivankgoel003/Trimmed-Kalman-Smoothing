"""
losses.py

Residual loss functions used by the trimmed smoothers.

Functions
---------
measurement_losses
    Per-timestep measurement loss:
        f_k(x) = 0.5 ||z_k - H x_k||_{R^{-1}}^2

compute_innovation_losses
    Per-transition scalar innovation loss:
        g_k(x) = 0.5 ||x_{k+1} - G x_k||_{Q^{-1}}^2

summarize_innovation_loss_near_jumps
    Convenience diagnostic: for each known jump index, report
    the peak innovation loss within a window around that transition.
    Returns a pandas DataFrame.

Usage
-----
    from losses import (
        measurement_losses,
        compute_innovation_losses,
        summarize_innovation_loss_near_jumps,
    )
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Measurement losses
# ---------------------------------------------------------------------------

def measurement_losses(H, R, z, x_hat):
    """
    Compute per-timestep measurement losses.

        f_k(x) = 0.5 ||z_k - H x_k||_{R^{-1}}^2

    Parameters
    ----------
    H : np.ndarray, shape (m, n)
        Measurement matrix.
    R : np.ndarray, shape (m, m)
        Measurement-noise covariance.
    z : np.ndarray, shape (N,) or (N, m)
        Measurements.
    x_hat : np.ndarray, shape (N, n)
        Current state estimate.

    Returns
    -------
    losses : np.ndarray, shape (N,)
        Per-measurement quadratic losses.
    """
    z = np.asarray(z)
    Rinv = np.linalg.inv(R)

    N = x_hat.shape[0]
    losses = np.zeros(N)

    for k in range(N):
        # Supports both scalar measurements z[k] and vector measurements z[k, :]
        zk = np.atleast_1d(z[k])
        residual = zk - H @ x_hat[k]
        losses[k] = 0.5 * float(residual.T @ Rinv @ residual)

    return losses


# ---------------------------------------------------------------------------
# Scalar innovation losses
# ---------------------------------------------------------------------------

def compute_innovation_losses(x_hat, G, Q):
    """
    Compute per-transition scalar innovation residual losses.

        g_k(x) = 0.5 ||x_{k+1} - G x_k||_{Q^{-1}}^2

    This is the correct loss/gradient for scalar innovation trimming:

        0.5 * sum_k nu_k * ||x_{k+1} - G x_k||_{Q^{-1}}^2

    where one scalar weight nu_k is assigned to the whole transition.

    Parameters
    ----------
    x_hat : np.ndarray, shape (N, n)
        Smoothed state estimates.
    G : np.ndarray, shape (n, n)
        State-transition matrix.
    Q : np.ndarray, shape (n, n)
        Process-noise covariance.

    Returns
    -------
    residuals : np.ndarray, shape (N-1, n)
        Innovation residuals:
            x_{k+1} - G x_k

    losses : np.ndarray, shape (N-1,)
        Full scalar innovation losses:
            0.5 * residual.T @ Q^{-1} @ residual
    """
    N = x_hat.shape[0]
    L = N - 1
    Qinv = np.linalg.inv(Q)

    residuals = np.zeros((L, x_hat.shape[1]))
    losses = np.zeros(L)

    for k in range(L):
        r = x_hat[k + 1] - G @ x_hat[k]
        residuals[k] = r
        losses[k] = 0.5 * float(r.T @ Qinv @ r)

    return residuals, losses


# ---------------------------------------------------------------------------
# Diagnostic: innovation loss near known jump transitions
# ---------------------------------------------------------------------------

def summarize_innovation_loss_near_jumps(innovation_losses, jump_indices, radius=3):
    """
    For each jump index, report the peak innovation loss within a
    window of `radius` transitions around the corresponding transition.

    Parameters
    ----------
    innovation_losses : np.ndarray, shape (N-1,)
        Per-transition innovation losses.
    jump_indices : array-like
        Indices of the true state jumps. If a jump occurs at state index k,
        the corresponding transition is k - 1.
    radius : int
        Half-width of the search window around each transition.

    Returns
    -------
    pd.DataFrame
        Columns:
            jump_idx,
            transition_idx,
            local_peak_idx,
            distance_from_jump_transition,
            innovation_loss_at_transition,
            max_innovation_loss_near_jump
    """
    rows = []

    for jump_idx in jump_indices:
        transition_idx = jump_idx - 1

        if transition_idx < 0 or transition_idx >= len(innovation_losses):
            continue

        left = max(0, transition_idx - radius)
        right = min(len(innovation_losses), transition_idx + radius + 1)

        local_losses = innovation_losses[left:right]
        local_argmax = np.argmax(local_losses)
        local_peak_idx = left + local_argmax

        rows.append({
            "jump_idx": jump_idx,
            "transition_idx": transition_idx,
            "local_peak_idx": local_peak_idx,
            "distance_from_jump_transition": abs(local_peak_idx - transition_idx),
            "innovation_loss_at_transition": innovation_losses[transition_idx],
            "max_innovation_loss_near_jump": innovation_losses[local_peak_idx],
        })

    return pd.DataFrame(rows)