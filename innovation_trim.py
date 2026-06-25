"""
innovation_trim.py
------------------
Alternating projected-gradient smoother for impulsive state jumps.

Where measurement_trim.py down-weights individual measurements,
this module down-weights individual *transitions* (innovation terms),
making the smoother robust to sudden changes in the latent state.

Algorithm (per iteration)
--------------------------
1. Fix nu -> solve the weighted batch smoother for x_hat.
2. Compute per-transition innovation losses g_k(x_hat).
3. Project-gradient update: nu <- proj_{Delta_{h_innov}}(nu - gamma * g).

Helper functions
----------------
warm_start_nu_from_jump_score
    Rough initialisation of nu based on consecutive measurement
    differences, so the smoother starts close to the jump locations.

detection_near_jumps
    Evaluate how many detected trims fall within `radius` steps of
    each true jump transition.

Usage
-----
    from innovation_trim import (
        innovation_trimmed_kalman_smoother,
        warm_start_nu_from_jump_score,
        detection_near_jumps,
    )
"""

import numpy as np

from projection import project_capped_simplex
from smoother import weighted_batch_smoother
from losses import compute_innovation_losses


# ---------------------------------------------------------------------------
# Main innovation-trimmed smoother
# ---------------------------------------------------------------------------

def innovation_trimmed_kalman_smoother(
    G,
    H,
    Q,
    R,
    z,
    h_innovation,
    gamma=None,
    gamma_factor=0.25,
    max_iter=100,
    tol=1e-8,
    x0_prior=None,
    P0=None,
    nu_init=None,
    verbose=True,
):
    """
    Alternating scalar innovation-trimmed Kalman smoother.

    Here one scalar weight nu_k is assigned to each transition:

        0.5 * nu_k * ||x_{k+1} - G x_k||_{Q^{-1}}^2

    Small nu_k means the transition k is allowed to violate the smooth
    dynamics, so it may represent a genuine state jump.

    Parameters
    ----------
    G, H, Q, R : np.ndarray
        State-space matrices.
    z : np.ndarray, shape (N,) or (N, m)
        Measurements.
    h_innovation : int or float
        Number of innovation transitions to keep.
        Since there are N-1 transitions, h_innovation = (N-1) - q
        trims q transitions.
    gamma : float or None
        Projected-gradient stepsize. If None, chosen automatically.
    gamma_factor : float
        Scale used for automatic gamma.
    max_iter : int
        Maximum alternating iterations.
    tol : float
        Stop when ||nu_new - nu|| < tol.
    x0_prior : np.ndarray, optional
        Prior mean for x_0.
    P0 : np.ndarray, optional
        Prior covariance for x_0.
    nu_init : np.ndarray, optional
        Initial innovation weights.
    verbose : bool
        Print iteration diagnostics.

    Returns
    -------
    x_hat : np.ndarray, shape (N, n)
        Final smoothed state estimate.
    nu : np.ndarray, shape (N-1,)
        Final scalar innovation weights.
    innovation_losses : np.ndarray, shape (N-1,)
        Final scalar innovation losses.
    innovation_residuals : np.ndarray, shape (N-1, n)
        Final innovation residuals.
    history : list of dict
        Per-iteration diagnostics.
    """
    N = len(z)
    L = N - 1

    if h_innovation < 0 or h_innovation > L:
        raise ValueError("h_innovation must satisfy 0 <= h_innovation <= N - 1.")

    omega = np.ones(N)

    # Initialize scalar innovation weights
    if nu_init is None:
        nu = np.full(L, h_innovation / L)
    else:
        nu = project_capped_simplex(nu_init, h_innovation)

    # --------------------------------------------------
    # Auto stepsize from initial scalar innovation losses
    # --------------------------------------------------
    if gamma is None:
        x_hat_init = weighted_batch_smoother(
            G=G,
            H=H,
            Q=Q,
            R=R,
            z=z,
            omega=omega,
            nu=nu,
            x0_prior=x0_prior,
            P0=P0,
        )

        innovation_residuals, init_full_losses = compute_innovation_losses(
            x_hat_init,
            G,
            Q,
        )

        max_init_loss = np.max(init_full_losses)
        gamma = gamma_factor / (max_init_loss + 1e-12)

        if verbose:
            print(f"Auto scalar-innovation gamma = {gamma:.6g}")
            print(f"Max initial scalar-innovation loss = {max_init_loss:.6g}")

    history = []

    for it in range(max_iter):
        # Step 1: weighted smoother with scalar innovation weights
        x_hat = weighted_batch_smoother(
            G=G,
            H=H,
            Q=Q,
            R=R,
            z=z,
            omega=omega,
            nu=nu,
            x0_prior=x0_prior,
            P0=P0,
        )

        # Step 2: full scalar innovation losses
        innovation_residuals, full_losses = compute_innovation_losses(
            x_hat,
            G,
            Q,
        )

        # Step 3: projected-gradient update
        nu_new = project_capped_simplex(
            nu - gamma * full_losses,
            h_innovation,
        )

        nu_change = np.linalg.norm(nu_new - nu)

        history.append({
            "iteration": it,
            "nu_change": nu_change,
            "min_nu": np.min(nu_new),
            "max_nu": np.max(nu_new),
            "max_innovation_loss": np.max(full_losses),
            "mean_innovation_loss": np.mean(full_losses),
            "gamma": gamma,
        })

        if verbose:
            print(
                f"iter {it:02d} | "
                f"nu change = {nu_change:.3e} | "
                f"min nu = {np.min(nu_new):.3f} | "
                f"max nu = {np.max(nu_new):.3f} | "
                f"max innovation loss = {np.max(full_losses):.3e}"
            )

        nu = nu_new

        if nu_change < tol:
            break

    # Final estimate at converged weights
    x_hat = weighted_batch_smoother(
        G=G,
        H=H,
        Q=Q,
        R=R,
        z=z,
        omega=omega,
        nu=nu,
        x0_prior=x0_prior,
        P0=P0,
    )

    innovation_residuals, full_losses = compute_innovation_losses(
        x_hat,
        G,
        Q,
    )

    return x_hat, nu, full_losses, innovation_residuals, history


# ---------------------------------------------------------------------------
# Warm-start helper
# ---------------------------------------------------------------------------

def warm_start_nu_from_jump_score(z, h_innovation, small_value=0.1):
    """
    Initialise nu by flagging transitions where consecutive measurements
    show a large relative difference.

    The heuristic one-sided jump score for transition k is:

        score_k = |z_{k+1} - z_k| / (1 + |z_k - z_{k-1}|)

    The q = (N-1) - h_innovation transitions with the highest scores are
    given weight `small_value`; the rest are set to 1 and the whole
    vector is then projected onto Delta_{h_innovation}.

    Parameters
    ----------
    z : np.ndarray, shape (N,)
        Measurements.
    h_innovation : int or float
        Number of transitions to keep.
    small_value : float
        Initial weight assigned to suspected jump transitions.

    Returns
    -------
    nu_init : np.ndarray, shape (N-1,)
        Projected initial innovation weights.
    trim_idx : np.ndarray
        Indices of the suspected jump transitions, sorted.
    jump_score : np.ndarray, shape (N-1,)
        Raw jump scores.
    """
    z = np.asarray(z)
    L = len(z) - 1

    if h_innovation < 0 or h_innovation > L:
        raise ValueError("h_innovation must satisfy 0 <= h_innovation <= N - 1.")

    q_innovation = int(round(L - h_innovation))

    if q_innovation <= 0:
        return np.ones(L), np.array([], dtype=int), None

    d = np.abs(np.diff(z))

    # Previous difference. Use median as boundary value at k = 0.
    d_prev = np.concatenate([[np.median(d)], d[:-1]])

    jump_score = d / (1.0 + d_prev)

    trim_idx = np.argsort(jump_score)[-q_innovation:]

    nu_init = np.ones(L)
    nu_init[trim_idx] = small_value
    nu_init = project_capped_simplex(nu_init, h_innovation)

    return nu_init, np.sort(trim_idx), jump_score


# ---------------------------------------------------------------------------
# Detection diagnostic
# ---------------------------------------------------------------------------

def detection_near_jumps(detected_idx, true_transition_idx, radius=2):
    """
    For each true jump transition, check whether at least one detected
    trim index falls within `radius` steps of it.

    Parameters
    ----------
    detected_idx : array-like
        Indices of transitions selected for trimming by the algorithm.
    true_transition_idx : array-like
        True jump transition indices, typically jump_idx - 1.
    radius : int
        Matching radius.

    Returns
    -------
    detection_rate : float
        Fraction of true jumps matched by at least one detected trim.
    matched : list of bool
        Per-jump match flags.
    """
    detected_idx = np.asarray(detected_idx)
    true_transition_idx = np.asarray(true_transition_idx)

    if len(true_transition_idx) == 0:
        return 0.0, []

    matched = []

    for j in true_transition_idx:
        near = detected_idx[np.abs(detected_idx - j) <= radius]
        matched.append(len(near) > 0)

    detection_rate = np.mean(matched)

    return detection_rate, matched