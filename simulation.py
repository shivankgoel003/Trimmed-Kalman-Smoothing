"""
simulation.py
-------------
Data-generation routines for the two test problems used in the notebook.

Functions
---------
simulate_constant_velocity_data
    1-D constant-velocity model with position-only measurements and
    randomly placed measurement outliers.

simulate_general_smooth_signal
    Generic derivative/signal model tracking a damped oscillatory signal
        s(t) = exp(-beta * t) * sin(alpha * t).
    No outliers are injected here; add them manually in the notebook.

Usage
-----
    from simulation import (
        simulate_constant_velocity_data,
        simulate_general_smooth_signal,
    )
"""

import numpy as np


# ---------------------------------------------------------------------------
# Constant-velocity model with measurement outliers
# ---------------------------------------------------------------------------

def simulate_constant_velocity_data(
    N=50,
    dt=0.1,
    process_std=0.05,
    meas_std=0.2,
    outlier_fraction=0.20,
    outlier_std=4.0,
    seed=1,
):
    """
    Simulate a 1-D constant-velocity state-space model.

    State:
        x_k = [velocity_k, position_k]

    Dynamics:
        x_k = G x_{k-1} + q_k,    q_k ~ N(0, Q)

    Measurement (position only):
        z_k = H x_k + r_k,         r_k ~ N(0, R)

    A random subset of measurements is corrupted with large outlier noise.

    Parameters
    ----------
    N : int
        Number of time steps.
    dt : float
        Sampling interval.
    process_std : float
        Standard deviation of the process noise (velocity-integrated).
    meas_std : float
        Standard deviation of the measurement noise.
    outlier_fraction : float
        Fraction of measurements to corrupt.
    outlier_std : float
        Standard deviation of the additional outlier noise
        (positive shift is also added to make outliers clearly visible).
    seed : int
        Random seed.

    Returns
    -------
    G : np.ndarray, shape (2, 2)
    H : np.ndarray, shape (1, 2)
    Q : np.ndarray, shape (2, 2)
    R : np.ndarray, shape (1, 1)
    x_true : np.ndarray, shape (N, 2)   -- [velocity, position]
    z : np.ndarray, shape (N,)
    outlier_idx : np.ndarray, shape (n_outliers,)
    """
    rng = np.random.default_rng(seed)

    G = np.array([[1, 0],
                  [dt, 1]])

    H = np.array([[0, 1]])      # observe position only

    sigma_q = process_std
    Q = sigma_q**2 * np.array([
        [dt,       dt**2 / 2],
        [dt**2 / 2, dt**3 / 3],
    ])

    R = np.array([[meas_std**2]])

    x_true = np.zeros((N, 2))
    z      = np.zeros(N)

    x_true[0] = np.array([1, 0])   # initial state: velocity=1, position=0

    for k in range(1, N):
        q = rng.multivariate_normal(mean=np.zeros(2), cov=Q)
        x_true[k] = G @ x_true[k - 1] + q

    for k in range(N):
        r    = rng.normal(0.0, meas_std)
        z[k] = (H @ x_true[k])[0] + r

    # Inject outliers
    num_outliers = int(outlier_fraction * N)
    outlier_idx  = rng.choice(N, size=num_outliers, replace=False)
    z[outlier_idx] += (
        np.abs(rng.normal(0.0, outlier_std, size=num_outliers)) + 5
    )

    return G, H, Q, R, x_true, z, outlier_idx


# ---------------------------------------------------------------------------
# General smooth damped oscillatory signal
# ---------------------------------------------------------------------------

def simulate_general_smooth_signal(
    N=150,
    t_max=8.0,
    alpha=4.0,
    beta=0.2,
    process_scale=2.0,
    meas_std=0.15,
    seed=1,
):
    """
    Generate noisy measurements of the general smooth signal

        s(t) = exp(-beta * t) * sin(alpha * t).

    State ordering:
        X_k = [s_dot(t_k), s(t_k)]

    The smoother model is linear:
        X_{k+1} = G X_k + w_k

    Measurement:
        z_k = H X_k + r_k,    H = [0, 1]

    No outliers are added here; corrupt z_general manually in the notebook.

    Parameters
    ----------
    N : int
        Number of time steps.
    t_max : float
        End time.
    alpha : float
        Angular frequency of the oscillation.
    beta : float
        Exponential decay rate.
    process_scale : float
        Scale factor for the integrated Brownian-motion process covariance.
    meas_std : float
        Standard deviation of the measurement noise.
    seed : int
        Random seed.

    Returns
    -------
    t : np.ndarray, shape (N,)
        Time grid.
    x_true : np.ndarray, shape (N, 2)
        True state [derivative, signal].
    z : np.ndarray, shape (N,)
        Noisy measurements of the signal.
    G : np.ndarray, shape (2, 2)
    H : np.ndarray, shape (1, 2)
    Q : np.ndarray, shape (2, 2)
    R : np.ndarray, shape (1, 1)
    """
    rng = np.random.default_rng(seed)

    t  = np.linspace(0.0, t_max, N)
    dt = t[1] - t[0]

    signal_true     = np.exp(-beta * t) * np.sin(alpha * t)
    derivative_true = np.exp(-beta * t) * (
        alpha * np.cos(alpha * t) - beta * np.sin(alpha * t)
    )

    # State ordering: [derivative, signal]
    x_true = np.column_stack([derivative_true, signal_true])

    G = np.array([
        [1.0, 0.0],
        [dt,  1.0],
    ])

    H = np.array([[0.0, 1.0]])      # observe the signal (column 1)

    Q = (process_scale**2) * np.array([
        [dt,         dt**2 / 2.0],
        [dt**2 / 2.0, dt**3 / 3.0],
    ])

    R = np.array([[meas_std**2]])

    z = signal_true + rng.normal(loc=0.0, scale=meas_std, size=N)

    return t, x_true, z, G, H, Q, R
