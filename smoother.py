"""
smoother.py
-----------
Weighted batch Kalman smoother via direct linear solve.

Solves the block-tridiagonal weighted least-squares problem:

    min  0.5 ||x_0 - x0_prior||_{P0^{-1}}^2
       + 0.5 sum_{k=0}^{N-2}  nu_k    ||x_{k+1} - G x_k||_{Q^{-1}}^2
       + 0.5 sum_{k=0}^{N-1}  omega_k ||z_k - H x_k||_{R^{-1}}^2

Setting omega = ones(N) and nu = ones(N-1) recovers the standard
Gaussian Kalman smoother.  Reducing individual omega_k or nu_k
down-weights the corresponding measurement or innovation term.

Usage
-----
    from smoother import weighted_batch_smoother
    x_hat = weighted_batch_smoother(G, H, Q, R, z)
"""

import numpy as np


def weighted_batch_smoother(
    G,
    H,
    Q,
    R,
    z,
    omega=None,     # measurement weights, shape (N,)
    nu=None,        # innovation weights, shape (N-1,)
    nu_deriv = None,  # derivative innovation weights, shape (N-1,)
    nu_signal = None, # signal innovation weights, shape (N-1,)
    x0_prior=None,
    P0=None,
):
    """
    Solve the weighted linear Gaussian smoothing problem.

    Parameters
    ----------
    G : np.ndarray, shape (n, n)
        State-transition matrix.
    H : np.ndarray, shape (m, n)
        Measurement matrix.
    Q : np.ndarray, shape (n, n)
        Process-noise covariance.
    R : np.ndarray, shape (m, m)
        Measurement-noise covariance.
    z : np.ndarray, shape (N,) or (N, m)
        Measurements.
    omega : np.ndarray, shape (N,), optional
        Per-measurement weights.  Defaults to all-ones.
    nu : np.ndarray, shape (N-1,), optional
        Per-innovation weights.  Defaults to all-ones.
    nu_deriv : np.ndarray, shape (N-1,), optional
        Per-transition weights on the derivative innovation component.
        Used together with nu_signal for component-wise trimming.
        When provided alongside nu_signal, takes priority over nu.
    nu_signal : np.ndarray, shape (N-1,), optional
        Per-transition weights on the signal innovation component.
        Set nu_signal[k] near 0 to allow a free signal jump at transition k
        while keeping the derivative constrained via nu_deriv.
    x0_prior : np.ndarray, shape (n,), optional
        Prior mean on the initial state.  Defaults to zeros.
    P0 : np.ndarray, shape (n, n), optional
        Prior covariance on the initial state.  Defaults to 10 * I.

    Returns
    -------
    x_hat : np.ndarray, shape (N, n)
        Smoothed state estimates.
    """
    z = np.asarray(z)
    N = len(z)
    n = G.shape[0]

    if omega is None:
        omega = np.ones(N)

    if nu is None:
        nu = np.ones(N - 1)

    if x0_prior is None:
        x0_prior = np.zeros(n)

    if P0 is None:
        P0 = 10.0 * np.eye(n)

    Qinv  = np.linalg.inv(Q)
    Rinv  = np.linalg.inv(R)
    P0inv = np.linalg.inv(P0)

    A = np.zeros((N * n, N * n))
    b = np.zeros(N * n)

    def block(k):
        return slice(k * n, (k + 1) * n)

    # Prior term at k = 0
    A[block(0), block(0)] += P0inv
    b[block(0)]           += P0inv @ x0_prior

    # Weighted innovation terms  (k -> k+1)
    for k in range(N - 1):
        curr = block(k)
        nxt  = block(k + 1)

        # --------------------------------------------------
        # Innovation precision matrix for transition k
        # --------------------------------------------------

        if nu_deriv is not None and nu_signal is not None:
            nd = np.clip(nu_deriv[k], 0.0, 1.0)
            ns = np.clip(nu_signal[k], 0.0, 1.0)
            W  = np.array([[nd,              np.sqrt(nd * ns)],
                   [np.sqrt(nd * ns), ns             ]])
            Pk = W * Qinv
        else:
            Pk = nu[k] * Qinv

    

        # --------------------------------------------------
        # Add innovation cost:
        # 0.5 * (X_{k+1} - G X_k)^T Pk (X_{k+1} - G X_k)
        # --------------------------------------------------

        A[curr, curr] += G.T @ Pk @ G
        A[curr, nxt]  += -G.T @ Pk
        A[nxt, curr]  += -Pk @ G
        A[nxt, nxt]   += Pk

    # Weighted measurement terms
    for k in range(N):
        idx = block(k)
        wk  = omega[k]

        A[idx, idx] += wk * (H.T @ Rinv @ H)
        b[idx]      += wk * (H.T @ Rinv * z[k]).reshape(-1)

    x_hat_flat = np.linalg.solve(A, b)
    return x_hat_flat.reshape(N, n)
