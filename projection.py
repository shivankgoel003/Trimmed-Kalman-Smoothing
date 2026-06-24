"""
projection.py
-------------
Capped-simplex projection.

The capped simplex is:
    Delta_h = {w : 0 <= w_i <= 1,  sum_i w_i = h}

Both measurement trimming and innovation trimming update their weights
via a projected-gradient step onto this set.  This module contains
only that primitive so it can be imported anywhere without pulling in
smoother or loss dependencies.

Usage
-----
    from projection import project_capped_simplex
    w = project_capped_simplex(y, h)
"""

import numpy as np


def project_capped_simplex(y, h, tol=1e-10, max_iter=200):
    """
    Project y onto the capped simplex:
        Delta_h = {w : 0 <= w_i <= 1, sum_i w_i = h}

    Solves:
        min_w  0.5 ||w - y||^2
        s.t.   0 <= w_i <= 1,  sum_i w_i = h

    Algorithm: bisection on the dual variable lambda such that
        sum_i clip(y_i - lambda, 0, 1) = h.

    Parameters
    ----------
    y : np.ndarray, shape (N,)
        Input vector.
    h : float
        Desired sum of the projected vector.  Must satisfy 0 <= h <= N.
    tol : float
        Bisection convergence tolerance.
    max_iter : int
        Maximum bisection iterations.

    Returns
    -------
    w : np.ndarray, shape (N,)
        Projection of y onto the capped simplex.
    """
    y = np.asarray(y, dtype=float)
    N = y.size

    if h < 0 or h > N:
        raise ValueError("h must satisfy 0 <= h <= len(y).")

    if h == 0:
        return np.zeros_like(y)

    if h == N:
        return np.ones_like(y)

    # Find lambda such that sum clip(y - lambda, 0, 1) = h.
    # Small lambda -> large weights; large lambda -> small weights.
    lo = np.min(y) - 1.0
    hi = np.max(y)

    for _ in range(max_iter):
        lam = 0.5 * (lo + hi)
        w = np.clip(y - lam, 0.0, 1.0)
        s = np.sum(w)

        if abs(s - h) < tol:
            break

        if s > h:   # lambda too small, weights too large
            lo = lam
        else:       # lambda too large, weights too small
            hi = lam

    return np.clip(y - lam, 0.0, 1.0)
