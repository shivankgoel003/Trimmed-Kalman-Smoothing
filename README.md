# Trimmed-Kalman-Smoothing

# Trimmed Kalman Smoother

Python implementation of measurement-trimmed and innovation-trimmed 
Kalman smoothers, developed as a part of research project with Professor Aleksandr Arkavin
at University of Washington.

Both methods reframe robust smoothing as projected-gradient descent on 
a capped simplex. The smoother alternates between solving a weighted 
block-tridiagonal least-squares system and updating the weights to 
down-weight outlying terms.

## Methods

**Measurement trimming** (`measurement_trim.py`): down-weights individual 
measurements robust to observation outliers.

**Innovation trimming** (`innovation_trim.py`): down-weights individual 
state transitions — detects sudden jumps in the latent state.

## File layout

| File | Contents |
|---|---|
| `projection.py` | Capped-simplex projection (bisection on dual variable) |
| `smoother.py` | Weighted batch smoother (direct linear solve) |
| `losses.py` | Measurement and innovation residual losses |
| `measurement_trim.py` | Measurement-trimmed smoother |
| `innovation_trim.py` | Innovation-trimmed smoother + warm-start helper |
| `simulation.py` | Synthetic data generators |
| `experiments.py` | Monte Carlo trial runner |
| `trimmed_kalman.ipynb` | Experiments notebook |

## Quickstart

Put all `.py` files in the same directory and open the notebook.
Or use the modules directly:

```python
from smoother import weighted_batch_smoother
from measurement_trim import trimmed_kalman_smoother

x_hat, omega, losses, history = trimmed_kalman_smoother(
    G, H, Q, R, z, h=h_keep
)
```

## Background

The core idea is that the standard Gaussian smoother solves:

$$\min \sum_k \frac{1}{2}\|z_k - H x_k\|_{R^{-1}}^2 + \frac{1}{2}\|x_{k+1} - G x_k\|_{Q^{-1}}^2$$

Trimming introduces per-term weights $\omega_k \in [0,1]$ constrained 
to $\sum \omega_k = h$, and alternates between solving the weighted 
system and projecting the weights onto the capped simplex $\Delta_h$.
