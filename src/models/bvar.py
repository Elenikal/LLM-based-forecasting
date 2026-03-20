"""
models/bvar.py
==============
Bayesian VAR with Minnesota (Litterman) prior implemented via dummy
observations (Banbura, Giannoni & Reichlin, 2010).

Key references:
- Litterman (1986): original Minnesota prior
- Banbura et al. (2010): dummy variable implementation (numerically stable)
- Carriero et al. (2019): FRED-MD BVAR forecasting benchmark

Implementation notes:
- Minnesota prior entered as dummy observations prepended to data
- Posterior is OLS on the augmented system → closed form, no MCMC needed
- Exogenous regressors (LLM PCs) handled via partitioned regression
- Rolling window: re-estimated at every forecast origin
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import BVAR_LAGS, BVAR_LAMBDA1, BVAR_LAMBDA2, BVAR_LAMBDA3


# ─────────────────────────────────────────────────────────────────────────────
# Data container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BVARResult:
    """Holds estimation output and provides forecast methods."""
    B_hat:       np.ndarray   # (K*p + 1 + n_exog, n_vars) coefficient matrix
    Sigma_hat:   np.ndarray   # (n_vars, n_vars) residual covariance
    n_vars:      int
    n_lags:      int
    n_exog:      int          # number of exogenous regressors (LLM PCs)
    var_names:   list
    last_obs:    np.ndarray   # last p observations for forecasting (n_lags, n_vars)
    last_exog:   Optional[np.ndarray] = None   # last known exog values
    diagnostics: dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: build regressor matrix (companion form)
# ─────────────────────────────────────────────────────────────────────────────

def build_X(Y: np.ndarray, p: int,
            exog: Optional[np.ndarray] = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the regressor matrix X and dependent variable matrix y_dep
    for a VAR(p) with optional exogenous variables.

    Parameters
    ----------
    Y    : (T, n) data matrix
    p    : lag order
    exog : (T, k) exogenous variables (aligned with Y rows, optional)

    Returns
    -------
    y_dep : (T-p, n)   LHS matrix
    X     : (T-p, n*p+1+k)  RHS matrix [Y_{t-1},...,Y_{t-p}, 1, exog_t]
    """
    T, n = Y.shape
    T_eff = T - p

    y_dep = Y[p:, :]           # (T-p, n)

    cols = []
    for lag in range(1, p + 1):
        cols.append(Y[p - lag: T - lag, :])   # (T-p, n)

    X_lags = np.hstack(cols)                   # (T-p, n*p)
    ones   = np.ones((T_eff, 1))
    X      = np.hstack([X_lags, ones])         # (T-p, n*p+1)

    if exog is not None:
        X = np.hstack([X, exog[p:, :]])        # (T-p, n*p+1+k)

    return y_dep, X


# ─────────────────────────────────────────────────────────────────────────────
# Minnesota prior via dummy observations
# ─────────────────────────────────────────────────────────────────────────────

def minnesota_dummies(Y: np.ndarray, p: int,
                      lambda1: float = BVAR_LAMBDA1,
                      lambda2: float = BVAR_LAMBDA2,
                      lambda3: float = BVAR_LAMBDA3) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate Minnesota prior dummy observations (Banbura et al., 2010).

    The dummy system encodes:
      - Own-lag-1 coefficient ~ N(1, lambda1^2)        [random walk prior]
      - Other-lag-l coefficients ~ N(0, (lambda1*lambda2 / s_i*l^lambda3)^2)
      - Intercept ~ N(0, (lambda1 * mean(|y|))^2) [diffuse]

    where s_i = std(y_i) is estimated from the in-sample data.

    Returns (y_d, X_d) dummy observation matrices to prepend to real data.
    """
    T, n = Y.shape
    sigma = np.std(Y[:min(T, 20), :], axis=0) + 1e-8  # robust scale estimate

    # Dummy 1: diagonal prior on lag-1 coefficients (n × n dummies per variable)
    # Shape: (n*p, n*p*n + 1 + ...) — simplified here to key blocks

    # ── Block 1: own-lag shrinkage (n × p dummies) ──
    y1_blocks = []
    X1_blocks = []
    for lag in range(1, p + 1):
        diag_y = np.diag(sigma) / (lambda1 * lag ** lambda3)  # scale / shrinkage
        y1_blocks.append(diag_y)                              # (n, n) Y dummy

        X_row = np.zeros((n, n * p + 1))
        for i in range(n):
            col_idx = (lag - 1) * n + i   # position of variable i at this lag
            X_row[i, col_idx] = sigma[i] / (lambda1 * lag ** lambda3)
        X1_blocks.append(X_row)

    y1 = np.vstack(y1_blocks)   # (n*p, n)
    X1 = np.vstack(X1_blocks)   # (n*p, n*p+1)

    # ── Block 2: co-persistence prior (1 dummy) ──
    y2 = np.mean(Y[:min(T, 4), :], axis=0, keepdims=True) / lambda1   # (1, n)
    X2 = np.zeros((1, n * p + 1))
    X2[0, :n * p] = np.tile(y2 / n, p)       # small weight on all lags
    X2[0, -1] = 0.0                           # intercept

    # ── Combine ──
    y_d = np.vstack([y1, y2])   # (n*p+1, n)
    X_d = np.vstack([X1, X2])   # (n*p+1, n*p+1)

    return y_d, X_d


# ─────────────────────────────────────────────────────────────────────────────
# Estimation
# ─────────────────────────────────────────────────────────────────────────────

def estimate_bvar(Y: np.ndarray, p: int = BVAR_LAGS,
                  lambda1: float = BVAR_LAMBDA1,
                  lambda2: float = BVAR_LAMBDA2,
                  lambda3: float = BVAR_LAMBDA3,
                  exog: Optional[np.ndarray] = None,
                  var_names: Optional[list] = None) -> BVARResult:
    """
    Estimate a BVAR(p) with Minnesota prior on Y.

    Parameters
    ----------
    Y         : (T, n) data matrix — endogenous variables
    p         : lag order
    lambda1-3 : Minnesota hyperparameters
    exog      : (T, k) exogenous regressors (e.g. LLM PCs); optional
    var_names : list of variable name strings

    Returns
    -------
    BVARResult with posterior mean coefficients B_hat and residual covariance
    """
    T, n = Y.shape
    k_exog = exog.shape[1] if exog is not None else 0

    if var_names is None:
        var_names = [f"y{i}" for i in range(n)]

    # ── Build data matrices ──
    y_dep, X = build_X(Y, p, exog)    # (T-p, n), (T-p, n*p+1+k_exog)

    # ── Minnesota dummies (exog not in prior — flat prior on Gamma) ──
    y_d, X_d = minnesota_dummies(Y, p, lambda1, lambda2, lambda3)

    # Extend dummy X to match full X width (add zeros for exog columns)
    if k_exog > 0:
        X_d = np.hstack([X_d, np.zeros((X_d.shape[0], k_exog))])

    # ── Augmented system: stack dummies on top of real data ──
    Y_aug = np.vstack([y_d, y_dep])   # (n*p+1 + T-p, n)
    X_aug = np.vstack([X_d, X])       # same rows

    # ── Posterior mean = OLS on augmented system ──
    XtX = X_aug.T @ X_aug
    XtY = X_aug.T @ Y_aug

    # Ridge-like regularisation for numerical stability (small diagonal add)
    XtX += np.eye(XtX.shape[0]) * 1e-8

    B_hat = np.linalg.solve(XtX, XtY)   # (n*p+1+k_exog, n)

    # ── Residual covariance ──
    resid    = Y_aug - X_aug @ B_hat
    T_eff    = Y_aug.shape[0]
    K        = X_aug.shape[1]
    Sigma    = (resid.T @ resid) / max(T_eff - K, 1)

    # ── Store last p obs for forecasting ──
    last_obs  = Y[-p:, :].copy()
    last_exog = exog[-1:, :].copy() if exog is not None else None

    diagnostics = {
        "T": T, "n": n, "p": p, "k_exog": k_exog,
        "T_eff_aug": T_eff, "condition_number": np.linalg.cond(XtX),
    }

    return BVARResult(
        B_hat=B_hat, Sigma_hat=Sigma,
        n_vars=n, n_lags=p, n_exog=k_exog,
        var_names=var_names,
        last_obs=last_obs, last_exog=last_exog,
        diagnostics=diagnostics,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Forecasting
# ─────────────────────────────────────────────────────────────────────────────

def forecast_bvar(result: BVARResult, h: int,
                  future_exog: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Produce h-step-ahead point forecast from a fitted BVAR.

    Uses recursive substitution (no simulation/draws).
    Exogenous regressors assumed to equal their last known value if not provided.

    Parameters
    ----------
    result     : fitted BVARResult
    h          : forecast horizon
    future_exog: (h, k) future exogenous values; if None, uses last observed

    Returns
    -------
    forecasts : (h, n_vars) array of point forecasts
    """
    n = result.n_vars
    p = result.n_lags
    B = result.B_hat          # (n*p+1+k_exog, n)

    # Initialise with last p observed values
    history = result.last_obs.copy()   # (p, n) newest-last

    # If future_exog not provided, repeat last known
    if result.n_exog > 0:
        if future_exog is None:
            future_exog = np.tile(result.last_exog, (h, 1))  # (h, k_exog)
        assert future_exog.shape == (h, result.n_exog), \
            f"future_exog shape mismatch: {future_exog.shape}"

    forecasts = np.zeros((h, n))
    for step in range(h):
        # Build regressor vector for this step
        # [y_{t-1}, y_{t-2}, ..., y_{t-p}, 1, exog_t]
        lags_vec = history[::-1, :].flatten()   # stack lags, most recent first
        x_t      = np.concatenate([lags_vec, [1.0]])

        if result.n_exog > 0:
            x_t = np.concatenate([x_t, future_exog[step, :]])

        y_hat = x_t @ B   # (n,)
        forecasts[step, :] = y_hat

        # Append forecast to history for next step
        history = np.vstack([history[1:, :], y_hat.reshape(1, -1)])

    return forecasts


# ─────────────────────────────────────────────────────────────────────────────
# AR(p) baseline
# ─────────────────────────────────────────────────────────────────────────────

def fit_ar(y: np.ndarray, p: int = 4) -> dict:
    """
    Fit an OLS AR(p) model for a single series.
    Returns dict with coefficients, for h-step ahead forecast.
    """
    T = len(y)
    Y = np.array([y[i:T - p + i] for i in range(p)]).T   # (T-p, p)
    y_dep = y[p:]
    X = np.hstack([Y, np.ones((T - p, 1))])
    b = np.linalg.lstsq(X, y_dep, rcond=None)[0]
    resid = y_dep - X @ b
    return {"b": b, "p": p, "last_obs": y[-p:].copy(), "sigma2": np.var(resid)}


def forecast_ar(fit: dict, h: int) -> np.ndarray:
    """h-step recursive forecast from AR fit."""
    p     = fit["p"]
    b     = fit["b"]
    hist  = list(fit["last_obs"])
    preds = []
    for _ in range(h):
        x    = hist[-p:][::-1] + [1.0]   # lags, most recent first, + intercept
        yhat = np.dot(b, x)
        preds.append(yhat)
        hist.append(yhat)
    return np.array(preds)


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(42)
    T, n = 60, 3
    Y = np.cumsum(np.random.randn(T, n) * 0.5, axis=0)   # random walk data
    # Synthetic exogenous (LLM PCs)
    exog = np.random.randn(T, 2)

    # Estimate without exog
    res0 = estimate_bvar(Y, p=4)
    fc0  = forecast_bvar(res0, h=2)
    print("BVAR (no exog):")
    print(f"  B_hat shape : {res0.B_hat.shape}")
    print(f"  Sigma shape : {res0.Sigma_hat.shape}")
    print(f"  Forecast h=2:\n{fc0}")

    # Estimate with LLM PCs
    res1 = estimate_bvar(Y, p=4, exog=exog)
    fc1  = forecast_bvar(res1, h=2, future_exog=np.zeros((2, 2)))
    print("\nBVAR (with 2 exog LLM PCs):")
    print(f"  B_hat shape : {res1.B_hat.shape}")
    print(f"  Forecast h=2:\n{fc1}")

    # AR baseline
    ar_fit = fit_ar(Y[:, 0], p=4)
    ar_fc  = forecast_ar(ar_fit, h=2)
    print(f"\nAR(4) forecast for y0: {ar_fc}")
    print(f"\nCondition number (BVAR): {res1.diagnostics['condition_number']:.2f}")
