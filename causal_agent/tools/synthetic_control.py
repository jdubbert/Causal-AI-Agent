import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy import stats
from typing import Optional
from ..models import MethodResult, DiagnosticResult, CausalMethod, DataProfile


def _fit_synthetic_control(
    treated_pre: np.ndarray,
    donor_pre: np.ndarray,
) -> np.ndarray:
    """
    Find optimal donor weights via constrained optimization:
    minimize ||treated_pre - donor_pre @ w||^2
    subject to: sum(w) = 1, w_i >= 0
    """
    n_donors = donor_pre.shape[1]

    def objective(w):
        return np.sum((treated_pre - donor_pre @ w) ** 2)

    def gradient(w):
        residuals = treated_pre - donor_pre @ w
        return -2 * donor_pre.T @ residuals

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0, 1)] * n_donors
    w0 = np.ones(n_donors) / n_donors

    result = minimize(
        objective,
        w0,
        jac=gradient,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-10, "maxiter": 1000}
    )

    return result.x


def run_synthetic_control(data_path: str, profile: DataProfile) -> MethodResult:
    """
    Implements the Abadie-Diamond-Hainmueller synthetic control method.
    
    Expects wide-format OR long-format data. For long format, needs:
    - time column
    - unit/geo column
    - outcome column
    - treatment indicator (marks which unit is treated)
    
    Diagnostics:
    1. Pre-period fit quality (RMSPE)
    2. In-space placebo tests (leave-one-out donor placebo)
    3. Post/pre RMSPE ratio
    """
    df = pd.read_csv(data_path)
    diagnostics = []

    outcome_col = profile.outcome_col
    time_col = profile.time_col

    # Try to find geo/unit column — prefer explicit geo over generic treatment/group
    geo_candidates = ["geo", "dma", "state", "region", "market", "unit", "zip", "city"]
    unit_col = None
    for cand in geo_candidates:
        matches = [c for c in df.columns if cand.lower() in c.lower()]
        if matches:
            unit_col = matches[0]
            break
    if unit_col is None:
        unit_col = profile.group_col or profile.treatment_col

    if not all([outcome_col, time_col, unit_col]):
        raise ValueError(
            f"Synthetic control requires outcome, time, and unit columns. "
            f"Got: outcome={outcome_col}, time={time_col}, unit={unit_col}"
        )

    df = df.dropna(subset=[outcome_col, time_col, unit_col])

    # Identify treated unit
    treatment_col = profile.treatment_col
    if treatment_col and treatment_col in df.columns and treatment_col != unit_col:
        # Find the unit(s) that are treated
        treated_units = df[df[treatment_col].astype(str).str.lower().isin(
            ["1", "true", "treatment", "treated", "test"]
        )][unit_col].unique()
        if len(treated_units) == 0:
            treated_units = df[unit_col].unique()[:1]
    else:
        # Assume first unit alphabetically is treated (user should specify)
        treated_units = sorted(df[unit_col].unique())[:1]

    treated_unit = treated_units[0]
    donor_units = [u for u in df[unit_col].unique() if u != treated_unit]

    if len(donor_units) < 2:
        raise ValueError(f"Synthetic control needs ≥2 donor units. Found: {len(donor_units)}")

    # Pivot to wide format: rows = time, cols = units
    pivot = df.pivot_table(index=time_col, columns=unit_col, values=outcome_col, aggfunc="mean")
    pivot = pivot.sort_index()

    time_vals = list(pivot.index)
    n_periods = len(time_vals)

    # Determine pre/post split
    post_col = next((c for c in df.columns if c.lower() in ("post", "is_post", "after", "post_period")), None)
    if post_col and post_col in df.columns:
        post_times = df[df[post_col].astype(str).isin(["1", "true", "True"])][time_col].unique()
        pre_times = [t for t in time_vals if t not in post_times]
    else:
        split_idx = n_periods // 2
        pre_times = time_vals[:split_idx]
        post_times = time_vals[split_idx:]

    if len(pre_times) < 2:
        raise ValueError("Need ≥2 pre-treatment periods for Synthetic Control")

    # Extract matrices
    treated_series = pivot[treated_unit].values
    donor_matrix = pivot[donor_units].values

    pre_idx = [i for i, t in enumerate(time_vals) if t in pre_times]
    post_idx = [i for i, t in enumerate(time_vals) if t in post_times]

    treated_pre = treated_series[pre_idx]
    donor_pre = donor_matrix[pre_idx, :]

    # Fit weights
    weights = _fit_synthetic_control(treated_pre, donor_pre)

    # Synthetic counterfactual
    synthetic = donor_matrix @ weights

    # ATT = avg treatment effect on treated (post-period)
    treated_post = treated_series[post_idx]
    synthetic_post = synthetic[post_idx]
    gaps = treated_post - synthetic_post
    estimate = float(np.mean(gaps))

    # Pre-period fit
    pre_gaps = treated_pre - (donor_pre @ weights)
    rmspe_pre = float(np.sqrt(np.mean(pre_gaps ** 2)))
    rmspe_post = float(np.sqrt(np.mean(gaps ** 2)))
    rmspe_ratio = rmspe_post / rmspe_pre if rmspe_pre > 0 else np.inf

    # Relative effect
    synthetic_post_mean = float(np.mean(synthetic_post))
    relative_effect = estimate / synthetic_post_mean if synthetic_post_mean != 0 else None

    # Naive SE via post-period variation
    se = float(np.std(gaps, ddof=1) / np.sqrt(len(gaps))) if len(gaps) > 1 else 0.0
    t_stat = estimate / se if se > 0 else 0.0
    df_t = len(gaps) - 1
    p_value = float(2 * (1 - stats.t.cdf(abs(t_stat), df=df_t))) if df_t > 0 else 1.0
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se

    # ── Diagnostic 1: Pre-period fit ──────────────────────────────────────────
    pre_fit_good = rmspe_ratio > 2  # post RMSPE >> pre RMSPE suggests real effect
    treated_pre_mean = np.mean(np.abs(treated_pre))
    rmspe_pct = (rmspe_pre / treated_pre_mean * 100) if treated_pre_mean > 0 else np.inf

    diagnostics.append(DiagnosticResult(
        name="Pre-Period Fit (RMSPE)",
        passed=rmspe_pct < 10,  # within 10% of treated mean
        statistic=round(rmspe_pre, 6),
        interpretation=(
            f"Pre-period RMSPE = {rmspe_pre:.4f} ({rmspe_pct:.1f}% of treated mean). "
            + ("Good synthetic control fit." if rmspe_pct < 10 else
               "⚠️ Poor pre-period fit — synthetic control may not be reliable. "
               "Consider different donor pool or DiD instead.")
        ),
        details={
            "rmspe_pre": round(rmspe_pre, 6),
            "rmspe_post": round(rmspe_post, 6),
            "rmspe_ratio": round(rmspe_ratio, 3),
            "top_donors": {
                str(donor_units[i]): round(float(weights[i]), 4)
                for i in np.argsort(weights)[::-1][:5]
                if weights[i] > 0.01
            }
        }
    ))

    # ── Diagnostic 2: Post/Pre RMSPE Ratio ───────────────────────────────────
    diagnostics.append(DiagnosticResult(
        name="Post/Pre RMSPE Ratio",
        passed=rmspe_ratio > 2,
        statistic=round(rmspe_ratio, 3),
        interpretation=(
            f"Post/Pre RMSPE ratio = {rmspe_ratio:.2f}. "
            + (f"Ratio > 2 suggests the post-period divergence is real, not noise."
               if rmspe_ratio > 2 else
               f"⚠️ Low ratio ({rmspe_ratio:.2f}) — post-period divergence is similar to pre-period noise. "
               "Effect may not be distinguishable from random variation.")
        ),
    ))

    # ── Diagnostic 3: Leave-one-out donor sensitivity ────────────────────────
    if len(donor_units) >= 3:
        loo_estimates = []
        for i in range(len(donor_units)):
            loo_donors = [d for j, d in enumerate(donor_units) if j != i]
            loo_donor_matrix = pivot[loo_donors].values
            loo_donor_pre = loo_donor_matrix[pre_idx, :]
            try:
                loo_weights = _fit_synthetic_control(treated_pre, loo_donor_pre)
                loo_synthetic_post = loo_donor_matrix[post_idx, :] @ loo_weights
                loo_estimate = float(np.mean(treated_post - loo_synthetic_post))
                loo_estimates.append(loo_estimate)
            except Exception:
                pass

        if loo_estimates:
            loo_std = np.std(loo_estimates)
            loo_range = max(loo_estimates) - min(loo_estimates)
            sensitivity_ok = loo_std / abs(estimate) < 0.3 if estimate != 0 else True

            diagnostics.append(DiagnosticResult(
                name="Leave-One-Out Donor Sensitivity",
                passed=sensitivity_ok,
                statistic=round(loo_std, 6),
                interpretation=(
                    f"LOO estimates range: [{min(loo_estimates):.4f}, {max(loo_estimates):.4f}]. "
                    f"Std dev = {loo_std:.4f} ({loo_std/abs(estimate)*100:.1f}% of estimate). "
                    + ("Results are stable across donor pool variations."
                       if sensitivity_ok else
                       "⚠️ Results are sensitive to donor pool composition. Interpret with caution.")
                ),
                details={"loo_estimates": [round(e, 4) for e in loo_estimates]}
            ))

    all_passed = all(d.passed for d in diagnostics)
    top_weights = {
        str(donor_units[i]): round(float(weights[i]), 4)
        for i in np.argsort(weights)[::-1][:10]
        if weights[i] > 0.01
    }

    return MethodResult(
        method=CausalMethod.SYNTHETIC_CONTROL,
        estimate=round(estimate, 6),
        std_error=round(se, 6) if se else None,
        ci_lower=round(ci_lower, 6),
        ci_upper=round(ci_upper, 6),
        p_value=round(p_value, 6),
        relative_effect=round(relative_effect, 4) if relative_effect is not None else None,
        diagnostics=diagnostics,
        diagnostics_passed=all_passed,
        raw_output={
            "treated_unit": str(treated_unit),
            "n_donor_units": len(donor_units),
            "n_pre_periods": len(pre_times),
            "n_post_periods": len(post_times),
            "donor_weights": top_weights,
            "rmspe_pre": round(rmspe_pre, 6),
            "rmspe_post": round(rmspe_post, 6),
            "rmspe_ratio": round(rmspe_ratio, 3),
            "outcome_col": outcome_col,
            "unit_col": unit_col,
            "time_col": time_col,
        }
    )
