import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy import stats
from typing import Optional
from ..models import MethodResult, DiagnosticResult, CausalMethod, DataProfile


def _fit_synthetic_control(
    treated_pre: np.ndarray,
    donor_pre: np.ndarray,
    n_restarts: int = 5,
) -> np.ndarray:
    """
    Find optimal donor weights via constrained optimization with multiple
    random restarts to avoid local minima:

        minimize  ||treated_pre − donor_pre @ w||²
        s.t.      sum(w) = 1,  w_i ≥ 0
    """
    n_donors = donor_pre.shape[1]

    def objective(w):
        return np.sum((treated_pre - donor_pre @ w) ** 2)

    def gradient(w):
        residuals = treated_pre - donor_pre @ w
        return -2 * donor_pre.T @ residuals

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0, 1)] * n_donors

    best_weights = None
    best_obj = np.inf

    for i in range(n_restarts):
        if i == 0:
            w0 = np.ones(n_donors) / n_donors  # uniform start
        else:
            # Dirichlet random start (lives on the simplex)
            w0 = np.random.dirichlet(np.ones(n_donors))

        result = minimize(
            objective,
            w0,
            jac=gradient,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-12, "maxiter": 2000},
        )

        if result.fun < best_obj:
            best_obj = result.fun
            best_weights = result.x

    return best_weights


def _run_sc_for_unit(
    treated_pre: np.ndarray,
    donor_pre: np.ndarray,
    donor_post: np.ndarray,
    treated_post: np.ndarray,
) -> dict:
    """
    Run SC for a single treated unit and return gap statistics.
    Used by both the main analysis and in-space placebos.
    """
    weights = _fit_synthetic_control(treated_pre, donor_pre, n_restarts=3)
    synthetic_pre = donor_pre @ weights
    synthetic_post = donor_post @ weights

    pre_gaps = treated_pre - synthetic_pre
    post_gaps = treated_post - synthetic_post

    rmspe_pre = float(np.sqrt(np.mean(pre_gaps ** 2)))
    rmspe_post = float(np.sqrt(np.mean(post_gaps ** 2)))
    rmspe_ratio = rmspe_post / rmspe_pre if rmspe_pre > 0 else np.inf
    att = float(np.mean(post_gaps))

    return {
        "weights": weights,
        "att": att,
        "rmspe_pre": rmspe_pre,
        "rmspe_post": rmspe_post,
        "rmspe_ratio": rmspe_ratio,
        "pre_gaps": pre_gaps,
        "post_gaps": post_gaps,
        "synthetic_pre": synthetic_pre,
        "synthetic_post": synthetic_post,
    }


def run_synthetic_control(data_path: str, profile: DataProfile) -> MethodResult:
    """
    Implements the Abadie-Diamond-Hainmueller synthetic control method.

    Key improvements over a basic SC implementation:
    - Multiple random restarts for weight optimization
    - In-space placebo tests (Fisher permutation p-value)
    - Time placebo test (backdate treatment)
    - Leave-one-out donor sensitivity
    - Pre-period fit quality diagnostics

    Diagnostics:
    1. Pre-period fit quality (RMSPE as % of treated mean)
    2. In-space placebo tests → Fisher permutation p-value
    3. Post/pre RMSPE ratio
    4. Time placebo (backdated treatment)
    5. Leave-one-out donor sensitivity
    """
    df = pd.read_csv(data_path)
    diagnostics = []

    outcome_col = profile.outcome_col
    time_col = profile.time_col

    # Try to find geo/unit column
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

    # ── Identify treated unit ────────────────────────────────────────────────
    treatment_col = profile.treatment_col
    if treatment_col and treatment_col in df.columns and treatment_col != unit_col:
        treated_units = df[
            df[treatment_col].astype(str).str.lower().isin(
                ["1", "true", "treatment", "treated", "test"]
            )
        ][unit_col].unique()
        if len(treated_units) == 0:
            treated_units = df[unit_col].unique()[:1]
    else:
        treated_units = sorted(df[unit_col].unique())[:1]

    treated_unit = treated_units[0]
    donor_units = [u for u in df[unit_col].unique() if u != treated_unit]

    if len(donor_units) < 2:
        raise ValueError(f"Synthetic control needs ≥2 donor units. Found: {len(donor_units)}")

    # ── Pivot to wide format ─────────────────────────────────────────────────
    pivot = df.pivot_table(
        index=time_col, columns=unit_col, values=outcome_col, aggfunc="mean"
    )
    pivot = pivot.sort_index()

    time_vals = list(pivot.index)
    n_periods = len(time_vals)

    # Determine pre/post split
    post_col_name = next(
        (c for c in df.columns if c.lower() in ("post", "is_post", "after", "post_period")),
        None,
    )
    if post_col_name and post_col_name in df.columns:
        post_times = df[df[post_col_name].astype(str).isin(["1", "true", "True"])][
            time_col
        ].unique()
        pre_times = [t for t in time_vals if t not in post_times]
        post_times = [t for t in time_vals if t in post_times]
    else:
        split_idx = n_periods // 2
        pre_times = time_vals[:split_idx]
        post_times = time_vals[split_idx:]

    if len(pre_times) < 2:
        raise ValueError("Need ≥2 pre-treatment periods for Synthetic Control")

    # ── Build index arrays ───────────────────────────────────────────────────
    pre_idx = [i for i, t in enumerate(time_vals) if t in pre_times]
    post_idx = [i for i, t in enumerate(time_vals) if t in post_times]

    treated_series = pivot[treated_unit].values
    donor_matrix = pivot[donor_units].values

    treated_pre = treated_series[pre_idx]
    treated_post = treated_series[post_idx]
    donor_pre = donor_matrix[pre_idx, :]
    donor_post = donor_matrix[post_idx, :]

    # ── Fit primary SC ───────────────────────────────────────────────────────
    primary = _run_sc_for_unit(treated_pre, donor_pre, donor_post, treated_post)
    weights = primary["weights"]
    estimate = primary["att"]
    rmspe_pre = primary["rmspe_pre"]
    rmspe_post = primary["rmspe_post"]
    rmspe_ratio = primary["rmspe_ratio"]
    gaps = primary["post_gaps"]
    synthetic_post = primary["synthetic_post"]

    # Relative effect
    synthetic_post_mean = float(np.mean(synthetic_post))
    relative_effect = estimate / synthetic_post_mean if synthetic_post_mean != 0 else None

    # ── Diagnostic 1: Pre-period fit ─────────────────────────────────────────
    treated_pre_mean = np.mean(np.abs(treated_pre))
    rmspe_pct = (rmspe_pre / treated_pre_mean * 100) if treated_pre_mean > 0 else np.inf

    diagnostics.append(DiagnosticResult(
        name="Pre-Period Fit (RMSPE)",
        passed=rmspe_pct < 10,
        statistic=round(rmspe_pre, 6),
        interpretation=(
            f"Pre-period RMSPE = {rmspe_pre:.4f} ({rmspe_pct:.1f}% of treated mean). "
            + (
                "Good synthetic control fit."
                if rmspe_pct < 10
                else
                "⚠️ Poor pre-period fit — synthetic control may not be reliable. "
                "Consider different donor pool or DiD instead."
            )
        ),
        details={
            "rmspe_pre": round(rmspe_pre, 6),
            "rmspe_post": round(rmspe_post, 6),
            "rmspe_ratio": round(rmspe_ratio, 3),
            "rmspe_pct_of_mean": round(rmspe_pct, 2),
            "top_donors": {
                str(donor_units[i]): round(float(weights[i]), 4)
                for i in np.argsort(weights)[::-1][:5]
                if weights[i] > 0.01
            },
        },
    ))

    # ── Diagnostic 2: In-space placebo (Fisher permutation p-value) ──────────
    # Run SC for each donor unit as if IT were the treated unit
    placebo_rmspe_ratios = []
    placebo_atts = []
    n_better = 0  # donors with ratio ≥ treated ratio

    if len(donor_units) >= 3:
        for donor in donor_units:
            # This donor is "pseudo-treated"; remaining donors + real treated are the pool
            pseudo_treated_series = pivot[donor].values
            pool_units = [u for u in df[unit_col].unique() if u != donor]
            pool_matrix = pivot[pool_units].values

            pseudo_pre = pseudo_treated_series[pre_idx]
            pseudo_post = pseudo_treated_series[post_idx]
            pool_pre = pool_matrix[pre_idx, :]
            pool_post = pool_matrix[post_idx, :]

            try:
                placebo = _run_sc_for_unit(pseudo_pre, pool_pre, pool_post, pseudo_post)

                # Only include placebos with reasonable pre-period fit
                # (standard practice: exclude if pre-RMSPE > X× treated pre-RMSPE)
                if rmspe_pre > 0 and placebo["rmspe_pre"] / rmspe_pre <= 5:
                    placebo_rmspe_ratios.append(placebo["rmspe_ratio"])
                    placebo_atts.append(placebo["att"])
                    if placebo["rmspe_ratio"] >= rmspe_ratio:
                        n_better += 1
            except Exception:
                continue

        n_placebos = len(placebo_rmspe_ratios)
        if n_placebos > 0:
            # Fisher exact p-value: fraction of placebos with RMSPE ratio ≥ treated
            fisher_p = (n_better + 1) / (n_placebos + 1)  # +1 for treated unit itself
            fisher_passed = fisher_p < 0.1

            # Also compute a two-sided permutation p-value on the ATT
            n_extreme = sum(1 for a in placebo_atts if abs(a) >= abs(estimate))
            perm_p_att = (n_extreme + 1) / (n_placebos + 1)

            diagnostics.append(DiagnosticResult(
                name="In-Space Placebo Test (Fisher Permutation)",
                passed=fisher_passed,
                statistic=round(rmspe_ratio, 3),
                p_value=round(fisher_p, 4),
                interpretation=(
                    f"Treated unit RMSPE ratio ({rmspe_ratio:.2f}) ranks in top "
                    f"{fisher_p*100:.0f}% of {n_placebos} placebo permutations. "
                    + (
                        f"Permutation p-value = {fisher_p:.3f} — the effect is unlikely "
                        "to be due to chance."
                        if fisher_passed
                        else
                        f"⚠️ Permutation p-value = {fisher_p:.3f} — the treated unit's "
                        "post-period divergence is not extreme relative to donor placebos. "
                        "Effect may not be distinguishable from noise."
                    )
                ),
                details={
                    "n_placebos_used": n_placebos,
                    "n_donors_excluded_poor_fit": len(donor_units) - n_placebos,
                    "fisher_p_rmspe_ratio": round(fisher_p, 4),
                    "permutation_p_att": round(perm_p_att, 4),
                    "placebo_rmspe_ratios_sorted": sorted(
                        [round(r, 3) for r in placebo_rmspe_ratios], reverse=True
                    )[:10],
                },
            ))

            # Use Fisher p-value as the primary p-value (replaces the naive t-test)
            p_value = fisher_p
        else:
            p_value = 1.0
            diagnostics.append(DiagnosticResult(
                name="In-Space Placebo Test",
                passed=False,
                interpretation=(
                    "⚠️ No valid placebo permutations produced. "
                    "Cannot assess statistical significance via permutation."
                ),
            ))
    else:
        # Not enough donors for placebos — fall back to naive inference
        se = float(np.std(gaps, ddof=1) / np.sqrt(len(gaps))) if len(gaps) > 1 else 0.0
        t_stat = estimate / se if se > 0 else 0.0
        df_t = len(gaps) - 1
        p_value = float(2 * (1 - stats.t.cdf(abs(t_stat), df=df_t))) if df_t > 0 else 1.0
        diagnostics.append(DiagnosticResult(
            name="In-Space Placebo Test",
            passed=False,
            interpretation=(
                "Fewer than 3 donor units — cannot run in-space placebos. "
                "Falling back to naive t-test on post-period gaps (less rigorous)."
            ),
        ))

    # ── Diagnostic 3: Post/Pre RMSPE Ratio ───────────────────────────────────
    diagnostics.append(DiagnosticResult(
        name="Post/Pre RMSPE Ratio",
        passed=rmspe_ratio > 2,
        statistic=round(rmspe_ratio, 3),
        interpretation=(
            f"Post/Pre RMSPE ratio = {rmspe_ratio:.2f}. "
            + (
                "Ratio > 2 suggests the post-period divergence is real, not noise."
                if rmspe_ratio > 2
                else
                f"⚠️ Low ratio ({rmspe_ratio:.2f}) — post-period divergence is similar "
                "to pre-period noise. Effect may not be distinguishable from random variation."
            )
        ),
    ))

    # ── Diagnostic 4: Time Placebo (backdate treatment) ──────────────────────
    if len(pre_times) >= 6:
        # Split pre-period in half: use first half as "pre", second half as fake "post"
        placebo_split = len(pre_times) // 2
        time_placebo_pre_idx = pre_idx[:placebo_split]
        time_placebo_post_idx = pre_idx[placebo_split:]

        tp_treated_pre = treated_series[time_placebo_pre_idx]
        tp_treated_post = treated_series[time_placebo_post_idx]
        tp_donor_pre = donor_matrix[time_placebo_pre_idx, :]
        tp_donor_post = donor_matrix[time_placebo_post_idx, :]

        try:
            tp_result = _run_sc_for_unit(
                tp_treated_pre, tp_donor_pre, tp_donor_post, tp_treated_post
            )
            tp_att = tp_result["att"]
            # Is the time-placebo effect small relative to the real effect?
            tp_ratio = abs(tp_att / estimate) if estimate != 0 else 0
            tp_passed = tp_ratio < 0.5

            diagnostics.append(DiagnosticResult(
                name="Time Placebo (Backdated Treatment)",
                passed=tp_passed,
                statistic=round(tp_att, 6),
                interpretation=(
                    f"Backdated SC estimate = {tp_att:+.4f} "
                    f"({tp_ratio * 100:.1f}% of real estimate). "
                    + (
                        "Minimal effect under fake treatment timing — supports "
                        "causal interpretation."
                        if tp_passed
                        else
                        "⚠️ Substantial 'effect' detected with fake treatment timing. "
                        "The real effect may partly reflect pre-existing dynamics."
                    )
                ),
                details={
                    "time_placebo_att": round(tp_att, 6),
                    "time_placebo_rmspe_pre": round(tp_result["rmspe_pre"], 6),
                    "ratio_to_real_effect": round(tp_ratio, 4),
                },
            ))
        except Exception:
            diagnostics.append(DiagnosticResult(
                name="Time Placebo",
                passed=True,
                interpretation="Could not run time placebo (optimization failed). Skipped.",
            ))
    else:
        diagnostics.append(DiagnosticResult(
            name="Time Placebo",
            passed=True,
            interpretation=(
                "Insufficient pre-periods for time placebo (need ≥6). Skipped."
            ),
        ))

    # ── Diagnostic 5: Leave-one-out donor sensitivity ────────────────────────
    if len(donor_units) >= 3:
        loo_estimates = []
        for i in range(len(donor_units)):
            loo_donors = [d for j, d in enumerate(donor_units) if j != i]
            loo_donor_matrix = pivot[loo_donors].values
            loo_donor_pre = loo_donor_matrix[pre_idx, :]
            loo_donor_post = loo_donor_matrix[post_idx, :]
            try:
                loo_result = _run_sc_for_unit(
                    treated_pre, loo_donor_pre, loo_donor_post, treated_post
                )
                loo_estimates.append(loo_result["att"])
            except Exception:
                pass

        if loo_estimates:
            loo_std = np.std(loo_estimates)
            sensitivity_ok = loo_std / abs(estimate) < 0.3 if estimate != 0 else True

            diagnostics.append(DiagnosticResult(
                name="Leave-One-Out Donor Sensitivity",
                passed=sensitivity_ok,
                statistic=round(float(loo_std), 6),
                interpretation=(
                    f"LOO estimates range: [{min(loo_estimates):.4f}, "
                    f"{max(loo_estimates):.4f}]. "
                    f"Std dev = {loo_std:.4f} "
                    f"({loo_std / abs(estimate) * 100:.1f}% of estimate). "
                    + (
                        "Results are stable across donor pool variations."
                        if sensitivity_ok
                        else
                        "⚠️ Results are sensitive to donor pool composition. "
                        "Interpret with caution."
                    )
                ),
                details={"loo_estimates": [round(e, 4) for e in loo_estimates]},
            ))

    all_passed = all(d.passed for d in diagnostics)

    # Build CI from permutation distribution if available, else from gaps
    if placebo_atts and len(placebo_atts) >= 5:
        ci_lower = float(np.percentile(placebo_atts, 2.5))
        ci_upper = float(np.percentile(placebo_atts, 97.5))
        # Shift so the CI is centered on the estimate (placebo-based CI)
        # Actually: a more interpretable approach — use the gap SE
        se = float(np.std(gaps, ddof=1) / np.sqrt(len(gaps))) if len(gaps) > 1 else 0.0
        ci_lower = estimate - 1.96 * se
        ci_upper = estimate + 1.96 * se
    else:
        se = float(np.std(gaps, ddof=1) / np.sqrt(len(gaps))) if len(gaps) > 1 else 0.0
        ci_lower = estimate - 1.96 * se
        ci_upper = estimate + 1.96 * se

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
        p_value=round(float(p_value), 6),
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
            "inference_method": (
                "Fisher permutation (in-space placebos)"
                if len(donor_units) >= 3
                else "Naive t-test on post-period gaps"
            ),
            "outcome_col": outcome_col,
            "unit_col": unit_col,
            "time_col": time_col,
        },
    )
