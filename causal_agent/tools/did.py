import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf
from typing import Optional
from ..models import MethodResult, DiagnosticResult, CausalMethod, DataProfile


def run_did(data_path: str, profile: DataProfile) -> MethodResult:
    """
    Runs a two-way fixed effects DiD:
        Y = α + β·treated + γ·post + δ·(treated × post) + ε
    
    Diagnostics:
    1. Parallel trends test (pre-period trend comparison via regression)
    2. Placebo test (fake treatment in pre-period)
    3. Balance check on pre-period means
    """
    df = pd.read_csv(data_path)
    diagnostics = []

    treatment_col = profile.treatment_col
    outcome_col = profile.outcome_col
    time_col = profile.time_col

    if not all([treatment_col, outcome_col, time_col]):
        raise ValueError(
            f"DiD requires treatment, outcome, and time columns. "
            f"Got: treatment={treatment_col}, outcome={outcome_col}, time={time_col}"
        )

    df = df.dropna(subset=[treatment_col, outcome_col, time_col])

    # Normalize treatment to 0/1
    unique_vals = sorted(df[treatment_col].unique(), key=str)
    if len(unique_vals) != 2:
        raise ValueError(f"Expected 2 treatment groups, found {len(unique_vals)}")

    control_label, treatment_label = unique_vals[0], unique_vals[1]
    if str(control_label).lower() in ("treatment", "test", "1", "true", "exposed"):
        control_label, treatment_label = treatment_label, control_label

    df["treated"] = (df[treatment_col] == treatment_label).astype(int)

    # Determine pre/post periods
    time_vals = sorted(df[time_col].unique())
    n_periods = len(time_vals)

    if n_periods < 2:
        raise ValueError("DiD requires at least 2 time periods")

    # Try to find a treatment period marker or split at midpoint
    # Check if there's an 'is_post' or 'post' column
    post_col = next((c for c in df.columns if c.lower() in ("post", "is_post", "after", "post_period")), None)

    if post_col is not None:
        df["post"] = df[post_col].astype(int)
    else:
        # Split at midpoint
        midpoint_idx = n_periods // 2
        treatment_start = time_vals[midpoint_idx]
        df["post"] = (df[time_col] >= treatment_start).astype(int)

    df["did_interaction"] = df["treated"] * df["post"]

    # ── Primary DiD estimate ─────────────────────────────────────────────────
    try:
        model = smf.ols(
            f"{outcome_col} ~ treated + post + did_interaction",
            data=df
        ).fit(cov_type="HC3")

        estimate = model.params["did_interaction"]
        se = model.bse["did_interaction"]
        p_value = model.pvalues["did_interaction"]
        ci = model.conf_int().loc["did_interaction"]
        ci_lower, ci_upper = ci[0], ci[1]

        control_mean_pre = df[(df["treated"] == 0) & (df["post"] == 0)][outcome_col].mean()
        relative_effect = estimate / control_mean_pre if control_mean_pre != 0 else None

    except Exception as e:
        raise ValueError(f"DiD regression failed: {e}")

    # ── Diagnostic 1: Parallel Trends ────────────────────────────────────────
    pre_df = df[df["post"] == 0].copy()
    pre_time_vals = sorted(pre_df[time_col].unique())

    if len(pre_time_vals) >= 2:
        # Encode time numerically for trend regression
        time_map = {t: i for i, t in enumerate(pre_time_vals)}
        pre_df["time_numeric"] = pre_df[time_col].map(time_map)
        pre_df["trend_interaction"] = pre_df["treated"] * pre_df["time_numeric"]

        try:
            trend_model = smf.ols(
                f"{outcome_col} ~ treated + time_numeric + trend_interaction",
                data=pre_df
            ).fit(cov_type="HC3")

            trend_coef = trend_model.params["trend_interaction"]
            trend_p = trend_model.pvalues["trend_interaction"]
            pt_passed = trend_p > 0.1  # non-significant interaction = parallel trends holds

            diagnostics.append(DiagnosticResult(
                name="Parallel Trends Test",
                passed=pt_passed,
                statistic=round(trend_coef, 6),
                p_value=round(trend_p, 4),
                interpretation=(
                    f"Parallel trends assumption holds (p={trend_p:.3f}). "
                    "Pre-period trends not significantly different between groups."
                    if pt_passed else
                    f"⚠️ Parallel trends may be violated (p={trend_p:.3f}). "
                    "Pre-period trends differ between treatment and control. "
                    "Consider Synthetic Control or adding covariates."
                ),
                details={"n_pre_periods": len(pre_time_vals), "trend_coef": round(trend_coef, 6)}
            ))
        except Exception as e:
            diagnostics.append(DiagnosticResult(
                name="Parallel Trends Test",
                passed=False,
                interpretation=f"Could not run parallel trends test: {e}"
            ))
    else:
        diagnostics.append(DiagnosticResult(
            name="Parallel Trends Test",
            passed=False,
            interpretation="Only 1 pre-period available — cannot test parallel trends. Use with caution."
        ))

    # ── Diagnostic 2: Placebo Test ────────────────────────────────────────────
    if len(pre_time_vals) >= 3:
        # Use first half of pre-period, treat second half of pre-period as fake post
        placebo_midpoint = pre_time_vals[len(pre_time_vals) // 2]
        pre_df["placebo_post"] = (pre_df[time_col] >= placebo_midpoint).astype(int)
        pre_df["placebo_interaction"] = pre_df["treated"] * pre_df["placebo_post"]

        try:
            placebo_model = smf.ols(
                f"{outcome_col} ~ treated + placebo_post + placebo_interaction",
                data=pre_df
            ).fit(cov_type="HC3")

            placebo_est = placebo_model.params["placebo_interaction"]
            placebo_p = placebo_model.pvalues["placebo_interaction"]
            placebo_passed = placebo_p > 0.1

            diagnostics.append(DiagnosticResult(
                name="Placebo Test (Pre-Period)",
                passed=placebo_passed,
                statistic=round(placebo_est, 6),
                p_value=round(placebo_p, 4),
                interpretation=(
                    f"Placebo test passed (p={placebo_p:.3f}). "
                    "No spurious effect detected in pre-period — supports DiD validity."
                    if placebo_passed else
                    f"⚠️ Placebo test failed (p={placebo_p:.3f}, est={placebo_est:.4f}). "
                    "Effect detected in pre-period where none should exist. "
                    "Results may reflect pre-existing trends rather than treatment effect."
                ),
                details={"placebo_estimate": round(placebo_est, 6), "placebo_midpoint": str(placebo_midpoint)}
            ))
        except Exception as e:
            diagnostics.append(DiagnosticResult(
                name="Placebo Test",
                passed=False,
                interpretation=f"Could not run placebo test: {e}"
            ))
    else:
        diagnostics.append(DiagnosticResult(
            name="Placebo Test",
            passed=True,
            interpretation="Insufficient pre-periods for placebo test (need ≥3). Skipped."
        ))

    # ── Diagnostic 3: Pre-Period Balance ─────────────────────────────────────
    pre_control = df[(df["treated"] == 0) & (df["post"] == 0)][outcome_col]
    pre_treatment = df[(df["treated"] == 1) & (df["post"] == 0)][outcome_col]

    if len(pre_control) > 1 and len(pre_treatment) > 1:
        _, p_balance = stats.ttest_ind(pre_treatment, pre_control, equal_var=False)
        balance_passed = True  # pre-period means can differ in DiD — this is informational
        diagnostics.append(DiagnosticResult(
            name="Pre-Period Mean Balance",
            passed=balance_passed,
            p_value=round(p_balance, 4),
            statistic=round(pre_treatment.mean() - pre_control.mean(), 6),
            interpretation=(
                f"Pre-period means: treatment={pre_treatment.mean():.4f}, "
                f"control={pre_control.mean():.4f}. "
                f"Difference p={p_balance:.3f}. "
                "(DiD accounts for level differences — informational only.)"
            )
        ))

    all_passed = all(d.passed for d in diagnostics)

    return MethodResult(
        method=CausalMethod.DID,
        estimate=round(estimate, 6),
        std_error=round(se, 6),
        ci_lower=round(ci_lower, 6),
        ci_upper=round(ci_upper, 6),
        p_value=round(p_value, 6),
        relative_effect=round(relative_effect, 4) if relative_effect is not None else None,
        diagnostics=diagnostics,
        diagnostics_passed=all_passed,
        raw_output={
            "model_r2": round(model.rsquared, 4),
            "n_obs": int(model.nobs),
            "control_mean_pre": round(control_mean_pre, 6),
            "n_pre_periods": len(pre_time_vals),
            "outcome_col": outcome_col,
            "treatment_col": treatment_col,
            "time_col": time_col,
        }
    )
