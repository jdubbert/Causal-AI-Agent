import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm
from typing import Optional
from ..models import MethodResult, DiagnosticResult, CausalMethod, DataProfile


def run_did(data_path: str, profile: DataProfile) -> MethodResult:
    """
    Runs a Difference-in-Differences analysis with:
        - Entity (unit) and time fixed effects when panel structure allows
        - Cluster-robust standard errors at the unit level
        - Falls back to pooled OLS with HC3 if panel is too thin

    Diagnostics:
    1. Parallel trends — event-study pre-period coefficients (joint F-test)
    2. Placebo test — fake treatment in the pre-period
    3. Pre-period balance check
    4. Anticipation effects check
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

    # ── Normalize treatment to 0/1 ───────────────────────────────────────────
    unique_vals = sorted(df[treatment_col].unique(), key=str)
    if len(unique_vals) != 2:
        raise ValueError(f"Expected 2 treatment groups, found {len(unique_vals)}")

    control_label, treatment_label = unique_vals[0], unique_vals[1]
    if str(control_label).lower() in ("treatment", "test", "1", "true", "exposed"):
        control_label, treatment_label = treatment_label, control_label

    df["treated"] = (df[treatment_col] == treatment_label).astype(int)

    # ── Determine pre/post periods ───────────────────────────────────────────
    time_vals = sorted(df[time_col].unique())
    n_periods = len(time_vals)

    if n_periods < 2:
        raise ValueError("DiD requires at least 2 time periods")

    post_col = next(
        (c for c in df.columns if c.lower() in ("post", "is_post", "after", "post_period")),
        None,
    )

    if post_col is not None:
        df["post"] = df[post_col].astype(int)
    else:
        midpoint_idx = n_periods // 2
        treatment_start = time_vals[midpoint_idx]
        df["post"] = (df[time_col] >= treatment_start).astype(int)

    df["did_interaction"] = df["treated"] * df["post"]

    # Detect panel unit column (market, region, etc.) for clustering / FE
    geo_candidates = ["geo", "dma", "state", "region", "market", "unit", "zip", "city"]
    unit_col = None
    for cand in geo_candidates:
        matches = [c for c in df.columns if cand.lower() in c.lower()]
        if matches:
            unit_col = matches[0]
            break
    if unit_col is None:
        unit_col = profile.group_col  # may still be None

    # ── Primary DiD estimate ─────────────────────────────────────────────────
    # Decide between TWFE (unit + time FE with clustering) or pooled OLS
    use_twfe = (
        unit_col is not None
        and unit_col in df.columns
        and df[unit_col].nunique() >= 3
        and n_periods >= 3
    )

    try:
        if use_twfe:
            # True two-way fixed effects: absorb unit + time, cluster at unit
            formula = f"{outcome_col} ~ did_interaction + C({unit_col}) + C({time_col})"
            model = smf.ols(formula, data=df).fit(
                cov_type="cluster",
                cov_kwds={"groups": df[unit_col]},
            )
        else:
            # Pooled 2×2 with HC3 (few units or no panel id)
            formula = f"{outcome_col} ~ treated + post + did_interaction"
            model = smf.ols(formula, data=df).fit(cov_type="HC3")

        estimate = model.params["did_interaction"]
        se = model.bse["did_interaction"]
        p_value = model.pvalues["did_interaction"]
        ci = model.conf_int().loc["did_interaction"]
        ci_lower, ci_upper = float(ci[0]), float(ci[1])

        control_mean_pre = df[(df["treated"] == 0) & (df["post"] == 0)][outcome_col].mean()
        relative_effect = estimate / control_mean_pre if control_mean_pre != 0 else None

    except Exception as e:
        raise ValueError(f"DiD regression failed: {e}")

    # ── Diagnostic 1: Parallel Trends (Event Study) ──────────────────────────
    pre_df = df[df["post"] == 0].copy()
    pre_time_vals = sorted(pre_df[time_col].unique())
    event_study_coeffs = {}

    if len(pre_time_vals) >= 2:
        # Build event-time dummies for the FULL panel
        post_time_vals = sorted(df[df["post"] == 1][time_col].unique())
        # Anchor = last pre-period (omitted / reference)
        anchor_period = pre_time_vals[-1]
        all_event_periods = [t for t in time_vals if t != anchor_period]

        df_es = df.copy()
        for t in all_event_periods:
            col_name = f"t_{t}"
            df_es[col_name] = ((df_es[time_col] == t) & (df_es["treated"] == 1)).astype(int)

        event_cols = [f"t_{t}" for t in all_event_periods]

        try:
            if use_twfe:
                es_formula = (
                    f"{outcome_col} ~ {' + '.join(event_cols)} + C({unit_col}) + C({time_col})"
                )
                es_model = smf.ols(es_formula, data=df_es).fit(
                    cov_type="cluster",
                    cov_kwds={"groups": df_es[unit_col]},
                )
            else:
                es_formula = f"{outcome_col} ~ treated + {' + '.join(event_cols)}"
                es_model = smf.ols(es_formula, data=df_es).fit(cov_type="HC3")

            # Collect pre-period coefficients for the parallel trends test
            pre_event_cols = [f"t_{t}" for t in pre_time_vals if t != anchor_period]
            pre_coefs = []
            pre_pvals = []
            for col in pre_event_cols:
                if col in es_model.params:
                    event_study_coeffs[col] = {
                        "coef": round(float(es_model.params[col]), 6),
                        "se": round(float(es_model.bse[col]), 6),
                        "p": round(float(es_model.pvalues[col]), 4),
                    }
                    pre_coefs.append(float(es_model.params[col]))
                    pre_pvals.append(float(es_model.pvalues[col]))

            # Also collect post-period coefficients for the report
            post_event_cols = [f"t_{t}" for t in post_time_vals]
            for col in post_event_cols:
                if col in es_model.params:
                    event_study_coeffs[col] = {
                        "coef": round(float(es_model.params[col]), 6),
                        "se": round(float(es_model.bse[col]), 6),
                        "p": round(float(es_model.pvalues[col]), 4),
                    }

            # Joint F-test on all pre-period interactions
            if len(pre_event_cols) >= 1:
                r_matrix = np.zeros((len(pre_event_cols), len(es_model.params)))
                for i, col in enumerate(pre_event_cols):
                    if col in es_model.params.index:
                        idx = list(es_model.params.index).index(col)
                        r_matrix[i, idx] = 1

                try:
                    f_result = es_model.f_test(r_matrix)
                    f_stat = float(f_result.fvalue)
                    f_pval = float(f_result.pvalue)
                except Exception:
                    # Fallback: Bonferroni correction on individual p-values
                    f_stat = None
                    f_pval = min(pre_pvals) * len(pre_pvals) if pre_pvals else 1.0

                pt_passed = (f_pval > 0.1) if f_pval is not None else True

                diagnostics.append(DiagnosticResult(
                    name="Parallel Trends (Event Study F-test)",
                    passed=pt_passed,
                    statistic=round(f_stat, 4) if f_stat is not None else None,
                    p_value=round(f_pval, 4) if f_pval is not None else None,
                    interpretation=(
                        f"Joint F-test on pre-period event-study coefficients: "
                        f"F={f_stat:.3f}, p={f_pval:.3f}. "
                        + (
                            "Pre-treatment coefficients are jointly insignificant — "
                            "parallel trends assumption is supported."
                            if pt_passed
                            else
                            "⚠️ Pre-treatment coefficients are jointly significant — "
                            "parallel trends may be violated. Consider Synthetic Control "
                            "or inspect the event-study plot."
                        )
                    ),
                    details={
                        "n_pre_periods": len(pre_time_vals),
                        "anchor_period": str(anchor_period),
                        "pre_period_coeffs": {
                            col: event_study_coeffs[col]
                            for col in pre_event_cols
                            if col in event_study_coeffs
                        },
                    },
                ))
            else:
                diagnostics.append(DiagnosticResult(
                    name="Parallel Trends (Event Study)",
                    passed=True,
                    interpretation=(
                        "Only 1 pre-period before anchor — cannot test. Use with caution."
                    ),
                ))

        except Exception as e:
            # Fallback to simpler linear trend test
            time_map = {t: i for i, t in enumerate(pre_time_vals)}
            pre_df["time_numeric"] = pre_df[time_col].map(time_map)
            pre_df["trend_interaction"] = pre_df["treated"] * pre_df["time_numeric"]
            try:
                trend_model = smf.ols(
                    f"{outcome_col} ~ treated + time_numeric + trend_interaction",
                    data=pre_df,
                ).fit(cov_type="HC3")
                trend_p = trend_model.pvalues["trend_interaction"]
                diagnostics.append(DiagnosticResult(
                    name="Parallel Trends (Linear Trend Fallback)",
                    passed=trend_p > 0.1,
                    statistic=round(float(trend_model.params["trend_interaction"]), 6),
                    p_value=round(float(trend_p), 4),
                    interpretation=(
                        f"Linear trend test p={trend_p:.3f}. "
                        + (
                            "Parallel trends supported."
                            if trend_p > 0.1
                            else "⚠️ Trends may diverge pre-treatment."
                        )
                    ),
                ))
            except Exception as e2:
                diagnostics.append(DiagnosticResult(
                    name="Parallel Trends Test",
                    passed=False,
                    interpretation=f"Could not run parallel trends test: {e2}",
                ))
    else:
        diagnostics.append(DiagnosticResult(
            name="Parallel Trends Test",
            passed=False,
            interpretation=(
                "Only 1 pre-period available — cannot test parallel trends. Use with caution."
            ),
        ))

    # ── Diagnostic 2: Placebo Test ───────────────────────────────────────────
    if len(pre_time_vals) >= 3:
        placebo_midpoint = pre_time_vals[len(pre_time_vals) // 2]
        pre_placebo = pre_df.copy()
        pre_placebo["placebo_post"] = (pre_placebo[time_col] >= placebo_midpoint).astype(int)
        pre_placebo["placebo_interaction"] = pre_placebo["treated"] * pre_placebo["placebo_post"]

        try:
            if use_twfe and unit_col in pre_placebo.columns:
                placebo_formula = (
                    f"{outcome_col} ~ placebo_interaction + C({unit_col}) + C({time_col})"
                )
                placebo_model = smf.ols(placebo_formula, data=pre_placebo).fit(
                    cov_type="cluster",
                    cov_kwds={"groups": pre_placebo[unit_col]},
                )
            else:
                placebo_formula = (
                    f"{outcome_col} ~ treated + placebo_post + placebo_interaction"
                )
                placebo_model = smf.ols(placebo_formula, data=pre_placebo).fit(cov_type="HC3")

            placebo_est = float(placebo_model.params["placebo_interaction"])
            placebo_p = float(placebo_model.pvalues["placebo_interaction"])
            placebo_passed = placebo_p > 0.1

            diagnostics.append(DiagnosticResult(
                name="Placebo Test (Pre-Period)",
                passed=placebo_passed,
                statistic=round(placebo_est, 6),
                p_value=round(placebo_p, 4),
                interpretation=(
                    f"Placebo test passed (p={placebo_p:.3f}). "
                    "No spurious effect in pre-period — supports DiD validity."
                    if placebo_passed
                    else
                    f"⚠️ Placebo test failed (p={placebo_p:.3f}, est={placebo_est:.4f}). "
                    "Spurious effect detected in pre-period. "
                    "Results may reflect pre-existing dynamics, not treatment."
                ),
                details={
                    "placebo_estimate": round(placebo_est, 6),
                    "placebo_midpoint": str(placebo_midpoint),
                },
            ))
        except Exception as e:
            diagnostics.append(DiagnosticResult(
                name="Placebo Test",
                passed=False,
                interpretation=f"Could not run placebo test: {e}",
            ))
    else:
        diagnostics.append(DiagnosticResult(
            name="Placebo Test",
            passed=True,
            interpretation="Insufficient pre-periods for placebo test (need ≥3). Skipped.",
        ))

    # ── Diagnostic 3: Pre-Period Balance ─────────────────────────────────────
    pre_control = df[(df["treated"] == 0) & (df["post"] == 0)][outcome_col]
    pre_treatment = df[(df["treated"] == 1) & (df["post"] == 0)][outcome_col]

    if len(pre_control) > 1 and len(pre_treatment) > 1:
        _, p_balance = stats.ttest_ind(pre_treatment, pre_control, equal_var=False)
        diagnostics.append(DiagnosticResult(
            name="Pre-Period Mean Balance",
            passed=True,  # informational — DiD accounts for level differences
            p_value=round(float(p_balance), 4),
            statistic=round(float(pre_treatment.mean() - pre_control.mean()), 6),
            interpretation=(
                f"Pre-period means: treatment={pre_treatment.mean():.4f}, "
                f"control={pre_control.mean():.4f}. "
                f"Difference p={p_balance:.3f}. "
                "(DiD accounts for level differences — informational only.)"
            ),
        ))

    # ── Diagnostic 4: Anticipation Effects ───────────────────────────────────
    if len(pre_time_vals) >= 2:
        last_pre = pre_time_vals[-1]
        pen_pre = pre_time_vals[-2]
        t_last = df[(df[time_col] == last_pre) & (df["treated"] == 1)][outcome_col]
        c_last = df[(df[time_col] == last_pre) & (df["treated"] == 0)][outcome_col]
        t_pen = df[(df[time_col] == pen_pre) & (df["treated"] == 1)][outcome_col]
        c_pen = df[(df[time_col] == pen_pre) & (df["treated"] == 0)][outcome_col]

        if all(len(s) > 0 for s in [t_last, c_last, t_pen, c_pen]):
            gap_last = t_last.mean() - c_last.mean()
            gap_pen = t_pen.mean() - c_pen.mean()
            anticipation_jump = gap_last - gap_pen

            antici_ratio = abs(anticipation_jump / estimate) if estimate != 0 else 0
            antici_ok = antici_ratio < 0.3

            diagnostics.append(DiagnosticResult(
                name="Anticipation Effects Check",
                passed=antici_ok,
                statistic=round(float(anticipation_jump), 6),
                interpretation=(
                    f"Gap change in final pre-period: {anticipation_jump:+.4f} "
                    f"({antici_ratio * 100:.1f}% of DiD estimate). "
                    + (
                        "No evidence of anticipation."
                        if antici_ok
                        else
                        "⚠️ Large shift in the last pre-period suggests possible anticipation "
                        "effects. Treatment units may have started reacting before the "
                        "official start date."
                    )
                ),
                details={
                    "gap_last_pre": round(float(gap_last), 4),
                    "gap_penultimate_pre": round(float(gap_pen), 4),
                },
            ))

    all_passed = all(d.passed for d in diagnostics)

    return MethodResult(
        method=CausalMethod.DID,
        estimate=round(float(estimate), 6),
        std_error=round(float(se), 6),
        ci_lower=round(ci_lower, 6),
        ci_upper=round(ci_upper, 6),
        p_value=round(float(p_value), 6),
        relative_effect=round(float(relative_effect), 4) if relative_effect is not None else None,
        diagnostics=diagnostics,
        diagnostics_passed=all_passed,
        raw_output={
            "model_type": (
                "TWFE (unit + time FE, cluster-robust SE)" if use_twfe else "Pooled OLS (HC3)"
            ),
            "model_r2": round(float(model.rsquared), 4),
            "n_obs": int(model.nobs),
            "n_units": (
                int(df[unit_col].nunique()) if unit_col and unit_col in df.columns else None
            ),
            "control_mean_pre": round(float(control_mean_pre), 6),
            "n_pre_periods": len(pre_time_vals),
            "n_post_periods": n_periods - len(pre_time_vals),
            "outcome_col": outcome_col,
            "treatment_col": treatment_col,
            "time_col": time_col,
            "unit_col": unit_col,
            "event_study_coefficients": event_study_coeffs if event_study_coeffs else None,
        },
    )
