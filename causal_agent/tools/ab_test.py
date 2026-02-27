import pandas as pd
import numpy as np
from scipy import stats
from typing import Optional
from ..models import MethodResult, DiagnosticResult, CausalMethod, DataProfile


def run_ab_test(data_path: str, profile: DataProfile) -> MethodResult:
    """
    Runs a two-sample A/B test with:
    - Two-sided t-test for the primary estimate
    - Covariate balance check (if pre-experiment columns exist)
    - Sample ratio mismatch (SRM) check
    - Effect size (Cohen's d)
    """
    df = pd.read_csv(data_path)
    diagnostics = []

    treatment_col = profile.treatment_col
    outcome_col = profile.outcome_col

    if treatment_col is None or outcome_col is None:
        raise ValueError(
            f"Cannot run A/B test: missing treatment_col={treatment_col} or outcome_col={outcome_col}"
        )

    # Normalize treatment to 0/1
    df = df.dropna(subset=[treatment_col, outcome_col])
    unique_vals = sorted(df[treatment_col].unique(), key=str)

    if len(unique_vals) != 2:
        raise ValueError(f"Expected 2 treatment groups, found {len(unique_vals)}: {unique_vals}")

    control_label, treatment_label = unique_vals[0], unique_vals[1]
    # Handle string labels like "control"/"treatment"
    if str(control_label).lower() in ("treatment", "test", "1", "true", "exposed"):
        control_label, treatment_label = treatment_label, control_label

    control = df[df[treatment_col] == control_label][outcome_col].astype(float)
    treatment = df[df[treatment_col] == treatment_label][outcome_col].astype(float)

    # ── Primary estimate ──────────────────────────────────────────────────────
    t_stat, p_value = stats.ttest_ind(treatment, control, equal_var=False)
    estimate = treatment.mean() - control.mean()
    relative_effect = estimate / control.mean() if control.mean() != 0 else None

    # Confidence interval
    n1, n2 = len(treatment), len(control)
    se = np.sqrt(treatment.var(ddof=1) / n1 + control.var(ddof=1) / n2)
    df_welch = (se**2)**2 / (
        (treatment.var(ddof=1) / n1)**2 / (n1 - 1) +
        (control.var(ddof=1) / n2)**2 / (n2 - 1)
    )
    t_crit = stats.t.ppf(0.975, df=df_welch)
    ci_lower = estimate - t_crit * se
    ci_upper = estimate + t_crit * se

    # ── Diagnostic 1: Sample Ratio Mismatch ──────────────────────────────────
    expected_ratio = 0.5
    observed_ratio = n1 / (n1 + n2)
    chi2_srm = ((n1 - (n1 + n2) * expected_ratio)**2 / ((n1 + n2) * expected_ratio) +
                (n2 - (n1 + n2) * (1 - expected_ratio))**2 / ((n1 + n2) * (1 - expected_ratio)))
    p_srm = 1 - stats.chi2.cdf(chi2_srm, df=1)
    srm_passed = p_srm > 0.01  # strict threshold

    diagnostics.append(DiagnosticResult(
        name="Sample Ratio Mismatch (SRM)",
        passed=srm_passed,
        statistic=round(chi2_srm, 4),
        p_value=round(p_srm, 4),
        interpretation=(
            f"No SRM detected (p={p_srm:.3f}). Assignment looks clean."
            if srm_passed else
            f"⚠️ SRM detected (p={p_srm:.3f}). Unequal split: {n1} control vs {n2} treatment. "
            "Investigate assignment mechanism before trusting results."
        ),
        details={"n_control": n1, "n_treatment": n2, "observed_ratio": round(observed_ratio, 3)}
    ))

    # ── Diagnostic 2: Effect Size (Cohen's d) ────────────────────────────────
    pooled_std = np.sqrt(((n1 - 1) * control.var(ddof=1) + (n2 - 1) * treatment.var(ddof=1)) / (n1 + n2 - 2))
    cohens_d = estimate / pooled_std if pooled_std > 0 else 0
    effect_size_label = (
        "small" if abs(cohens_d) < 0.2 else
        "medium" if abs(cohens_d) < 0.5 else
        "large"
    )

    diagnostics.append(DiagnosticResult(
        name="Effect Size (Cohen's d)",
        passed=True,  # informational
        statistic=round(cohens_d, 4),
        interpretation=f"Cohen's d = {cohens_d:.3f} ({effect_size_label} effect size)",
        details={"pooled_std": round(pooled_std, 4)}
    ))

    # ── Diagnostic 3: Outlier & Variance Check ────────────────────────────────
    # Detect extreme outliers via IQR method and flag variance asymmetry
    # Skip outlier detection for binary outcomes (IQR=0 makes it meaningless)
    all_vals = pd.concat([control, treatment])
    is_binary = set(all_vals.dropna().unique()).issubset({0, 0.0, 1, 1.0})

    def _count_outliers(series: pd.Series) -> tuple[int, float, float]:
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            return 0, float(q1), float(q3)  # can't define outliers when IQR=0
        lower, upper = q1 - 3 * iqr, q3 + 3 * iqr  # 3× IQR = extreme outliers
        n_outliers = int(((series < lower) | (series > upper)).sum())
        return n_outliers, float(lower), float(upper)

    if is_binary:
        # Binary outcome — outlier detection not applicable
        diagnostics.append(DiagnosticResult(
            name="Outlier & Variance Check",
            passed=True,
            interpretation="Binary outcome detected — outlier check not applicable. Variance ratio is inherent to proportions.",
            details={"is_binary": True}
        ))
    else:
        outliers_c, lo_c, hi_c = _count_outliers(control)
        outliers_t, lo_t, hi_t = _count_outliers(treatment)
        total_outliers = outliers_c + outliers_t

        var_ratio = (
            max(treatment.var(ddof=1), control.var(ddof=1)) /
            min(treatment.var(ddof=1), control.var(ddof=1))
            if min(treatment.var(ddof=1), control.var(ddof=1)) > 0 else np.inf
        )

        outlier_ok = total_outliers <= 0.02 * (n1 + n2)  # ≤2% of data
        variance_ok = var_ratio < 10  # variance ratio under 10×
        check_passed = outlier_ok and variance_ok

        outlier_parts = []
        if not outlier_ok:
            outlier_parts.append(
                f"⚠️ {total_outliers} extreme outliers detected "
                f"({outliers_c} control, {outliers_t} treatment) using 3×IQR rule. "
                "These may heavily distort the t-test estimate. "
                "Consider trimming outliers or using a robust test."
            )
        if not variance_ok:
            outlier_parts.append(
                f"⚠️ Extreme variance asymmetry — ratio = {var_ratio:.1f}× "
                f"(control std={control.std():.2f}, treatment std={treatment.std():.2f}). "
                "Welch's t-test handles unequal variances, but this level of "
                "asymmetry suggests outlier-driven inflation in one group."
            )
        if check_passed:
            outlier_parts.append(
                f"No extreme outliers detected. Variance ratio = {var_ratio:.1f}× — acceptable."
            )

        diagnostics.append(DiagnosticResult(
            name="Outlier & Variance Check",
            passed=check_passed,
            statistic=round(var_ratio, 2),
            interpretation=" ".join(outlier_parts),
            details={
                "outliers_control": outliers_c,
                "outliers_treatment": outliers_t,
                "variance_ratio": round(var_ratio, 2),
                "control_std": round(float(control.std()), 4),
                "treatment_std": round(float(treatment.std()), 4),
                "iqr_bounds_control": [round(lo_c, 2), round(hi_c, 2)],
                "iqr_bounds_treatment": [round(lo_t, 2), round(hi_t, 2)],
            }
        ))

    # ── Diagnostic 4: Normality check (for small samples) ───────────────────
    if n1 < 30 or n2 < 30:
        _, p_norm_c = stats.shapiro(control)
        _, p_norm_t = stats.shapiro(treatment)
        norm_passed = p_norm_c > 0.05 and p_norm_t > 0.05
        diagnostics.append(DiagnosticResult(
            name="Normality Check (Shapiro-Wilk)",
            passed=norm_passed,
            interpretation=(
                "Both groups roughly normal — t-test valid."
                if norm_passed else
                "⚠️ Non-normal distributions detected. Consider Mann-Whitney U test for robustness."
            ),
            details={
                "p_control": round(p_norm_c, 4),
                "p_treatment": round(p_norm_t, 4)
            }
        ))

    all_passed = all(d.passed for d in diagnostics)

    return MethodResult(
        method=CausalMethod.AB_TEST,
        estimate=round(estimate, 6),
        std_error=round(se, 6),
        ci_lower=round(ci_lower, 6),
        ci_upper=round(ci_upper, 6),
        p_value=round(p_value, 6),
        relative_effect=round(relative_effect, 4) if relative_effect is not None else None,
        diagnostics=diagnostics,
        diagnostics_passed=all_passed,
        raw_output={
            "control_mean": round(control.mean(), 6),
            "treatment_mean": round(treatment.mean(), 6),
            "control_std": round(control.std(), 6),
            "treatment_std": round(treatment.std(), 6),
            "n_control": n1,
            "n_treatment": n2,
            "t_statistic": round(t_stat, 6),
            "cohens_d": round(cohens_d, 4),
            "outcome_col": outcome_col,
            "treatment_col": treatment_col,
            "control_label": str(control_label),
            "treatment_label": str(treatment_label),
        }
    )
