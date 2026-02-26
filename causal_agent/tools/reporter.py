from ..models import AgentState, MethodResult, CausalMethod, DiagnosticResult
from typing import Optional
import datetime


def _significance_label(p_value: Optional[float]) -> str:
    if p_value is None:
        return "unknown"
    if p_value < 0.001:
        return "highly significant (p < 0.001)"
    elif p_value < 0.01:
        return "significant (p < 0.01)"
    elif p_value < 0.05:
        return "significant (p < 0.05)"
    elif p_value < 0.10:
        return "marginally significant (p < 0.10)"
    else:
        return f"not significant (p = {p_value:.3f})"


def _format_diagnostic(d: DiagnosticResult) -> str:
    status = "✅" if d.passed else "⚠️"
    lines = [f"{status} **{d.name}**"]
    lines.append(f"   {d.interpretation}")
    return "\n".join(lines)


def _format_method_section(result: MethodResult, label: str = "Primary") -> str:
    method_names = {
        CausalMethod.AB_TEST: "A/B Test (Two-Sample T-Test)",
        CausalMethod.DID: "Difference-in-Differences",
        CausalMethod.SYNTHETIC_CONTROL: "Synthetic Control",
    }
    method_name = method_names.get(result.method, str(result.method))

    lines = []
    lines.append(f"### {label} Analysis: {method_name}\n")

    # Core estimates
    lines.append("**Effect Estimate**\n")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| ATT Estimate | {result.estimate:+.4f} |")
    if result.relative_effect is not None:
        lines.append(f"| Relative Lift | {result.relative_effect*100:+.2f}% |")
    if result.std_error is not None:
        lines.append(f"| Std Error | {result.std_error:.4f} |")
    if result.ci_lower is not None and result.ci_upper is not None:
        lines.append(f"| 95% CI | [{result.ci_lower:.4f}, {result.ci_upper:.4f}] |")
    if result.p_value is not None:
        lines.append(f"| P-value | {result.p_value:.4f} |")
        lines.append(f"| Significance | {_significance_label(result.p_value)} |")

    lines.append("")

    # Raw context
    if result.raw_output:
        raw = result.raw_output
        if "control_mean" in raw and "treatment_mean" in raw:
            lines.append(f"Control mean: `{raw['control_mean']:.4f}` | Treatment mean: `{raw['treatment_mean']:.4f}` | "
                         f"N: {raw.get('n_control', '?')} control, {raw.get('n_treatment', '?')} treatment\n")
        elif "control_mean_pre" in raw:
            lines.append(f"Control mean (pre): `{raw['control_mean_pre']:.4f}` | "
                         f"Observations: {raw.get('n_obs', '?')}\n")
        elif "treated_unit" in raw:
            lines.append(f"Treated unit: `{raw['treated_unit']}` | "
                         f"Donors: {raw.get('n_donor_units', '?')} | "
                         f"Pre-periods: {raw.get('n_pre_periods', '?')} | "
                         f"Post-periods: {raw.get('n_post_periods', '?')}\n")

        # Donor weights for SC
        if "donor_weights" in raw and raw["donor_weights"]:
            lines.append("**Synthetic Control Donor Weights**\n")
            for unit, weight in sorted(raw["donor_weights"].items(), key=lambda x: -x[1]):
                bar = "█" * int(weight * 20)
                lines.append(f"- `{unit}`: {weight:.3f} {bar}")
            lines.append("")

    # Diagnostics
    lines.append("**Diagnostics**\n")
    for d in result.diagnostics:
        lines.append(_format_diagnostic(d))
        lines.append("")

    overall = "✅ All diagnostics passed" if result.diagnostics_passed else "⚠️ Some diagnostics failed — interpret with caution"
    lines.append(f"**Overall diagnostic status:** {overall}\n")

    return "\n".join(lines)


def generate_report(state: AgentState) -> str:
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = []

    lines.append(f"# Causal Inference Analysis Report")
    lines.append(f"*Generated: {now}*\n")

    lines.append(f"## Business Question\n> {state.question}\n")

    # Data profile summary
    if state.data_profile:
        p = state.data_profile
        lines.append("## Data Profile\n")
        lines.append(f"- **Rows**: {p.n_rows:,} | **Columns**: {p.n_cols}")
        lines.append(f"- **Treatment column**: `{p.treatment_col}`")
        lines.append(f"- **Outcome column**: `{p.outcome_col}`")
        if p.time_col:
            lines.append(f"- **Time column**: `{p.time_col}`")
        if p.group_col:
            lines.append(f"- **Group column**: `{p.group_col}`")
        for note in p.notes:
            lines.append(f"- {note}")
        lines.append("")

    # Method selection reasoning
    lines.append("## Agent Reasoning Log\n")
    for i, log in enumerate(state.reasoning_log, 1):
        lines.append(f"{i}. {log}")
    lines.append("")

    # Primary result
    if state.method_result:
        lines.append("## Results\n")
        lines.append(_format_method_section(state.method_result, label="Primary"))

    # Fallback result
    if state.fallback_result:
        lines.append(_format_method_section(state.fallback_result, label="Fallback / Robustness Check"))

    # Interpretation & recommendations
    lines.append("## Interpretation & Recommendations\n")

    primary = state.method_result
    if primary:
        is_significant = primary.p_value is not None and primary.p_value < 0.05
        diags_ok = primary.diagnostics_passed

        if is_significant and diags_ok:
            lines.append(
                f"**Finding**: The analysis finds a **statistically significant causal effect** of "
                f"`{primary.estimate:+.4f}` "
                + (f"({primary.relative_effect*100:+.1f}% relative lift)" if primary.relative_effect else "") +
                f". All diagnostic checks passed, supporting the credibility of this estimate."
            )
        elif is_significant and not diags_ok:
            lines.append(
                f"**Finding**: A statistically significant effect of `{primary.estimate:+.4f}` was detected, "
                f"but **one or more diagnostic checks failed**. The estimate should be treated with caution "
                f"until the flagged assumption violations are addressed."
            )
        elif not is_significant and diags_ok:
            lines.append(
                f"**Finding**: **No statistically significant effect** detected (p={primary.p_value:.3f}). "
                f"Diagnostics are clean, so this null result appears reliable. "
                f"Consider whether the experiment was adequately powered."
            )
        else:
            lines.append(
                f"**Finding**: Ambiguous — no significant effect detected and diagnostic issues present. "
                f"Results are not conclusive."
            )

        lines.append("")

        if state.fallback_result:
            fr = state.fallback_result
            if abs(fr.estimate - primary.estimate) / max(abs(primary.estimate), 1e-10) < 0.2:
                lines.append(
                    f"**Robustness**: The fallback {fr.method} estimate (`{fr.estimate:+.4f}`) "
                    f"is consistent with the primary estimate, strengthening confidence in the finding."
                )
            else:
                lines.append(
                    f"**Robustness**: ⚠️ The fallback {fr.method} estimate (`{fr.estimate:+.4f}`) "
                    f"diverges from the primary estimate (`{primary.estimate:+.4f}`). "
                    f"Investigate the source of this discrepancy before reporting results."
                )

    # Errors
    if state.errors:
        lines.append("\n## Errors & Warnings\n")
        for err in state.errors:
            lines.append(f"- ⚠️ {err}")

    lines.append("\n---")
    lines.append("*Report generated by the Causal Inference Agent*")

    report = "\n".join(lines)
    return report
