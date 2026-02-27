import pandas as pd
import numpy as np
from typing import Optional
from ..models import DataProfile, CausalMethod


COMMON_TIME_COLS = ["date", "week", "wk", "month", "period", "time", "year", "quarter", "day", "dt"]
COMMON_GROUP_COLS = ["group", "grp", "variant", "treatment", "arm", "cell", "segment", "assigned",
                    "condition", "cohort", "bucket","cell",'test_cell']
COMMON_GEO_COLS = ["geo", "dma", "state", "region", "market", "zip", "city"]
COMMON_TREATMENT_COLS = ["treated", "treatment", "is_treatment", "exposed", "is_exposed",
                         "grp", "group", "variant", "condition"]
COMMON_OUTCOME_COLS = ["revenue", "conversions", "clicks", "opens", "response", "outcome",
                       "activated", "enrolled", "metric", "kpi", "value", "val", "sales",
                       "amount", "score", "rate", "result"]


def _find_col(df_cols: list[str], candidates: list[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in df_cols}
    for cand in candidates:
        if cand in cols_lower:
            return cols_lower[cand]
    # partial match
    for cand in candidates:
        for col_lower, col in cols_lower.items():
            if cand in col_lower or col_lower in cand:
                return col
    return None


def profile_dataset(data_path: str) -> DataProfile:
    """
    Loads a dataset and returns a structured profile including column detection,
    data shape, and suggested causal methods.
    """
    df = pd.read_csv(data_path)
    cols = list(df.columns)

    time_col = _find_col(cols, COMMON_TIME_COLS)
    group_col = _find_col(cols, COMMON_GROUP_COLS)
    geo_col = _find_col(cols, COMMON_GEO_COLS)
    treatment_col = _find_col(cols, COMMON_TREATMENT_COLS)
    outcome_col = _find_col(cols, COMMON_OUTCOME_COLS)

    # Resolve treatment col (could be same as group col)
    if treatment_col is None and group_col is not None:
        treatment_col = group_col

    notes = []
    suggested_methods = []

    # Value-based detection: if no treatment col found, look for columns with
    # test/control, treatment/control, A/B, 0/1 values
    if treatment_col is None:
        treatment_values = {"test", "control", "treatment", "treated", "exposed",
                           "unexposed", "placebo", "a", "b", "0", "1"}
        for col in cols:
            unique_vals = set(str(v).lower().strip() for v in df[col].dropna().unique())
            if len(unique_vals) == 2 and unique_vals.issubset(treatment_values):
                treatment_col = col
                group_col = group_col or col
                notes.append(f"Detected '{col}' as treatment column based on values: {sorted(unique_vals)}")
                break

    # Count treatment/control
    n_treatment = n_control = None
    if treatment_col is not None and treatment_col in df.columns:
        vc = df[treatment_col].value_counts()
        # try to identify binary treatment
        unique_vals = df[treatment_col].dropna().unique()
        if set(str(v).lower() for v in unique_vals).issubset({"0", "1", "true", "false", "treatment", "control", "test"}):
            # binary or two-group
            sorted_vals = sorted(unique_vals, key=str)
            if len(sorted_vals) >= 2:
                n_control = int(vc.iloc[-1]) if len(vc) > 1 else None
                n_treatment = int(vc.iloc[0])

    # Count time periods
    n_pre_periods = n_post_periods = None
    if time_col is not None and time_col in df.columns:
        n_periods = df[time_col].nunique()
        notes.append(f"Found {n_periods} unique time periods in '{time_col}'")

    # Method suggestion logic
    has_time = time_col is not None
    has_group = group_col is not None
    has_geo = geo_col is not None
    has_treatment = treatment_col is not None

    if has_treatment and not has_time:
        suggested_methods.append(CausalMethod.AB_TEST)
        notes.append("Cross-sectional treatment/control structure detected → A/B test appropriate")

    if has_treatment and has_time:
        suggested_methods.append(CausalMethod.AB_TEST)
        suggested_methods.append(CausalMethod.DID)
        notes.append("Panel structure with treatment detected → A/B test or DiD both viable")

    if has_geo and has_time:
        suggested_methods.append(CausalMethod.SYNTHETIC_CONTROL)
        notes.append("Geographic + time structure detected → Synthetic Control viable")

    if has_time and not has_treatment and has_geo:
        suggested_methods.append(CausalMethod.DID)
        suggested_methods.append(CausalMethod.SYNTHETIC_CONTROL)
        notes.append("No explicit treatment column — geo-level DiD or Synthetic Control suggested")

    if not suggested_methods:
        suggested_methods.append(CausalMethod.AB_TEST)
        notes.append("Could not confidently detect structure — defaulting to A/B test suggestion")

    # Deduplicate
    seen = set()
    deduped = []
    for m in suggested_methods:
        if m not in seen:
            seen.add(m)
            deduped.append(m)

    return DataProfile(
        n_rows=len(df),
        n_cols=len(cols),
        columns=cols,
        has_time_column=has_time,
        has_group_column=has_group,
        has_geo_column=has_geo,
        treatment_col=treatment_col,
        outcome_col=outcome_col,
        time_col=time_col,
        group_col=group_col,
        n_treatment=n_treatment,
        n_control=n_control,
        suggested_methods=deduped,
        notes=notes,
    )
