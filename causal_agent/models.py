from pydantic import BaseModel, Field
from typing import Optional, Literal, Any
from enum import Enum


class CausalMethod(str, Enum):
    AB_TEST = "ab_test"
    DID = "difference_in_differences"
    SYNTHETIC_CONTROL = "synthetic_control"
    UNKNOWN = "unknown"


class DiagnosticResult(BaseModel):
    name: str
    passed: bool
    statistic: Optional[float] = None
    p_value: Optional[float] = None
    interpretation: str
    details: Optional[dict] = None


class MethodResult(BaseModel):
    method: CausalMethod
    estimate: float
    std_error: Optional[float] = None
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    p_value: Optional[float] = None
    relative_effect: Optional[float] = None  # % lift
    diagnostics: list[DiagnosticResult] = []
    diagnostics_passed: bool = True
    raw_output: Optional[dict] = None


class DataProfile(BaseModel):
    n_rows: int
    n_cols: int
    columns: list[str]
    has_time_column: bool
    has_group_column: bool
    has_geo_column: bool
    treatment_col: Optional[str] = None
    outcome_col: Optional[str] = None
    time_col: Optional[str] = None
    group_col: Optional[str] = None
    n_treatment: Optional[int] = None
    n_control: Optional[int] = None
    n_pre_periods: Optional[int] = None
    n_post_periods: Optional[int] = None
    suggested_methods: list[CausalMethod] = []
    notes: list[str] = []


class AgentState(BaseModel):
    # Inputs
    question: str
    data_path: str

    # Progressive state
    data_profile: Optional[DataProfile] = None
    selected_method: Optional[CausalMethod] = None
    method_result: Optional[MethodResult] = None
    fallback_method: Optional[CausalMethod] = None
    fallback_result: Optional[MethodResult] = None
    report: Optional[str] = None

    # Control flow
    iteration: int = 0
    max_iterations: int = 3
    errors: list[str] = []
    reasoning_log: list[str] = []

    class Config:
        use_enum_values = True
