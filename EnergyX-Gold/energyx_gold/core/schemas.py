"""Pydantic models for the task manifest and for gold results.

`uv run python scripts/export_schemas.py` writes them to schema/ as JSON Schema.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    ValidationError,
    model_validator,
)

AnswerType = Literal["numeric", "set", "scalar_text", "record", "refusal", "method"]
OutputType = Literal[
    "scalar", "boolean", "categorical", "list", "ranked_list", "table", "time_series",
    "forecast", "anomaly_set", "cluster_assignment", "visualization_data", "decision_support",
    "refusal",
]
Status = Literal["proposed", "reviewed", "accepted_seed", "gold_candidate", "gold_verified", "rejected"]
Stack = Literal["sql", "pandas", "kg", "sklearn", "sktime", "pytorch", "lightgbm", "statsmodels", "scipy", "ruptures", "tslearn", "statsforecast"]
RequiredData = Literal[
    "Building Metadata", "Electricity Consumption Data", "Weather Data", "Dataset/Catalog Metadata",
]

# The evaluation methods each answer type may use.
EVALUATION_METHODS: dict[str, tuple[str, ...]] = {
    "numeric": ("numeric",),
    "set": ("set_match",),
    "scalar_text": ("scalar_text",),
    "record": ("record",),
    "refusal": ("refusal",),
    "method": ("rubric", "unscored"),
}

NOT_ANSWERABLE = "Not answerable"


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


# -- manifest ------------------------------------------------------------------


class Evaluation(Model):
    method: str
    field: str | None = None
    tolerance: float | None = None


class AnswerFormat(Model):
    """How an agent must give its answer. `instructions` is shown with the question."""

    instructions: str
    unit: str | None = None
    decimals: int | None = None
    order: Literal["any", "ranked"] | None = None
    key: str | None = None


class Task(Model):
    task_id: str = Field(pattern=r"^Q\d{2}$")
    serial: int | None = None  # S.No in the final question list
    question: str
    category: int = Field(ge=1, le=10)
    category_name: str
    secondary_categories: list[str] = []
    sub_task: str | None = None
    user_roles: list[str] = []
    knowledge_source: str | None = None
    source_reference: str | None = None
    seed_origin: dict[str, Any] = {}
    ai_capability: str | None = None
    difficulty: Literal["L1", "L2", "L3", "L4", "L5"]
    required_data: list[RequiredData]
    temporal_scope: str | None = None
    expected_output_format: str | None = None
    answer_type: AnswerType
    output_type: OutputType
    execution_stack: list[Stack] = []
    answer_format: AnswerFormat
    evaluation: Evaluation
    definition: str | None = None
    caveats: str | None = None
    data_version: str
    reference_module: str
    status: Status
    status_note: str | None = None  # why a task isn't gold_verified yet
    draft: dict[str, Any] = {}

    @model_validator(mode="after")
    def _method_fits_answer_type(self) -> Task:
        allowed = EVALUATION_METHODS[self.answer_type]
        if self.evaluation.method not in allowed:
            raise ValueError(f"evaluation.method {self.evaluation.method!r} doesn't fit "
                             f"answer_type {self.answer_type!r}; expected one of {allowed}")
        return self


# -- results -------------------------------------------------------------------


class NumericResult(Model):
    answer: StrictInt | StrictFloat
    unit: str | None = None


class SetResult(Model):
    answer: list[str]
    count: int

    @model_validator(mode="after")
    def _sorted_unique_counted(self) -> SetResult:
        if self.answer != sorted(set(self.answer)):
            raise ValueError("members must be sorted and unique")
        if self.count != len(self.answer):
            raise ValueError("count must equal the number of members")
        return self


class ScalarTextResult(Model):
    answer: str


class RecordResult(Model):
    """Several named values, e.g. {"MAE": 1.2, "RMSE": 2.3}. Numbers are compared field by field."""

    answer: dict[str, Any]
    units: dict[str, str] = {}


class RefusalResult(Model):
    refused: Literal[True]
    predicate: str


class MethodResult(Model):
    answer: None
    method: str


RESULT_MODELS: dict[str, type[Model]] = {
    "numeric": NumericResult,
    "set": SetResult,
    "scalar_text": ScalarTextResult,
    "record": RecordResult,
    "refusal": RefusalResult,
    "method": MethodResult,
}
Result = NumericResult | SetResult | ScalarTextResult | RecordResult | RefusalResult | MethodResult


class Alternate(Model):
    """A second accepted answer, for a question with two agreed readings."""

    reading: str
    result: Result


class AbsenceCheck(Model):
    """Evidence that a refusal question can't be answered: a query over the data that returns no rows."""

    query: str
    rows: Literal[0]


class Provenance(Model):
    data_version: str
    snapshot_sha256: str
    code_commit: str | None
    code_dirty: bool | None
    uv_lock_sha256: str | None
    python: str
    packages: dict[str, str]
    platform: str


class GoldResult(Model):
    """One line of outputs/gold_results.jsonl."""

    task_id: str
    status: Literal["success"]
    result: Result
    alternates: list[Alternate] = []
    evidence: dict[str, Any]
    provenance: Provenance | None = None


def check_output(task: Task, output: dict) -> list[str]:
    """Return the problems with a task's output. An empty list means it is well formed."""
    try:
        gold = GoldResult.model_validate(output)
        RESULT_MODELS[task.answer_type].model_validate(output["result"])
        for alternate in output.get("alternates", []):
            RESULT_MODELS[task.answer_type].model_validate(alternate["result"])
        if task.answer_type == "refusal":
            AbsenceCheck.model_validate(output["evidence"].get("absence_check"))
    except ValidationError as err:
        return [f"{'.'.join(map(str, e['loc'])) or 'output'}: {e['msg']}" for e in err.errors()]
    if gold.task_id != task.task_id:
        return [f"task_id is {gold.task_id!r}, expected {task.task_id!r}"]
    return []
