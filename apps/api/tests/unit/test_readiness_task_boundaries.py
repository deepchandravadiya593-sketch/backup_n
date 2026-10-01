
"""NUMPA v2 - ML readiness task-specific boundary tests.

Tests the four supported ML tasks and their input boundaries.

Coverage:
1. Classification accepts a valid categorical target.
2. Regression accepts a valid numeric target.
3. Forecasting accepts a valid datetime column.
4. Clustering works without a target column.
5. Classification requires a target.
6. Regression requires a target.
7. Forecasting requires a datetime column.
8. Clustering rejects a target column.
9. Unknown target columns are rejected by the engine.
10. Assessments do not modify the source dataframe.

These tests use the public readiness engine and request schema.
"""

from __future__ import annotations

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from pydantic import ValidationError

from app.schemas.readiness import ReadinessAssessmentRequest
from app.services.readiness.engine import assess_readiness


# ---------------------------------------------------------
# Shared dataset
# ---------------------------------------------------------

def sample_dataframe() -> pd.DataFrame:
    """Build a dataset suitable for testing all four tasks."""

    row_count = 160

    return pd.DataFrame(
        {
            "event_time": pd.date_range(
                "2025-01-01",
                periods=row_count,
                freq="D",
            ),
            "customer_id": [
                f"CUST-{index:04d}"
                for index in range(row_count)
            ],
            "age": [
                20 + index % 40
                for index in range(row_count)
            ],
            "visits": [
                1 + index % 12
                for index in range(row_count)
            ],
            "monthly_spend": [
                100.0 + index * 2.5
                for index in range(row_count)
            ],
            "churn": [
                index % 2
                for index in range(row_count)
            ],
        }
    )


def make_request(
    task: str,
    *,
    target_column: str | None = None,
    datetime_column: str | None = None,
) -> ReadinessAssessmentRequest:
    """Create a task-specific assessment request."""

    return ReadinessAssessmentRequest(
        dataset_id="task-boundary-test",
        task=task,
        target_column=target_column,
        datetime_column=datetime_column,
        validation_strategy="auto",
    )


def run_assessment(
    request: ReadinessAssessmentRequest,
    dataframe: pd.DataFrame | None = None,
) -> dict:
    """Run the existing readiness engine."""

    if dataframe is None:
        dataframe = sample_dataframe()

    return assess_readiness(
        dataframe,
        request,
    )


def assert_structured_assessment(
    result: dict,
    expected_task: str,
) -> None:
    """Check the common response contract."""

    assert result["task"] == expected_task

    assert result["assessment_id"]

    assert result["assessment_status"] == "provisional"

    assert isinstance(
        result["dimensions"],
        list,
    )

    assert result["dimensions"]

    assert isinstance(
        result["task_diagnostics"],
        dict,
    )

    assert isinstance(
        result["issues"],
        list,
    )

    assert isinstance(
        result["audit"],
        dict,
    )


# ---------------------------------------------------------
# Test 1: Classification
# ---------------------------------------------------------

def test_classification_accepts_valid_target() -> None:
    """Classification must assess a valid target column."""

    request = make_request(
        "classification",
        target_column="churn",
    )

    result = run_assessment(request)

    assert_structured_assessment(
        result,
        "classification",
    )

    assert result["target_column"] == "churn"


# ---------------------------------------------------------
# Test 2: Regression
# ---------------------------------------------------------

def test_regression_accepts_numeric_target() -> None:
    """Regression must assess a numeric target column."""

    request = make_request(
        "regression",
        target_column="monthly_spend",
    )

    result = run_assessment(request)

    assert_structured_assessment(
        result,
        "regression",
    )

    assert result["target_column"] == "monthly_spend"


# ---------------------------------------------------------
# Test 3: Forecasting
# ---------------------------------------------------------

def test_forecasting_accepts_datetime_column() -> None:
    """Forecasting must accept a valid time-series setup."""

    request = make_request(
        "forecasting",
        target_column="monthly_spend",
        datetime_column="event_time",
    )

    result = run_assessment(request)

    assert_structured_assessment(
        result,
        "forecasting",
    )

    assert result["target_column"] == "monthly_spend"


# ---------------------------------------------------------
# Test 4: Clustering
# ---------------------------------------------------------

def test_clustering_accepts_no_target() -> None:
    """Unsupervised clustering must not require a target."""

    request = make_request(
        "clustering",
    )

    result = run_assessment(request)

    assert_structured_assessment(
        result,
        "clustering",
    )

    assert result["target_column"] is None


# ---------------------------------------------------------
# Test 5: Classification target requirement
# ---------------------------------------------------------

def test_classification_requires_target() -> None:
    """Classification requests without a target must fail."""

    with pytest.raises(ValidationError):
        make_request(
            "classification",
        )


# ---------------------------------------------------------
# Test 6: Regression target requirement
# ---------------------------------------------------------

def test_regression_requires_target() -> None:
    """Regression requests without a target must fail."""

    with pytest.raises(ValidationError):
        make_request(
            "regression",
        )


# ---------------------------------------------------------
# Test 7: Forecasting datetime requirement
# ---------------------------------------------------------

def test_forecasting_requires_datetime() -> None:
    """Forecasting requests without a datetime column must fail."""

    with pytest.raises(ValidationError):
        make_request(
            "forecasting",
            target_column="monthly_spend",
        )


# ---------------------------------------------------------
# Test 8: Clustering rejects targets
# ---------------------------------------------------------

def test_clustering_rejects_target_column() -> None:
    """Clustering must reject supervised target configuration."""

    with pytest.raises(ValidationError):
        make_request(
            "clustering",
            target_column="churn",
        )


# ---------------------------------------------------------
# Test 9: Unknown target
# ---------------------------------------------------------

def test_unknown_target_is_rejected() -> None:
    """The engine must not silently assess a nonexistent target."""

    request = make_request(
        "classification",
        target_column="does_not_exist",
    )

    with pytest.raises(
        (ValueError, KeyError),
    ):
        run_assessment(request)


# ---------------------------------------------------------
# Test 10: Dataframe immutability
# ---------------------------------------------------------

@pytest.mark.parametrize(
    "task,target_column,datetime_column",
    [
        ("classification", "churn", None),
        ("regression", "monthly_spend", None),
        ("forecasting", "monthly_spend", "event_time"),
        ("clustering", None, None),
    ],
)
def test_task_assessment_does_not_mutate_dataframe(
    task: str,
    target_column: str | None,
    datetime_column: str | None,
) -> None:
    """All four task assessments must preserve the source dataset."""

    dataframe = sample_dataframe()

    original = dataframe.copy(deep=True)

    request = make_request(
        task,
        target_column=target_column,
        datetime_column=datetime_column,
    )

    run_assessment(
        request,
        dataframe,
    )

    assert_frame_equal(
        dataframe,
        original,
    )
