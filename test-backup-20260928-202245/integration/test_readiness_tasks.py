
"""NUMPA v2 - Task-specific ML readiness integration tests.

Covers:
- Classification.
- Regression.
- Time-series forecasting.
- Clustering.
- Required task-specific request fields.
- Missing dataset and column errors.
- Repeatability of measured dimension scores.

Assessments may remain provisional while validation
or leakage checks are pending.
"""

from __future__ import annotations

from uuid import uuid4

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.storage.datasets import (
    delete_dataset,
    save_dataset,
)


# ---------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------

@pytest.fixture
def client():
    """Provide a FastAPI test client."""

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def task_dataset():
    """Create one dataset suitable for testing four ML tasks."""

    # Use deterministic data to make test failures reproducible.
    rng = np.random.default_rng(42)
    row_count = 180

    dataframe = pd.DataFrame(
        {
            "event_time": pd.date_range(
                start="2025-01-01",
                periods=row_count,
                freq="D",
            ).astype(str),
            "customer_id": np.arange(
                1,
                row_count + 1,
            ),
            "age": rng.integers(
                18,
                65,
                size=row_count,
            ),
            "visits": rng.integers(
                1,
                20,
                size=row_count,
            ),
            "monthly_spend": np.round(
                rng.uniform(
                    500,
                    5000,
                    size=row_count,
                ),
                2,
            ),
            "churn": [
                index % 2
                for index in range(row_count)
            ],
        }
    )

    metadata = save_dataset(
        dataframe,
        name=f"task_tests_{uuid4().hex[:8]}.csv",
    )

    dataset_id = metadata["dataset_id"]

    yield dataset_id

    # Remove only the dataset created by this fixture.
    try:
        delete_dataset(dataset_id)
    except (KeyError, LookupError, ValueError):
        pass


# ---------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------

def assess(
    client: TestClient,
    dataset_id: str,
    task: str,
    **options,
) -> dict:
    """Submit a task-specific assessment request."""

    payload = {
        "dataset_id": dataset_id,
        "task": task,
        **options,
    }

    response = client.post(
        "/api/v1/readiness/assess",
        json=payload,
    )

    assert response.status_code == 200, response.text

    result = response.json()

    assert result["assessment_id"]
    assert result["dataset_id"] == dataset_id
    assert result["task"] == task

    return result


def assert_valid_assessment(
    assessment: dict,
) -> None:
    """Verify the shared readiness response contract."""

    assert assessment["assessment_status"] in (
        "provisional",
        "final",
    )

    assert isinstance(
        assessment["dimensions"],
        list,
    )

    assert isinstance(
        assessment["issues"],
        list,
    )

    assert isinstance(
        assessment["task_diagnostics"],
        dict,
    )

    assert isinstance(
        assessment["audit"],
        dict,
    )

    # A provisional assessment must not expose
    # an unverified final score.
    if assessment["assessment_status"] == "provisional":
        assert assessment["overall_score"] is None

    # All measured dimensions must have valid scores.
    for dimension in assessment["dimensions"]:
        assert 0 <= dimension["score"] <= 100
        assert 0 <= dimension["weight"] <= 100


# ---------------------------------------------------------
# Test 1: Classification
# ---------------------------------------------------------

def test_classification_assessment(
    client: TestClient,
    task_dataset: str,
) -> None:
    """Assess a dataset with a binary classification target."""

    result = assess(
        client,
        task_dataset,
        "classification",
        target_column="churn",
        excluded_columns=[
            "customer_id",
            "event_time",
        ],
        validation_strategy="stratified_split",
    )

    assert result["target_column"] == "churn"

    assert_valid_assessment(result)


# ---------------------------------------------------------
# Test 2: Regression
# ---------------------------------------------------------

def test_regression_assessment(
    client: TestClient,
    task_dataset: str,
) -> None:
    """Assess a dataset with a numeric regression target."""

    result = assess(
        client,
        task_dataset,
        "regression",
        target_column="monthly_spend",
        excluded_columns=[
            "customer_id",
            "event_time",
        ],
        validation_strategy="random_split",
    )

    assert result["target_column"] == "monthly_spend"

    assert_valid_assessment(result)


# ---------------------------------------------------------
# Test 3: Time-series forecasting
# ---------------------------------------------------------

def test_forecasting_assessment(
    client: TestClient,
    task_dataset: str,
) -> None:
    """Assess chronological data for forecasting."""

    result = assess(
        client,
        task_dataset,
        "forecasting",
        target_column="monthly_spend",
        datetime_column="event_time",
        excluded_columns=[
            "customer_id",
        ],
        validation_strategy="time_series_split",
    )

    assert result["target_column"] == "monthly_spend"

    assert result["validation_strategy"] == (
        "time_series_split"
    )

    assert_valid_assessment(result)


# ---------------------------------------------------------
# Test 4: Clustering
# ---------------------------------------------------------

def test_clustering_assessment(
    client: TestClient,
    task_dataset: str,
) -> None:
    """Assess numeric features without a target column."""

    result = assess(
        client,
        task_dataset,
        "clustering",
        excluded_columns=[
            "customer_id",
            "event_time",
            "churn",
        ],
        validation_strategy="none",
    )

    assert result["target_column"] is None

    assert_valid_assessment(result)


# ---------------------------------------------------------
# Test 5: Classification requires a target
# ---------------------------------------------------------

def test_classification_requires_target(
    client: TestClient,
    task_dataset: str,
) -> None:
    """Reject classification without a target column."""

    response = client.post(
        "/api/v1/readiness/assess",
        json={
            "dataset_id": task_dataset,
            "task": "classification",
        },
    )

    assert response.status_code == 422


# ---------------------------------------------------------
# Test 6: Regression requires a target
# ---------------------------------------------------------

def test_regression_requires_target(
    client: TestClient,
    task_dataset: str,
) -> None:
    """Reject regression without a target column."""

    response = client.post(
        "/api/v1/readiness/assess",
        json={
            "dataset_id": task_dataset,
            "task": "regression",
        },
    )

    assert response.status_code == 422


# ---------------------------------------------------------
# Test 7: Forecasting requires a datetime column
# ---------------------------------------------------------

def test_forecasting_requires_datetime(
    client: TestClient,
    task_dataset: str,
) -> None:
    """Reject forecasting without a datetime column."""

    response = client.post(
        "/api/v1/readiness/assess",
        json={
            "dataset_id": task_dataset,
            "task": "forecasting",
            "target_column": "monthly_spend",
        },
    )

    assert response.status_code == 422


# ---------------------------------------------------------
# Test 8: Clustering forbids a target
# ---------------------------------------------------------

def test_clustering_rejects_target(
    client: TestClient,
    task_dataset: str,
) -> None:
    """Clustering requests must not specify a target."""

    response = client.post(
        "/api/v1/readiness/assess",
        json={
            "dataset_id": task_dataset,
            "task": "clustering",
            "target_column": "churn",
        },
    )

    assert response.status_code == 422


# ---------------------------------------------------------
# Test 9: Unknown dataset
# ---------------------------------------------------------

def test_unknown_dataset_returns_404(
    client: TestClient,
) -> None:
    """An assessment cannot use an unknown dataset."""

    response = client.post(
        "/api/v1/readiness/assess",
        json={
            "dataset_id": str(uuid4()),
            "task": "classification",
            "target_column": "churn",
        },
    )

    assert response.status_code == 404


# ---------------------------------------------------------
# Test 10: Unknown target column
# ---------------------------------------------------------

def test_unknown_target_is_rejected(
    client: TestClient,
    task_dataset: str,
) -> None:
    """Reject a target column absent from the dataset."""

    response = client.post(
        "/api/v1/readiness/assess",
        json={
            "dataset_id": task_dataset,
            "task": "classification",
            "target_column": "does_not_exist",
        },
    )

    assert response.status_code == 422, response.text


# ---------------------------------------------------------
# Test 11: Repeated assessments
# ---------------------------------------------------------

def test_identical_assessments_have_consistent_scores(
    client: TestClient,
    task_dataset: str,
) -> None:
    """Repeated assessments must give the same measured scores."""

    options = {
        "target_column": "churn",
        "excluded_columns": [
            "customer_id",
            "event_time",
        ],
        "validation_strategy": "stratified_split",
    }

    first = assess(
        client,
        task_dataset,
        "classification",
        **options,
    )

    second = assess(
        client,
        task_dataset,
        "classification",
        **options,
    )

    # Separate assessments should have distinct IDs.
    assert (
        first["assessment_id"]
        != second["assessment_id"]
    )

    # IDs and timestamps may differ, but measured
    # scores should be deterministic.
    first_scores = {
        item["name"]: item["score"]
        for item in first["dimensions"]
    }

    second_scores = {
        item["name"]: item["score"]
        for item in second["dimensions"]
    }

    assert first_scores == second_scores

    assert (
        first["provisional_score"]
        == second["provisional_score"]
    )

    assert (
        first["pending_dimensions"]
        == second["pending_dimensions"]
    )
