
"""NUMPA v2 - ML readiness workflow integration tests.

Tests the core workflow:
1. Assess the original dataset.
2. Clean the dataset.
3. Assess the cleaned dataset.
4. Compare the two assessments.
5. Download the comparison PDF.
6. Reject invalid assessment references.

Current readiness assessments may be provisional.
Tests must not assume a final readiness score exists.
"""

from __future__ import annotations

from uuid import uuid4

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app

from app.services.storage.datasets import (
    delete_dataset,
    get_dataset,
    save_dataset,
)

from app.services.storage.executions import (
    ExecutionNotFoundError,
    delete_execution,
)


# ---------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------

@pytest.fixture
def client():
    """Create a FastAPI test client."""

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def workflow_dataset():
    """Create a classification dataset for the workflow.

    The data includes one duplicated row so that
    duplicate removal has a measurable effect.
    """

    rows = []

    # Use enough records for basic classification
    # checks without relying on a large test file.
    for index in range(120):
        rows.append(
            {
                "customer_id": index + 1,
                "age": 20 + (index % 40),
                "monthly_spend": float(
                    500 + (index * 17) % 3000
                ),
                "visits": 1 + (index % 12),
                "city": [
                    "Surat",
                    "Ahmedabad",
                    "Vadodara",
                ][index % 3],
                "churn": index % 2,
            }
        )

    # Add one exact duplicate record.
    rows.append(rows[0].copy())

    dataframe = pd.DataFrame(rows)

    metadata = save_dataset(
        dataframe,
        name=f"readiness_{uuid4().hex[:8]}.csv",
    )

    resources = {
        "original_id": metadata["dataset_id"],
        "cleaned_ids": [],
        "execution_ids": [],
        "dataframe": dataframe,
    }

    yield resources

    # Remove resources created during the test.
    for execution_id in resources["execution_ids"]:
        try:
            delete_execution(execution_id)
        except ExecutionNotFoundError:
            pass

    for cleaned_id in resources["cleaned_ids"]:
        try:
            delete_dataset(cleaned_id)
        except (KeyError, LookupError, ValueError):
            pass

    try:
        delete_dataset(resources["original_id"])
    except (KeyError, LookupError, ValueError):
        pass


# ---------------------------------------------------------
# Shared API helpers
# ---------------------------------------------------------

def assess_classification(
    client: TestClient,
    dataset_id: str,
) -> dict:
    """Create an ML readiness assessment."""

    response = client.post(
        "/api/v1/readiness/assess",
        json={
            "dataset_id": dataset_id,
            "task": "classification",
            "target_column": "churn",
            "validation_strategy": "stratified_split",
            "excluded_columns": ["customer_id"],
        },
    )

    assert response.status_code == 200, response.text

    assessment = response.json()

    assert assessment["assessment_id"]
    assert assessment["dataset_id"] == dataset_id
    assert assessment["task"] == "classification"

    return assessment


def execute_duplicate_cleaning(
    client: TestClient,
    resources: dict,
) -> dict:
    """Clean the original dataset through the API."""

    response = client.post(
        "/api/v1/cleaning/execute",
        json={
            "dataset_id": resources["original_id"],
            "confirmed": True,
            "pipeline": {
                "name": "Readiness Workflow Cleaning",
                "steps": [
                    {
                        "step_id": "remove-duplicates",
                        "operation": "duplicates",
                        "config": {},
                    }
                ],
            },
        },
    )

    assert response.status_code == 200, response.text

    result = response.json()

    assert result["status"] == "completed"

    # Register resources immediately for cleanup.
    resources["cleaned_ids"].append(
        result["cleaned_dataset_id"]
    )

    resources["execution_ids"].append(
        result["execution_id"]
    )

    return result


def compare_readiness(
    client: TestClient,
    before_id: str,
    after_id: str,
) -> dict:
    """Compare two saved readiness assessments."""

    response = client.post(
        "/api/v1/readiness/compare",
        json={
            "before_assessment_id": before_id,
            "after_assessment_id": after_id,
        },
    )

    assert response.status_code == 200, response.text

    return response.json()


# ---------------------------------------------------------
# Test 1: Initial assessment
# ---------------------------------------------------------

def test_initial_readiness_assessment(
    client: TestClient,
    workflow_dataset: dict,
) -> None:
    """The original dataset must support assessment."""

    assessment = assess_classification(
        client,
        workflow_dataset["original_id"],
    )

    assert assessment["task"] == "classification"

    assert assessment["target_column"] == "churn"

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

    assert "audit" in assessment

    # A provisional score must remain distinct
    # from a final readiness score.
    if assessment["assessment_status"] == "provisional":
        assert assessment["overall_score"] is None


# ---------------------------------------------------------
# Test 2: Assessment after cleaning
# ---------------------------------------------------------

def test_cleaned_dataset_can_be_reassessed(
    client: TestClient,
    workflow_dataset: dict,
) -> None:
    """Cleaning must produce a separately assessable dataset."""

    before = assess_classification(
        client,
        workflow_dataset["original_id"],
    )

    execution = execute_duplicate_cleaning(
        client,
        workflow_dataset,
    )

    cleaned_id = execution["cleaned_dataset_id"]

    # The cleaned dataset must exist independently.
    cleaned = get_dataset(cleaned_id)

    assert isinstance(cleaned, pd.DataFrame)

    assert len(cleaned) == 120

    after = assess_classification(
        client,
        cleaned_id,
    )

    assert (
        before["assessment_id"]
        != after["assessment_id"]
    )

    assert before["dataset_id"] != after["dataset_id"]

    assert after["dataset_id"] == cleaned_id


# ---------------------------------------------------------
# Test 3: Before-and-after comparison
# ---------------------------------------------------------

def test_readiness_comparison(
    client: TestClient,
    workflow_dataset: dict,
) -> None:
    """Compare the initial and cleaned dataset assessments."""

    before = assess_classification(
        client,
        workflow_dataset["original_id"],
    )

    execution = execute_duplicate_cleaning(
        client,
        workflow_dataset,
    )

    after = assess_classification(
        client,
        execution["cleaned_dataset_id"],
    )

    comparison = compare_readiness(
        client,
        before["assessment_id"],
        after["assessment_id"],
    )

    assert (
        comparison["before_assessment_id"]
        == before["assessment_id"]
    )

    assert (
        comparison["after_assessment_id"]
        == after["assessment_id"]
    )

    assert (
        comparison["before_dataset_id"]
        == workflow_dataset["original_id"]
    )

    assert (
        comparison["after_dataset_id"]
        == execution["cleaned_dataset_id"]
    )

    assert comparison["task"] == "classification"

    assert isinstance(
        comparison["dimension_changes"],
        list,
    )

    assert isinstance(
        comparison["comparability_reasons"],
        list,
    )

    # A score change may be calculated only if
    # the comparison engine establishes comparability.
    if not comparison["comparable"]:
        assert comparison["score_change"] is None


# ---------------------------------------------------------
# Test 4: Comparison PDF
# ---------------------------------------------------------

def test_readiness_comparison_pdf(
    client: TestClient,
    workflow_dataset: dict,
) -> None:
    """Download a PDF for the before-and-after comparison."""

    before = assess_classification(
        client,
        workflow_dataset["original_id"],
    )

    execution = execute_duplicate_cleaning(
        client,
        workflow_dataset,
    )

    after = assess_classification(
        client,
        execution["cleaned_dataset_id"],
    )

    response = client.post(
        "/api/v1/readiness/compare/report/pdf",
        json={
            "before_assessment_id": before["assessment_id"],
            "after_assessment_id": after["assessment_id"],
        },
    )

    assert response.status_code == 200, response.text

    assert response.content.startswith(b"%PDF-")

    assert (
        response.headers["content-type"]
        == "application/pdf"
    )

    assert "attachment" in (
        response.headers["content-disposition"]
    )


# ---------------------------------------------------------
# Test 5: Identical assessment IDs
# ---------------------------------------------------------

def test_comparison_rejects_identical_ids(
    client: TestClient,
    workflow_dataset: dict,
) -> None:
    """An assessment cannot be compared with itself."""

    assessment = assess_classification(
        client,
        workflow_dataset["original_id"],
    )

    assessment_id = assessment["assessment_id"]

    response = client.post(
        "/api/v1/readiness/compare",
        json={
            "before_assessment_id": assessment_id,
            "after_assessment_id": assessment_id,
        },
    )

    assert response.status_code == 422


# ---------------------------------------------------------
# Test 6: Missing assessment
# ---------------------------------------------------------

def test_missing_assessment_returns_404(
    client: TestClient,
) -> None:
    """An unknown assessment must not return another report."""

    missing_id = str(uuid4())

    response = client.get(
        f"/api/v1/readiness/assessments/{missing_id}"
    )

    assert response.status_code == 404


# ---------------------------------------------------------
# Test 7: Readiness assessment PDF
# ---------------------------------------------------------

def test_initial_assessment_pdf(
    client: TestClient,
    workflow_dataset: dict,
) -> None:
    """The initial assessment must have a downloadable PDF."""

    assessment = assess_classification(
        client,
        workflow_dataset["original_id"],
    )

    assessment_id = assessment["assessment_id"]

    response = client.get(
        "/api/v1/readiness/assessments/"
        f"{assessment_id}/report/pdf"
    )

    assert response.status_code == 200, response.text

    assert response.content.startswith(b"%PDF-")

    assert (
        response.headers["content-type"]
        == "application/pdf"
    )
