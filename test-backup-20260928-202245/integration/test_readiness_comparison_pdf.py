
"""NUMPA v2 - Readiness comparison PDF integration tests.

Tests:
- PDF generation for compatible assessments.
- PDF generation for incompatible assessments.
- PDF generation when issues change.
- PDF generation from two assessment records.
- Rejection of malformed comparison data.
- PDF export through the readiness API.

The PDF tests validate file structure and generation.
They do not rely on PDF text-extraction libraries.
"""

from __future__ import annotations

from copy import deepcopy
from uuid import uuid4

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.readiness import DEFAULT_DIMENSION_WEIGHTS
from app.services.reports.comparison import compare_assessments
from app.services.reports.comparison_pdf import (
    generate_comparison_pdf_from_assessments,
    generate_readiness_comparison_pdf,
)
from app.services.storage.datasets import (
    delete_dataset,
    save_dataset,
)


# ---------------------------------------------------------
# Fixtures
# ---------------------------------------------------------

@pytest.fixture
def client():
    """Create an API test client."""

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def dataset_id():
    """Store a small deterministic classification dataset."""

    dataframe = pd.DataFrame(
        {
            "age": [20 + index % 40 for index in range(120)],
            "visits": [1 + index % 12 for index in range(120)],
            "spend": [
                100.0 + index * 2.5
                for index in range(120)
            ],
            "churn": [index % 2 for index in range(120)],
        }
    )

    metadata = save_dataset(
        dataframe,
        name=f"comparison_pdf_{uuid4().hex[:8]}.csv",
    )

    stored_id = metadata["dataset_id"]

    yield stored_id

    # Remove the test dataset without touching user data.
    try:
        delete_dataset(stored_id)
    except (KeyError, LookupError, ValueError):
        pass


# ---------------------------------------------------------
# Synthetic assessment helpers
# ---------------------------------------------------------

def make_assessment(
    *,
    score: float = 75.0,
    configuration_version: int = 1,
    issues: list[dict] | None = None,
) -> dict:
    """Build a synthetic provisional assessment."""

    weights = deepcopy(DEFAULT_DIMENSION_WEIGHTS)

    return {
        "assessment_id": str(uuid4()),
        "dataset_id": str(uuid4()),
        "task": "classification",
        "target_column": "churn",
        "intended_algorithm": None,
        "validation_strategy": "stratified_split",
        "configuration": {
            "configuration_id": "comparison-pdf-test",
            "name": "PDF Test Configuration",
            "version": configuration_version,
        },
        "assessment_status": "provisional",
        "overall_score": None,
        "provisional_score": score,
        "measured_weight_percentage": 100.0,
        "pending_weight_percentage": 0.0,
        "dimensions": [
            {
                "name": name,
                "score": score,
                "weight": weight,
                "weighted_score": round(
                    score * weight / 100.0,
                    4,
                ),
                "status": "ready",
                "explanation": "Synthetic PDF test result.",
                "issues": [],
                "metrics": {},
            }
            for name, weight in weights.items()
        ],
        "pending_dimensions": [],
        "issues": deepcopy(issues or []),
        "blocking_issues": [],
        "recommendations": [],
        "audit": {
            "configuration_weights": weights,
            "measured_dimensions": list(weights),
            "pending_dimensions": [],
        },
    }


def assert_pdf(pdf: bytes) -> None:
    """Verify that a PDF generator returned PDF bytes."""

    assert isinstance(pdf, bytes)

    # A PDF must begin with the standard file signature.
    assert pdf.startswith(b"%PDF-")

    # This is a basic integrity check, not full PDF parsing.
    assert len(pdf) > 500

    # ReportLab normally includes an EOF marker.
    assert b"%%EOF" in pdf[-1024:]


def create_api_assessment(
    client: TestClient,
    dataset_id: str,
) -> dict:
    """Create a real assessment through the readiness API."""

    response = client.post(
        "/api/v1/readiness/assess",
        json={
            "dataset_id": dataset_id,
            "task": "classification",
            "target_column": "churn",
            "validation_strategy": "stratified_split",
        },
    )

    assert response.status_code == 200, response.text

    result = response.json()

    assert result["assessment_id"]

    return result


# ---------------------------------------------------------
# Test 1: Compatible comparison PDF
# ---------------------------------------------------------

def test_compatible_comparison_generates_pdf() -> None:
    """Generate a PDF for compatible provisional scores."""

    before = make_assessment(score=70.0)
    after = make_assessment(score=85.0)

    comparison = compare_assessments(before, after)

    assert comparison["comparable"] is True
    assert comparison["score_kind"] == "provisional"

    pdf = generate_readiness_comparison_pdf(
        comparison
    )

    assert_pdf(pdf)


# ---------------------------------------------------------
# Test 2: Incompatible comparison PDF
# ---------------------------------------------------------

def test_incompatible_comparison_generates_pdf() -> None:
    """Generate a report without claiming a score improvement."""

    before = make_assessment(
        score=70.0,
        configuration_version=1,
    )

    after = make_assessment(
        score=90.0,
        configuration_version=2,
    )

    comparison = compare_assessments(before, after)

    assert comparison["comparable"] is False
    assert comparison["score_change"] is None
    assert comparison["comparability_reasons"]

    pdf = generate_readiness_comparison_pdf(
        comparison
    )

    assert_pdf(pdf)


# ---------------------------------------------------------
# Test 3: Comparison with changing issues
# ---------------------------------------------------------

def test_issue_changes_generate_pdf() -> None:
    """Generate a PDF with resolved and newly detected issues."""

    duplicate_issue = {
        "code": "DUPLICATE_ROWS",
        "severity": "medium",
        "dimension": "record_integrity",
        "message": "Duplicate rows detected.",
        "column": None,
    }

    missing_issue = {
        "code": "MISSING_VALUES",
        "severity": "medium",
        "dimension": "completeness",
        "message": "Missing values detected.",
        "column": "age",
    }

    before = make_assessment(
        score=65.0,
        issues=[duplicate_issue],
    )

    after = make_assessment(
        score=80.0,
        issues=[missing_issue],
    )

    comparison = compare_assessments(before, after)

    assert len(comparison["resolved_issues"]) == 1
    assert len(comparison["new_issues"]) == 1

    pdf = generate_readiness_comparison_pdf(
        comparison
    )

    assert_pdf(pdf)


# ---------------------------------------------------------
# Test 4: Generate directly from assessments
# ---------------------------------------------------------

def test_generate_pdf_from_assessment_records() -> None:
    """Generate the comparison and PDF in one operation."""

    before = make_assessment(score=60.0)
    after = make_assessment(score=75.0)

    pdf = generate_comparison_pdf_from_assessments(
        before,
        after,
    )

    assert_pdf(pdf)


# ---------------------------------------------------------
# Test 5: Invalid comparison input
# ---------------------------------------------------------

def test_invalid_comparison_is_rejected() -> None:
    """Reject input that lacks essential comparison data."""

    with pytest.raises(
        (TypeError, ValueError, KeyError)
    ):
        generate_readiness_comparison_pdf(
            {
                "comparable": True,
                "score_change": 20.0,
            }
        )


# ---------------------------------------------------------
# Test 6: API PDF export
# ---------------------------------------------------------

def test_comparison_pdf_api(
    client: TestClient,
    dataset_id: str,
) -> None:
    """Export a PDF using two saved API assessments."""

    before = create_api_assessment(
        client,
        dataset_id,
    )

    after = create_api_assessment(
        client,
        dataset_id,
    )

    response = client.post(
        "/api/v1/readiness/compare/report/pdf",
        json={
            "before_assessment_id": before["assessment_id"],
            "after_assessment_id": after["assessment_id"],
        },
    )

    assert response.status_code == 200, response.text

    assert response.headers["content-type"].startswith(
        "application/pdf"
    )

    assert_pdf(response.content)


# ---------------------------------------------------------
# Test 7: Unknown assessment IDs
# ---------------------------------------------------------

def test_comparison_pdf_rejects_unknown_assessments(
    client: TestClient,
) -> None:
    """Unknown assessment IDs must not produce a PDF."""

    response = client.post(
        "/api/v1/readiness/compare/report/pdf",
        json={
            "before_assessment_id": str(uuid4()),
            "after_assessment_id": str(uuid4()),
        },
    )

    assert response.status_code == 404


# ---------------------------------------------------------
# Test 8: Original assessments remain unchanged
# ---------------------------------------------------------

def test_pdf_generation_does_not_modify_assessments() -> None:
    """PDF generation must not mutate saved assessment data."""

    before = make_assessment(score=60.0)
    after = make_assessment(score=85.0)

    original_before = deepcopy(before)
    original_after = deepcopy(after)

    generate_comparison_pdf_from_assessments(
        before,
        after,
    )

    assert before == original_before
    assert after == original_after
