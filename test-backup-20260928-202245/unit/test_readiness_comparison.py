
"""NUMPA v2 - ML readiness comparison safety tests.

Tests:
- Compatible provisional assessments.
- Different ML tasks.
- Different target columns.
- Different validation strategies.
- Different configuration versions.
- Different scoring weights.
- Different measurement coverage.
- Missing scores.
- Resolved, remaining and new issues.
- Rejection of identical assessment IDs.

These tests use synthetic assessments and do not
claim that a provisional score is a final ML score.
"""

from __future__ import annotations

from copy import deepcopy
from uuid import uuid4

import pytest

from app.schemas.readiness import DEFAULT_DIMENSION_WEIGHTS
from app.services.reports.comparison import compare_assessments


# ---------------------------------------------------------
# Synthetic assessment factory
# ---------------------------------------------------------

def make_assessment(
    *,
    dataset_id: str | None = None,
    score: float = 80.0,
    task: str = "classification",
    target_column: str | None = "churn",
    validation_strategy: str = "stratified_split",
    configuration_version: int = 1,
    issues: list[dict] | None = None,
) -> dict:
    """Create a synthetic provisional assessment."""

    weights = deepcopy(DEFAULT_DIMENSION_WEIGHTS)

    dimensions = [
        {
            "name": name,
            "score": score,
            "weight": weight,
            "weighted_score": round(
                score * weight / 100.0,
                4,
            ),
            "status": "ready",
            "explanation": "Synthetic test dimension.",
            "issues": [],
            "metrics": {},
        }
        for name, weight in weights.items()
    ]

    return {
        "assessment_id": str(uuid4()),
        "dataset_id": (
            dataset_id or str(uuid4())
        ),
        "task": task,
        "target_column": target_column,
        "intended_algorithm": None,
        "validation_strategy": validation_strategy,
        "configuration": {
            "configuration_id": "test-configuration",
            "name": "Comparison Safety Test",
            "version": configuration_version,
        },
        "assessment_status": "provisional",
        "overall_score": None,
        "provisional_score": score,
        "measured_weight_percentage": 100.0,
        "pending_weight_percentage": 0.0,
        "dimensions": dimensions,
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


def assert_not_comparable(result: dict) -> None:
    """Verify that an invalid comparison has no score delta."""

    assert result["comparable"] is False
    assert result["score_change"] is None
    assert result["comparability_reasons"]


# ---------------------------------------------------------
# Test 1: Compatible provisional assessments
# ---------------------------------------------------------

def test_compatible_provisional_assessments() -> None:
    """Compatible provisional scores may have a provisional delta."""

    before = make_assessment(score=70.0)
    after = make_assessment(score=85.0)

    result = compare_assessments(before, after)

    assert result["comparable"] is True
    assert result["score_kind"] == "provisional"
    assert result["before_score"] == pytest.approx(70.0)
    assert result["after_score"] == pytest.approx(85.0)
    assert result["score_change"] == pytest.approx(15.0)

    # The comparison must retain its provisional label.
    assert result["score_kind"] != "final"


# ---------------------------------------------------------
# Test 2: Different ML tasks
# ---------------------------------------------------------

def test_different_tasks_are_not_comparable() -> None:
    """Classification and regression scores must not be mixed."""

    before = make_assessment(
        task="classification",
        target_column="churn",
    )

    after = make_assessment(
        task="regression",
        target_column="revenue",
    )

    result = compare_assessments(before, after)

    assert_not_comparable(result)


# ---------------------------------------------------------
# Test 3: Different target columns
# ---------------------------------------------------------

def test_different_targets_are_not_comparable() -> None:
    """Changing the prediction target changes the assessment."""

    before = make_assessment(target_column="churn")
    after = make_assessment(target_column="retention")

    result = compare_assessments(before, after)

    assert_not_comparable(result)


# ---------------------------------------------------------
# Test 4: Different validation strategies
# ---------------------------------------------------------

def test_different_validation_strategies() -> None:
    """Scores using different validation plans are incompatible."""

    before = make_assessment(
        validation_strategy="stratified_split"
    )

    after = make_assessment(
        validation_strategy="random_split"
    )

    result = compare_assessments(before, after)

    assert_not_comparable(result)


# ---------------------------------------------------------
# Test 5: Different configuration versions
# ---------------------------------------------------------

def test_different_configuration_versions() -> None:
    """A configuration change must be disclosed."""

    before = make_assessment(
        configuration_version=1
    )

    after = make_assessment(
        configuration_version=2
    )

    result = compare_assessments(before, after)

    assert_not_comparable(result)


# ---------------------------------------------------------
# Test 6: Different scoring weights
# ---------------------------------------------------------

def test_different_scoring_weights() -> None:
    """Changed weights must prevent direct score comparison."""

    before = make_assessment()
    after = make_assessment()

    # Move five percentage points between dimensions.
    after["audit"]["configuration_weights"][
        "completeness"
    ] = 15.0

    after["audit"]["configuration_weights"][
        "leakage_validation"
    ] = 20.0

    # Keep the displayed dimension weights aligned
    # with the modified configuration.
    for dimension in after["dimensions"]:
        name = dimension["name"]

        dimension["weight"] = (
            after["audit"]["configuration_weights"][name]
        )

        dimension["weighted_score"] = round(
            dimension["score"]
            * dimension["weight"]
            / 100.0,
            4,
        )

    result = compare_assessments(before, after)

    assert_not_comparable(result)


# ---------------------------------------------------------
# Test 7: Different measurement coverage
# ---------------------------------------------------------

def test_different_measurement_coverage() -> None:
    """Pending checks must prevent an invalid score delta."""

    before = make_assessment()
    after = make_assessment()

    after["dimensions"] = [
        dimension
        for dimension in after["dimensions"]
        if dimension["name"] != "leakage_validation"
    ]

    after["pending_dimensions"] = [
        {
            "dimension": "leakage_validation",
            "reason": "Validation checks remain pending.",
        }
    ]

    after["measured_weight_percentage"] = 85.0
    after["pending_weight_percentage"] = 15.0

    after["audit"]["measured_dimensions"] = [
        dimension["name"]
        for dimension in after["dimensions"]
    ]

    after["audit"]["pending_dimensions"] = [
        "leakage_validation"
    ]

    result = compare_assessments(before, after)

    assert_not_comparable(result)


# ---------------------------------------------------------
# Test 8: Missing provisional score
# ---------------------------------------------------------

def test_missing_score_prevents_comparison() -> None:
    """An unavailable score must not produce a numeric delta."""

    before = make_assessment()
    after = make_assessment()

    after["provisional_score"] = None

    result = compare_assessments(before, after)

    assert_not_comparable(result)


# ---------------------------------------------------------
# Test 9: Issue tracking
# ---------------------------------------------------------

def test_resolved_remaining_and_new_issues() -> None:
    """The comparison must classify issue changes."""

    resolved = {
        "code": "DUPLICATE_ROWS",
        "severity": "medium",
        "dimension": "record_integrity",
        "message": "Duplicate rows were detected.",
        "column": None,
    }

    remaining = {
        "code": "MISSING_VALUES",
        "severity": "medium",
        "dimension": "completeness",
        "message": "Missing values were detected.",
        "column": "age",
    }

    new = {
        "code": "FEATURE_ENCODING_REQUIRED",
        "severity": "low",
        "dimension": "feature_suitability",
        "message": "A feature requires encoding.",
        "column": "city",
    }

    before = make_assessment(
        issues=[resolved, remaining]
    )

    after = make_assessment(
        issues=[remaining, new]
    )

    result = compare_assessments(before, after)

    summary = result["issue_summary"]

    assert summary["resolved_count"] == 1
    assert summary["remaining_count"] == 1
    assert summary["new_count"] == 1

    assert len(result["resolved_issues"]) == 1
    assert len(result["remaining_issues"]) == 1
    assert len(result["new_issues"]) == 1


# ---------------------------------------------------------
# Test 10: Identical assessment IDs
# ---------------------------------------------------------

def test_identical_assessment_ids_are_rejected() -> None:
    """An assessment must not be compared with itself."""

    before = make_assessment()
    after = deepcopy(before)

    with pytest.raises(ValueError):
        compare_assessments(before, after)


# ---------------------------------------------------------
# Test 11: Input dictionaries remain unchanged
# ---------------------------------------------------------

def test_comparison_does_not_mutate_inputs() -> None:
    """Comparison must not modify stored assessments."""

    before = make_assessment(score=70.0)
    after = make_assessment(score=85.0)

    original_before = deepcopy(before)
    original_after = deepcopy(after)

    compare_assessments(before, after)

    assert before == original_before
    assert after == original_after
