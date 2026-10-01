
"""NUMPA v2 - ML readiness scoring safety tests.

Tests the final scoring gate, including:
- Incomplete validation evidence.
- Incomplete leakage evidence.
- Critical leakage findings.
- Missing scoring dimensions.
- Invalid configuration weights.
- Deterministic scoring.

These tests use synthetic scoring inputs.
They do not claim to validate the statistical quality
of the readiness algorithm.
"""

from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import ValidationError

from app.schemas.readiness import (
    DEFAULT_DIMENSION_WEIGHTS,
    ReadinessConfiguration,
    ReadinessDimensionResult,
    ReadinessIssue,
    ScoringDimensionConfig,
)

from app.services.readiness.scoring import (
    FinalScoringRequest,
    LeakageEvidence,
    ValidationEvidence,
    evaluate_final_scoring,
)


# ---------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------

@pytest.fixture
def configuration() -> ReadinessConfiguration:
    """Build NUMPA's default seven-dimension configuration."""

    return ReadinessConfiguration(
        name="Scoring Safety Test",
        version=1,
    )


@pytest.fixture
def dimensions() -> list[ReadinessDimensionResult]:
    """Create synthetic measured results for all dimensions."""

    return [
        ReadinessDimensionResult(
            name=name,
            score=85.0,
            weight=weight,
            weighted_score=round(
                85.0 * weight / 100.0,
                4,
            ),
            status="ready",
            explanation=(
                "Synthetic result for scoring-gate testing."
            ),
            issues=[],
            metrics={},
        )
        for name, weight in DEFAULT_DIMENSION_WEIGHTS.items()
    ]


def complete_validation() -> ValidationEvidence:
    """Create explicit evidence for validation checks."""

    return ValidationEvidence(
        completed=True,
        strategy="stratified_split",
        actual_split_checked=True,
        split_overlap_checked=True,
        preprocessing_boundary_checked=True,
        feature_availability_checked=True,
        temporal_order_checked=True,
        group_isolation_checked=True,
        unsupervised_protocol_checked=True,
        notes=[
            "Synthetic validation evidence for unit tests."
        ],
    )


def complete_leakage() -> LeakageEvidence:
    """Create explicit evidence for leakage checks."""

    return LeakageEvidence(
        completed=True,
        target_leakage_checked=True,
        split_leakage_checked=True,
        preprocessing_leakage_checked=True,
        prediction_time_availability_checked=True,
        notes=[
            "Synthetic leakage evidence for unit tests."
        ],
    )


def build_request(
    configuration: ReadinessConfiguration,
    dimensions: list[ReadinessDimensionResult],
    *,
    validation: ValidationEvidence | None = None,
    leakage: LeakageEvidence | None = None,
    issues: list[ReadinessIssue] | None = None,
) -> FinalScoringRequest:
    """Build a classification scoring request."""

    return FinalScoringRequest(
        task="classification",
        configuration=configuration,
        dimensions=deepcopy(dimensions),
        issues=deepcopy(issues or []),
        validation=(
            validation
            if validation is not None
            else complete_validation()
        ),
        leakage=(
            leakage
            if leakage is not None
            else complete_leakage()
        ),
    )


def assert_final_score_blocked(result: dict) -> None:
    """Check the core safety requirement."""

    assert result["publishable"] is False
    assert result["final_score"] is None
    assert result["blocking_reasons"]


# ---------------------------------------------------------
# Test 1: Incomplete validation
# ---------------------------------------------------------

def test_incomplete_validation_blocks_final_score(
    configuration,
    dimensions,
) -> None:
    """A final score requires completed validation."""

    validation = complete_validation().model_copy(
        update={
            "completed": False,
            "actual_split_checked": False,
        }
    )

    request = build_request(
        configuration,
        dimensions,
        validation=validation,
    )

    result = evaluate_final_scoring(request)

    assert_final_score_blocked(result)


# ---------------------------------------------------------
# Test 2: Incomplete leakage checks
# ---------------------------------------------------------

def test_incomplete_leakage_checks_block_final_score(
    configuration,
    dimensions,
) -> None:
    """A final score requires completed leakage checks."""

    leakage = complete_leakage().model_copy(
        update={
            "completed": False,
            "target_leakage_checked": False,
        }
    )

    request = build_request(
        configuration,
        dimensions,
        leakage=leakage,
    )

    result = evaluate_final_scoring(request)

    assert_final_score_blocked(result)


# ---------------------------------------------------------
# Test 3: Critical target leakage
# ---------------------------------------------------------

def test_critical_target_leakage_blocks_final_score(
    configuration,
    dimensions,
) -> None:
    """A critical leakage finding prevents publication."""

    critical_issue = ReadinessIssue(
        code="TARGET_DUPLICATED_IN_FEATURE",
        severity="critical",
        dimension="leakage_validation",
        message=(
            "A feature duplicates the prediction target."
        ),
        column="target_copy",
        recommendation=(
            "Remove the leaking feature and reassess."
        ),
        evidence={
            "synthetic_test": True,
        },
    )

    request = build_request(
        configuration,
        dimensions,
        issues=[critical_issue],
    )

    result = evaluate_final_scoring(request)

    assert_final_score_blocked(result)

    assert result["critical_issue_count"] >= 1


# ---------------------------------------------------------
# Test 4: Missing dimension
# ---------------------------------------------------------

def test_missing_dimension_blocks_final_score(
    configuration,
    dimensions,
) -> None:
    """All enabled dimensions must be measured."""

    incomplete_dimensions = [
        dimension
        for dimension in dimensions
        if dimension.name != "leakage_validation"
    ]

    request = build_request(
        configuration,
        incomplete_dimensions,
    )

    result = evaluate_final_scoring(request)

    assert_final_score_blocked(result)

    assert result["measured_weight_percentage"] < 100


# ---------------------------------------------------------
# Test 5: Invalid configuration weights
# ---------------------------------------------------------

def test_invalid_configuration_weights_rejected() -> None:
    """Enabled dimension weights must total 100 percent."""

    invalid_dimensions = {
        name: ScoringDimensionConfig(
            weight=weight,
        )
        for name, weight in DEFAULT_DIMENSION_WEIGHTS.items()
    }

    # Increase one weight without adjusting the others.
    invalid_dimensions["completeness"] = (
        ScoringDimensionConfig(weight=25.0)
    )

    with pytest.raises(ValidationError):
        ReadinessConfiguration(
            name="Invalid Weight Test",
            dimensions=invalid_dimensions,
        )


# ---------------------------------------------------------
# Test 6: Complete evidence
# ---------------------------------------------------------

def test_complete_evidence_allows_scoring(
    configuration,
    dimensions,
) -> None:
    """Complete evidence permits scoring without critical issues.

    This test checks the scoring gate only.
    The evidence is synthetic, not a real validation run.
    """

    request = build_request(
        configuration,
        dimensions,
    )

    result = evaluate_final_scoring(request)

    assert result["publishable"] is True
    assert result["final_score"] is not None

    assert 0 <= result["final_score"] <= 100

    assert result["measured_weight_percentage"] == pytest.approx(
        100.0
    )

    assert result["critical_issue_count"] == 0


# ---------------------------------------------------------
# Test 7: Deterministic scoring
# ---------------------------------------------------------

def test_scoring_is_deterministic(
    configuration,
    dimensions,
) -> None:
    """Identical inputs must produce identical score results."""

    request = build_request(
        configuration,
        dimensions,
    )

    first = evaluate_final_scoring(request)

    second = evaluate_final_scoring(request)

    assert first["publishable"] == second["publishable"]

    assert first["final_score"] == second["final_score"]

    assert (
        first["measured_weight_percentage"]
        == second["measured_weight_percentage"]
    )


# ---------------------------------------------------------
# Test 8: Duplicate dimension names
# ---------------------------------------------------------

def test_duplicate_dimensions_rejected(
    configuration,
    dimensions,
) -> None:
    """The same dimension cannot be counted twice."""

    duplicate_dimensions = deepcopy(dimensions)

    duplicate_dimensions.append(
        deepcopy(dimensions[0])
    )

    with pytest.raises(ValidationError):
        build_request(
            configuration,
            duplicate_dimensions,
        )
