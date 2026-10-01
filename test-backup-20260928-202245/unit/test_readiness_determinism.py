
"""NUMPA v2 - ML readiness determinism and audit tests.

Verifies that:
- Repeated assessments produce consistent scoring results.
- Assessment identifiers are unique.
- The source dataframe is never modified.
- Assessment metadata identifies the selected ML task.
- Every measured dimension has an explainable score.
- Weighted scores agree with dimension scores and weights.
- Pending checks remain visible.
- Changing the input dataset does not reuse stale results.

This file tests the existing readiness engine directly.
"""

from __future__ import annotations

from copy import deepcopy

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from app.schemas.readiness import (
    ReadinessAssessmentRequest,
)
from app.services.readiness.engine import (
    assess_readiness,
)


# ---------------------------------------------------------
# Test dataset and assessment helpers
# ---------------------------------------------------------

def sample_dataframe() -> pd.DataFrame:
    """Build a deterministic classification dataset."""

    return pd.DataFrame(
        {
            "age": [
                20 + index % 40
                for index in range(160)
            ],
            "visits": [
                1 + index % 12
                for index in range(160)
            ],
            "monthly_spend": [
                100.0 + index * 2.5
                for index in range(160)
            ],
            "city": [
                ("Surat", "Rajkot", "Ahmedabad", "Vadodara")[
                    index % 4
                ]
                for index in range(160)
            ],
            "churn": [
                index % 2
                for index in range(160)
            ],
        }
    )


def assessment_request() -> ReadinessAssessmentRequest:
    """Create a valid classification assessment request."""

    return ReadinessAssessmentRequest(
        dataset_id="determinism-test-dataset",
        task="classification",
        target_column="churn",
        validation_strategy="stratified_split",
    )


def assess(
    dataframe: pd.DataFrame,
) -> dict:
    """Run the existing readiness engine."""

    return assess_readiness(
        dataframe,
        assessment_request(),
    )


def scoring_snapshot(
    assessment: dict,
) -> dict:
    """Select deterministic fields, excluding generated metadata."""

    return {
        "task": assessment["task"],
        "target_column": assessment["target_column"],
        "validation_strategy": assessment[
            "validation_strategy"
        ],
        "assessment_status": assessment[
            "assessment_status"
        ],
        "overall_score": assessment["overall_score"],
        "provisional_score": assessment[
            "provisional_score"
        ],
        "measured_weight_percentage": assessment[
            "measured_weight_percentage"
        ],
        "pending_weight_percentage": assessment[
            "pending_weight_percentage"
        ],
        "dimensions": deepcopy(
            assessment["dimensions"]
        ),
        "pending_dimensions": deepcopy(
            assessment["pending_dimensions"]
        ),
        "issues": deepcopy(
            assessment["issues"]
        ),
        "blocking_issues": deepcopy(
            assessment["blocking_issues"]
        ),
        "recommendations": deepcopy(
            assessment["recommendations"]
        ),
    }


# ---------------------------------------------------------
# Test 1: Repeated scoring is deterministic
# ---------------------------------------------------------

def test_repeated_assessments_have_same_scoring() -> None:
    """Identical inputs must produce identical scoring results."""

    dataframe = sample_dataframe()

    first = assess(
        dataframe.copy(deep=True)
    )

    second = assess(
        dataframe.copy(deep=True)
    )

    assert scoring_snapshot(first) == scoring_snapshot(
        second
    )


# ---------------------------------------------------------
# Test 2: Assessment IDs are unique
# ---------------------------------------------------------

def test_assessment_ids_are_unique() -> None:
    """Each assessment must have its own identifier."""

    dataframe = sample_dataframe()

    first = assess(dataframe)
    second = assess(dataframe)

    assert first["assessment_id"]
    assert second["assessment_id"]

    assert (
        first["assessment_id"]
        != second["assessment_id"]
    )


# ---------------------------------------------------------
# Test 3: Source data remains unchanged
# ---------------------------------------------------------

def test_assessment_does_not_mutate_dataframe() -> None:
    """Readiness analysis must not modify uploaded data."""

    dataframe = sample_dataframe()
    snapshot = dataframe.copy(deep=True)

    assess(dataframe)

    assert_frame_equal(
        dataframe,
        snapshot,
    )


# ---------------------------------------------------------
# Test 4: Assessment identifies its task
# ---------------------------------------------------------

def test_assessment_contains_task_metadata() -> None:
    """An assessment must retain its task and target."""

    result = assess(
        sample_dataframe()
    )

    assert result["dataset_id"] == (
        "determinism-test-dataset"
    )

    assert result["task"] == "classification"

    assert result["target_column"] == "churn"

    assert result["validation_strategy"] == (
        "stratified_split"
    )

    assert result["configuration"]["name"]
    assert result["configuration"]["version"] >= 1

    assert result["audit"]


# ---------------------------------------------------------
# Test 5: Dimension scores are explainable
# ---------------------------------------------------------

def test_measured_dimensions_are_explainable() -> None:
    """Measured dimensions must expose scores and explanations."""

    result = assess(
        sample_dataframe()
    )

    assert result["dimensions"]

    for dimension in result["dimensions"]:
        assert dimension["name"]

        assert 0.0 <= dimension["score"] <= 100.0

        assert 0.0 <= dimension["weight"] <= 100.0

        assert isinstance(
            dimension["explanation"],
            str,
        )

        assert dimension["explanation"].strip()


# ---------------------------------------------------------
# Test 6: Weighted scores are consistent
# ---------------------------------------------------------

def test_dimension_weighted_scores_are_consistent() -> None:
    """A dimension's contribution must match its weight."""

    result = assess(
        sample_dataframe()
    )

    for dimension in result["dimensions"]:
        expected = (
            dimension["score"]
            * dimension["weight"]
            / 100.0
        )

        assert dimension["weighted_score"] == (
            pytest.approx(expected, abs=0.02)
        )


# ---------------------------------------------------------
# Test 7: Pending checks are disclosed
# ---------------------------------------------------------

def test_provisional_assessment_discloses_pending_checks() -> None:
    """An incomplete assessment must not publish a final score."""

    result = assess(
        sample_dataframe()
    )

    assert result["assessment_status"] == (
        "provisional"
    )

    assert result["overall_score"] is None

    assert isinstance(
        result["pending_dimensions"],
        list,
    )

    assert result["measured_weight_percentage"] >= 0

    assert result["pending_weight_percentage"] >= 0

    assert (
        result["measured_weight_percentage"]
        + result["pending_weight_percentage"]
    ) == pytest.approx(100.0, abs=0.02)


# ---------------------------------------------------------
# Test 8: Changed input is assessed independently
# ---------------------------------------------------------

def test_changed_dataset_gets_new_assessment() -> None:
    """The engine must assess the dataframe it actually receives."""

    original = sample_dataframe()

    changed = original.copy(deep=True)

    # Introduce substantial missingness in a feature
    # without changing the prediction target.
    changed.loc[:119, "monthly_spend"] = None

    before = assess(original)
    after = assess(changed)

    assert (
        before["assessment_id"]
        != after["assessment_id"]
    )

    before_completeness = next(
        dimension
        for dimension in before["dimensions"]
        if dimension["name"] == "completeness"
    )

    after_completeness = next(
        dimension
        for dimension in after["dimensions"]
        if dimension["name"] == "completeness"
    )

    assert (
        after_completeness["score"]
        < before_completeness["score"]
    )
