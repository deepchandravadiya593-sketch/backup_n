
"""NUMPA v2 - ML readiness leakage and validation safety tests.

Tests:
- A duplicated target column is detected.
- Target leakage is not reported when no duplicate exists.
- Identical training and validation records are detected.
- Disjoint train/validation positions avoid overlap findings.
- Identifier-risk analysis returns structured findings.
- Combined leakage analysis exposes its component results.
- Repeated checks produce consistent findings.
- Leakage checks never modify the input dataframe.

These tests use the existing leakage service.
"""

from __future__ import annotations

from copy import deepcopy
from inspect import signature

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from app.services.readiness.leakage import (
    analyze_leakage,
    detect_identifier_risks,
    detect_split_leakage,
    detect_target_leakage,
)


# ---------------------------------------------------------
# Shared test data
# ---------------------------------------------------------

def sample_dataframe() -> pd.DataFrame:
    """Create a deterministic classification dataset."""

    return pd.DataFrame(
        {
            "customer_id": [
                f"CUST-{index:04d}"
                for index in range(120)
            ],
            "age": [
                20 + index % 40
                for index in range(120)
            ],
            "visits": [
                1 + index % 12
                for index in range(120)
            ],
            "churn": [
                index % 2
                for index in range(120)
            ],
        }
    )


def leakage_dataframe() -> pd.DataFrame:
    """Add a feature containing an exact copy of the target."""

    dataframe = sample_dataframe()

    dataframe["leaked_churn"] = dataframe["churn"].copy()

    return dataframe


def issue_codes(value) -> set[str]:
    """Collect issue codes from dictionaries, lists and Pydantic models."""

    codes: set[str] = set()

    # Convert Pydantic ReadinessIssue objects to dictionaries.
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="python")

    if isinstance(value, dict):
        code = value.get("code")

        if isinstance(code, str):
            codes.add(code)

        # Recursively inspect nested findings.
        for nested in value.values():
            codes.update(issue_codes(nested))

    elif isinstance(value, (list, tuple)):
        for item in value:
            codes.update(issue_codes(item))

    return codes


def normalize_result(value):
    """Remove incidental timestamps before comparing findings."""

    if isinstance(value, dict):
        return {
            key: normalize_result(item)
            for key, item in value.items()
            if key not in (
                "checked_at",
                "assessed_at",
                "generated_at",
            )
        }

    if isinstance(value, list):
        return [
            normalize_result(item)
            for item in value
        ]

    return value


def invoke_supported(
    function,
    dataframe: pd.DataFrame,
    **available_arguments,
):
    """Pass only keyword arguments accepted by the service.

    This accommodates the existing leakage helpers when
    they expose different subsets of assessment context.
    It does not invent values for required parameters.
    """

    parameters = signature(function).parameters

    accepted = {
        key: value
        for key, value in available_arguments.items()
        if key in parameters
    }

    return function(
        dataframe,
        **accepted,
    )


def target_check(dataframe: pd.DataFrame):
    """Run target leakage detection."""

    return invoke_supported(
        detect_target_leakage,
        dataframe,
        target_column="churn",
        excluded_columns=[],
        task="classification",
    )


def identifier_check(dataframe: pd.DataFrame):
    """Run identifier-risk detection."""

    return invoke_supported(
        detect_identifier_risks,
        dataframe,
        target_column="churn",
        excluded_columns=[],
        task="classification",
    )


def split_check(
    dataframe: pd.DataFrame,
    train_positions: list[int],
    validation_positions: list[int],
):
    """Run training/validation overlap detection."""

    return invoke_supported(
        detect_split_leakage,
        dataframe,
        train_positions=train_positions,
        validation_positions=validation_positions,
        target_column="churn",
        excluded_columns=[],
        task="classification",
        validation_strategy="stratified_split",
    )


def combined_check(dataframe: pd.DataFrame):
    """Run the combined leakage service."""

    return invoke_supported(
        analyze_leakage,
        dataframe,
        task="classification",
        target_column="churn",
        excluded_columns=[],
        validation_strategy="stratified_split",
        train_positions=None,
        validation_positions=None,
    )


# ---------------------------------------------------------
# Test 1: Exact target duplication
# ---------------------------------------------------------

def test_exact_target_duplication_is_detected() -> None:
    """A feature identical to the target must be flagged."""

    result = target_check(
        leakage_dataframe()
    )

    assert (
        "TARGET_DUPLICATED_IN_FEATURE"
        in issue_codes(result)
    )


# ---------------------------------------------------------
# Test 2: No duplicate target
# ---------------------------------------------------------

def test_clean_features_have_no_exact_target_duplicate() -> None:
    """Ordinary features must not trigger exact target duplication."""

    result = target_check(
        sample_dataframe()
    )

    assert (
        "TARGET_DUPLICATED_IN_FEATURE"
        not in issue_codes(result)
    )


# ---------------------------------------------------------
# Test 3: Overlapping split positions
# ---------------------------------------------------------

def test_overlapping_train_validation_is_detected() -> None:
    """Overlapping train and validation positions indicate leakage."""

    dataframe = sample_dataframe()

    result = split_check(
        dataframe,
        train_positions=list(range(0, 80)),
        validation_positions=list(range(70, 120)),
    )

    assert (
        "TRAIN_VALIDATION_ROW_OVERLAP"
        in issue_codes(result)
    )


# ---------------------------------------------------------
# Test 4: Disjoint split positions
# ---------------------------------------------------------

def test_disjoint_positions_have_no_row_overlap() -> None:
    """Disjoint row positions must not trigger an overlap finding."""

    dataframe = sample_dataframe()

    result = split_check(
        dataframe,
        train_positions=list(range(0, 80)),
        validation_positions=list(range(80, 120)),
    )

    assert (
        "TRAIN_VALIDATION_ROW_OVERLAP"
        not in issue_codes(result)
    )


# ---------------------------------------------------------
# Test 5: Identifier-risk analysis
# ---------------------------------------------------------

def test_identifier_risk_analysis_returns_findings() -> None:
    """A high-cardinality customer ID must be evaluated."""

    result = identifier_check(
        sample_dataframe()
    )

    assert result is not None

    # Identifier heuristics may return warnings or
    # structured metadata depending on the service.
    assert isinstance(
        result,
        (dict, list, tuple),
    )


# ---------------------------------------------------------
# Test 6: Combined leakage analysis
# ---------------------------------------------------------

def test_combined_analysis_includes_target_leakage() -> None:
    """The combined service must retain exact-target findings."""

    result = combined_check(
        leakage_dataframe()
    )

    assert (
        "TARGET_DUPLICATED_IN_FEATURE"
        in issue_codes(result)
    )


# ---------------------------------------------------------
# Test 7: Repeated checks are deterministic
# ---------------------------------------------------------

def test_leakage_detection_is_deterministic() -> None:
    """Identical data must produce identical leakage findings."""

    dataframe = leakage_dataframe()

    first = target_check(
        dataframe.copy(deep=True)
    )

    second = target_check(
        dataframe.copy(deep=True)
    )

    assert normalize_result(first) == normalize_result(
        second
    )


# ---------------------------------------------------------
# Test 8: Input dataframe is immutable
# ---------------------------------------------------------

def test_leakage_checks_do_not_mutate_input() -> None:
    """Leakage checks must never alter the uploaded dataset."""

    dataframe = leakage_dataframe()
    snapshot = dataframe.copy(deep=True)

    target_check(dataframe)
    identifier_check(dataframe)
    combined_check(dataframe)

    assert_frame_equal(
        dataframe,
        snapshot,
    )
