
"""NUMPA v2 — Duplicate-cleaning edge-case tests.

Verify duplicate detection and removal independently of the
cleaning pipeline.

Coverage:
1. Keep the first duplicate.
2. Keep the last duplicate.
3. Remove all members of duplicate groups.
4. Detect duplicates using selected columns.
5. Normalize text when explicitly enabled.
6. Preserve case-sensitive matching by default.
7. Preserve the original DataFrame.
8. Handle repeated DataFrame indices.
9. Handle datasets without duplicate rows.
10. Handle empty datasets with defined columns.
11. Reject unknown selected columns.
12. Reject duplicate column names in the configuration.
13. Reject unsupported removal strategies.
14. Reject unknown configuration fields.
"""

from __future__ import annotations

import pandas as pd
import pytest

from pandas.testing import assert_frame_equal
from pydantic import ValidationError

from app.services.cleaning.operations.duplicates import (
    DuplicateConfig,
    clean_duplicates,
    detect_duplicates,
)


def sample_dataframe() -> pd.DataFrame:
    """Return a dataset containing one duplicate row."""

    return pd.DataFrame(
        {
            "customer_id": [101, 102, 102, 103],
            "city": [
                "Surat",
                "Rajkot",
                "Rajkot",
                "Ahmedabad",
            ],
            "spend": [1200, 1800, 1800, 2400],
        }
    )


def test_keep_first_duplicate_strategy() -> None:
    """The first occurrence must survive."""

    dataframe = sample_dataframe()

    config = DuplicateConfig(
        strategy="keep_first",
    )

    cleaned, report = clean_duplicates(
        dataframe,
        config,
    )

    expected = dataframe.iloc[
        [0, 1, 3]
    ].copy()

    assert_frame_equal(
        cleaned,
        expected,
    )

    assert report["rows_removed"] == 1
    assert report["removed_row_positions"] == [2]
    assert report["strategy"] == "keep_first"


def test_keep_last_duplicate_strategy() -> None:
    """The last occurrence must survive."""

    dataframe = sample_dataframe()

    config = DuplicateConfig(
        strategy="keep_last",
    )

    cleaned, report = clean_duplicates(
        dataframe,
        config,
    )

    expected = dataframe.iloc[
        [0, 2, 3]
    ].copy()

    assert_frame_equal(
        cleaned,
        expected,
    )

    assert report["rows_removed"] == 1
    assert report["removed_row_positions"] == [1]


def test_remove_all_duplicate_strategy() -> None:
    """Every member of a duplicate group must be removed."""

    dataframe = sample_dataframe()

    config = DuplicateConfig(
        strategy="remove_all",
    )

    cleaned, report = clean_duplicates(
        dataframe,
        config,
    )

    expected = dataframe.iloc[
        [0, 3]
    ].copy()

    assert_frame_equal(
        cleaned,
        expected,
    )

    assert report["rows_removed"] == 2
    assert report["removed_row_positions"] == [1, 2]


def test_duplicate_detection_with_selected_columns() -> None:
    """Only approved columns should determine duplicate groups."""

    dataframe = pd.DataFrame(
        {
            "customer_id": [101, 101, 102],
            "city": ["Surat", "Surat", "Rajkot"],
            "spend": [1200, 1500, 1800],
        }
    )

    config = DuplicateConfig(
        strategy="keep_first",
        columns=["customer_id", "city"],
    )

    detection = detect_duplicates(
        dataframe,
        config,
    )

    cleaned, report = clean_duplicates(
        dataframe,
        config,
    )

    assert detection["extra_duplicate_rows"] == 1
    assert detection["rows_in_duplicate_groups"] == 2

    assert cleaned["spend"].tolist() == [
        1200,
        1800,
    ]

    assert report["selected_columns"] == [
        "customer_id",
        "city",
    ]


def test_text_normalization_when_enabled() -> None:
    """Whitespace and case differences should be ignored when approved."""

    dataframe = pd.DataFrame(
        {
            "city": [
                " Surat ",
                "surat",
                "RAJKOT",
                "Rajkot ",
            ],
            "sales": [100, 100, 200, 200],
        }
    )

    config = DuplicateConfig(
        strategy="keep_first",
        normalize_text=True,
    )

    cleaned, report = clean_duplicates(
        dataframe,
        config,
    )

    assert len(cleaned) == 2

    # Cleaning must not rewrite the surviving original values.
    assert cleaned["city"].tolist() == [
        " Surat ",
        "RAJKOT",
    ]

    assert report["rows_removed"] == 2
    assert report["normalize_text"] is True


def test_text_matching_is_case_sensitive_by_default() -> None:
    """Without normalization, differently cased text remains distinct."""

    dataframe = pd.DataFrame(
        {
            "city": [
                "Surat",
                "surat",
                "SURAT",
            ]
        }
    )

    config = DuplicateConfig()

    cleaned, report = clean_duplicates(
        dataframe,
        config,
    )

    assert_frame_equal(
        cleaned,
        dataframe,
    )

    assert report["rows_removed"] == 0


def test_duplicate_cleaning_does_not_mutate_input() -> None:
    """Cleaning must preserve the uploaded source DataFrame."""

    dataframe = sample_dataframe()
    original = dataframe.copy(deep=True)

    config = DuplicateConfig(
        strategy="remove_all",
    )

    clean_duplicates(
        dataframe,
        config,
    )

    assert_frame_equal(
        dataframe,
        original,
    )


def test_duplicate_cleaning_with_repeated_indices() -> None:
    """Row selection must use positions rather than index labels."""

    dataframe = sample_dataframe()

    dataframe.index = [
        10,
        10,
        10,
        20,
    ]

    config = DuplicateConfig(
        strategy="keep_first",
    )

    cleaned, report = clean_duplicates(
        dataframe,
        config,
    )

    expected = dataframe.iloc[
        [0, 1, 3]
    ].copy()

    assert_frame_equal(
        cleaned,
        expected,
    )

    assert report["removed_row_positions"] == [2]


def test_dataset_without_duplicates() -> None:
    """A dataset without duplicates should remain unchanged."""

    dataframe = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "value": [10, 20, 30],
        }
    )

    config = DuplicateConfig()

    detection = detect_duplicates(
        dataframe,
        config,
    )

    cleaned, report = clean_duplicates(
        dataframe,
        config,
    )

    assert_frame_equal(
        cleaned,
        dataframe,
    )

    assert detection["extra_duplicate_rows"] == 0
    assert report["rows_removed"] == 0
    assert report["status"] == "completed"


def test_empty_dataset_with_defined_columns() -> None:
    """An empty dataset with a valid schema should be handled safely."""

    dataframe = pd.DataFrame(
        {
            "id": pd.Series(dtype="int64"),
            "city": pd.Series(dtype="object"),
        }
    )

    config = DuplicateConfig()

    detection = detect_duplicates(
        dataframe,
        config,
    )

    cleaned, report = clean_duplicates(
        dataframe,
        config,
    )

    assert_frame_equal(
        cleaned,
        dataframe,
    )

    assert detection["total_rows"] == 0
    assert detection["extra_duplicate_rows"] == 0
    assert report["rows_removed"] == 0


def test_unknown_selected_column_is_rejected() -> None:
    """Unknown column names must not be silently ignored."""

    dataframe = sample_dataframe()

    config = DuplicateConfig(
        columns=["nonexistent_column"],
    )

    with pytest.raises(
        ValueError,
        match="Unknown columns",
    ):
        detect_duplicates(
            dataframe,
            config,
        )


def test_repeated_configuration_columns_are_rejected() -> None:
    """A column cannot appear twice in the matching configuration."""

    with pytest.raises(
        ValidationError,
    ):
        DuplicateConfig(
            columns=[
                "customer_id",
                "customer_id",
            ],
        )


def test_unsupported_duplicate_strategy_is_rejected() -> None:
    """Unsupported strategies must fail configuration validation."""

    with pytest.raises(
        ValidationError,
    ):
        DuplicateConfig.model_validate(
            {
                "strategy": "unsupported_strategy",
            }
        )


def test_unknown_configuration_field_is_rejected() -> None:
    """Misspelled or obsolete fields must not be silently accepted."""

    with pytest.raises(
        ValidationError,
    ):
        DuplicateConfig.model_validate(
            {
                "keep": "first",
            }
        )
