
"""NUMPA v2 — Duplicate-cleaning data integrity tests.

Verify that duplicate removal preserves important dataset
properties and produces stable, reproducible results.

Coverage:
1. Column order.
2. Column data types.
3. Missing values.
4. Original row order.
5. Original row indices.
6. Source dataset immutability.
7. Idempotency.
8. Repeated execution.
9. Selected-column matching.
10. Empty-result schema preservation.
11. Mixed numeric and text columns.
12. Duplicate detection without mutation.
"""

from __future__ import annotations

import pandas as pd
import pytest

from pandas.testing import assert_frame_equal

from app.services.cleaning.operations.duplicates import (
    DuplicateConfig,
    clean_duplicates,
    detect_duplicates,
)


def sample_dataframe() -> pd.DataFrame:
    """Build a dataset containing one exact duplicate."""

    return pd.DataFrame(
        {
            "customer_id": pd.Series(
                [101, 102, 102, 103],
                dtype="int64",
            ),
            "city": pd.Series(
                ["Surat", "Rajkot", "Rajkot", "Ahmedabad"],
                dtype="string",
            ),
            "spend": pd.Series(
                [1200.5, 1800.0, 1800.0, 2400.5],
                dtype="float64",
            ),
        }
    )


def test_column_order_is_preserved() -> None:
    """Cleaning must not rearrange dataset columns."""

    dataframe = sample_dataframe()

    cleaned, _ = clean_duplicates(
        dataframe,
        DuplicateConfig(),
    )

    assert cleaned.columns.tolist() == dataframe.columns.tolist()


def test_column_dtypes_are_preserved() -> None:
    """Duplicate removal must preserve original column types."""

    dataframe = sample_dataframe()

    cleaned, _ = clean_duplicates(
        dataframe,
        DuplicateConfig(),
    )

    assert cleaned.dtypes.equals(dataframe.dtypes)


def test_missing_values_are_preserved() -> None:
    """Surviving records must retain their original missing values."""

    dataframe = pd.DataFrame(
        {
            "id": [1, 2, 2, 3],
            "city": ["Surat", None, None, "Rajkot"],
            "spend": [100.0, None, None, 300.0],
        }
    )

    cleaned, report = clean_duplicates(
        dataframe,
        DuplicateConfig(),
    )

    expected = dataframe.iloc[[0, 1, 3]].copy()

    assert_frame_equal(cleaned, expected)
    assert report["rows_removed"] == 1

    # The retained customer must still have missing values.
    assert pd.isna(cleaned.loc[1, "city"])
    assert pd.isna(cleaned.loc[1, "spend"])


def test_original_row_order_is_preserved() -> None:
    """Surviving rows must remain in their original relative order."""

    dataframe = sample_dataframe()

    cleaned, _ = clean_duplicates(
        dataframe,
        DuplicateConfig(strategy="keep_first"),
    )

    assert cleaned["customer_id"].tolist() == [
        101,
        102,
        103,
    ]


def test_original_row_indices_are_preserved() -> None:
    """Cleaning must not silently reset source row indices."""

    dataframe = sample_dataframe()

    dataframe.index = [10, 20, 30, 40]

    cleaned, report = clean_duplicates(
        dataframe,
        DuplicateConfig(strategy="keep_first"),
    )

    assert cleaned.index.tolist() == [10, 20, 40]

    # Audit positions refer to source row positions, not labels.
    assert report["removed_row_positions"] == [2]


@pytest.mark.parametrize(
    "strategy",
    [
        "keep_first",
        "keep_last",
        "remove_all",
    ],
)
def test_source_dataset_is_never_mutated(
    strategy: str,
) -> None:
    """Every supported strategy must protect the uploaded dataset."""

    dataframe = sample_dataframe()
    original = dataframe.copy(deep=True)

    clean_duplicates(
        dataframe,
        DuplicateConfig.model_validate(
            {"strategy": strategy}
        ),
    )

    assert_frame_equal(dataframe, original)


def test_keep_first_is_idempotent() -> None:
    """Repeating keep_first must not remove additional rows."""

    dataframe = sample_dataframe()
    config = DuplicateConfig(strategy="keep_first")

    first_cleaned, first_report = clean_duplicates(
        dataframe,
        config,
    )

    second_cleaned, second_report = clean_duplicates(
        first_cleaned,
        config,
    )

    assert_frame_equal(first_cleaned, second_cleaned)

    assert first_report["rows_removed"] == 1
    assert second_report["rows_removed"] == 0


def test_repeated_execution_is_deterministic() -> None:
    """Identical inputs must produce identical outputs and reports."""

    dataframe = sample_dataframe()
    config = DuplicateConfig(strategy="keep_last")

    first_cleaned, first_report = clean_duplicates(
        dataframe,
        config,
    )

    second_cleaned, second_report = clean_duplicates(
        dataframe,
        config,
    )

    assert_frame_equal(first_cleaned, second_cleaned)
    assert first_report == second_report


def test_selected_column_matching_preserves_other_values() -> None:
    """Columns excluded from matching must retain their actual values."""

    dataframe = pd.DataFrame(
        {
            "customer_id": [101, 101, 102],
            "city": ["Surat", "Surat", "Rajkot"],
            "spend": [100, 200, 300],
        }
    )

    cleaned, report = clean_duplicates(
        dataframe,
        DuplicateConfig(
            strategy="keep_last",
            columns=["customer_id"],
        ),
    )

    expected = dataframe.iloc[[1, 2]].copy()

    assert_frame_equal(cleaned, expected)

    # The later customer's spend must not be overwritten.
    assert cleaned["spend"].tolist() == [200, 300]

    assert report["selected_columns"] == ["customer_id"]


def test_empty_result_preserves_dataset_schema() -> None:
    """Removing every duplicate-group row must preserve column types."""

    dataframe = pd.DataFrame(
        {
            "customer_id": pd.Series(
                [101, 101],
                dtype="int64",
            ),
            "city": pd.Series(
                ["Surat", "Surat"],
                dtype="string",
            ),
        }
    )

    cleaned, report = clean_duplicates(
        dataframe,
        DuplicateConfig(strategy="remove_all"),
    )

    assert cleaned.empty
    assert cleaned.columns.tolist() == dataframe.columns.tolist()
    assert cleaned.dtypes.equals(dataframe.dtypes)

    assert report["rows_before"] == 2
    assert report["rows_after"] == 0
    assert report["rows_removed"] == 2


def test_mixed_numeric_and_text_columns() -> None:
    """Matching must work across numeric and text columns."""

    dataframe = pd.DataFrame(
        {
            "id": [1, 1, 2],
            "name": ["A", "A", "B"],
            "score": [10.5, 10.5, 20.0],
        }
    )

    cleaned, report = clean_duplicates(
        dataframe,
        DuplicateConfig(),
    )

    expected = dataframe.iloc[[0, 2]].copy()

    assert_frame_equal(cleaned, expected)
    assert report["rows_removed"] == 1


def test_duplicate_detection_does_not_mutate_source() -> None:
    """Detection alone must not change any dataset values."""

    dataframe = sample_dataframe()
    original = dataframe.copy(deep=True)

    report = detect_duplicates(
        dataframe,
        DuplicateConfig(),
    )

    assert report["extra_duplicate_rows"] == 1
    assert report["rows_in_duplicate_groups"] == 2

    assert_frame_equal(dataframe, original)
