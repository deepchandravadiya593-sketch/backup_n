
"""NUMPA v2 — Duplicate-cleaning audit report tests.

Verify that duplicate-removal reports are accurate, deterministic,
and consistent with the actual cleaned dataset.

The tests use the public duplicate-cleaning functions and do not
modify production code.
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
    """Return a dataset with two distinct duplicate groups."""

    return pd.DataFrame(
        {
            "customer_id": [101, 102, 102, 103, 104, 104],
            "city": [
                "Surat",
                "Rajkot",
                "Rajkot",
                "Ahmedabad",
                "Vadodara",
                "Vadodara",
            ],
            "spend": [100, 200, 200, 300, 400, 400],
        }
    )


def test_detection_report_counts() -> None:
    """Detection must distinguish extra rows from duplicate groups."""

    dataframe = sample_dataframe()

    report = detect_duplicates(
        dataframe,
        DuplicateConfig(),
    )

    assert report["total_rows"] == 6
    assert report["extra_duplicate_rows"] == 2
    assert report["rows_in_duplicate_groups"] == 4
    assert report["unique_rows"] == 4


def test_keep_first_report_is_accurate() -> None:
    """The report must identify rows removed by keep_first."""

    dataframe = sample_dataframe()

    cleaned, report = clean_duplicates(
        dataframe,
        DuplicateConfig(strategy="keep_first"),
    )

    assert report["rows_before"] == 6
    assert report["rows_after"] == 4
    assert report["rows_removed"] == 2
    assert report["removed_row_positions"] == [2, 5]
    assert report["strategy"] == "keep_first"

    assert_frame_equal(
        cleaned,
        dataframe.iloc[[0, 1, 3, 4]].copy(),
    )


def test_keep_last_report_is_accurate() -> None:
    """The report must identify rows removed by keep_last."""

    dataframe = sample_dataframe()

    cleaned, report = clean_duplicates(
        dataframe,
        DuplicateConfig(strategy="keep_last"),
    )

    assert report["rows_before"] == 6
    assert report["rows_after"] == 4
    assert report["rows_removed"] == 2
    assert report["removed_row_positions"] == [1, 4]
    assert report["strategy"] == "keep_last"

    assert_frame_equal(
        cleaned,
        dataframe.iloc[[0, 2, 3, 5]].copy(),
    )


def test_remove_all_report_is_accurate() -> None:
    """The report must include every removed duplicate-group row."""

    dataframe = sample_dataframe()

    cleaned, report = clean_duplicates(
        dataframe,
        DuplicateConfig(strategy="remove_all"),
    )

    assert report["rows_before"] == 6
    assert report["rows_after"] == 2
    assert report["rows_removed"] == 4

    assert report["removed_row_positions"] == [
        1,
        2,
        4,
        5,
    ]

    assert_frame_equal(
        cleaned,
        dataframe.iloc[[0, 3]].copy(),
    )


@pytest.mark.parametrize(
    "strategy",
    [
        "keep_first",
        "keep_last",
        "remove_all",
    ],
)
def test_report_row_counts_are_consistent(
    strategy: str,
) -> None:
    """Every strategy must report mathematically consistent counts."""

    dataframe = sample_dataframe()

    cleaned, report = clean_duplicates(
        dataframe,
        DuplicateConfig.model_validate(
            {"strategy": strategy}
        ),
    )

    assert (
        report["rows_before"] - report["rows_removed"]
        == report["rows_after"]
    )

    assert report["rows_after"] == len(cleaned)

    assert (
        len(report["removed_row_positions"])
        == report["rows_removed"]
    )


def test_report_contains_required_audit_fields() -> None:
    """Reports must contain the fields needed for an audit trail."""

    dataframe = sample_dataframe()

    _, report = clean_duplicates(
        dataframe,
        DuplicateConfig(),
    )

    required_fields = {
        "operation",
        "strategy",
        "selected_columns",
        "normalize_text",
        "rows_before",
        "rows_after",
        "rows_removed",
        "extra_duplicate_rows_before",
        "rows_in_duplicate_groups_before",
        "removed_row_positions",
        "status",
    }

    assert required_fields.issubset(report.keys())

    assert report["operation"] == "remove_duplicates"
    assert report["status"] == "completed"


def test_selected_columns_are_recorded() -> None:
    """The report must identify which columns determined duplicates."""

    dataframe = pd.DataFrame(
        {
            "customer_id": [101, 101, 102],
            "spend": [100, 200, 300],
        }
    )

    config = DuplicateConfig(
        columns=["customer_id"],
        strategy="keep_first",
    )

    cleaned, report = clean_duplicates(
        dataframe,
        config,
    )

    assert report["selected_columns"] == ["customer_id"]
    assert report["removed_row_positions"] == [1]

    assert cleaned["spend"].tolist() == [100, 300]


def test_normalization_setting_is_recorded() -> None:
    """The audit report must record whether text was normalized."""

    dataframe = pd.DataFrame(
        {
            "city": [" Surat ", "surat", "Rajkot"],
        }
    )

    cleaned, report = clean_duplicates(
        dataframe,
        DuplicateConfig(
            normalize_text=True,
        ),
    )

    assert report["normalize_text"] is True
    assert report["rows_removed"] == 1

    # Normalization is used for matching only.
    assert cleaned["city"].tolist() == [
        " Surat ",
        "Rajkot",
    ]


def test_report_is_deterministic() -> None:
    """Repeated execution must produce identical audit reports."""

    dataframe = sample_dataframe()
    config = DuplicateConfig(strategy="keep_first")

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


def test_no_duplicates_report() -> None:
    """A clean dataset must produce a successful zero-removal report."""

    dataframe = pd.DataFrame(
        {
            "customer_id": [101, 102, 103],
            "spend": [100, 200, 300],
        }
    )

    cleaned, report = clean_duplicates(
        dataframe,
        DuplicateConfig(),
    )

    assert_frame_equal(cleaned, dataframe)

    assert report["rows_removed"] == 0
    assert report["removed_row_positions"] == []
    assert report["status"] == "completed"


def test_report_positions_with_repeated_indices() -> None:
    """Removed positions must remain correct when index labels repeat."""

    dataframe = sample_dataframe()

    dataframe.index = [10, 10, 20, 20, 30, 30]

    cleaned, report = clean_duplicates(
        dataframe,
        DuplicateConfig(strategy="keep_first"),
    )

    assert report["removed_row_positions"] == [2, 5]

    assert_frame_equal(
        cleaned,
        dataframe.iloc[[0, 1, 3, 4]].copy(),
    )


def test_report_generation_preserves_source() -> None:
    """Producing an audit report must never mutate the input dataset."""

    dataframe = sample_dataframe()
    original = dataframe.copy(deep=True)

    clean_duplicates(
        dataframe,
        DuplicateConfig(strategy="remove_all"),
    )

    assert_frame_equal(dataframe, original)
