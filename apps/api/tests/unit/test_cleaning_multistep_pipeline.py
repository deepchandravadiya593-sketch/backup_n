
"""NUMPA v2 — Multi-step cleaning pipeline tests.

Verify that multiple cleaning steps execute in the approved order.

Coverage:
- Sequential duplicate-removal operations.
- Different matching columns in successive steps.
- Disabled steps.
- Pipeline execution order.
- Repeated execution and deterministic results.
- Source dataset protection.
- Independent executions.
- Empty pipelines.
- Pipelines that remove every row.
"""

from __future__ import annotations

import pandas as pd

from pandas.testing import assert_frame_equal

from app.services.cleaning.pipeline import (
    CleaningPipelineConfig,
    execute_pipeline,
)


def sample_dataframe() -> pd.DataFrame:
    """Create a dataset containing different duplicate groups."""

    return pd.DataFrame(
        {
            "customer_id": [101, 101, 102, 103, 103],
            "city": [
                "Surat",
                "Surat",
                "Rajkot",
                "Ahmedabad",
                "Ahmedabad",
            ],
            "spend": [
                100,
                200,
                300,
                400,
                400,
            ],
        }
    )


def duplicate_step(
    step_id: str,
    *,
    strategy: str = "keep_first",
    columns: list[str] | None = None,
    enabled: bool = True,
) -> dict:
    """Create a duplicate-removal step using the pipeline contract."""

    parameters = {
        "strategy": strategy,
    }

    if columns is not None:
        parameters["columns"] = columns

    return {
        "step_id": step_id,
        "operation": "duplicates",
        "parameters": parameters,
        "enabled": enabled,
    }


def make_pipeline(
    steps: list[dict],
) -> CleaningPipelineConfig:
    """Build a validated pipeline with ordered steps."""

    return CleaningPipelineConfig.model_validate(
        {
            "name": "Multi-Step Cleaning Test",
            "steps": steps,
            "stop_on_error": True,
        }
    )


def run_pipeline(
    dataframe: pd.DataFrame,
    steps: list[dict],
) -> pd.DataFrame:
    """Execute a pipeline and return its cleaned dataset."""

    pipeline = make_pipeline(steps)

    cleaned, report, artifacts = execute_pipeline(
        dataframe,
        pipeline,
    )

    return cleaned


def test_two_sequential_duplicate_steps() -> None:
    """The second step must receive the first step's output."""

    dataframe = sample_dataframe()

    cleaned = run_pipeline(
        dataframe,
        [
            duplicate_step(
                "first",
                columns=["customer_id", "city", "spend"],
            ),
            duplicate_step(
                "second",
                columns=["customer_id"],
            ),
        ],
    )

    # Step 1 removes the exact duplicate at position 4.
    # Step 2 removes the second occurrence of customer 101.
    expected = dataframe.iloc[
        [0, 2, 3]
    ].copy()

    assert_frame_equal(cleaned, expected)


def test_reversing_steps_can_change_results() -> None:
    """Verify that changing step order can change surviving source rows."""

    dataframe = sample_dataframe()

    # First remove repeated customers, keeping their last record.
    first_order = run_pipeline(
        dataframe,
        [
            duplicate_step(
                "by-customer",
                columns=["customer_id"],
                strategy="keep_last",
            ),
            duplicate_step(
                "exact-match",
                strategy="keep_first",
            ),
        ],
    )

    # First remove exact duplicates, then keep the last
    # remaining record for each customer.
    second_order = run_pipeline(
        dataframe,
        [
            duplicate_step(
                "exact-match",
                strategy="keep_first",
            ),
            duplicate_step(
                "by-customer",
                columns=["customer_id"],
                strategy="keep_last",
            ),
        ],
    )

    # Both pipelines produce the same customer and spending values.
    assert first_order["spend"].tolist() == [200, 300, 400]
    assert second_order["spend"].tolist() == [200, 300, 400]

    # However, the surviving source rows are different.
    assert first_order.index.tolist() == [1, 2, 4]
    assert second_order.index.tolist() == [1, 2, 3]

    # The row values match when original indices are excluded.
    assert_frame_equal(
        first_order.reset_index(drop=True),
        second_order.reset_index(drop=True),
    )

    # The original uploaded dataset remains unchanged.
    assert_frame_equal(
        dataframe,
        sample_dataframe(),
    )


def test_disabled_first_step_is_skipped() -> None:
    """A disabled first step must not affect the next operation."""

    dataframe = sample_dataframe()

    cleaned = run_pipeline(
        dataframe,
        [
            duplicate_step(
                "disabled-exact-match",
                enabled=False,
            ),
            duplicate_step(
                "customer-deduplication",
                columns=["customer_id"],
            ),
        ],
    )

    expected = dataframe.iloc[
        [0, 2, 3]
    ].copy()

    assert_frame_equal(cleaned, expected)


def test_all_disabled_steps_preserve_dataset() -> None:
    """A pipeline containing only disabled steps is a no-op."""

    dataframe = sample_dataframe()
    original = dataframe.copy(deep=True)

    cleaned = run_pipeline(
        dataframe,
        [
            duplicate_step(
                "disabled-one",
                enabled=False,
            ),
            duplicate_step(
                "disabled-two",
                strategy="remove_all",
                enabled=False,
            ),
        ],
    )

    assert_frame_equal(cleaned, original)
    assert_frame_equal(dataframe, original)


def test_empty_pipeline_preserves_dataset() -> None:
    """An empty pipeline must not modify or remove any records."""

    dataframe = sample_dataframe()
    original = dataframe.copy(deep=True)

    cleaned = run_pipeline(
        dataframe,
        [],
    )

    assert_frame_equal(cleaned, original)
    assert_frame_equal(dataframe, original)


def test_repeated_execution_is_deterministic() -> None:
    """Identical inputs and configurations must produce identical results."""

    dataframe = sample_dataframe()

    steps = [
        duplicate_step(
            "exact-match",
        ),
        duplicate_step(
            "by-customer",
            columns=["customer_id"],
        ),
    ]

    first = run_pipeline(dataframe, steps)
    second = run_pipeline(dataframe, steps)

    assert_frame_equal(first, second)


def test_multistep_pipeline_preserves_source() -> None:
    """Multiple cleaning operations must never mutate uploaded data."""

    dataframe = sample_dataframe()
    original = dataframe.copy(deep=True)

    run_pipeline(
        dataframe,
        [
            duplicate_step(
                "exact-match",
            ),
            duplicate_step(
                "by-customer",
                columns=["customer_id"],
            ),
        ],
    )

    assert_frame_equal(dataframe, original)


def test_pipeline_can_remove_all_duplicate_groups() -> None:
    """A later step may remove every member of a duplicate group."""

    dataframe = sample_dataframe()

    cleaned = run_pipeline(
        dataframe,
        [
            duplicate_step(
                "remove-exact-duplicates",
                strategy="keep_first",
            ),
            duplicate_step(
                "remove-repeated-customers",
                strategy="remove_all",
                columns=["customer_id"],
            ),
        ],
    )

    # After exact deduplication, customer 101 still appears twice.
    # The second step removes both occurrences.
    expected = dataframe.iloc[
        [2, 3]
    ].copy()

    assert_frame_equal(cleaned, expected)


def test_independent_pipelines_do_not_share_state() -> None:
    """One pipeline's execution must not affect another pipeline."""

    dataframe = sample_dataframe()

    first = run_pipeline(
        dataframe,
        [
            duplicate_step(
                "keep-first",
                columns=["customer_id"],
                strategy="keep_first",
            ),
        ],
    )

    second = run_pipeline(
        dataframe,
        [
            duplicate_step(
                "keep-last",
                columns=["customer_id"],
                strategy="keep_last",
            ),
        ],
    )

    assert first["spend"].tolist() == [
        100,
        300,
        400,
    ]

    assert second["spend"].tolist() == [
        200,
        300,
        400,
    ]


def test_repeated_indices_survive_multistep_execution() -> None:
    """Pipeline row selection must work with non-unique index labels."""

    dataframe = sample_dataframe()

    dataframe.index = [
        10,
        10,
        20,
        20,
        20,
    ]

    cleaned = run_pipeline(
        dataframe,
        [
            duplicate_step(
                "exact-match",
            ),
            duplicate_step(
                "by-customer",
                columns=["customer_id"],
            ),
        ],
    )

    expected = dataframe.iloc[
        [0, 2, 3]
    ].copy()

    assert_frame_equal(cleaned, expected)
