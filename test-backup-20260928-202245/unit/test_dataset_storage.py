
"""NUMPA v2 - Dataset storage and lifecycle safety tests.

Tests:
- Saving and retrieving a dataset.
- Preserving the caller's original dataframe.
- Keeping retrieved datasets independent of stored data.
- Recording cleaned-dataset parent relationships.
- Deleting a dataset.
- Rejecting unknown dataset IDs.
- Keeping separately saved datasets independent.

Uses the existing in-memory dataset storage service.
"""

from __future__ import annotations

from uuid import uuid4

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from app.services.storage.datasets import (
    delete_dataset,
    get_dataset,
    save_dataset,
)


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def sample_dataframe() -> pd.DataFrame:
    """Create a deterministic dataset for storage tests."""

    return pd.DataFrame(
        {
            "customer_id": [101, 102, 103],
            "age": [24.0, None, 38.0],
            "city": ["Surat", "Rajkot", "Ahmedabad"],
        }
    )


def unique_filename(prefix: str = "storage_test") -> str:
    """Avoid name collisions between tests."""

    return f"{prefix}_{uuid4().hex[:10]}.csv"


def stored_dataframe(record) -> pd.DataFrame:
    """Extract a dataframe from the storage service result."""

    if isinstance(record, pd.DataFrame):
        return record

    if isinstance(record, dict):
        for key in ("dataframe", "df", "data"):
            value = record.get(key)

            if isinstance(value, pd.DataFrame):
                return value

    raise AssertionError(
        "get_dataset() did not return a dataframe "
        "or a record containing one."
    )


def save_test_dataset(
    dataframe: pd.DataFrame,
    *,
    kind: str | None = None,
    parent_dataset_id: str | None = None,
) -> dict:
    """Save a test dataset using the public storage API."""

    options = {
        "name": unique_filename(),
    }

    if kind is not None:
        options["kind"] = kind

    if parent_dataset_id is not None:
        options["parent_dataset_id"] = parent_dataset_id

    metadata = save_dataset(
        dataframe,
        **options,
    )

    assert isinstance(metadata, dict)
    assert isinstance(metadata.get("dataset_id"), str)

    return metadata


@pytest.fixture
def saved_dataset():
    """Save a dataset and remove it after each test."""

    original = sample_dataframe()

    metadata = save_test_dataset(original)

    dataset_id = metadata["dataset_id"]

    yield {
        "dataset_id": dataset_id,
        "original": original,
        "metadata": metadata,
    }

    # The dataset may already have been deleted by a test.
    try:
        delete_dataset(dataset_id)
    except LookupError:
        pass


# ---------------------------------------------------------
# Test 1: Save and retrieve
# ---------------------------------------------------------

def test_save_and_retrieve_dataset(saved_dataset) -> None:
    """Stored values must match the uploaded dataframe."""

    dataset_id = saved_dataset["dataset_id"]

    record = get_dataset(dataset_id)

    restored = stored_dataframe(record)

    assert_frame_equal(
        restored,
        saved_dataset["original"],
    )


# ---------------------------------------------------------
# Test 2: Saving must not mutate the input
# ---------------------------------------------------------

def test_save_does_not_modify_input() -> None:
    """Storage must preserve the caller's dataframe."""

    original = sample_dataframe()
    snapshot = original.copy(deep=True)

    metadata = save_test_dataset(original)

    try:
        assert_frame_equal(original, snapshot)

    finally:
        delete_dataset(metadata["dataset_id"])


# ---------------------------------------------------------
# Test 3: Retrieved data must be independent
# ---------------------------------------------------------

def test_retrieved_dataframe_is_defensive_copy(
    saved_dataset,
) -> None:
    """Changing a retrieved dataframe must not alter storage."""

    dataset_id = saved_dataset["dataset_id"]

    first = stored_dataframe(
        get_dataset(dataset_id)
    )

    # Mutate the retrieved dataframe only.
    first.loc[0, "age"] = 999.0

    second = stored_dataframe(
        get_dataset(dataset_id)
    )

    assert_frame_equal(
        second,
        saved_dataset["original"],
    )


# ---------------------------------------------------------
# Test 4: Cleaned dataset lineage
# ---------------------------------------------------------

def test_cleaned_dataset_tracks_parent(
    saved_dataset,
) -> None:
    """A cleaned dataset must retain its parent dataset ID."""

    parent_id = saved_dataset["dataset_id"]

    cleaned = (
        saved_dataset["original"]
        .dropna(subset=["age"])
        .reset_index(drop=True)
    )

    metadata = save_test_dataset(
        cleaned,
        kind="cleaned",
        parent_dataset_id=parent_id,
    )

    child_id = metadata["dataset_id"]

    try:
        assert child_id != parent_id

        # The lineage may be included in the save response.
        # It must identify the original parent.
        assert metadata["parent_dataset_id"] == parent_id

        child = stored_dataframe(
            get_dataset(child_id)
        )

        assert_frame_equal(child, cleaned)

        # Saving a child must not overwrite its parent.
        parent = stored_dataframe(
            get_dataset(parent_id)
        )

        assert_frame_equal(
            parent,
            saved_dataset["original"],
        )

    finally:
        delete_dataset(child_id)


# ---------------------------------------------------------
# Test 5: Delete a dataset
# ---------------------------------------------------------

def test_deleted_dataset_cannot_be_retrieved() -> None:
    """Deletion must make a dataset unavailable."""

    metadata = save_test_dataset(
        sample_dataframe()
    )

    dataset_id = metadata["dataset_id"]

    delete_dataset(dataset_id)

    with pytest.raises(LookupError):
        get_dataset(dataset_id)


# ---------------------------------------------------------
# Test 6: Unknown dataset ID
# ---------------------------------------------------------

def test_unknown_dataset_id_is_rejected() -> None:
    """Retrieving an unknown ID must raise a controlled error."""

    unknown_id = str(uuid4())

    with pytest.raises(LookupError):
        get_dataset(unknown_id)


# ---------------------------------------------------------
# Test 7: Separately saved datasets
# ---------------------------------------------------------

def test_separate_datasets_are_independent() -> None:
    """Two saved datasets must not share mutable data."""

    first_dataframe = sample_dataframe()

    second_dataframe = sample_dataframe()
    second_dataframe.loc[0, "age"] = 55.0

    first_metadata = save_test_dataset(
        first_dataframe
    )

    second_metadata = save_test_dataset(
        second_dataframe
    )

    first_id = first_metadata["dataset_id"]
    second_id = second_metadata["dataset_id"]

    try:
        assert first_id != second_id

        first_saved = stored_dataframe(
            get_dataset(first_id)
        )

        second_saved = stored_dataframe(
            get_dataset(second_id)
        )

        assert_frame_equal(
            first_saved,
            first_dataframe,
        )

        assert_frame_equal(
            second_saved,
            second_dataframe,
        )

    finally:
        delete_dataset(first_id)
        delete_dataset(second_id)
