
"""NUMPA v2 - Cleaning preview and execution consistency tests.

Tests:
- Preview and execution produce consistent results.
- Preview does not create a cleaned dataset.
- Preview does not create an execution-history record.
- The original dataset remains unchanged.
- Duplicate removal is consistent between modes.
- Multiple cleaning steps behave consistently.
- Preview requests do not require execution confirmation.

Uses the existing cleaning API and in-memory storage.
"""

from __future__ import annotations

from uuid import uuid4

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pandas.testing import assert_frame_equal

from app.main import app
from app.services.storage.datasets import (
    delete_dataset,
    get_dataset,
    save_dataset,
)
from app.services.storage.executions import (
    get_execution,
)


# ---------------------------------------------------------
# Fixtures
# ---------------------------------------------------------

@pytest.fixture
def client():
    """Provide a FastAPI test client."""

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def dataset():
    """Create a dataset with missing values and duplicates."""

    dataframe = pd.DataFrame(
        {
            "age": [
                20.0,
                None,
                30.0,
                30.0,
                45.0,
            ],
            "income": [
                1000.0,
                2000.0,
                3000.0,
                3000.0,
                5000.0,
            ],
            "city": [
                "Surat",
                "Ahmedabad",
                "Rajkot",
                "Rajkot",
                "Vadodara",
            ],
        }
    )

    metadata = save_dataset(
        dataframe,
        name=f"preview_test_{uuid4().hex[:8]}.csv",
    )

    dataset_id = metadata["dataset_id"]

    yield {
        "dataset_id": dataset_id,
        "original": dataframe.copy(deep=True),
    }

    # Remove only the fixture's original dataset.
    try:
        delete_dataset(dataset_id)
    except (KeyError, LookupError, ValueError):
        pass


# ---------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------

def pipeline(
    steps: list[dict],
    *,
    name: str = "Preview Consistency Test",
) -> dict:
    """Build a cleaning pipeline request."""

    return {
        "name": name,
        "steps": steps,
    }


def duplicate_step() -> dict:
    """Return a duplicate-removal pipeline step."""

    return {
        "step_id": "remove-duplicates",
        "operation": "duplicates",
        "parameters": {},
    }


def missing_step() -> dict:
    """Return a mean-imputation pipeline step."""

    return {
        "step_id": "fill-age",
        "operation": "missing_values",
        "parameters": {
            "strategy": "mean",
            "columns": ["age"],
        },
    }


def preview(
    client: TestClient,
    dataset_id: str,
    cleaning_pipeline: dict,
) -> dict:
    """Preview a cleaning pipeline."""

    response = client.post(
        "/api/v1/cleaning/preview",
        json={
            "dataset_id": dataset_id,
            "pipeline": cleaning_pipeline,
            "preview_rows": 10,
        },
    )

    assert response.status_code == 200, response.text

    return response.json()


def execute(
    client: TestClient,
    dataset_id: str,
    cleaning_pipeline: dict,
) -> dict:
    """Execute a confirmed cleaning pipeline."""

    response = client.post(
        "/api/v1/cleaning/execute",
        json={
            "dataset_id": dataset_id,
            "confirmed": True,
            "pipeline": cleaning_pipeline,
        },
    )

    assert response.status_code == 200, response.text

    return response.json()


def get_dataframe(dataset_id: str) -> pd.DataFrame:
    """Retrieve a saved dataframe from dataset storage."""

    stored = get_dataset(dataset_id)

    # Existing storage may return either the dataframe
    # directly or a record containing the dataframe.
    if isinstance(stored, pd.DataFrame):
        return stored

    return stored["dataframe"]


def assert_same_dataset_shape(
    first: dict,
    second: dict,
) -> None:
    """Compare the documented before/after summaries."""

    assert (
        first["before"]["rows"]
        == second["before"]["rows"]
    )

    assert (
        first["before"]["columns"]
        == second["before"]["columns"]
    )

    assert (
        first["after"]["rows"]
        == second["after"]["rows"]
    )

    assert (
        first["after"]["columns"]
        == second["after"]["columns"]
    )


# ---------------------------------------------------------
# Test 1: Duplicate-removal consistency
# ---------------------------------------------------------

def test_preview_matches_duplicate_execution(
    client: TestClient,
    dataset,
) -> None:
    """Preview and execution must agree on duplicate removal."""

    dataset_id = dataset["dataset_id"]

    cleaning_pipeline = pipeline(
        [duplicate_step()]
    )

    preview_result = preview(
        client,
        dataset_id,
        cleaning_pipeline,
    )

    execution_result = execute(
        client,
        dataset_id,
        cleaning_pipeline,
    )

    assert_same_dataset_shape(
        preview_result,
        execution_result,
    )

    cleaned_id = execution_result["cleaned_dataset_id"]

    assert cleaned_id

    cleaned = get_dataframe(cleaned_id)

    assert len(cleaned) == 4


# ---------------------------------------------------------
# Test 2: Preview does not save a dataset
# ---------------------------------------------------------

def test_preview_does_not_save_cleaned_dataset(
    client: TestClient,
    dataset,
) -> None:
    """A preview must not create a cleaned dataset."""

    result = preview(
        client,
        dataset["dataset_id"],
        pipeline([duplicate_step()]),
    )

    assert result["cleaned_dataset_id"] is None


# ---------------------------------------------------------
# Test 3: Preview does not save execution history
# ---------------------------------------------------------

def test_preview_does_not_create_execution_history(
    client: TestClient,
    dataset,
) -> None:
    """Preview execution IDs must not appear in saved history."""

    result = preview(
        client,
        dataset["dataset_id"],
        pipeline([duplicate_step()]),
    )

    execution_id = result["execution_id"]

    # The preview response may contain an execution ID
    # for traceability, but it must not be persisted.
    with pytest.raises(LookupError):
        get_execution(execution_id)


# ---------------------------------------------------------
# Test 4: Original dataset is unchanged
# ---------------------------------------------------------

def test_preview_and_execution_preserve_original(
    client: TestClient,
    dataset,
) -> None:
    """Neither operation may mutate the original dataset."""

    dataset_id = dataset["dataset_id"]

    cleaning_pipeline = pipeline(
        [duplicate_step()]
    )

    preview(
        client,
        dataset_id,
        cleaning_pipeline,
    )

    execute(
        client,
        dataset_id,
        cleaning_pipeline,
    )

    saved_original = get_dataframe(dataset_id)

    assert_frame_equal(
        saved_original,
        dataset["original"],
    )


# ---------------------------------------------------------
# Test 5: Multiple-step consistency
# ---------------------------------------------------------

def test_multistep_preview_matches_execution(
    client: TestClient,
    dataset,
) -> None:
    """Imputation and duplicate removal must agree across modes."""

    dataset_id = dataset["dataset_id"]

    cleaning_pipeline = pipeline(
        [
            missing_step(),
            duplicate_step(),
        ],
        name="Impute Then Deduplicate",
    )

    preview_result = preview(
        client,
        dataset_id,
        cleaning_pipeline,
    )

    execution_result = execute(
        client,
        dataset_id,
        cleaning_pipeline,
    )

    assert_same_dataset_shape(
        preview_result,
        execution_result,
    )

    cleaned = get_dataframe(
        execution_result["cleaned_dataset_id"]
    )

    # The missing age value should be filled.
    assert cleaned["age"].isna().sum() == 0

    # The duplicated Rajkot record should be removed.
    assert len(cleaned) == 4


# ---------------------------------------------------------
# Test 6: Preview needs no confirmation
# ---------------------------------------------------------

def test_preview_does_not_require_confirmation(
    client: TestClient,
    dataset,
) -> None:
    """Preview is a read-only operation."""

    response = client.post(
        "/api/v1/cleaning/preview",
        json={
            "dataset_id": dataset["dataset_id"],
            "pipeline": pipeline(
                [duplicate_step()]
            ),
        },
    )

    assert response.status_code == 200, response.text


# ---------------------------------------------------------
# Test 7: Execution requires confirmation
# ---------------------------------------------------------

def test_execution_requires_confirmation(
    client: TestClient,
    dataset,
) -> None:
    """Execution must reject an unconfirmed request."""

    response = client.post(
        "/api/v1/cleaning/execute",
        json={
            "dataset_id": dataset["dataset_id"],
            "confirmed": False,
            "pipeline": pipeline(
                [duplicate_step()]
            ),
        },
    )

    assert response.status_code == 422

