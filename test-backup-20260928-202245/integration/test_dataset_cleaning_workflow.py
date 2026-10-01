
"""NUMPA v2 - Dataset and cleaning workflow integration tests.

Verifies:
1. Cleaning creates a separate child dataset.
2. The original dataset remains unchanged.
3. Parent-child lineage is preserved.
4. Cleaned datasets can be exported as CSV.
5. Validation and statistics work after cleaning.
6. Original datasets cannot be deleted while they
   still have a saved cleaned child.

Run:
    python -m pytest \
        tests/integration/test_dataset_cleaning_workflow.py -v
"""

from __future__ import annotations

from io import BytesIO
from uuid import uuid4

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app

from app.services.storage.datasets import (
    delete_dataset,
    get_dataset,
    get_dataset_metadata,
    save_dataset,
)

from app.services.storage.executions import (
    delete_execution,
)


# ---------------------------------------------------------
# Fixtures
# ---------------------------------------------------------

@pytest.fixture
def client():
    """Provide an isolated FastAPI test client."""

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def workflow_dataset():
    """Create an original dataset for workflow testing."""

    dataframe = pd.DataFrame(
        {
            "customer_id": [101, 102, 102, 103, 104],
            "age": [24, 31, 31, 45, 28],
            "monthly_spend": [
                1200.0,
                1800.0,
                1800.0,
                2400.0,
                1500.0,
            ],
            "city": [
                "Surat",
                "Ahmedabad",
                "Ahmedabad",
                "Vadodara",
                "Rajkot",
            ],
        }
    )

    metadata = save_dataset(
        dataframe,
        name=f"workflow_{uuid4().hex[:8]}.csv",
    )

    original_id = metadata["dataset_id"]

    # Track every resource created by this fixture.
    resources = {
        "original_id": original_id,
        "cleaned_ids": [],
        "execution_ids": [],
        "original_dataframe": dataframe,
    }

    yield resources

    # Remove execution reports before datasets.
    for execution_id in resources["execution_ids"]:
        try:
            delete_execution(execution_id)
        except (KeyError, LookupError):
            pass

    # Remove child datasets before the original.
    for cleaned_id in resources["cleaned_ids"]:
        try:
            delete_dataset(cleaned_id)
        except (KeyError, LookupError):
            pass

    try:
        delete_dataset(original_id)
    except (KeyError, LookupError, ValueError):
        pass


# ---------------------------------------------------------
# Shared workflow helper
# ---------------------------------------------------------

def execute_test_pipeline(
    client: TestClient,
    resources: dict,
) -> dict:
    """Execute duplicate removal and track created resources."""

    original_id = resources["original_id"]

    # Use the duplicate-cleaning operation implemented
    # earlier in the NUMPA cleaning pipeline.
    payload = {
        "dataset_id": original_id,
        "confirmed": True,
        "pipeline": {
            "name": "Dataset Workflow Test",
            "steps": [
                {
                    "step_id": "remove-duplicate-rows",
                    "operation": "duplicates",
                    "config": {},
                }
            ],
        },
    }

    response = client.post(
        "/api/v1/cleaning/execute",
        json=payload,
    )

    assert response.status_code == 200, (
        f"Cleaning failed: {response.text}"
    )

    result = response.json()

    assert result["status"] == "completed"
    assert result["execution_id"]
    assert result["cleaned_dataset_id"]

    # Track the resources immediately so they are
    # removed even if a later assertion fails.
    resources["execution_ids"].append(
        result["execution_id"]
    )

    resources["cleaned_ids"].append(
        result["cleaned_dataset_id"]
    )

    return result


# ---------------------------------------------------------
# Test 1: Original dataset is preserved
# ---------------------------------------------------------

def test_cleaning_preserves_original_dataset(
    client: TestClient,
    workflow_dataset: dict,
) -> None:
    """Cleaning must not overwrite the original data."""

    original_id = workflow_dataset["original_id"]

    expected_original = (
        workflow_dataset["original_dataframe"]
    )

    result = execute_test_pipeline(
        client,
        workflow_dataset,
    )

    cleaned_id = result["cleaned_dataset_id"]

    assert cleaned_id != original_id

    # Verify the original DataFrame is unchanged.
    stored_original = get_dataset(
        original_id
    )

    pd.testing.assert_frame_equal(
        stored_original,
        expected_original,
    )

    # The cleaned result must exist independently.
    cleaned = get_dataset(
        cleaned_id
    )

    assert isinstance(cleaned, pd.DataFrame)

    # The input includes one duplicate record.
    # Duplicate removal should reduce five rows to four.
    assert len(stored_original) == 5
    assert len(cleaned) == 4


# ---------------------------------------------------------
# Test 2: Parent-child lineage
# ---------------------------------------------------------

def test_cleaned_dataset_has_correct_lineage(
    client: TestClient,
    workflow_dataset: dict,
) -> None:
    """The cleaned dataset must reference its parent."""

    original_id = workflow_dataset["original_id"]

    result = execute_test_pipeline(
        client,
        workflow_dataset,
    )

    cleaned_id = result["cleaned_dataset_id"]

    metadata = get_dataset_metadata(
        cleaned_id
    )

    assert metadata["dataset_id"] == cleaned_id
    assert metadata["kind"] == "cleaned"

    assert (
        metadata["parent_dataset_id"]
        == original_id
    )


# ---------------------------------------------------------
# Test 3: Cleaned dataset CSV export
# ---------------------------------------------------------

def test_cleaned_dataset_csv_export(
    client: TestClient,
    workflow_dataset: dict,
) -> None:
    """Exported CSV must contain the cleaned data."""

    result = execute_test_pipeline(
        client,
        workflow_dataset,
    )

    cleaned_id = result["cleaned_dataset_id"]

    response = client.get(
        f"/api/v1/datasets/{cleaned_id}/export"
    )

    assert response.status_code == 200, response.text

    # Read the downloaded CSV back into pandas.
    exported = pd.read_csv(
        BytesIO(response.content)
    )

    stored_cleaned = get_dataset(
        cleaned_id
    )

    # Compare the exported data with the saved data.
    pd.testing.assert_frame_equal(
        exported.reset_index(drop=True),
        stored_cleaned.reset_index(drop=True),
        check_dtype=False,
    )

    assert len(exported) == 4


# ---------------------------------------------------------
# Test 4: Validation after cleaning
# ---------------------------------------------------------

def test_cleaned_dataset_validation(
    client: TestClient,
    workflow_dataset: dict,
) -> None:
    """The validation API must accept cleaned datasets."""

    result = execute_test_pipeline(
        client,
        workflow_dataset,
    )

    cleaned_id = result["cleaned_dataset_id"]

    response = client.get(
        f"/api/v1/datasets/{cleaned_id}/validation"
    )

    assert response.status_code == 200, response.text

    validation = response.json()

    assert isinstance(validation, dict)
    assert validation["rows"] == 4
    assert validation["columns"] == 4

    # The duplicate row in the source dataset
    # should have been removed by the pipeline.
    issue_text = str(
        validation.get("issues", [])
    ).lower()

    assert (
        "duplicate rows" not in issue_text
        and "duplicate_rows" not in issue_text
    )


# ---------------------------------------------------------
# Test 5: Statistics after cleaning
# ---------------------------------------------------------

def test_cleaned_dataset_statistics(
    client: TestClient,
    workflow_dataset: dict,
) -> None:
    """Statistics must be available for cleaned datasets."""

    result = execute_test_pipeline(
        client,
        workflow_dataset,
    )

    cleaned_id = result["cleaned_dataset_id"]

    response = client.get(
        f"/api/v1/datasets/{cleaned_id}/statistics"
    )

    assert response.status_code == 200, response.text

    statistics = response.json()

    assert isinstance(statistics, dict)
    assert statistics

    # The statistics endpoint should return a
    # nonempty report for the cleaned dataset.


# ---------------------------------------------------------
# Test 6: Prevent deletion of a dataset with children
# ---------------------------------------------------------

def test_parent_deletion_is_blocked(
    client: TestClient,
    workflow_dataset: dict,
) -> None:
    """A parent must remain while its cleaned child exists."""

    original_id = workflow_dataset["original_id"]

    result = execute_test_pipeline(
        client,
        workflow_dataset,
    )

    cleaned_id = result["cleaned_dataset_id"]

    # Deleting the original now would break lineage.
    response = client.delete(
        f"/api/v1/datasets/{original_id}"
    )

    assert response.status_code in (400, 409, 422)

    # Both datasets must still exist.
    assert isinstance(
        get_dataset(original_id),
        pd.DataFrame,
    )

    assert isinstance(
        get_dataset(cleaned_id),
        pd.DataFrame,
    )
