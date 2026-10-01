
"""Integration tests for NUMPA cleaning execution history.

Tests the complete API workflow:
1. Upload an original dataset.
2. Execute a cleaning pipeline.
3. Verify the cleaned dataset was saved.
4. Retrieve its execution history.
5. Download its PDF report.
6. Verify rollback when history storage fails.

Run with:
    python -m pytest tests/integration/test_cleaning_history.py -v
"""

from __future__ import annotations

from io import BytesIO
from uuid import uuid4

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.api.v1.endpoints import cleaning
from app.services.storage.datasets import (
    delete_dataset,
    get_dataset,
    save_dataset,
)
from app.services.storage.executions import (
    ExecutionStorageLimitError,
    delete_execution,
    get_execution,
)


# ---------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------

@pytest.fixture
def client() -> TestClient:
    """Create a FastAPI test client."""

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def original_dataset():
    """Create a unique dataset and clean it up afterward."""

    dataframe = pd.DataFrame(
        {
            "customer_id": [1, 2, 2, 3, 4],
            "age": [21, 35, 35, 42, 29],
            "city": [
                "Surat",
                "Vadodara",
                "Vadodara",
                "Ahmedabad",
                "Surat",
            ],
        }
    )

    metadata = save_dataset(
        dataframe,
        name=f"integration_{uuid4().hex[:8]}.csv",
    )

    dataset_id = metadata["dataset_id"]

    yield dataset_id, dataframe

    # The original dataset is deleted after any
    # cleaned child datasets have been removed.
    try:
        delete_dataset(dataset_id)
    except (KeyError, LookupError, ValueError):
        pass


def _execution_payload(
    dataset_id: str,
) -> dict:
    """Build a cleaning request using a duplicates step."""

    return {
        "dataset_id": dataset_id,
        "confirmed": True,
        "pipeline": {
            "name": "Integration Test Pipeline",
            "steps": [
                {
                    "step_id": "remove-duplicates",
                    "operation": "duplicates",
                    "parameters": {},
                }
            ],
        },
    }


def _cleanup_execution(
    execution_id: str | None,
    cleaned_dataset_id: str | None,
) -> None:
    """Remove only resources created by the current test."""

    if execution_id:
        try:
            delete_execution(execution_id)
        except (KeyError, LookupError):
            pass

    if cleaned_dataset_id:
        try:
            delete_dataset(cleaned_dataset_id)
        except (KeyError, LookupError):
            pass


# ---------------------------------------------------------
# Test 1: End-to-end cleaning and execution history
# ---------------------------------------------------------

def test_execute_saves_dataset_and_history(
    client: TestClient,
    original_dataset,
) -> None:
    """A successful execution must save both records."""

    dataset_id, original = original_dataset

    execution_id = None
    cleaned_dataset_id = None

    try:
        # Execute a real cleaning pipeline through the API.
        response = client.post(
            "/api/v1/cleaning/execute",
            json=_execution_payload(dataset_id),
        )

        assert response.status_code == 200, response.text

        result = response.json()

        execution_id = result["execution_id"]
        cleaned_dataset_id = result["cleaned_dataset_id"]

        assert result["status"] == "completed"
        assert execution_id
        assert cleaned_dataset_id

        # Verify the cleaned dataset exists in storage.
        cleaned = get_dataset(cleaned_dataset_id)

        assert isinstance(cleaned, pd.DataFrame)
        assert len(cleaned) <= len(original)

        # Verify that the original dataset was not changed.
        stored_original = get_dataset(dataset_id)

        pd.testing.assert_frame_equal(
            stored_original,
            original,
        )

        # Verify automatic execution-history storage.
        saved_execution = get_execution(execution_id)

        assert saved_execution["execution_id"] == execution_id
        assert saved_execution["dataset_id"] == dataset_id
        assert (
            saved_execution["cleaned_dataset_id"]
            == cleaned_dataset_id
        )

        # Verify the history-list API.
        history_response = client.get(
            "/api/v1/cleaning/history",
            params={"dataset_id": dataset_id},
        )

        assert history_response.status_code == 200

        history = history_response.json()

        assert any(
            item["execution_id"] == execution_id
            for item in history["executions"]
        )

        # Verify retrieval through the API.
        detail_response = client.get(
            f"/api/v1/cleaning/history/{execution_id}"
        )

        assert detail_response.status_code == 200
        assert (
            detail_response.json()["execution_id"]
            == execution_id
        )

        # Verify PDF generation from saved history.
        pdf_response = client.get(
            "/api/v1/cleaning/history/"
            f"{execution_id}/report/pdf"
        )

        assert pdf_response.status_code == 200
        assert pdf_response.content.startswith(b"%PDF-")
        assert (
            pdf_response.headers["content-type"]
            == "application/pdf"
        )

    finally:
        _cleanup_execution(
            execution_id,
            cleaned_dataset_id,
        )


# ---------------------------------------------------------
# Test 2: Missing execution IDs
# ---------------------------------------------------------

def test_missing_execution_returns_404(
    client: TestClient,
) -> None:
    """Unknown execution IDs must not return another report."""

    missing_id = f"missing-{uuid4().hex}"

    detail = client.get(
        f"/api/v1/cleaning/history/{missing_id}"
    )

    assert detail.status_code == 404

    pdf = client.get(
        "/api/v1/cleaning/history/"
        f"{missing_id}/report/pdf"
    )

    assert pdf.status_code == 404


# ---------------------------------------------------------
# Test 3: Rollback when execution-history storage fails
# ---------------------------------------------------------

def test_history_failure_rolls_back_cleaned_dataset(
    client: TestClient,
    original_dataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A history-storage failure must not leave a child dataset."""

    dataset_id, original = original_dataset

    # Capture the newly created cleaned dataset ID
    # without changing the real dataset-storage behavior.
    created_ids: list[str] = []

    real_save_dataset = cleaning.save_dataset

    def tracked_save_dataset(*args, **kwargs):
        metadata = real_save_dataset(*args, **kwargs)

        created_ids.append(
            metadata["dataset_id"]
        )

        return metadata

    # Simulate an execution-history capacity failure.
    def failing_save_execution(*args, **kwargs):
        raise ExecutionStorageLimitError(
            "Test: execution storage is full."
        )

    monkeypatch.setattr(
        cleaning,
        "save_dataset",
        tracked_save_dataset,
    )

    monkeypatch.setattr(
        cleaning,
        "save_execution",
        failing_save_execution,
    )

    try:
        response = client.post(
            "/api/v1/cleaning/execute",
            json=_execution_payload(dataset_id),
        )

        # The API should report the storage failure.
        assert response.status_code == 507, response.text

        # The pipeline must have created a cleaned dataset
        # before reaching the simulated history failure.
        assert len(created_ids) == 1

        # Verify that the newly created dataset was removed.
        with pytest.raises((KeyError, LookupError)):
            get_dataset(created_ids[0])

        # The original dataset must remain available.
        stored_original = get_dataset(dataset_id)

        pd.testing.assert_frame_equal(
            stored_original,
            original,
        )

    finally:
        # Defensive cleanup if an assertion fails.
        for cleaned_id in created_ids:
            try:
                delete_dataset(cleaned_id)
            except (KeyError, LookupError):
                pass

