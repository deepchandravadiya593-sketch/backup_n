
"""NUMPA v2 - Execution history API and edge-case tests.

Tests:
- History listing and pagination.
- Original and cleaned dataset filtering.
- Invalid pagination parameters.
- Missing execution IDs.
- Duplicate execution protection.
- Defensive copies of stored reports.
- Execution-storage capacity limits.

Each test creates unique execution IDs and removes
only the records that it created.
"""

from __future__ import annotations

from copy import deepcopy
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.main import app

from app.services.storage import executions

from app.services.storage.executions import (
    ExecutionAlreadyExistsError,
    ExecutionNotFoundError,
    ExecutionStorageLimitError,
    delete_execution,
    get_execution,
    save_execution,
)


# ---------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------

@pytest.fixture
def client():
    """Provide a FastAPI test client."""

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def execution_factory():
    """Create temporary executions and remove them afterward."""

    created_ids: list[str] = []

    def create(
        *,
        dataset_id: str | None = None,
        cleaned_dataset_id: str | None = None,
    ) -> dict:
        """Create and store a unique execution report."""

        execution_id = str(uuid4())

        report = {
            "execution_id": execution_id,
            "status": "completed",
            "mode": "execute",
            "pipeline_name": "History API Test",
            "dataset_id": (
                dataset_id
                or f"original-{uuid4().hex}"
            ),
            "cleaned_dataset_id": (
                cleaned_dataset_id
                or f"cleaned-{uuid4().hex}"
            ),
            "before": {
                "rows": 100,
                "columns": 4,
            },
            "after": {
                "rows": 95,
                "columns": 4,
            },
            "pipeline_report": {
                "steps": [
                    {
                        "operation": "duplicates",
                        "removed_rows": 5,
                    }
                ]
            },
        }

        saved = save_execution(report)

        created_ids.append(execution_id)

        return saved

    yield create

    # Never clear the entire execution store:
    # other tests may have created their own records.
    for execution_id in reversed(created_ids):
        try:
            delete_execution(execution_id)
        except ExecutionNotFoundError:
            pass


# ---------------------------------------------------------
# Test 1: History pagination
# ---------------------------------------------------------

def test_history_pagination(
    client: TestClient,
    execution_factory,
) -> None:
    """Verify limit, offset and total-count behavior."""

    # A unique dataset ID isolates this test from
    # records created by other tests.
    dataset_id = f"pagination-{uuid4().hex}"

    created = [
        execution_factory(dataset_id=dataset_id)
        for _ in range(5)
    ]

    base = "/api/v1/cleaning/history"

    first_page = client.get(
        base,
        params={
            "dataset_id": dataset_id,
            "limit": 2,
            "offset": 0,
        },
    )

    assert first_page.status_code == 200

    first_data = first_page.json()

    assert first_data["total"] == 5
    assert first_data["limit"] == 2
    assert first_data["offset"] == 0
    assert len(first_data["executions"]) == 2

    second_page = client.get(
        base,
        params={
            "dataset_id": dataset_id,
            "limit": 2,
            "offset": 2,
        },
    )

    assert second_page.status_code == 200

    second_data = second_page.json()

    assert second_data["total"] == 5
    assert len(second_data["executions"]) == 2

    first_ids = {
        item["execution_id"]
        for item in first_data["executions"]
    }

    second_ids = {
        item["execution_id"]
        for item in second_data["executions"]
    }

    # Different pages must not repeat execution IDs.
    assert first_ids.isdisjoint(second_ids)

    last_page = client.get(
        base,
        params={
            "dataset_id": dataset_id,
            "limit": 2,
            "offset": 4,
        },
    )

    assert last_page.status_code == 200

    last_data = last_page.json()

    assert last_data["total"] == 5
    assert len(last_data["executions"]) == 1

    # Every created execution must appear across
    # the three pages.
    all_ids = (
        first_ids
        | second_ids
        | {
            item["execution_id"]
            for item in last_data["executions"]
        }
    )

    expected_ids = {
        item["execution_id"]
        for item in created
    }

    assert all_ids == expected_ids


# ---------------------------------------------------------
# Test 2: Dataset filtering
# ---------------------------------------------------------

def test_history_filters_original_and_cleaned_datasets(
    client: TestClient,
    execution_factory,
) -> None:
    """History must be searchable by both dataset IDs."""

    original_id = f"original-{uuid4().hex}"
    cleaned_id = f"cleaned-{uuid4().hex}"

    saved = execution_factory(
        dataset_id=original_id,
        cleaned_dataset_id=cleaned_id,
    )

    base = "/api/v1/cleaning/history"

    for dataset_id in (
        original_id,
        cleaned_id,
    ):
        response = client.get(
            base,
            params={
                "dataset_id": dataset_id,
            },
        )

        assert response.status_code == 200

        execution_ids = {
            item["execution_id"]
            for item in response.json()["executions"]
        }

        assert saved["execution_id"] in execution_ids

    # A completely unrelated dataset should not
    # return this execution.
    unrelated = client.get(
        base,
        params={
            "dataset_id": f"unrelated-{uuid4().hex}",
        },
    )

    assert unrelated.status_code == 200
    assert unrelated.json()["total"] == 0


# ---------------------------------------------------------
# Test 3: Invalid pagination
# ---------------------------------------------------------

@pytest.mark.parametrize(
    "parameters",
    [
        {"limit": 0},
        {"limit": -1},
        {"limit": 201},
        {"offset": -1},
        {"limit": "invalid"},
        {"offset": "invalid"},
    ],
)
def test_history_rejects_invalid_pagination(
    client: TestClient,
    parameters: dict,
) -> None:
    """Invalid query parameters must return HTTP 422."""

    response = client.get(
        "/api/v1/cleaning/history",
        params=parameters,
    )

    assert response.status_code == 422


# ---------------------------------------------------------
# Test 4: Missing execution
# ---------------------------------------------------------

def test_missing_execution_returns_404(
    client: TestClient,
) -> None:
    """Unknown IDs must return 404 for JSON and PDF."""

    missing_id = f"missing-{uuid4().hex}"

    base = "/api/v1/cleaning/history"

    detail = client.get(
        f"{base}/{missing_id}"
    )

    assert detail.status_code == 404

    pdf = client.get(
        f"{base}/{missing_id}/report/pdf"
    )

    assert pdf.status_code == 404


# ---------------------------------------------------------
# Test 5: Duplicate execution protection
# ---------------------------------------------------------

def test_duplicate_execution_id_is_rejected(
    execution_factory,
) -> None:
    """A stored execution cannot be silently overwritten."""

    saved = execution_factory()

    duplicate = deepcopy(saved)

    duplicate["status"] = "modified"

    with pytest.raises(
        ExecutionAlreadyExistsError
    ):
        save_execution(duplicate)

    # The original report must remain unchanged.
    stored = get_execution(
        saved["execution_id"]
    )

    assert stored["status"] == "completed"


# ---------------------------------------------------------
# Test 6: Defensive copies
# ---------------------------------------------------------

def test_execution_storage_returns_independent_copies(
    execution_factory,
) -> None:
    """Mutating a retrieved report must not change storage."""

    saved = execution_factory()

    execution_id = saved["execution_id"]

    retrieved = get_execution(
        execution_id
    )

    # Modify a nested value in the retrieved copy.
    retrieved["after"]["rows"] = 9999

    retrieved["pipeline_report"]["steps"][0][
        "removed_rows"
    ] = 9999

    stored = get_execution(
        execution_id
    )

    assert stored["after"]["rows"] == 95

    assert (
        stored["pipeline_report"]["steps"][0][
            "removed_rows"
        ]
        == 5
    )


# ---------------------------------------------------------
# Test 7: Execution-storage capacity
# ---------------------------------------------------------

def test_execution_storage_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The storage service must reject writes at capacity.

    This test temporarily reduces the capacity to
    the number of records already present. It does
    not create or delete unrelated records.
    """

    original_count = executions.count_executions()

    # The next save must fail because the temporary
    # capacity equals the current number of records.
    monkeypatch.setattr(
        executions,
        "MAX_EXECUTIONS",
        original_count,
    )

    report = {
        "execution_id": str(uuid4()),
        "status": "completed",
    }

    with pytest.raises(
        ExecutionStorageLimitError
    ):
        save_execution(report)

    assert (
        executions.count_executions()
        == original_count
    )


# ---------------------------------------------------------
# Test 8: API response structure
# ---------------------------------------------------------

def test_history_response_structure(
    client: TestClient,
    execution_factory,
) -> None:
    """Verify fields required by the history UI."""

    saved = execution_factory()

    response = client.get(
        "/api/v1/cleaning/history",
        params={
            "dataset_id": saved["dataset_id"],
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert "executions" in data
    assert "total" in data
    assert "limit" in data
    assert "offset" in data

    item = next(
        entry
        for entry in data["executions"]
        if entry["execution_id"]
        == saved["execution_id"]
    )

    required_fields = {
        "execution_id",
        "status",
        "mode",
        "pipeline_name",
        "dataset_id",
        "cleaned_dataset_id",
        "executed_at",
        "stored_at",
    }

    assert required_fields.issubset(
        item.keys()
    )


# ---------------------------------------------------------
# Test 9: PDF response headers
# ---------------------------------------------------------

def test_saved_execution_pdf_headers(
    client: TestClient,
    execution_factory,
) -> None:
    """A saved report must download as a valid PDF."""

    saved = execution_factory()

    response = client.get(
        "/api/v1/cleaning/history/"
        f"{saved['execution_id']}/report/pdf"
    )

    assert response.status_code == 200, response.text

    assert response.content.startswith(
        b"%PDF-"
    )

    assert (
        response.headers["content-type"]
        == "application/pdf"
    )

    assert "attachment" in (
        response.headers["content-disposition"]
    )

    assert (
        response.headers["cache-control"]
        == "no-store"
    )
